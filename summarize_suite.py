"""Summarise run_evaluation_suite.sh. For every condition: 1 cm success of variant B, v3 (if run) and the
expert with Wilson 95% CIs, paired exact McNemar tests (B vs expert, B vs v3), median final error, and
per-step controller compute time; plus model size. Writes <run-dir>/suite/summary.json and summary.md."""
import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

CONTROLLERS = ("B", "v3", "expert")


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n))


def load(folder):
    rows = [json.loads(p.read_text()) for p in sorted((folder / "rollouts").glob("*/result.json"))]
    return {r["id"]: r for r in rows}


def paired(a, b, scenes):
    x = np.array([a.get(i, {}).get("success_1cm", False) for i in scenes], dtype=bool)
    y = np.array([b.get(i, {}).get("success_1cm", False) for i in scenes], dtype=bool)
    only_a, only_b = int((x & ~y).sum()), int((~x & y).sum())
    return dict(only_first=only_a, only_second=only_b, mcnemar_p=mcnemar_exact(only_a, only_b))


def model_info(path):
    with np.load(path, allow_pickle=False) as data:
        gru = int(sum(data[k].size for k in ("w_ih", "w_hh", "b_ih", "b_hh", "w_out", "b_out")))
        park = int(sum(data[k].size for k in data.files if k[:6] in ("park_w", "park_b")))
    return dict(file=str(path), file_kib=path.stat().st_size / 1024, gru_parameters=gru, park_mlp_parameters=park)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    suite = args.run_dir / "suite"
    conditions = sorted({p.name.rsplit("_", 1)[0] for p in suite.iterdir()
                         if p.is_dir() and p.name.rsplit("_", 1)[-1] in CONTROLLERS},
                        key=lambda n: (not n.startswith("fresh"), n))
    rows = []
    for cond in conditions:
        results = {c: load(suite / f"{cond}_{c}") for c in CONTROLLERS if (suite / f"{cond}_{c}").is_dir()}
        scenes = sorted(json.loads(p.read_text())["id"] for p in (suite / f"{cond}_B" / "scenes").glob("*.json"))
        n = len(scenes)
        row = dict(condition=cond, scenes=n)
        for c, res in results.items():
            if c != "expert" and any(r.get("expert_module_loaded") for r in res.values()):
                raise RuntimeError(f"{cond}_{c}: expert module loaded in a learned rollout")
            k = sum(res.get(i, {}).get("success_1cm", False) for i in scenes)
            row[c] = dict(success=int(k), rate=k / n, wilson95=wilson(int(k), n), process_errors=n - len(res),
                          median_final_error_m=float(np.median([r["final_error_m"] for r in res.values()])),
                          mean_success_time_s=(float(np.mean([r["elapsed_s"] for r in res.values() if r["success_1cm"]]))
                                               if k else None),
                          policy_time_mean_us=float(np.mean([r["policy_time_mean_us"] for r in res.values()])))
        if "expert" in results:
            row["B_vs_expert"] = paired(results["B"], results["expert"], scenes)
        if "v3" in results:
            row["B_vs_v3"] = paired(results["B"], results["v3"], scenes)
        rows.append(row)

    models = {name: model_info(args.run_dir / f"gru_{folder}" / "gru.npz")
              for name, folder in (("B", "B_park"), ("v3", "v3_bearing"))}
    (suite / "summary.json").write_text(json.dumps(dict(conditions=rows, models=models), indent=2) + "\n")

    pct = lambda d: f"{d['success']} ({100 * d['rate']:.1f}%, {100 * d['wilson95'][0]:.0f}–{100 * d['wilson95'][1]:.0f})"
    lines = ["# Evaluation suite (main: B, baseline: v3, reference: expert)", "",
             "1 cm successes (rate, Wilson 95% CI). McNemar: exact two-sided, discordant pairs shown as first/second.", "",
             "| condition | n | B | v3 | expert | B vs expert (p) | B vs v3 (p) | median err B / expert (m) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cell = lambda c: pct(r[c]) if c in r else "-"
        test = lambda k: (f"{r[k]['only_first']}/{r[k]['only_second']} ({r[k]['mcnemar_p']:.2g})" if k in r else "-")
        lines.append(f"| {r['condition']} | {r['scenes']} | {cell('B')} | {cell('v3')} | {cell('expert')} | "
                     f"{test('B_vs_expert')} | {test('B_vs_v3')} | {r['B']['median_final_error_m']:.4f} / "
                     f"{r['expert']['median_final_error_m'] if 'expert' in r else float('nan'):.4f} |")
    fresh = next((r for r in rows if r["condition"] == "fresh"), None)
    lines.append("")
    if fresh:
        lines.append("Controller compute per 1 kHz step on the fresh test (features, GRU update every 10th step, "
                     "PD + gravity; expert: IK + PD + gravity): "
                     + ", ".join(f"{c} {fresh[c]['policy_time_mean_us']:.0f} µs" for c in CONTROLLERS if c in fresh) + ".")
    for name, m in models.items():
        lines.append(f"Model {name}: {m['file_kib']:.0f} KiB, GRU {m['gru_parameters']} parameters"
                     + (f" + park MLP {m['park_mlp_parameters']}" if m["park_mlp_parameters"] else "") + ".")
    (suite / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
