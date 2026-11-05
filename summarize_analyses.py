"""Summarise run_bc_analyses.sh: training-seed spread, learning curve (training-data fraction),
gravity ablation (validation) and the 2 s hold check (fresh test). Writes <run-dir>/analysis/summary.json
and summary.md."""
import argparse
import json
from pathlib import Path

import numpy as np

from summarize_suite import load, mcnemar_exact, paired, wilson


def val_row(folder):
    m = json.loads((folder / "metrics.json").read_text())
    return dict(success=m["success_1cm"], scenes=m["scenes"], wilson95=wilson(m["success_1cm"], m["scenes"]),
                mean_success_time_s=m["mean_success_time_s"])


def train_info(gru_json):
    d = json.loads(gru_json.read_text())["metrics"]
    keys = ("best_epoch", "epochs_run", "validation_wheel_loss", "validation_arm_loss", "validation_stop_loss",
            "latch_ticks", "training_wall_s")
    return {k: d.get(k) for k in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run, a = args.run_dir, args.run_dir / "analysis"
    grid = {r["name"]: r for r in json.loads((run / "grid_B_park" / "grid.json").read_text())["table"]}
    chosen = json.loads((run / "grid_B_park" / "grid.json").read_text())["chosen"]
    frozen_val = run / "grid_B_park" / chosen["name"]
    n_train_full = len(json.loads((run / "base" / "split.json").read_text())["splits"]["train"])
    out = {}

    # 1. seeds (GRU seed only; split, park MLP and gains fixed)
    seeds = {"42 (frozen)": dict(**val_row(frozen_val), **train_info(run / "gru_B_park" / "gru.json"))}
    for s in (1, 2):
        if (a / f"val_seed{s}" / "metrics.json").exists():
            seeds[str(s)] = dict(**val_row(a / f"val_seed{s}"), **train_info(a / f"seed{s}" / "gru.json"))
    out["seeds"] = seeds

    # 2. learning curve (park/stop MLPs and GRU retrained on nested subsets)
    curve = {}
    for f in ("0.1", "0.25", "0.5"):
        if (a / f"val_frac{f}" / "metrics.json").exists():
            n = len(json.loads((a / f"frac{f}" / "base" / "split.json").read_text())["splits"]["train"])
            curve[f] = dict(train_trajectories=n, **val_row(a / f"val_frac{f}"), **train_info(a / f"frac{f}" / "gru.json"))
    curve["1.0"] = dict(train_trajectories=n_train_full, **seeds["42 (frozen)"])
    out["learning_curve"] = curve

    # 3. gravity ablation, paired with the same gains from the validation grid
    gravity = {}
    for name, ref in (("gravity_zero", chosen["name"]), ("gravity_zero_kp50_kd30_lead0.1", "kp50_kd30_lead0.1")):
        folder = a / f"val_{name}"
        if (folder / "metrics.json").exists():
            with_g, without = load(run / "grid_B_park" / ref), load(folder)
            scenes = sorted(with_g)
            gravity[ref] = dict(with_gravity=val_row(run / "grid_B_park" / ref), without_gravity=val_row(folder),
                                paired=paired(with_g, without, scenes),
                                median_final_error_without_m=float(np.median([r["final_error_m"] for r in without.values()])))
    out["gravity_ablation"] = gravity

    # 4. hold check on the fresh test (unfiltered, 300 scenes)
    hold = {}
    for c in ("B", "v3", "expert", "diag_v3_kp100_kd30_lead0.1", "diag_B_kp100_kd10_lead0.1"):
        folder = a / f"hold_{c}"
        if not (folder / "metrics.json").exists():
            continue
        res = load(folder)
        held = [r["hold"] for r in res.values() if r["success_1cm"] and r.get("hold")]
        full = [h for h in held if h["completed_s"] >= h["requested_s"] - 1e-6]
        worst = np.array([h["max_error_m"] for h in full])
        # Last second of the hold, from the per-step error log (1 ms steps).
        last = np.array([np.load(folder / "rollouts" / i / "rollout.npz")["ee_error"][-1000:].max()
                         for i, r in sorted(res.items())
                         if r["success_1cm"] and r.get("hold") and r["hold"]["completed_s"] >= r["hold"]["requested_s"] - 1e-6])
        hold[c] = dict(scenes=len(res), success_1cm=sum(r["success_1cm"] for r in res.values()),
                       hold_completed=len(full), held_within_1cm=int(sum(h["within_1cm"] for h in full)),
                       held_within_2cm=int(sum(h["within_2cm"] for h in full)),
                       hold_max_error_median_m=float(np.median(worst)) if len(worst) else None,
                       hold_max_error_p95_m=float(np.quantile(worst, 0.95)) if len(worst) else None,
                       end_error_median_m=float(np.median([h["end_error_m"] for h in full])) if full else None,
                       end_error_p95_m=float(np.quantile([h["end_error_m"] for h in full], 0.95)) if full else None,
                       last_second_max_median_m=float(np.median(last)) if len(last) else None,
                       last_second_max_p95_m=float(np.quantile(last, 0.95)) if len(last) else None,
                       last_second_within_1cm=int((last < 0.01).sum()))
        hold[c]["success_and_held_1cm"] = hold[c]["held_within_1cm"]
        hold[c]["wilson95_success_and_held_1cm"] = wilson(hold[c]["held_within_1cm"], len(res))
    ok = lambda r: bool(r.get("success_1cm") and r.get("hold") and r["hold"]["within_1cm"])
    for other in ("expert", "v3"):
        if "B" in hold and other in hold:
            b, e = load(a / "hold_B"), load(a / f"hold_{other}")
            scenes = sorted(b)
            x, y = np.array([ok(b.get(i, {})) for i in scenes]), np.array([ok(e.get(i, {})) for i in scenes])
            only_b, only_e = int((x & ~y).sum()), int((~x & y).sum())
            hold[f"B_vs_{other}_held_1cm"] = dict(only_first=only_b, only_second=only_e,
                                                  mcnemar_p=mcnemar_exact(only_b, only_e))
    out["hold"] = hold
    (a / "summary.json").write_text(json.dumps(out, indent=2) + "\n")

    pct = lambda d: f"{d['success']}/{d['scenes']} ({100 * d['wilson95'][0]:.0f}–{100 * d['wilson95'][1]:.0f}%)"
    L = ["# Additional analyses of variant B (frozen model unchanged)", "",
         "## Training seeds (validation, chosen gains)", "",
         "| GRU seed | 1 cm success (95% CI) | best epoch | val arm loss | latch |", "|---|---|---|---|---|"]
    for s, r in seeds.items():
        L.append(f"| {s} | {pct(r)} | {r['best_epoch']} | {r['validation_arm_loss']:.4g} | {r['latch_ticks']} |")
    L += ["", "## Learning curve (validation, chosen gains; park/stop MLPs and GRU retrained on nested subsets)", "",
          "| fraction | train trajectories | 1 cm success (95% CI) | val wheel loss | val arm loss | val stop loss | train time (min) |",
          "|---|---|---|---|---|---|---|"]
    for f, r in sorted(curve.items(), key=lambda kv: float(kv[0])):
        L.append(f"| {f} | {r['train_trajectories']} | {pct(r)} | {r['validation_wheel_loss']:.4g} | "
                 f"{r['validation_arm_loss']:.4g} | {r['validation_stop_loss']:.4g} | {r['training_wall_s'] / 60:.0f} |")
    L += ["", "## Gravity term of the PD layer (validation, paired)", "",
          "| gains | with gravity | without | discordant with/without (p) | median final error without |", "|---|---|---|---|---|"]
    for g, r in gravity.items():
        p = r["paired"]
        L.append(f"| {g} | {pct(r['with_gravity'])} | {pct(r['without_gravity'])} | "
                 f"{p['only_first']}/{p['only_second']} ({p['mcnemar_p']:.2g}) | {r['median_final_error_without_m']:.4f} m |")
    L += ["", "## Holding after success (fresh test, 2 s more simulation)", "",
          "| controller | 1 cm success | held within 1 cm (whole 2 s) | held within 2 cm | within 1 cm over the last 1 s | "
          "worst error in the last 1 s: median / p95 | error at the end: median / p95 |",
          "|---|---|---|---|---|---|---|"]
    for c in ("B", "v3", "expert", "diag_v3_kp100_kd30_lead0.1", "diag_B_kp100_kd10_lead0.1"):
        if c in hold:
            h = hold[c]
            L.append(f"| {c} | {h['success_1cm']}/{h['scenes']} | {h['held_within_1cm']} | {h['held_within_2cm']} | "
                     f"{h['last_second_within_1cm']} | {1000 * h['last_second_max_median_m']:.1f} / {1000 * h['last_second_max_p95_m']:.1f} mm | "
                     f"{1000 * h['end_error_median_m']:.1f} / {1000 * h['end_error_p95_m']:.1f} mm |")
    L.append("")
    for other in ("expert", "v3"):
        if f"B_vs_{other}_held_1cm" in hold:
            p = hold[f"B_vs_{other}_held_1cm"]
            L.append(f"Success and held within 1 cm for the whole 2 s, B vs {other}: discordant "
                     f"{p['only_first']}/{p['only_second']}, exact McNemar p = {p['mcnemar_p']:.2g}.")
    (a / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
