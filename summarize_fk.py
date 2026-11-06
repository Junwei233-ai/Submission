"""Summarise the B-FK runs (learned forward kinematics, no gravity term) in results/final_run/fk_run/.

Reads the per-scene results.json that run_fk_study.sh (and the hold run) copy into fk_run/<run>/, plus
two per-scene references from the URDF study in fk_run/reference/ (variant B on the fresh test and the
expert's 2 s hold run), and writes fk_run/summary.json and summary.md: success with Wilson 95% CIs,
exact McNemar tests on identical scenes, memory-study mechanisms, hold statistics and compute time.
All runs summarised here were produced on one machine; closed-loop outcomes of borderline scenes can
differ across platforms (floating point), see README Section 7.1.1.

Usage: python summarize_fk.py [--fk-run results/final_run/fk_run] [--rollouts artifacts/fk_study]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from summarize_memory import mechanisms
from summarize_suite import paired, wilson

ROOT = Path(__file__).resolve().parent
CONDITIONS = ["noise003", "noise010", "torque080", "pushee30", "pushbase20"]
FAMILIES = ["far_target", "high_target", "low_target", "wide_start"]


def load(folder):
    path = folder / "results.json"
    return {r["id"]: r for r in json.loads(path.read_text())} if path.exists() else None


def row(res, ref=None):
    k = sum(r["success_1cm"] for r in res.values())
    ok = [r["elapsed_s"] for r in res.values() if r["success_1cm"]]
    out = dict(success=k, scenes=len(res), wilson95=wilson(k, len(res)),
               mean_success_time_s=float(np.mean(ok)) if ok else None,
               policy_time_mean_us=float(np.mean([r["policy_time_mean_us"] for r in res.values()])))
    if ref is not None:
        out["paired"] = paired(res, ref, sorted(res))
    return out


def hold_stats(res):
    full = [r for r in res.values() if r["success_1cm"] and r.get("hold")
            and r["hold"]["completed_s"] >= r["hold"]["requested_s"] - 1e-6]
    return dict(success_1cm=sum(r["success_1cm"] for r in res.values()), hold_completed=len(full),
                held_within_1cm=int(sum(r["hold"]["within_1cm"] for r in full)),
                hold_max_error_median_m=float(np.median([r["hold"]["max_error_m"] for r in full])),
                end_error_median_m=float(np.median([r["hold"]["end_error_m"] for r in full])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fk-run", type=Path, default=ROOT / "results" / "final_run" / "fk_run")
    parser.add_argument("--rollouts", type=Path, default=ROOT / "artifacts" / "fk_study",
                        help="Full roll-outs of run_fk_study.sh, only for the memory mechanisms (optional)")
    args = parser.parse_args()
    f = args.fk_run
    run = {p.name: load(p) for p in f.iterdir() if p.is_dir() and (p / "results.json").exists()}
    ref = {p.name: load(p) for p in (f / "reference").iterdir() if (p / "results.json").exists()}
    for name, res in ref.items():  # expert runs are not affected by the FK model; reused from the URDF study
        if name.endswith("_expert"):
            run.setdefault(name, res)
    out = {}
    # Accuracy, against the expert and against variant B (URDF) on the same 300 fresh scenes.
    out["fresh"] = dict(B_FK=row(run["test_L3"], run.get("test_expert")),
                        expert=row(run["test_expert"]) if "test_expert" in run else None,
                        B_FK_vs_B_urdf=paired(run["test_L3"], ref["fresh_B_urdf"], sorted(run["test_L3"]))
                        if "fresh_B_urdf" in ref else None)
    out["validation"] = dict(B_FK=row(run["val_L3"]))
    # Ablation: the GRU trained on recorded (URDF) end-effector positions, run with the learned FK.
    u = f / "urdf_trained_gru"
    out["gru_trained_on_urdf_ee"] = {s: row(load(u / n)) for s, n in (("validation", "val_L3"), ("fresh", "test_L3"))
                                     if (u / n / "results.json").exists()}
    # Memory study with the learned forward kinematics (paired against L3 = B-FK).
    out["memory"] = {}
    for split, prefix in (("validation", "val"), ("fresh", "test")):
        for lvl in ("L1", "L2", "L3"):
            name = f"{prefix}_{lvl}"
            if name in run:
                r = row(run[name], None if lvl == "L3" else run[f"{prefix}_L3"])
                folder = args.rollouts / name
                if (folder / "rollouts").is_dir():
                    r["mechanisms"] = mechanisms(folder, sorted(run[name]))
                out["memory"][f"{split}_{lvl}"] = r
    # Robustness and out-of-distribution, paired against the expert on identical scenes and perturbations.
    for key, names in (("robustness", CONDITIONS), ("ood", FAMILIES)):
        out[key] = {}
        for c in names:
            stem = f"robust_{c}" if key == "robustness" else f"ood_{c}"
            if f"{stem}_L3" in run:
                out[key][c] = dict(B_FK=row(run[f"{stem}_L3"], run.get(f"{stem}_expert")),
                                   expert=row(run[f"{stem}_expert"]) if f"{stem}_expert" in run else None)
    # Holding for 2 s after success (fresh test), against the expert's hold run.
    if "hold_L3" in run:
        out["hold"] = dict(B_FK=hold_stats(run["hold_L3"]))
        if "hold_expert" in ref:
            e = ref["hold_expert"]
            out["hold"]["expert"] = hold_stats(e)
            ok = lambda r: bool(r["success_1cm"] and r.get("hold") and r["hold"]["within_1cm"])
            out["hold"]["B_FK_vs_expert_held"] = paired({k: dict(success_1cm=ok(v)) for k, v in run["hold_L3"].items()},
                                                        {k: dict(success_1cm=ok(v)) for k, v in e.items()},
                                                        sorted(run["hold_L3"]))
    fk = json.loads((f / "fk.json").read_text())
    out["fk_model"] = {k: fk[k] for k in ("validation_final_frame", "validation_after_stop", "best_epoch")}
    with np.load(f / "fk.npz") as d:
        out["fk_model"]["parameters"] = int(sum(d[k].size for k in d.files if k[3] in "wb"))
    (f / "summary.json").write_text(json.dumps(out, indent=2) + "\n")

    fmt = lambda r: f"{r['success']}/{r['scenes']} ({100 * r['wilson95'][0]:.1f}–{100 * r['wilson95'][1]:.1f}%)"
    pv = lambda p: f"{p['only_first']}/{p['only_second']}, p = {p['mcnemar_p']:.2g}" if p else "–"
    lines = ["# B-FK: learned forward kinematics, no gravity term", "",
             f"Fresh test: B-FK {fmt(out['fresh']['B_FK'])}, expert "
             f"{fmt(out['fresh']['expert']) if out['fresh']['expert'] else '–'}; B-FK vs expert {pv(out['fresh']['B_FK'].get('paired'))}; "
             f"B-FK vs B (URDF) {pv(out['fresh']['B_FK_vs_B_urdf'])}.",
             f"Validation: B-FK {fmt(out['validation']['B_FK'])}.",
             f"Compute per 1 kHz step (fresh test, mean): B-FK {out['fresh']['B_FK']['policy_time_mean_us']:.0f} µs.", "",
             "## Memory study (learned FK for every level)", "",
             "| Level | Split | Success | vs L3 (only level / only L3) | never done | restarts |", "|---|---|---|---|---|---|"]
    for key, r in out["memory"].items():
        split, lvl = key.split("_")
        m = r.get("mechanisms", {})
        lines.append(f"| {lvl} | {split} | {fmt(r)} | {pv(r.get('paired'))} | {m.get('never_done', '–')} | {m.get('restarts', '–')} |")
    for key, title in (("robustness", "Robustness (60 fresh scenes)"), ("ood", "Out of distribution (40 scenes each)")):
        lines += ["", f"## {title}", "", "| Condition | B-FK | Expert | B-FK vs expert |", "|---|---|---|---|"]
        for c, r in out[key].items():
            lines.append(f"| {c} | {r['B_FK']['success']} | {r['expert']['success'] if r['expert'] else '–'} | {pv(r['B_FK'].get('paired'))} |")
    if "hold" in out:
        h = out["hold"]
        lines += ["", "## Holding for 2 s after success (fresh test)", "",
                  f"B-FK: {h['B_FK']['held_within_1cm']} of {h['B_FK']['success_1cm']} successes stay within 1 cm"]
        if "expert" in h:
            lines[-1] += (f"; expert: {h['expert']['held_within_1cm']} of {h['expert']['success_1cm']}; "
                          f"success and held, B-FK vs expert {pv(h['B_FK_vs_expert_held'])}.")
    m = out["fk_model"]
    lines += ["", f"Forward-kinematics MLP: {m['parameters']:,} parameters; validation final-frame error median "
                  f"{m['validation_final_frame']['median_mm']:.1f} mm, p95 {m['validation_final_frame']['p95_mm']:.1f} mm, "
                  f"max {m['validation_final_frame']['max_mm']:.1f} mm."]
    (f / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
