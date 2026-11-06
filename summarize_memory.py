"""Summarise run_memory_study.sh: what does the cloned policy need to remember?

Plain BC (L0, run_direct_baseline.sh: linear and MLP regressors from the 21 raw inputs to the 11 raw
actions, no structure; "done" = all wheel commands below 0.05 rad/s, as for L2) and three levels, all trained on the final run's data and split and evaluated on the same scenes:
  L1 four single-frame MLPs + hand-coded latch on the stop MLP (no learned memory),
  L2 GRU without stop head (implicit memory: the stage must live in the hidden state),
  L3 variant B: GRU + stop head + latch (explicit memory).
Per level: 1 cm successes on 208 validation and 300 fresh scenes (Wilson 95% CI), paired exact McNemar
against B, and failure mechanisms measured from the logged roll-outs:
  never_done       the base never reaches its "done" state (latch never fires / wheels never below
                   0.05 rad/s);
  restarts         after the first "done" the wheels start again (impossible with a latch);
  arm_drift_rad    largest joint deviation from the start pose before the first "done" (the arm
                   should wait while the base drives);
  creep_m          base displacement between the first "done" and the end of the episode.
Also summarises the direct-torque ablation (B's GRU, stop head and inputs, arm head regressing
recorded torques with no PD layer) on validation, if present.
Writes <run-dir>/memory/summary.json and summary.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from summarize_suite import load, paired, wilson

LEVELS = {"L0lin": "Plain BC, linear (ridge) regressor, raw inputs -> raw actions",
          "L0mlp": "Plain BC, MLP regressor, raw inputs -> raw actions",
          "L1": "Four single-frame MLPs + latch", "L2": "GRU, no stop head", "L3": "B: GRU + stop head + latch"}


def mechanisms(folder, scenes):
    drift, creep, restarts, never_done = [], [], 0, 0
    for scene in scenes:
        with np.load(folder / "rollouts" / scene / "rollout.npz", allow_pickle=False) as d:
            q, base, done = d["inputs"][:, 4:11], d["inputs"][:, :2], d["done"].astype(bool)
        if not done.any():
            never_done += 1
            drift.append(float(np.abs(q - q[0]).max()))
            continue
        k = int(np.argmax(done))
        drift.append(float(np.abs(q[:k + 1] - q[0]).max()))
        creep.append(float(np.linalg.norm(base[-1] - base[k])))
        restarts += int((~done[k:]).any())
    return dict(never_done=never_done, restarts=restarts,
                arm_drift_rad_median=float(np.median(drift)), arm_drift_rad_p90=float(np.quantile(drift, 0.9)),
                creep_m_median=float(np.median(creep)) if creep else None,
                creep_m_p90=float(np.quantile(creep, 0.9)) if creep else None)


def row(folder, ref):
    res = load(folder)
    scenes = sorted(res)
    ok = [res[s] for s in scenes if res[s]["success_1cm"]]
    failed = [s for s in scenes if not res[s]["success_1cm"]]
    return dict(success=len(ok), scenes=len(scenes), wilson95=wilson(len(ok), len(scenes)),
                mean_success_time_s=float(np.mean([r["elapsed_s"] for r in ok])) if ok else None,
                vs_B=paired(res, ref, scenes) if ref is not None else None,
                all=mechanisms(folder, scenes), failures=mechanisms(folder, failed) if failed else None,
                failed_scenes=failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run, m = args.run_dir, args.run_dir / "memory"
    chosen = json.loads((run / "grid_B_park" / "grid.json").read_text())["chosen"]
    d = run / "direct"  # run_direct_baseline.sh
    folders = {"validation": dict(L0lin=d / "val_linear", L0mlp=d / "val_mlp",
                                  L1=m / "val_fournet", L2=m / "val_gru_nostop", L3=run / "grid_B_park" / chosen["name"]),
               "fresh": dict(L0lin=d / "fresh_linear", L0mlp=d / "fresh_mlp", L1=m / "fresh_fournet", L2=m / "fresh_gru_nostop", L3=run / "suite" / "fresh_B")}
    out = dict(levels=LEVELS, gains=dict(
        L1=json.loads((m / "grid_fournet_arm" / "grid.json").read_text())["chosen"], L2=chosen, L3=chosen))
    for split, f in folders.items():
        ref = load(f["L3"])
        out[split] = {lvl: row(path, None if lvl == "L3" else ref) for lvl, path in f.items()
                      if (path / "rollouts").is_dir()}
    # Open-loop fit of the plain regressors on recorded validation frames (R^2 per output group).
    out["L0_open_loop"] = {}
    for kind in ("linear", "mlp"):
        if (d / f"{kind}.json").exists():
            info = json.loads((d / f"{kind}.json").read_text())
            out["L0_open_loop"][kind] = dict(parameters=info["parameters"],
                                             **{k: info["open_loop_validation"][k] for k in ("r2_mean_wheels", "r2_mean_torques", "r2")})
    # Action-representation ablation (B with direct torque output instead of M e + PD), validation only.
    if (m / "val_gru_torque" / "rollouts").is_dir():
        t = load(m / "val_gru_torque")
        k = sum(r["success_1cm"] for r in t.values())
        out["torque_validation"] = dict(success=k, scenes=len(t), wilson95=wilson(k, len(t)),
                                        vs_B=paired(t, load(folders["validation"]["L3"]), sorted(t)),
                                        median_min_error_m=float(np.median([r["min_error_m"] for r in t.values()])))
    (m / "summary.json").write_text(json.dumps(out, indent=2) + "\n")

    lines = ["# Memory study", "", "| Level | Split | Success | 95% CI | vs B (only level / only B, p) | "
             "never done | restarts | arm drift before done, median (rad) | creep after done, median (cm) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for split in folders:
        for lvl, r in out[split].items():
            v = r["vs_B"]
            cmp = f"{v['only_first']}/{v['only_second']}, p={v['mcnemar_p']:.2g}" if v else "--"
            a = r["all"]
            creep = f"{100 * a['creep_m_median']:.1f}" if a["creep_m_median"] is not None else "--"
            lines.append(f"| {lvl} {LEVELS[lvl]} | {split} | {r['success']}/{r['scenes']} | "
                         f"{100 * r['wilson95'][0]:.1f}-{100 * r['wilson95'][1]:.1f}% | {cmp} | {a['never_done']} | "
                         f"{a['restarts']} | {a['arm_drift_rad_median']:.3f} | {creep} |")
    if "torque_validation" in out:
        t = out["torque_validation"]
        lines += ["", f"Direct-torque ablation (validation): {t['success']}/{t['scenes']}, "
                      f"median closest approach {100 * t['median_min_error_m']:.1f} cm"]
    for kind, r in out["L0_open_loop"].items():
        lines += [f"L0 {kind} open-loop validation R^2: wheels {r['r2_mean_wheels']:.3f}, torques {r['r2_mean_torques']:.3f}"]
    lines += ["", "Gains: " + ", ".join(f"{k} {v['kp']:g}/{v['kd']:g}/{v['lead']:g}" for k, v in out["gains"].items())]
    (m / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
