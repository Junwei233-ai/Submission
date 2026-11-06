"""Choose the arm PD gains and reference lead as hyperparameters on validation scenes.

Each combination runs the expert base with our arm, or with --gru-model the full GRU policy
(staged_evaluate.py). The selection rule is fixed before running: most 1 cm successes, then
shortest mean success time, then lowest mean fraction of torque-clipped steps.

With --fk-model the search uses the final configuration: learned forward kinematics and no gravity term.
--min-kp-lead then skips combinations whose largest reference torque, Kp x lead, cannot hold the arm up.
"""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--arm-model", type=Path, help="Staged arm .npz (runs with the expert base)")
    parser.add_argument("--gru-model", type=Path, help="GRU policy .npz (drives base and arm)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kp", type=float, nargs="+", default=[50, 100, 200])
    parser.add_argument("--kd", type=float, nargs="+", default=[10, 30])
    parser.add_argument("--lead", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--limit", type=int, default=40, help="Validation scenes per combination (0 = all)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--fk-model", type=Path, help="Learned forward kinematics; the search then has no gravity term")
    parser.add_argument("--min-kp-lead", type=float, default=0.0, help="Skip combinations with Kp x lead below this (N m)")
    parser.add_argument("--scenes-dir", type=Path, help="Scene files instead of the validation split")
    args = parser.parse_args()
    if (args.arm_model is None) == (args.gru_model is None):
        parser.error("Give exactly one of --arm-model or --gru-model")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.output.mkdir(parents=True)
    rows = []
    for kp, kd, lead in itertools.product(args.kp, args.kd, args.lead):
        if kp * lead < args.min_kp_lead:
            continue
        name = f"kp{kp:g}_kd{kd:g}_lead{lead:g}"
        run = args.output / name
        if args.gru_model:
            policy = ["--base", "gru", "--arm", "gru", "--gru-model", str(args.gru_model)]
        else:
            policy = ["--base", "expert", "--arm", "ours", "--arm-model", str(args.arm_model)]
        limit = ["--limit", str(args.limit)] if args.limit else []
        learned = ["--fk-model", str(args.fk_model), "--gravity", "zero"] if args.fk_model else []
        scenes = ["--scenes-dir", str(args.scenes_dir)] if args.scenes_dir else ["--split", "validation"]
        subprocess.run([sys.executable, str(ROOT / "staged_evaluate.py"), "--model-dir", str(args.model_dir),
                        *policy, *learned, *scenes, *limit, "--workers", str(args.workers),
                        "--kp", str(kp), "--kd", str(kd), "--lead", str(lead), "--output", str(run)],
                       stdout=subprocess.DEVNULL, check=True)
        metrics = json.loads((run / "metrics.json").read_text())
        results = json.loads((run / "results.json").read_text())
        row = dict(name=name, kp=kp, kd=kd, lead=lead, success_1cm=metrics["success_1cm"],
                   success_5cm=metrics["success_5cm"], scenes=metrics["scenes"],
                   mean_success_time_s=metrics["mean_success_time_s"],
                   mean_clipped_fraction=float(np.mean([r["clipped_step_fraction"] for r in results])),
                   median_min_error_m=float(np.median([r["min_error_m"] for r in results])))
        rows.append(row)
        print(json.dumps(row), flush=True)
    key = lambda r: (-r["success_1cm"], r["mean_success_time_s"] or np.inf, r["mean_clipped_fraction"])
    rows.sort(key=key)
    summary = dict(rule="max 1cm successes, then min mean success time, then min mean clipped fraction",
                   chosen=rows[0], table=rows)
    (args.output / "grid.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("chosen:", json.dumps(rows[0]), flush=True)


if __name__ == "__main__":
    main()
