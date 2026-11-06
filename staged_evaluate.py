"""Evaluate one base/arm combination on a split of the staged model's split.json, in parallel."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent


def wilson(successes, n, z=1.96):
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def summarise(rows):
    n = len(rows)
    finished = [r for r in rows if r["stop_reason"] != "process_error"]
    errors = np.array([r["final_error_m"] for r in finished])
    latched = [r for r in finished if r["base_done_s"] is not None]
    summary = dict(scenes=n, process_errors=n - len(finished))
    for key in ("success_1cm", "success_5cm"):
        k = sum(r[key] for r in finished)
        summary[key] = k
        summary[key.replace("success", "rate")] = k / n
        summary[key.replace("success", "wilson95")] = wilson(k, n)
    summary.update(
        median_final_error_m=float(np.median(errors)) if len(errors) else None,
        mean_success_time_s=(float(np.mean([r["elapsed_s"] for r in finished if r["success_1cm"]]))
                             if any(r["success_1cm"] for r in finished) else None),
        base_latched=len(latched),
        median_base_done_s=float(np.median([r["base_done_s"] for r in latched])) if latched else None,
        median_base_offset_from_rule_m=(float(np.median([r["base_done_offset_from_rule_m"] for r in latched]))
                                        if latched else None),
        median_abs_base_yaw_offset_rad=(float(np.median([abs(r["base_done_yaw_offset_rad"]) for r in latched]))
                                        if latched else None),
        stop_reasons={s: sum(r["stop_reason"] == s for r in rows) for s in sorted({r["stop_reason"] for r in rows})})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory with split.json (and base policy.npz)")
    parser.add_argument("--base", choices=("ours", "expert", "gru", "direct"), required=True)
    parser.add_argument("--arm", choices=("ours", "expert", "gru", "direct"), required=True)
    parser.add_argument("--direct-model", type=Path, help="Plain BC regressor (staged_direct_train.py)")
    parser.add_argument("--fk-model", type=Path, help="Learned forward kinematics (staged_fk_train.py); needs --gravity zero")
    parser.add_argument("--gru-model", type=Path)
    parser.add_argument("--arm-model", type=Path)
    parser.add_argument("--latch-steps", type=int)
    parser.add_argument("--kp", type=float)
    parser.add_argument("--kd", type=float)
    parser.add_argument("--lead", type=float)
    parser.add_argument("--nav-timeout", type=float, help="Force our base latch after this many seconds")
    parser.add_argument("--gains", type=Path, help="grid.json from staged_arm_grid.py: use its chosen kp, kd and lead")
    parser.add_argument("--split", choices=("train", "validation", "test", "test_unfiltered", "excluded"),
                        default="validation",
                        help="test_unfiltered: test plus the expert's failed runs assigned to test; "
                             "excluded: all source scenes outside train/validation/test (the expert's failed runs)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scenes-dir", type=Path,
                        help="Evaluate the scenario JSON files in this directory instead of a split (e.g. OOD targets)")
    for flag, kind in (("--obs-noise", float), ("--torque-scale", float), ("--push-base", float),
                       ("--push-time", float), ("--push-ee", float), ("--perturb-seed", int)):
        parser.add_argument(flag, type=kind, help="Robustness perturbation passed to staged_rollout.py")
    parser.add_argument("--hold-time", type=float, help="Post-success hold evaluation (s)")
    parser.add_argument("--gravity", choices=("model", "zero"), help="GRU torque layer gravity term (ablation)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.arm == "ours" and args.arm_model is None:
        parser.error("--arm ours needs --arm-model")
    if args.gains:
        if any(getattr(args, k) is not None for k in ("kp", "kd", "lead")):
            parser.error("Give either --gains or --kp/--kd/--lead")
        chosen = json.loads(args.gains.read_text())["chosen"]
        args.kp, args.kd, args.lead = chosen["kp"], chosen["kd"], chosen["lead"]
    split = json.loads((args.model_dir / "split.json").read_text())
    source = Path(split["source"])
    if args.split == "excluded":
        used = {name for part in ("train", "validation", "test") for name in split["splits"][part]}
        names = sorted(p.name for p in source.glob("*.npz") if p.name not in used)[:args.limit]
    elif args.split == "test_unfiltered":
        names = (split["splits"]["test"] + split["splits"]["test_expert_failed"])[:args.limit]
    else:
        names = split["splits"][args.split][:args.limit]
    for folder in ("scenes", "rollouts", "logs"):
        (args.output / folder).mkdir(parents=True)
    config = dict(vars(args), scenes=None if args.scenes_dir else names)
    (args.output / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    tasks = []
    if args.scenes_dir:
        scenes = [json.loads(path.read_text()) for path in sorted(args.scenes_dir.glob("*.json"))][:args.limit]
    else:
        scenes = []
        for name in names:
            with np.load(source / name, allow_pickle=False) as data:
                scenes.append(json.loads(str(data["scenario"])))
    for scene in scenes:
        (args.output / "scenes" / f"{scene['id']}.json").write_text(json.dumps(scene))
        tasks.append(scene["id"])

    def run_scene(scene_id):
        command = [sys.executable, str(ROOT / "staged_rollout.py"), "--scene", str(args.output / "scenes" / f"{scene_id}.json"),
                   "--output", str(args.output / "rollouts" / scene_id), "--base", args.base, "--arm", args.arm]
        if args.base == "ours":
            command += ["--base-model", str(args.model_dir / "policy.npz")]
        if args.arm == "ours":
            command += ["--arm-model", str(args.arm_model)]
        if args.base == "gru":
            command += ["--gru-model", str(args.gru_model)]
        if args.base == "direct":
            command += ["--direct-model", str(args.direct_model)]
        if args.latch_steps:
            command += ["--latch-steps", str(args.latch_steps)]
        for flag in ("kp", "kd", "lead", "nav_timeout", "obs_noise", "torque_scale", "push_base",
                     "push_time", "push_ee", "perturb_seed", "gravity", "hold_time", "fk_model"):
            if getattr(args, flag) is not None:
                command += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        with (args.output / "logs" / f"{scene_id}.log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        if process.returncode:
            return dict(id=scene_id, stop_reason="process_error", success_1cm=False, success_5cm=False,
                        final_error_m=None, base_done_s=None)
        return json.loads((args.output / "rollouts" / scene_id / "result.json").read_text())

    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_scene, scene_id): scene_id for scene_id in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            print(f"{len(rows)}/{len(tasks)} {row['id']}: {row['stop_reason']} error={row['final_error_m']}", flush=True)
    ordered = [rows[s] for s in tasks]
    (args.output / "results.json").write_text(json.dumps(ordered, indent=2) + "\n")
    summary = summarise(ordered)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["process_errors"]:
        raise SystemExit("Some simulations failed; inspect logs before interpreting rates.")


if __name__ == "__main__":
    main()
