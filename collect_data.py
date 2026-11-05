import argparse
import csv
import json
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
import numpy as np


ROOT = Path(__file__).resolve().parent

ARM_REFERENCE = [0.0, 1.57, 0.0, 1.0, 0.0, 1.4, 0.0]
SAMPLING = {
        "base_xy": [-0.6, 0.6],
        "yaw_degrees": [-180, 180],
        "arm_delta": [-0.15, 0.15],
        "target_low": [-1.6, -1.6, 0.10],
        "target_high": [1.6, 1.6, 1.3],
}

def generate_scenarios(n, seed):
    limits = SAMPLING
    rng = np.random.default_rng(seed)
    scenes = []
    for i in range(n):
        target = rng.uniform(limits["target_low"], limits["target_high"]).tolist()
        base_pos = rng.uniform(*limits["base_xy"], size=2).tolist() + [0.0]
        yaw = float(rng.uniform(*np.deg2rad(limits["yaw_degrees"])))
        arm_q = (np.array(ARM_REFERENCE) + rng.uniform(*limits["arm_delta"], size=7)).tolist()
        scenes.append({
            "id": f"scene_{seed}_{i + 1:04d}",
            "target": target,
            "base_pos": base_pos,
            "base_bearing": yaw,
            "arm_q": arm_q,
            "arm_qd": [0.0] * 7,
        })
    return {
        "n": n,
        "seed": seed,
        "sampling": {**limits, "arm_reference": ARM_REFERENCE, "base_z": 0.0,
                     "arm_qd": [0.0] * 7, "distribution": "uniform"},
        "scenarios": scenes,
    }


def collect_scene(scene, script, output, env):
    """Run one scene in its own process and return its summary and log."""
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--scene",
            json.dumps(scene),
            "--output",
            str(output),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log = result.stdout.decode("utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(
            f"Scene {scene['id']} failed with exit code {result.returncode}.\n{log}"
        )

    with np.load(output / f"{scene['id']}.npz", allow_pickle=False) as data:
        summary = json.loads(str(data["metadata"]))
    return scene["id"], summary, log


def main():
    parser = argparse.ArgumentParser(description="Generate scenes and collect expert trajectories")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of scenes to simulate concurrently (default: 4)",
    )

    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.gui and args.workers != 1:
        parser.error("--gui requires --workers 1; parallel collection runs headless")

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.output = (
            ROOT / "artifacts"
            / f"collection_{args.n}_seed{args.seed}_{timestamp}"
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"Results will be saved to: {output}", flush=True)
    
    document = generate_scenarios(args.n, args.seed)
    script = ROOT / "robotnik_arm_task" / "robotnik_arm_collect.py"

    env = os.environ.copy()
    env["ROBOENV_HEADLESS"] = "0" if args.gui else "1"
    env["ROBOENV_MAX_STEPS"] = "40000"
    print(
        f"Collecting {args.n} scenes with up to {args.workers} worker(s)...",
        flush=True,
    )

    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["id", "success", "stop_reason", "final_error_m"],
        )
        writer.writeheader()
        stream.flush()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(collect_scene, scene, script, output, env): scene
                for scene in document["scenarios"]
            }

            failure = None
            try:
                for future in as_completed(futures):
                    if future.cancelled():
                        continue
                    try:
                        scene_id, summary, log = future.result()
                    except Exception as exc:
                        if failure is None:
                            failure = exc
                            # Do not start more scenes after a child fails. Other
                            # active scenes may finish and will still be recorded.
                            for pending in futures:
                                pending.cancel()
                        continue

                    if log:
                        print(log, end="" if log.endswith("\n") else "\n", flush=True)
                    writer.writerow({"id": scene_id, **summary})
                    stream.flush()
                    print(f"{scene_id}: {summary}", flush=True)
            except BaseException:
                # Stop queued scenes after an interrupt. Already-running
                # simulations finish before the executor exits.
                for future in futures:
                    future.cancel()
                raise
            if failure is not None:
                raise failure


if __name__ == "__main__":
    main()
