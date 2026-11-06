"""Set the run-time arm torque limit: the largest torque magnitude in the training demonstrations.

Reads every recorded frame of the training trajectories listed in split.json and writes
torque_limit.json, which staged_rollout.py loads. One value is used for all joints.

Usage: python compute_torque_limit.py --model-dir artifacts/final_run/base [--source COLLECTION_DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory with split.json")
    parser.add_argument("--source", type=Path, help="Collection directory (default: the one in split.json)")
    parser.add_argument("--output", type=Path, default=ROOT / "torque_limit.json")
    args = parser.parse_args()
    split = json.loads((args.model_dir / "split.json").read_text())
    source = args.source or Path(split["source"])
    names = split["splits"]["train"]
    limit = 0.0
    for name in names:
        with np.load(source / name, allow_pickle=False) as data:
            limit = max(limit, float(np.abs(data["actions"][:, 4:11]).max()))
    args.output.write_text(json.dumps(dict(torque_limit=limit, training_trajectories=len(names)), indent=2) + "\n")
    print(f"torque limit {limit} N m from {len(names)} training trajectories -> {args.output}")


if __name__ == "__main__":
    main()
