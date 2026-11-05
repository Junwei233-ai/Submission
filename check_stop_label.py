"""Verification only (never used for training): does the public stop label match the expert's state?

The learner's stop label is staged_data.stop_index: the first step after which every recorded wheel
command stays zero, computed from recorded actions only. The collector also saved the expert's internal
navigation phase ("align", "drive", "orient", "done"), which the training code never reads. This script
compares the two on every successful demonstration of the ORIGINAL collection (the copy that still has
the phase field) and writes a small JSON report.

Usage: python check_stop_label.py --collection DIR [--output stop_label_check.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from staged_data import stop_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True, help="Collection with the 'phase' field")
    parser.add_argument("--output", type=Path, default=Path("stop_label_check.json"))
    args = parser.parse_args()
    offsets, checked = [], 0
    for path in sorted(args.collection.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "phase" not in data.files or not json.loads(str(data["metadata"])).get("success", False):
                continue
            phase, actions = data["phase"], data["actions"]
        done = np.flatnonzero(phase == "done")
        expert = int(done[0]) if len(done) else len(phase)
        offsets.append(stop_index(actions) - expert)
        checked += 1
    offsets = np.asarray(offsets)
    report = dict(trajectories=checked, exact_match=int((offsets == 0).sum()),
                  within_1_step=int((np.abs(offsets) <= 1).sum()), within_10_steps=int((np.abs(offsets) <= 10).sum()),
                  offset_steps_min=int(offsets.min()), offset_steps_max=int(offsets.max()),
                  offset_steps_median=float(np.median(offsets)))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
