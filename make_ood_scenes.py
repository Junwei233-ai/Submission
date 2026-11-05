"""Write out-of-distribution scenario JSON files (one directory per family) for staged_evaluate.py --scenes-dir.

Training scenes (collect_data.SAMPLING): target x, y in [-1.6, 1.6], z in [0.10, 1.30]; base xy in
[-0.6, 0.6]; arm angles = reference +- 0.15 rad. Each family moves exactly one factor outside that range
and keeps the others in-distribution:
  far_target  : max(|x|, |y|) in (1.6, 2.2]
  low_target  : z in [0.03, 0.10)
  high_target : z in (1.30, 1.40]
  wide_start  : base xy in [-1.2, 1.2] outside [-0.6, 0.6] on at least one axis; arm +-0.30 rad
"""
import argparse
import json
from pathlib import Path

import numpy as np

from collect_data import ARM_REFERENCE, SAMPLING


def in_dist(rng):
    return dict(target=rng.uniform(SAMPLING["target_low"], SAMPLING["target_high"]),
                base=np.append(rng.uniform(*SAMPLING["base_xy"], size=2), 0.0),
                yaw=float(rng.uniform(-np.pi, np.pi)),
                arm=np.array(ARM_REFERENCE) + rng.uniform(*SAMPLING["arm_delta"], size=7))


def family(name, rng):
    s = in_dist(rng)
    if name == "far_target":
        while np.max(np.abs(s["target"][:2])) <= 1.6:
            s["target"][:2] = rng.uniform(-2.2, 2.2, size=2)
    elif name == "low_target":
        s["target"][2] = rng.uniform(0.03, 0.10)
    elif name == "high_target":
        s["target"][2] = rng.uniform(1.30, 1.40)
    elif name == "wide_start":
        while np.max(np.abs(s["base"][:2])) <= 0.6:
            s["base"][:2] = rng.uniform(-1.2, 1.2, size=2)
        s["arm"] = np.array(ARM_REFERENCE) + rng.uniform(-0.30, 0.30, size=7)
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=40, help="Scenes per family")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    for index, name in enumerate(("far_target", "low_target", "high_target", "wide_start")):
        rng = np.random.default_rng([args.seed, index])
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=False)
        for i in range(args.n):
            s = family(name, rng)
            scene = dict(id=f"ood_{name}_{i + 1:03d}", target=s["target"].tolist(), base_pos=s["base"].tolist(),
                         base_bearing=s["yaw"], arm_q=s["arm"].tolist(), arm_qd=[0.0] * 7)
            (folder / f"{scene['id']}.json").write_text(json.dumps(scene))
    print("written", args.output)


if __name__ == "__main__":
    main()
