"""Labels for the staged policy, extracted only from public recorded signals.

Recorded `inputs` columns: base xyz 0:3, yaw 3, arm q 4:11, arm qd 11:18, target 18:21.
`actions`: 4 wheel velocities, then 7 arm torques. The expert's recorded `phase`
(an internal controller variable) is never read here.
"""
import csv
import json
from pathlib import Path

import numpy as np

WHEEL_EPS = 1e-3  # rad/s; below this all four wheels count as stopped
BASE_FEATURES = ["dist_to_park", "sin_heading_err", "cos_heading_err", "sin_yaw_err", "cos_yaw_err"]


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def stop_index(actions):
    """First step after which every wheel command stays zero: the public navigation-done time."""
    moving = np.flatnonzero(np.abs(actions[:, :4]).max(axis=1) > WHEEL_EPS)
    return int(moving[-1]) + 1 if len(moving) else 0


def base_features(base_xy, yaw, park_xy, park_yaw):
    """Distance/heading/yaw errors of the base relative to a parking pose (broadcasts over rows)."""
    delta = park_xy - base_xy
    heading = wrap_angle(np.arctan2(delta[..., 1], delta[..., 0]) - yaw)
    yaw_err = wrap_angle(park_yaw - yaw)
    return np.stack((np.linalg.norm(delta, axis=-1), np.sin(heading), np.cos(heading),
                     np.sin(yaw_err), np.cos(yaw_err)), axis=-1)


def successful_paths(source, success=True):
    """Trajectories whose summary.csv success flag equals `success` (False: the expert's failed runs)."""
    with (source / "summary.csv").open() as stream:
        ids = sorted(row["id"] for row in csv.DictReader(stream) if (row["success"] == "True") == success)
    return [source / f"{scene_id}.npz" for scene_id in ids]


def split_paths(paths, seed=42):
    """Whole-trajectory 70/15/15 split, decided before any frame is read."""
    order = np.random.default_rng(seed).permutation(len(paths))
    shuffled = [paths[i] for i in order]
    n_train, n_val = int(len(paths) * 0.70), int(len(paths) * 0.15)
    return dict(train=shuffled[:n_train], validation=shuffled[n_train:n_train + n_val],
                test=shuffled[n_train + n_val:])


def load_trajectory(path):
    with np.load(path, allow_pickle=False) as data:
        x, y = data["inputs"], data["actions"]
        scenario = json.loads(str(data["scenario"]))
    if x.shape[1] != 21 or y.shape != (len(x), 11) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"Malformed trajectory {path}")
    stop = stop_index(y)
    # stop == 0 is valid: the base can start already parked (scene_42_0597 never moves).
    if not 0 <= stop < len(x):
        raise ValueError(f"No public stop time in {path}")
    return dict(id=scenario["id"], inputs=x, actions=y, stop=stop, target=x[0, 18:21].copy(),
                park_xy=x[stop, :2].copy(), park_yaw=float(x[stop, 3]))


def write_split(path, source, splits, seed, train_fraction=1.0):
    """Split of the successful trajectories, plus test_expert_failed: the expert's failed runs that
    the same 70/15/15 rule puts in the test part. They are never trained on; together with test
    they form the unfiltered test set (staged_evaluate.py --split test_unfiltered)."""
    failed = split_paths(successful_paths(source, success=False), seed)["test"]
    path.write_text(json.dumps(dict(
        source=str(source), seed=seed, ratios=dict(train=0.70, validation=0.15, test=0.15),
        train_fraction=train_fraction,
        population="train/validation/test: successful expert trajectories (summary.csv success == True); "
                   "test_expert_failed: unsuccessful ones assigned to test by the same rule",
        splits={**{name: [p.name for p in paths] for name, paths in splits.items()},
                "test_expert_failed": [p.name for p in failed]},
    ), indent=2) + "\n")
