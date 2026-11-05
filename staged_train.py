"""Train the staged policy's base networks (parking pose, wheels, stop) from recorded trajectories.

Only public signals are used: base pose, target and wheel commands. The parking label is the
base pose at the public stop time (wheels zero from then on); the stop label is whether that
time has passed. Base features are measured against the parking net's own prediction, as at
run time, so the stop net learns the expert's arrival tolerance. Test trajectories are never read.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from staged_data import (BASE_FEATURES, WHEEL_EPS, base_features, load_trajectory, split_paths,
                         successful_paths, wrap_angle, write_split)
from staged_policy import NumpyMLP, StagedBase, park_pose

ROOT = Path(__file__).resolve().parent
LATCH_CANDIDATES = (1, 5, 10, 20, 50, 100)


def make_mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_out))


def normaliser(values):
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


def fit(name, x, y, val_x, val_y, args, classify=False):
    """Adam with early stopping on validation loss; returns NumPy arrays for export."""
    x_mean, x_scale = normaliser(x)
    # Classifier logits are left unscaled.
    y_mean, y_scale = (np.zeros(y.shape[1]), np.ones(y.shape[1])) if classify else normaliser(y)
    scale = lambda a, m, s: torch.from_numpy(((a - m) / s).astype(np.float32))
    X, Y = scale(x, x_mean, x_scale), scale(y, y_mean, y_scale)
    VX, VY = scale(val_x, x_mean, x_scale), scale(val_y, y_mean, y_scale)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    model = make_mlp(x.shape[1], y.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss() if classify else nn.MSELoss()
    # Small sets (one parking label per trajectory) still get ~20 updates per epoch.
    batch_size = min(args.batch_size, max(32, len(X) // 20))
    best, best_loss, best_epoch, history = None, np.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for index in torch.randperm(len(X), generator=generator).split(batch_size):
            optimiser.zero_grad()
            loss = loss_fn(model(X[index]), Y[index])
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(VX), VY))
        if not np.isfinite(val_loss):
            raise RuntimeError(f"{name}: nonfinite validation loss")
        history.append(dict(epoch=epoch, train_loss=float(loss), validation_loss=val_loss))
        if val_loss < best_loss - 1e-7:
            best, best_loss, best_epoch = copy.deepcopy(model.state_dict()), val_loss, epoch
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best)
    model.eval()
    arrays = {f"{name}_x_mean": x_mean, f"{name}_x_scale": x_scale,
              f"{name}_y_mean": y_mean, f"{name}_y_scale": y_scale}
    for i in (0, 2, 4):
        arrays[f"{name}_w{i}"] = model[i].weight.detach().numpy().astype(np.float64)
        arrays[f"{name}_b{i}"] = model[i].bias.detach().numpy().astype(np.float64)
    with torch.no_grad():
        expected = model(VX[:2000]).numpy() * y_scale + y_mean
    export_error = float(np.max(np.abs(NumpyMLP(arrays, name)(val_x[:2000]) - expected)))
    if export_error > 1e-4 * (1 + np.abs(expected).max()):
        raise RuntimeError(f"{name}: NumPy export differs from PyTorch by {export_error}")
    print(f"{name}: best epoch {best_epoch}/{len(history)}, validation loss {best_loss:.6g}", flush=True)
    return arrays, dict(best_epoch=best_epoch, epochs_run=len(history), validation_loss=best_loss,
                        export_max_error=export_error, samples=len(x), batch_size=batch_size), history


def frames(trajectories, park, stride):
    """Strided base features, wheel labels and stop labels for every trajectory."""
    features, wheels, stopped, before_stop = [], [], [], []
    for traj, (park_xy, park_yaw) in zip(trajectories, park):
        x = traj["base"][::stride]
        steps = np.arange(0, len(traj["base"]), stride)
        features.append(base_features(x[:, :2], x[:, 2], park_xy, park_yaw))
        wheels.append(traj["wheels"][::stride])
        stopped.append(steps >= traj["stop"])
    features, wheels, stopped = map(np.concatenate, (features, wheels, stopped))
    return features, wheels, stopped.astype(np.float64)[:, None]


def latch_timing(policy, trajectories, park):
    """Open-loop latch check on recorded validation states: fire step minus public stop step."""
    rows = {}
    for steps in LATCH_CANDIDATES:
        offsets = []
        for traj, (park_xy, park_yaw) in zip(trajectories, park):
            fires = policy.nets["stop"](base_features(traj["base"][:, :2], traj["base"][:, 2],
                                                      park_xy, park_yaw))[:, 0] > 0
            run = np.convolve(fires.astype(float), np.ones(steps), "valid") >= steps
            hit = np.flatnonzero(run)
            offsets.append(hit[0] + steps - 1 - traj["stop"] if len(hit) else np.nan)
        offsets = np.asarray(offsets, dtype=float)
        fired = np.isfinite(offsets)
        rows[steps] = dict(never_fired=int((~fired).sum()), early_over_100=int((offsets[fired] < -100).sum()),
                           median_offset=float(np.median(offsets[fired])) if fired.any() else None,
                           mean_abs_offset=float(np.mean(np.abs(offsets[fired]))) if fired.any() else None)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "artifacts/collection_1500_seed42_20261031_013545_761934")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=5, help="Use every Kth recorded step")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, help="Smoke test: first N trajectories of train and validation")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="Learning curve: keep this fraction of the training trajectories (nested subsets; "
                             "validation and test unchanged). The written split.json holds the reduced train set.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if min(args.stride, args.epochs, args.patience, args.batch_size) <= 0:
        parser.error("stride, epochs, patience and batch size must be positive")
    started = time.perf_counter()
    splits = split_paths(successful_paths(args.source), args.seed)
    if not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction must be in (0, 1]")
    if args.train_fraction < 1:
        # Fixed permutation (seed 0), so smaller fractions are subsets of larger ones.
        order = np.random.default_rng(0).permutation(len(splits["train"]))
        keep = max(1, round(args.train_fraction * len(order)))
        splits["train"] = [splits["train"][i] for i in sorted(order[:keep])]
    args.output.mkdir(parents=True)
    write_split(args.output / "split.json", args.source, splits, args.seed, train_fraction=args.train_fraction)

    data = {}
    for name in ("train", "validation"):
        data[name] = []
        for path in splits[name][:args.limit]:
            traj = load_trajectory(path)
            data[name].append(dict(id=traj["id"], stop=traj["stop"], target=traj["target"],
                                   park=np.r_[traj["park_xy"], np.sin(traj["park_yaw"]), np.cos(traj["park_yaw"])],
                                   base=traj["inputs"][:, [0, 1, 3]].copy(), wheels=traj["actions"][:, :4].copy()))
        print(f"Loaded {len(data[name])} {name} trajectories", flush=True)

    arrays, report, histories = {}, {}, {}
    targets = {n: np.array([t["target"] for t in data[n]]) for n in data}
    labels = {n: np.array([t["park"] for t in data[n]]) for n in data}
    park_arrays, report["park"], histories["park"] = fit("park", targets["train"], labels["train"],
                                                         targets["validation"], labels["validation"], args)
    arrays.update(park_arrays)
    park_net = NumpyMLP(arrays, "park")
    park = {n: list(zip(*park_pose(park_net, targets[n]))) for n in data}
    pos_err = np.linalg.norm(park_net(targets["validation"])[:, :2] - labels["validation"][:, :2], axis=1)
    yaw_err = np.abs(wrap_angle(np.array([p[1] for p in park["validation"]])
                                - np.arctan2(labels["validation"][:, 2], labels["validation"][:, 3])))
    report["park"].update(validation_position_error_m=dict(median=float(np.median(pos_err)),
                                                           p95=float(np.percentile(pos_err, 95))),
                          validation_yaw_error_rad=dict(median=float(np.median(yaw_err)),
                                                        p95=float(np.percentile(yaw_err, 95))))

    train_f, train_w, train_s = frames(data["train"], park["train"], args.stride)
    val_f, val_w, val_s = frames(data["validation"], park["validation"], args.stride)
    # Wheels are only commanded before the latch; after it the runtime outputs zeros.
    moving, val_moving = train_s[:, 0] == 0, val_s[:, 0] == 0
    wheel_arrays, report["wheels"], histories["wheels"] = fit(
        "wheels", train_f[moving], train_w[moving], val_f[val_moving], val_w[val_moving], args)
    stop_arrays, report["stop"], histories["stop"] = fit("stop", train_f, train_s, val_f, val_s, args, classify=True)
    arrays.update(wheel_arrays)
    arrays.update(stop_arrays)
    wheel_mae = np.abs(NumpyMLP(arrays, "wheels")(val_f[val_moving]) - val_w[val_moving]).mean(axis=0)
    report["wheels"]["validation_mae_rad_s"] = wheel_mae.tolist()
    report["stop"]["validation_accuracy"] = float(np.mean((NumpyMLP(arrays, "stop")(val_f) > 0) == (val_s > 0.5)))

    metadata = dict(policy_type="staged_base", base_features=BASE_FEATURES, wheel_eps=WHEEL_EPS,
                    stride=args.stride, seed=args.seed, latch_steps=1,
                    split_sha256=hashlib.sha256((args.output / "split.json").read_bytes()).hexdigest(),
                    labels="public only: base pose, target, wheel commands; expert phase not read")
    policy_path = args.output / "policy.npz"
    np.savez(policy_path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    timing = latch_timing(StagedBase(policy_path), data["validation"], park["validation"])
    # Smallest mean timing error without firing more than 0.1 s early on any validation trajectory.
    safe = [s for s, row in timing.items() if row["never_fired"] == 0 and row["early_over_100"] == 0]
    metadata["latch_steps"] = min(safe or timing, key=lambda s: timing[s]["mean_abs_offset"] or np.inf)
    policy_path.unlink()
    np.savez(policy_path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    report.update(latch_timing_validation_open_loop=timing, latch_steps=metadata["latch_steps"],
                  training_wall_s=time.perf_counter() - started, test_data_opened=False)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "history.json").write_text(json.dumps(histories) + "\n")
    print(json.dumps({k: report[k] for k in ("park", "wheels", "stop", "latch_steps")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
