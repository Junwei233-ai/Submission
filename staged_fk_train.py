"""Learn the arm's forward kinematics from recorded frames: q -> EE position in the base frame.

This network replaces the URDF forward kinematics at run time. Its data are expert runs that keep
going for 2 s after success (staged_evaluate.py --hold-time 2), so the converged poses near the
target are covered. Labels are the recorded EE positions; frames after the stop time are sampled
more densely because the arm barely moves before it.

Usage: python staged_fk_train.py --train-rollouts HOLD_TRAIN --validation-rollouts HOLD_VAL --output fk.npz
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from staged_data import stop_index
from staged_policy import NumpyMLP, local_position


def load_rollouts(folder):
    """Successful rollouts of one split as (inputs, actions, ee_pos) tuples."""
    items = []
    for path in sorted(Path(folder, "rollouts").iterdir()):
        if json.loads((path / "result.json").read_text())["stop_reason"] != "success":
            continue
        with np.load(path / "rollout.npz", allow_pickle=False) as data:
            items.append((data["inputs"], data["actions"], data["ee_pos"]))
    return items


def samples(items, stride, nav_stride):
    """Every stride-th frame after the stop; before it every nav_stride-th."""
    xs, ys = [], []
    for x, actions, ee in items:
        stop = stop_index(actions)
        index = np.concatenate((np.arange(0, stop, nav_stride), np.arange(stop, len(x), stride)))
        xs.append(x[index, 4:11])
        ys.append(local_position(ee[index], x[index, :3], x[index, 3]))
    return np.concatenate(xs), np.concatenate(ys)


def fit(train, validation, hidden, epochs, batch, lr):
    """Adam with a cosine learning rate; keeps the epoch with the lowest validation loss."""
    x, y = train
    x_mean, x_scale, y_mean, y_scale = x.mean(0), x.std(0), y.mean(0), y.std(0)
    to_tensor = lambda a, mean, scale: torch.from_numpy(((a - mean) / scale).astype(np.float32))
    tx, ty = to_tensor(x, x_mean, x_scale), to_tensor(y, y_mean, y_scale)
    vx, vy = to_tensor(validation[0], x_mean, x_scale), to_tensor(validation[1], y_mean, y_scale)
    model = nn.Sequential(nn.Linear(7, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    best, best_loss, best_epoch, started = None, np.inf, 0, time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(tx))
        for start in range(0, len(tx), batch):
            chunk = order[start:start + batch]
            optimiser.zero_grad()
            nn.functional.mse_loss(model(tx[chunk]), ty[chunk]).backward()
            optimiser.step()
        schedule.step()
        model.eval()
        with torch.no_grad():
            loss = float(nn.functional.mse_loss(model(vx), vy))
        if loss < best_loss:
            best, best_loss, best_epoch = copy.deepcopy(model.state_dict()), loss, epoch
        if epoch % 10 == 0:
            print(f"epoch {epoch}: validation {loss:.3e}, best {best_loss:.3e} ({time.perf_counter() - started:.0f}s)", flush=True)
    model.load_state_dict(best)
    arrays = dict(fk_x_mean=x_mean, fk_x_scale=x_scale, fk_y_mean=y_mean, fk_y_scale=y_scale)
    for i in (0, 2, 4):
        arrays[f"fk_w{i}"] = model[i].weight.detach().numpy().astype(np.float64)
        arrays[f"fk_b{i}"] = model[i].bias.detach().numpy().astype(np.float64)
    return arrays, dict(samples=len(x), validation_samples=len(validation[0]), best_epoch=best_epoch, epochs_run=epochs,
                        validation_loss=best_loss, wall_s=time.perf_counter() - started)


def summary(error):
    mm = np.asarray(error) * 1000
    return dict(median_mm=float(np.median(mm)), p95_mm=float(np.quantile(mm, 0.95)), max_mm=float(mm.max()),
                under_5mm=float(np.mean(mm < 5)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rollouts", type=Path, required=True)
    parser.add_argument("--validation-rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .npz path")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--stride", type=int, default=2, help="Training frames after the stop")
    parser.add_argument("--nav-stride", type=int, default=20, help="Training frames before the stop")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train, validation = load_rollouts(args.train_rollouts), load_rollouts(args.validation_rollouts)
    arrays, report = fit(samples(train, args.stride, args.nav_stride), samples(validation, 20, 200),
                         args.hidden, args.epochs, 4096, 1e-3)
    net = NumpyMLP(arrays, "fk")
    all_frames, after_stop, final = [], [], []
    for x, actions, ee in validation:
        stop = stop_index(actions)
        error = np.linalg.norm(net(x[:, 4:11]) - local_position(ee, x[:, :3], x[:, 3]), axis=1)
        all_frames.append(error[::50])
        after_stop.append(error[stop::10])
        final.append(error[-1])
    report.update(train_rollouts=len(train), validation_rollouts=len(validation),
                  parameters=int(sum(a.size for k, a in arrays.items() if k[3] in "wb")),
                  validation_all_frames=summary(np.concatenate(all_frames)),
                  validation_after_stop=summary(np.concatenate(after_stop)),
                  validation_final_frame=summary(final))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
