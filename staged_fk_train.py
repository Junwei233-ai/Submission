"""Learn the arm's forward kinematics without any recorded end-effector position.

When the expert has converged on a target, the end-effector is at that target. The final frame of
each expert run with a 2 s hold (staged_evaluate.py --hold-time 2) therefore gives one training sample:
joint angles -> target position in the base frame. Inputs and labels use only recorded joint angles,
base pose and target. The network replaces forward kinematics in training and at run time.

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

from staged_policy import LearnedFK, local_position


def final_frames(folder):
    """(joint angles, target in the base frame) from the last frame of each successful run."""
    xs, ys = [], []
    for path in sorted(Path(folder, "rollouts").iterdir()):
        if json.loads((path / "result.json").read_text())["stop_reason"] != "success":
            continue
        with np.load(path / "rollout.npz", allow_pickle=False) as data:
            last = data["inputs"][-1]
        xs.append(last[4:11])
        ys.append(local_position(last[18:21], last[:3], last[3]))
    return np.asarray(xs), np.asarray(ys)


def fit(train, validation, hidden, epochs, batch, lr, weight_decay):
    """Adam with a cosine learning rate; keeps the epoch with the lowest validation loss."""
    x, y = train
    x_mean, x_scale, y_mean, y_scale = x.mean(0), x.std(0), y.mean(0), y.std(0)
    to_tensor = lambda a, mean, scale: torch.tensor((a - mean) / scale, dtype=torch.float64)
    tx, ty = to_tensor(x, x_mean, x_scale), to_tensor(y, y_mean, y_scale)
    vx, vy = to_tensor(validation[0], x_mean, x_scale), to_tensor(validation[1], y_mean, y_scale)
    model = nn.Sequential(nn.Linear(7, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3)).double()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    best, best_loss, best_epoch, started = None, np.inf, 0, time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(tx))
        for start in range(0, len(tx), batch):
            chunk = order[start:start + batch]
            optimiser.zero_grad()
            nn.functional.mse_loss(model(tx[chunk]), ty[chunk]).backward()
            optimiser.step()
        schedule.step()
        with torch.no_grad():
            loss = float(nn.functional.mse_loss(model(vx), vy))
        if loss < best_loss:
            best, best_loss, best_epoch = copy.deepcopy(model.state_dict()), loss, epoch
    model.load_state_dict(best)
    arrays = dict(fk_x_mean=x_mean, fk_x_scale=x_scale, fk_y_mean=y_mean, fk_y_scale=y_scale)
    for i in (0, 2, 4):
        arrays[f"fk_w{i}"] = model[i].weight.detach().numpy()
        arrays[f"fk_b{i}"] = model[i].bias.detach().numpy()
    return arrays, dict(best_epoch=best_epoch, epochs_run=epochs, validation_loss=best_loss,
                        parameters=int(sum(p.numel() for p in model.parameters())),
                        wall_s=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rollouts", type=Path, required=True)
    parser.add_argument("--validation-rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .npz path")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train, validation = final_frames(args.train_rollouts), final_frames(args.validation_rollouts)
    arrays, report = fit(train, validation, args.hidden, args.epochs, args.batch_size, args.lr, args.weight_decay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    error = np.linalg.norm(LearnedFK(args.output).local(validation[0]) - validation[1], axis=1) * 1000
    report.update(train_samples=len(train[0]), validation_samples=len(validation[0]),
                  validation_final_pose_error=dict(median_mm=float(np.median(error)), p95_mm=float(np.quantile(error, 0.95)),
                                                   max_mm=float(error.max()), under_1cm=int((error < 10).sum())))
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
