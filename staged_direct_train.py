"""Plain behaviour cloning baseline (L0): one regressor from the 21 raw inputs to the 11 raw actions.

Inputs, exactly as recorded: base_pos (3), base_bearing (1), q_mes (7), qd_mes (7), TARGET_EE_POS (3).
Outputs, exactly as recorded: wheel_cmds (4) and torque_arm (7). No features, no parking model, no
stop head or latch, no PD layer, no gravity model: the regressor's output is applied directly.

Two regressors, both fitted on frames of the training trajectories (every --stride-th step) with
inputs and outputs standardised on the training data:
  linear  ridge regression, closed form; the ridge strength is chosen on validation MSE;
  mlp     two hidden ReLU layers of 128 units (staged_train.make_mlp / fit: Adam, early stopping on
          validation MSE).
Writes <output>/{linear,mlp}.npz (+ .json with open-loop validation errors). Test data is never read.

Usage: python staged_direct_train.py --model-dir artifacts/final_run/base --output artifacts/final_run/direct
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from staged_data import load_trajectory
from staged_policy import DirectPolicy
from staged_train import fit, normaliser

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
OUTPUTS = [f"wheel{i}" for i in range(4)] + [f"torque{i}" for i in range(1, 8)]


def frames(paths, stride):
    trajectories = [load_trajectory(p) for p in paths]
    x = np.concatenate([t["inputs"][::stride] for t in trajectories])
    y = np.concatenate([t["actions"][::stride] for t in trajectories])
    return x, y


def open_loop(policy, x, y):
    """Validation errors of the exported NumPy policy on recorded frames (no closed loop)."""
    pred = np.stack([np.concatenate(policy.act(row)) for row in x])
    mse = ((pred - y) ** 2).mean(axis=0)
    r2 = 1 - mse / y.var(axis=0)
    return dict(rmse={k: float(np.sqrt(v)) for k, v in zip(OUTPUTS, mse)},
                r2={k: float(v) for k, v in zip(OUTPUTS, r2)},
                r2_mean_wheels=float(r2[:4].mean()), r2_mean_torques=float(r2[4:].mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="staged_train.py output with split.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, help="Smoke test: first N trajectories per split")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.output.mkdir(parents=True)
    split = json.loads((args.model_dir / "split.json").read_text())
    source = Path(split["source"])
    x, y = frames([source / n for n in split["splits"]["train"][:args.limit]], args.stride)
    vx, vy = frames([source / n for n in split["splits"]["validation"][:args.limit]], args.stride)
    print(f"train frames {len(x)}, validation frames {len(vx)}", flush=True)
    meta = dict(inputs="base_pos(3), base_bearing(1), q_mes(7), qd_mes(7), TARGET_EE_POS(3)",
                outputs="wheel_cmds(4), torque_arm(7)", stride=args.stride, seed=args.seed,
                train_trajectories=len(split["splits"]["train"][:args.limit]), test_data_opened=False)

    # Linear (ridge) regression, closed form on standardised data.
    started = time.perf_counter()
    x_mean, x_scale = normaliser(x)
    y_mean, y_scale = normaliser(y)
    X, Y = (x - x_mean) / x_scale, (y - y_mean) / y_scale
    VX, VY = (vx - x_mean) / x_scale, (vy - y_mean) / y_scale
    X1, VX1 = np.hstack([X, np.ones((len(X), 1))]), np.hstack([VX, np.ones((len(VX), 1))])
    gram, cross = X1.T @ X1, X1.T @ Y
    scores = {}
    for alpha in ALPHAS:
        reg = alpha * np.eye(X1.shape[1])
        reg[-1, -1] = 0.0  # no penalty on the bias
        w = np.linalg.solve(gram + reg, cross)
        scores[alpha] = (float(((VX1 @ w - VY) ** 2).mean()), w)
    alpha = min(scores, key=lambda a: scores[a][0])
    w = scores[alpha][1]
    linear = dict(kind=np.asarray("linear"), x_mean=x_mean, x_scale=x_scale, y_mean=y_mean, y_scale=y_scale,
                  weight=w[:-1].T.copy(), bias=w[-1].copy())
    np.savez(args.output / "linear.npz", **linear)
    report = dict(meta, model="ridge", alpha=alpha, validation_mse_by_alpha={str(a): s[0] for a, s in scores.items()},
                  parameters=int(w.size), training_wall_s=time.perf_counter() - started,
                  open_loop_validation=open_loop(DirectPolicy(args.output / "linear.npz"), vx, vy))
    (args.output / "linear.json").write_text(json.dumps(report, indent=2) + "\n")
    print("linear:", json.dumps({k: report[k] for k in ("alpha", "parameters")}),
          json.dumps({k: v for k, v in report["open_loop_validation"].items() if k.startswith("r2_mean")}), flush=True)

    # MLP regressor.
    started = time.perf_counter()
    arrays, info, history = fit("direct", x, y, vx, vy, args)
    np.savez(args.output / "mlp.npz", kind=np.asarray("mlp"), **arrays)
    report = dict(meta, model="mlp 2x128 ReLU", **info,
                  parameters=int(sum(v.size for k, v in arrays.items() if k[7] in "wb")),
                  training_wall_s=time.perf_counter() - started, history=history,
                  open_loop_validation=open_loop(DirectPolicy(args.output / "mlp.npz"), vx, vy))
    (args.output / "mlp.json").write_text(json.dumps(report, indent=2) + "\n")
    print("mlp:", json.dumps({k: report[k] for k in ("best_epoch", "parameters")}),
          json.dumps({k: v for k, v in report["open_loop_validation"].items() if k.startswith("r2_mean")}), flush=True)


if __name__ == "__main__":
    main()
