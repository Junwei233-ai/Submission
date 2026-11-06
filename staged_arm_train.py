"""Train the staged policy's arm network from recorded trajectories (public signals only).

The network maps (end-effector error in the base yaw frame, 7 joint angles) to a 7x3 matrix M;
the desired joint velocity is v = M @ error, so it is exactly zero at zero error. Labels are the
joints' actual velocity, a central difference of the recorded joint angles (no expert internals,
no measured qd as input). Frames start at the public base stop time; test trajectories are never
read. Kp, Kd and the reference lead are fixed controller constants chosen separately on validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from staged_data import stop_index
from staged_policy import LearnedFK, NumpyMLP, arm_features
from staged_train import make_mlp, normaliser

DT = 0.001


def arm_frames(path, half_window, stride, fk=None):
    """Features and velocity labels for one trajectory, from the public stop time onwards.

    With fk (a LearnedFK) the end-effector position is the learned estimate, as at run time."""
    with np.load(path, allow_pickle=False) as data:
        x, actions = data["inputs"], data["actions"]
        ee = fk(x[:, 4:11], x[:, :3], x[:, 3]) if fk is not None else data["ee_pos"]
    q, yaw, target = x[:, 4:11], x[:, 3], x[0, 18:21]
    start = max(stop_index(actions), half_window)
    index = np.arange(start, len(x) - half_window, stride)
    error = target - ee[index]
    velocity = (q[index + half_window] - q[index - half_window]) / (2 * half_window * DT)
    return arm_features(error, yaw[index], q[index]), velocity


def load(paths, half_window, stride, fk=None):
    pairs = [arm_frames(p, half_window, stride, fk) for p in paths]
    return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="staged_train.py output with split.json")
    parser.add_argument("--output", type=Path, required=True, help="New arm policy .npz path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--half-window", type=int, default=10, help="Central difference spans +/- this many steps")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kp", type=float, default=100.0)
    parser.add_argument("--kd", type=float, default=30.0)
    parser.add_argument("--lead", type=float, default=0.15)
    parser.add_argument("--fk-model", type=Path,
                        help="Learned forward kinematics (staged_fk_train.py): features use its EE estimate, not the URDF")
    parser.add_argument("--limit", type=int, help="Smoke test: first N trajectories per split")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    started = time.perf_counter()
    split = json.loads((args.model_dir / "split.json").read_text())
    source = Path(split["source"])
    data = {name: load([source / n for n in split["splits"][name][:args.limit]], args.half_window, args.stride,
                        LearnedFK(args.fk_model) if args.fk_model else None)
            for name in ("train", "validation")}
    for name, (x, y) in data.items():
        print(f"{name}: {len(x)} frames", flush=True)
    (x, y), (vx, vy) = data["train"], data["validation"]
    x_mean, x_scale = normaliser(x)
    to_tensor = lambda a: torch.from_numpy(a.astype(np.float32))
    X, VX = to_tensor((x - x_mean) / x_scale), to_tensor((vx - x_mean) / x_scale)
    E, VE = to_tensor(x[:, :3]), to_tensor(vx[:, :3])
    Y, VY = to_tensor(y), to_tensor(vy)

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    net = make_mlp(10, 21)
    optimiser = torch.optim.Adam(net.parameters(), lr=args.lr)
    velocity = lambda inputs, error: torch.einsum("bij,bj->bi", net(inputs).view(-1, 7, 3), error)
    best, best_loss, best_epoch, history = None, np.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        net.train()
        for index in torch.randperm(len(X), generator=generator).split(args.batch_size):
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(velocity(X[index], E[index]), Y[index])
            loss.backward()
            optimiser.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.mse_loss(velocity(VX, VE), VY))
        history.append(dict(epoch=epoch, train_loss=float(loss), validation_loss=val_loss))
        if val_loss < best_loss - 1e-9:
            best, best_loss, best_epoch = {k: v.clone() for k, v in net.state_dict().items()}, val_loss, epoch
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch}: val MSE {val_loss:.6f} (best {best_loss:.6f} @ {best_epoch})", flush=True)
        if epoch - best_epoch >= args.patience:
            break
    net.load_state_dict(best)
    net.eval()

    arrays = dict(arm_x_mean=x_mean, arm_x_scale=x_scale, arm_y_mean=np.zeros(21), arm_y_scale=np.ones(21))
    for i in (0, 2, 4):
        arrays[f"arm_w{i}"] = net[i].weight.detach().numpy().astype(np.float64)
        arrays[f"arm_b{i}"] = net[i].bias.detach().numpy().astype(np.float64)
    runtime = NumpyMLP(arrays, "arm")
    with torch.no_grad():
        expected = velocity(VX[:2000], VE[:2000]).numpy()
    matrices = runtime(vx[:2000]).reshape(-1, 7, 3)
    exported = np.einsum("bij,bj->bi", matrices, vx[:2000, :3])
    export_error = float(np.max(np.abs(exported - expected)))
    if export_error > 1e-4:
        raise RuntimeError(f"NumPy export differs from PyTorch by {export_error}")

    with torch.no_grad():
        predicted = velocity(VX, VE).numpy()
    abs_err = np.abs(predicted - vy)
    distance = np.linalg.norm(vx[:, :3], axis=1)
    bands = {"near_lt_5cm": distance < 0.05, "mid_5cm_to_20cm": (distance >= 0.05) & (distance < 0.20),
             "far_ge_20cm": distance >= 0.20}
    report = dict(
        best_epoch=best_epoch, epochs_run=len(history), validation_mse=best_loss, export_max_error=export_error,
        validation_mae_rad_s=float(abs_err.mean()),
        validation_mae_by_distance={k: dict(frames=int(m.sum()), mae=float(abs_err[m].mean()) if m.any() else None,
                                            mean_abs_label=float(np.abs(vy[m]).mean()) if m.any() else None)
                                    for k, m in bands.items()},
        train_frames=len(x), validation_frames=len(vx), training_wall_s=time.perf_counter() - started,
        test_data_opened=False)
    metadata = dict(ee_source=str(args.fk_model) if args.fk_model else "recorded ee_pos", policy_type="staged_arm", kp=[args.kp] * 7, kd=[args.kd] * 7, reference_lead=args.lead,
                    velocity_limit=np.abs(y).max(axis=0).tolist(), half_window=args.half_window, stride=args.stride,
                    seed=args.seed, input="[ee error in base yaw frame (3), q (7)]; output 7x3 M, v = M @ error",
                    split_sha256=hashlib.sha256((args.model_dir / "split.json").read_bytes()).hexdigest(),
                    labels="central difference of recorded joint angles after the public base stop time")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    args.output.with_suffix(".json").write_text(json.dumps(dict(metrics=report, metadata=metadata, history=history), indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
