"""Train one GRU policy for base and arm from recorded trajectories (public signals only).

Every GRU_PERIOD recorded steps the GRU reads relative-coordinate features (gru_features) and
outputs 4 wheel commands and a 7x3 matrix M; the arm's desired velocity is M @ (EE error in the
base frame). There is no stop output, latch or parking model: stages must be learned in the
hidden state. Labels: recorded wheel commands and central-difference joint velocities on every
frame. Whole trajectories are used as sequences; test trajectories are never read.

With --stop-head the GRU also outputs a stop logit (label: the public stop time, after which all
wheel commands stay zero). At run time it is latched: the arm holds until the latch and the
wheels are zero after it, so the arm loss is only taken on frames after the stop time.

With --arm-output torque (ablation) the arm head outputs the 7 recorded joint torques directly,
applied with zero-order hold and no PD layer or gravity term.

With --features park the hand-made world-bearing and target-direction inputs are replaced by the
parking pose predicted by the learned park MLP in --model-dir (gru_park_features); its weights
are copied into the GRU file.
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

from staged_data import stop_index
from staged_policy import (GRU_FEATURES, GRU_PERIOD, PARK_GRU_FEATURES, GRUPolicy, LearnedFK, NumpyMLP,
                           gru_features, gru_park_features, park_pose)

LATCH_CANDIDATES = (1, 2, 3, 5, 10, 20)  # GRU updates (10 ms each)

DT = 0.001


def sequence(path, half_window, park_net=None, fk=None):
    """Features, wheel, arm velocity, stop and arm torque labels at every GRU_PERIOD-th step.

    With fk (a LearnedFK), the end-effector position in the features is the learned model's estimate
    from the recorded joint angles and base pose, exactly as at run time; the recorded (URDF) ee_pos is
    then not read."""
    with np.load(path, allow_pickle=False) as data:
        x, actions = data["inputs"], data["actions"]
        ee = fk(x[:, 4:11], x[:, :3], x[:, 3]) if fk is not None else data["ee_pos"]
    index = np.arange(0, len(x), GRU_PERIOD)
    lo, hi = np.maximum(index - half_window, 0), np.minimum(index + half_window, len(x) - 1)
    velocity = (x[hi, 4:11] - x[lo, 4:11]) / ((hi - lo)[:, None] * DT)
    if park_net is None:
        features = gru_features(x[index, :3], x[index, 3], x[index, 4:11], ee[index], x[index, 18:21])
    else:
        # Same as at run time: the park MLP's prediction from the target, not the recorded stop pose.
        features = gru_park_features(x[index, :3], x[index, 3], x[index, 4:11], ee[index], x[index, 18:21],
                                     *park_pose(park_net, x[0, 18:21]))
    stopped = (index >= stop_index(actions)).astype(np.float64)[:, None]
    return features, actions[index, :4], velocity, stopped, actions[index, 4:11]


class GRUNet(nn.Module):
    def __init__(self, hidden, stop_head, arm_size=21):
        super().__init__()
        self.gru = nn.GRU(len(GRU_FEATURES), hidden, batch_first=True)
        self.out = nn.Linear(hidden, 4 + arm_size + int(stop_head))

    def forward(self, x):
        return self.out(self.gru(x)[0])


def batches(items, size, generator=None):
    order = torch.randperm(len(items), generator=generator).tolist() if generator else range(len(items))
    order = list(order)
    for start in range(0, len(order), size):
        chunk = [items[i] for i in order[start:start + size]]
        length = max(len(c[0]) for c in chunk)
        pad = lambda a: np.pad(a, ((0, length - len(a)), (0, 0)))
        stack = lambda k: torch.from_numpy(np.stack([pad(c[k]) for c in chunk]).astype(np.float32))
        mask = torch.from_numpy(np.stack([np.arange(length) < len(c[0]) for c in chunk]))
        yield stack(0), stack(1), stack(2), stack(3), stack(4), mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="staged_train.py output with split.json")
    parser.add_argument("--output", type=Path, required=True, help="New GRU policy .npz path")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--half-window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--kp", type=float, default=100.0)
    parser.add_argument("--kd", type=float, default=30.0)
    parser.add_argument("--lead", type=float, default=0.1)
    parser.add_argument("--stop-head", action="store_true", help="Add a latched stop output (see module doc)")
    parser.add_argument("--arm-output", choices=("matrix", "torque"), default="matrix",
                        help="matrix: v = M @ error tracked by PD; torque: direct joint torques (ablation)")
    parser.add_argument("--features", choices=("bearing", "park"), default="bearing",
                        help="bearing: gru_features; park: gru_park_features with the learned park MLP")
    parser.add_argument("--fk-model", type=Path,
                        help="Learned forward kinematics (staged_fk_train.py): features use its EE estimate, not the URDF")
    parser.add_argument("--limit", type=int, help="Smoke test: first N trajectories per split")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    split = json.loads((args.model_dir / "split.json").read_text())
    source = Path(split["source"])
    park_arrays, park_net = {}, None
    if args.features == "park":
        with np.load(args.model_dir / "policy.npz", allow_pickle=False) as data:
            park_arrays = {k: data[k] for k in data.files if k.startswith("park_")}
        park_net = NumpyMLP(park_arrays, "park")
    fk = LearnedFK(args.fk_model) if args.fk_model else None
    raw = {name: [sequence(source / n, args.half_window, park_net, fk) for n in split["splits"][name][:args.limit]]
           for name in ("train", "validation")}
    train_x = np.concatenate([s[0] for s in raw["train"]])
    x_mean, x_scale = train_x.mean(axis=0), train_x.std(axis=0)
    x_scale[x_scale == 0] = 1.0
    wheels = np.concatenate([s[1] for s in raw["train"]])
    wheel_mean, wheel_scale = wheels.mean(axis=0), wheels.std(axis=0)
    v_scale = float(np.concatenate([s[2] for s in raw["train"]]).std())
    torque = args.arm_output == "torque"
    if torque:
        # Normalised on the frames the arm loss uses: after the stop time with a stop head, else all.
        tq = np.concatenate([s[4][s[3][:, 0] > 0.5] if args.stop_head else s[4] for s in raw["train"]])
        torque_mean, torque_scale = tq.mean(axis=0), tq.std(axis=0)
        arm_label = lambda v, t: (t - torque_mean) / torque_scale
    else:
        arm_label = lambda v, t: v
    # Item: normalised features, normalised wheel labels, arm labels (velocity, or normalised torque),
    # raw EE error (for M @ error), stop.
    data = {name: [((f - x_mean) / x_scale, (w - wheel_mean) / wheel_scale, arm_label(v, t), f[:, 8:11], st)
                   for f, w, v, st, t in items] for name, items in raw.items()}
    print({n: (len(v), sum(len(s[0]) for s in v)) for n, v in data.items()}, "(sequences, GRU steps)", flush=True)

    def loss_terms(model, x, wheel, arm_target, error, stopped, mask):
        out = model(x)
        if torque:
            arm, scale = out[..., 4:11], 1.0
        else:
            matrix = out[..., 4:25].reshape(*out.shape[:2], 7, 3)
            arm, scale = torch.einsum("btij,btj->bti", matrix, error), v_scale
        m = mask.unsqueeze(-1).float()
        wheel_loss = (((out[..., :4] - wheel) ** 2) * m).sum() / (m.sum() * 4)
        if not args.stop_head:
            arm_loss = ((((arm - arm_target) / scale) ** 2) * m).sum() / (m.sum() * 7)
            return wheel_loss, arm_loss, torch.zeros(())
        # The arm only acts after the latch, so it is only trained on frames after the stop time.
        arm_mask = m * stopped
        arm_loss = ((((arm - arm_target) / scale) ** 2) * arm_mask).sum() / (arm_mask.sum() * 7).clamp(min=1)
        stop_loss = (nn.functional.binary_cross_entropy_with_logits(out[..., -1:], stopped, reduction="none")
                     * m).sum() / m.sum()
        return wheel_loss, arm_loss, stop_loss

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    model = GRUNet(args.hidden, args.stop_head, 7 if torque else 21)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_loss, best_epoch, history = None, np.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, wheel, velocity, error, stopped, mask in batches(data["train"], args.batch_size, generator):
            optimiser.zero_grad()
            wheel_loss, arm_loss, stop_loss = loss_terms(model, x, wheel, velocity, error, stopped, mask)
            (wheel_loss + arm_loss + stop_loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            totals = np.zeros(4)
            for x, wheel, velocity, error, stopped, mask in batches(data["validation"], 64):
                wl, al, sl = loss_terms(model, x, wheel, velocity, error, stopped, mask)
                n = float(mask.sum())
                totals += [float(wl) * n, float(al) * n, float(sl) * n, n]
        val_wheel, val_arm, val_stop = totals[:3] / totals[3]
        val_loss = val_wheel + val_arm + val_stop
        history.append(dict(epoch=epoch, validation_wheel=val_wheel, validation_arm=val_arm, validation_stop=val_stop,
                            elapsed_s=time.perf_counter() - started))
        print(f"epoch {epoch}: val wheel {val_wheel:.5f} arm {val_arm:.5f} stop {val_stop:.5f} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
        if val_loss < best_loss - 1e-7:
            best, best_loss, best_epoch = copy.deepcopy(model.state_dict()), val_loss, epoch
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best)
    model.eval()

    arrays = dict(x_mean=x_mean, x_scale=x_scale, wheel_mean=wheel_mean, wheel_scale=wheel_scale,
                  w_ih=model.gru.weight_ih_l0.detach().numpy().astype(np.float64),
                  w_hh=model.gru.weight_hh_l0.detach().numpy().astype(np.float64),
                  b_ih=model.gru.bias_ih_l0.detach().numpy().astype(np.float64),
                  b_hh=model.gru.bias_hh_l0.detach().numpy().astype(np.float64),
                  w_out=model.out.weight.detach().numpy().astype(np.float64),
                  b_out=model.out.bias.detach().numpy().astype(np.float64))
    if torque:
        arrays.update(torque_mean=torque_mean, torque_scale=torque_scale)
    arrays.update(park_arrays)
    labels = np.concatenate([s[2] for s in raw["train"]])
    metadata = dict(policy_type="gru", feature_set=args.features,
                    features=PARK_GRU_FEATURES if park_net else GRU_FEATURES, period=GRU_PERIOD, hidden=args.hidden,
                    stop_head=args.stop_head, latch_ticks=1, arm_output=args.arm_output,
                    kp=args.kp, kd=args.kd, reference_lead=args.lead,
                    velocity_limit=np.abs(labels).max(axis=0).tolist(), half_window=args.half_window, seed=args.seed,
                    split_sha256=hashlib.sha256((args.model_dir / "split.json").read_bytes()).hexdigest(),
                    ee_source=("learned forward kinematics, sha256 "
                               + hashlib.sha256(args.fk_model.read_bytes()).hexdigest()) if args.fk_model else "recorded ee_pos",
                    labels=("recorded wheel commands; arm: "
                            + ("recorded joint torques" if torque else "central-difference joint velocity")
                            + ("; stop: public stop time from recorded wheels (arm loss after it only)"
                               if args.stop_head else "; no stop label")
                            + "; expert phase never read"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))

    # Export check: step the NumPy GRU through one validation sequence and compare with PyTorch.
    features = raw["validation"][0][0]
    with torch.no_grad():
        expected = model(torch.from_numpy(((features - x_mean) / x_scale).astype(np.float32))[None])[0].numpy()
    runtime = GRUPolicy(args.output)
    runtime.reset(np.zeros(7))
    got = []
    for f in features:
        runtime._update(f)
        got.append(runtime.raw.copy())
    export_error = float(np.max(np.abs(np.asarray(got) - expected)))
    if export_error > 1e-3:
        raise RuntimeError(f"NumPy GRU differs from PyTorch by {export_error}")
    report = dict()
    if args.stop_head:
        # Open-loop latch choice on recorded validation sequences: fire tick minus public stop tick.
        timing = {}
        for ticks in LATCH_CANDIDATES:
            offsets = []
            for features, _, _, stopped, _ in raw["validation"]:
                runtime = GRUPolicy(args.output, latch_ticks=ticks)
                runtime.reset(np.zeros(7))
                fire = None
                for k, f in enumerate(features):
                    runtime._update(f)
                    if runtime.latched:
                        fire = k
                        break
                truth = int(np.argmax(stopped[:, 0] > 0.5)) if stopped.any() else len(stopped)
                offsets.append(np.nan if fire is None else fire - truth)
            offsets = np.asarray(offsets, dtype=float)
            fired = np.isfinite(offsets)
            timing[ticks] = dict(never_fired=int((~fired).sum()), early_over_10=int((offsets[fired] < -10).sum()),
                                 median_offset=float(np.median(offsets[fired])) if fired.any() else None,
                                 mean_abs_offset=float(np.mean(np.abs(offsets[fired]))) if fired.any() else None)
        safe = [t for t, row in timing.items() if row["never_fired"] == 0 and row["early_over_10"] == 0]
        metadata["latch_ticks"] = min(safe or timing, key=lambda t: timing[t]["mean_abs_offset"] or np.inf)
        args.output.unlink()
        np.savez(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        report.update(latch_timing_validation_open_loop=timing, latch_ticks=metadata["latch_ticks"])
    report.update(best_epoch=best_epoch, epochs_run=len(history), validation_wheel_loss=history[best_epoch - 1]["validation_wheel"],
                  validation_arm_loss=history[best_epoch - 1]["validation_arm"],
                  validation_stop_loss=history[best_epoch - 1]["validation_stop"], export_max_error=export_error,
                  parameters=int(sum(p.numel() for p in model.parameters())),
                  training_wall_s=time.perf_counter() - started, test_data_opened=False)
    args.output.with_suffix(".json").write_text(json.dumps(dict(metrics=report, metadata=metadata, history=history), indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
