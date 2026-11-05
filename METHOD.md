# Method: GRU behaviour cloning policy

This document describes the final method (variant B) and the baseline (v3). The two are identical
except for four input features (Section 3). All numbers come from the final run; the result
summaries are in `results/final_run/`.

```
1500 random scenes ──expert in simulation──▶ 1391 successful trajectories (1 kHz)
        │
        ▼ whole-trajectory split 70/15/15 (seed 42): train 973 / validation 208 / test 210
        │
        ▼ every 10th step (100 Hz): 18 features; labels: wheel commands, joint velocities, stopped
        │
        ▼ GRU(18 → 128) + linear → 4 wheel commands + 7×3 matrix M + 1 stop logit
        │      loss = wheel MSE + arm velocity MSE (after the stop time) + stop BCE
        │
        ▼ export to NumPy; choose latch length N open-loop on validation
        │
        ▼ choose PD gains in closed loop on validation (12 combinations)
        │
        ▼ closed loop: GRU at 100 Hz, joint PD layer at 1 kHz → wheel_cmds, torque_arm
```

## 1. Data collection (`collect_data.py`, `robotnik_arm_task/robotnik_arm_collect.py`)

1500 scenes are sampled uniformly (seed 42):

| Quantity | Range |
|---|---|
| Target x, y | [−1.6, 1.6] m |
| Target z | [0.10, 1.30] m |
| Base start x, y | [−0.6, 0.6] m |
| Base start heading | [−180°, 180°] |
| Arm start angles | reference posture `[0, 1.57, 0, 1.0, 0, 1.4, 0]` ± 0.15 rad per joint |
| Arm start velocities | 0 |

The expert from the template runs each scene in PyBullet. Every 1 ms we record:

- `inputs` (21 columns): base x, y, z, yaw; 7 joint angles; 7 joint velocities; target x, y, z;
- `actions` (11 columns): 4 wheel commands, 7 joint torques;
- `ee_pos`: end-effector position, from forward kinematics of the measured state.

The collector also stores the expert's internal `phase` and `time`. **The learner never reads them.**
Only the expert's successful runs are kept as demonstrations (1391 of 1500).

## 2. Split (`staged_data.split_paths`)

- Split by **whole trajectory**, not by frame, so that neighbouring frames cannot leak between sets.
- 70/15/15 with seed 42: 973 / 208 / 210.
- All models share one `split.json`, so comparisons on the same scenes are paired.
- Normalisation statistics come from the training set only.

## 3. Input features (`staged_policy.gru_park_features`, `gru_features`)

18 features, expressed in the base frame (rotated by the base yaw):

| Dim | Variant B (final) | v3 (baseline) | Purpose |
|---|---|---|---|
| 3 | target relative to the base: x, y, dz | same | where to drive |
| 1 | horizontal distance to the target | same | how far |
| 2 | **parking position** relative to the base, predicted by the parking MLP | sin/cos of the target direction in the base frame | where to park |
| 2 | sin/cos of **parking yaw error** (predicted parking yaw − yaw) | sin/cos of the world bearing (atan2(target_y, target_x) − yaw) | which way to face |
| 3 | end-effector error (target − end-effector), base frame | same | where the arm should move |
| 7 | joint angles q | same | arm configuration |

Why these choices:

- **Relative coordinates.** The expert's base control law uses relative quantities only: distance to
  the parking pose, heading error and final yaw error. Giving the network relative features saves it
  from learning the rotation into the base frame.
- **The parking pose is not translation invariant.** The expert parks on the ray from the world origin
  to the target, at a standoff that depends on the target height, facing along that ray
  (`robotnik_arm_collect.py`). Relative coordinates alone do not say which side of the target to park on.
  - v3 adds the world-bearing feature. It is the same angle the expert uses, written by hand after
    reading the expert code.
  - Variant B learns the parking pose from data instead: an MLP maps the target to the recorded base
    pose at the public stop time (Section 4). It also drops the target-direction angle, which flips
    when the base passes under a high target.
- **Arm in the base frame.** The expert maps a world-frame error through an analytic Jacobian that
  includes the base pose. The network has no Jacobian, so the error is rotated into the base frame to
  remove the yaw dependence.
- **Angles as sin/cos**, to avoid the jump at ±π.
- **No joint velocities as input.** They are noisy, and the PD layer already uses them.

## 4. Labels (`staged_gru_train.sequence`, `staged_data.stop_index`)

Every 10th recorded step is used (`GRU_PERIOD = 10`, 100 Hz).

| Label | Source |
|---|---|
| Wheel commands (4) | recorded `actions[:, :4]` |
| Desired joint velocity (7) | central difference of recorded joint angles: (q[t+10] − q[t−10]) / 20 ms. These are observed states, not the expert's `qd_des` |
| Stopped (1) | 1 if the step is at or after the public stop time, else 0 |
| Parking pose (MLP) | recorded base x, y and yaw at the public stop time, one sample per trajectory |

**Public stop time:** the first step after which every recorded wheel command stays at 0
(|ω| ≤ 1e-3 rad/s). It needs only the recorded actions. A one-off check (`check_stop_label.py`, the only
script that reads the recorded phase) finds it equal to the expert's first `done` step on 1390 of 1391
demonstrations and 1 ms later on the other.

## 5. Networks

**GRU policy** (`staged_gru_train.GRUNet`, run-time `staged_policy.GRUPolicy`)

```
x_k (18) ─▶ GRU (hidden 128) ─▶ h_k ─▶ Linear(128 → 26)
                                      ├─ [0:4]   wheel commands (normalised)
                                      ├─ [4:25]  7×3 matrix M
                                      └─ [25]    stop logit
```

- 60,186 parameters: GRU 56,832, output layer 3,354.
- **What must be remembered (memory study, README Section 7.9).** A single frame cannot tell "almost
  there, slowing down" from "already parked": the expert's stage is not observed. Three levels on the
  same data and scenes (validation /208, fresh /300):
  - L1, four single-frame MLPs with a hand-coded latch: 189 / 272;
  - L2, this GRU without the stop output (stage only in the hidden state): 54 / 83. The wheels restart
    after stopping in 188 of 300 fresh runs, and the ungated arm drifts a median 0.31 rad while driving;
  - L3, this GRU with the latched stop output (B): 207 / 296.
  The stage is a discrete, irreversible event: it is best learned as an explicit, supervised and latched
  output, while the hidden state carries continuous context.
- **Why M·e and not torques.** The arm's desired velocity is v = M·e.
  - v = 0 whenever e = 0, so the arm stops at the target without drift.
  - M has the shape of a damped least-squares inverse kinematics map J^T(JJ^T + λ²I)^{-1}, learned
    from data without a Jacobian.
  - Torques mix gravity and dynamics. The same GRU with its arm head regressing torques directly
    (no PD layer) reaches 0/208 on validation.

**Parking MLP** (variant B; `staged_train.py`, stored in `base/policy.npz`)

- Input: the target (x, y, z). Output: parking x, y, and sin/cos of the parking yaw.
- Two hidden layers of 128 ReLU units, 17,540 parameters.
- Trained with MSE in normalised space: Adam with learning rate 1e-3, up to 200 epochs, patience 15.
  The best validation epoch was 187.

## 6. Loss (`loss_terms`, all weights 1)

With mask m (real frame vs padding), stop label c, and σ_v the standard deviation of the training
velocity labels:

```
L_wheel = Σ m ‖ŵ − w̃‖² / (4 Σ m)
L_arm   = Σ m c ‖(M e − v) / σ_v‖² / (7 Σ m c)        # only after the stop time
L_stop  = Σ m BCE(s, c) / Σ m
L       = L_wheel + L_arm + L_stop
```

- The arm loss uses only frames after the stop time, because at run time the arm holds its start pose
  until the latch.
- Dividing by σ_v brings the arm term to the same scale as the wheel term.
- The stop term is a Bernoulli negative log-likelihood, so σ(s_k) estimates
  P(stopped | x_1, …, x_k).
- The arm loss is computed only after the stop time, but its gradient flows back through the hidden
  state (full BPTT). The GRU can therefore store, while driving, information the arm needs later.

## 7. Training

| Item | Setting |
|---|---|
| Optimiser | Adam, learning rate 1e-3 |
| Batch | 32 whole trajectories, zero-padded with a mask |
| Back-propagation | full-sequence BPTT, no truncation |
| Gradient clipping | norm ≤ 1.0 |
| Early stopping | sum of the three validation losses, patience 8, at most 80 epochs; best epoch kept |
| Seed | 42 (seeds 1 and 2 also checked, Section 10) |
| Variant B | best epoch 78 of 80, 31 min on CPU |
| v3 | best epoch 68, stopped at 76, 29 min on CPU |

**Export.** The weights are saved to `gru.npz` and the run-time forward pass is written in NumPy, with
the gate order r, z, n as in PyTorch. On one validation sequence the largest difference from PyTorch
is 7.7e-6.

## 8. Stop latch (chosen open-loop on validation)

The robot latches once the stop logit has been positive for **N consecutive** GRU steps, and the latch
is never released. The recorded validation features are fed to the GRU, and the firing step is
compared with the public stop time (in GRU steps of 10 ms):

| N | B: never fired / early by > 10 / mean abs. offset | v3: never fired / early by > 10 / mean abs. offset |
|---|---|---|
| 1 | 0 / 10 / **12.2** | 0 / 18 / 19.1 |
| 2 | 0 / 9 / 12.6 | 0 / 14 / **11.3** |
| 3 | 0 / 9 / 13.1 | 0 / 13 / 11.8 |
| 5 | 0 / 9 / 14.4 | 0 / 10 / 13.0 |
| 10 | 0 / 8 / 18.5 | 0 / 8 / 17.0 |
| 20 | 0 / 5 / 27.7 | 0 / 4 / 21.3 |

The rule fixed in advance prefers a candidate that never misses and never fires more than 10 steps
early. No candidate meets it, so the code falls back to the smallest mean absolute offset: N = 1 for B,
N = 2 for v3. The latch turns the non-monotonic logit into a monotonic decision.

## 9. Run-time control (`GRUPolicy.step`)

- The GRU updates the wheel commands and M at 100 Hz and holds them between updates.
- The end-effector error e is recomputed every 1 ms, so v = M·e runs at 1 kHz.

Before the latch:
- the wheels follow the GRU;
- the arm holds its start pose q0 (reference q0, v = 0).

After the latch:
- the wheels are set to 0;
- the reference is integrated: `q_ref ← clip(q_ref + v·dt, q − lead, q + lead)`;
- the torque is `τ = Kp (q_ref − q) + Kd (v − qd) + g(q)`, with g(q) from pinocchio
  (`computeGeneralizedGravity`).
  - v is capped per joint at the largest absolute training label.
  - The simulator clips wheel speed and torque (±30 N·m).

**Gains.** Kp, Kd and lead are chosen for each model on all 208 validation scenes, from
Kp ∈ {50, 100, 200} × Kd ∈ {10, 30} × lead ∈ {0.1, 0.2} (`staged_arm_grid.py`).
- Rule, fixed in advance: most 1 cm successes, then shortest mean success time, then lowest fraction
  of torque-clipped steps.
- Chosen gains: B 200/30/0.2 (207/208); v3 100/10/0.1 (205/208).
- The result is insensitive to the gains: 10 of B's 12 combinations reach 206–207.
- The gains are not trained with the network. The labels are velocities, so the loss does not depend
  on them, and closed-loop success is not differentiable through PyBullet. They are treated as
  hyperparameters, as in the PID tuning lab.

**Gravity term.** At the chosen gains g(q) is not needed: with `--gravity zero`, B still reaches
207/208 on validation. The integrated reference behaves like an integral term, and Kp × lead = 40 N·m
exceeds the shoulder's gravity load (median recorded torque 17.7 N·m). At 50/30/0.1 (5 N·m) removing
g(q) drops B to 57/208.

## 10. Evaluation and results

See README Section 7 and `results/final_run/suite/summary.md` and `analysis/summary.md`. In brief:

- **Fresh test** (300 scenes, seed 2026, unfiltered): B 296, v3 289, expert 293. B vs expert p = 0.55;
  B vs v3 p = 0.092.
- **Robustness:** under noise, weaker actuators and end-effector pushes, B scores 59/60 and the expert
  60/60. Under a base push, B scores 56/60 and the expert 38/60.
- **Out of distribution:** higher targets B 40/40, expert 15/40; the other families are similar for
  both.
- **Seeds 42 / 1 / 2:** 207 / 207 / 208 on validation.
- **Learning curve** with 10 / 25 / 50 / 100% of the training trajectories: 157 / 193 / 195 / 207.
- **Holding for 2 s after success:** B 286/296, expert 286/293, v3 171/289. v3's oscillation comes
  from its gains (Kd = 10), not from its policy.

## 11. What is learned and what is fixed

| Learned from demonstrations | Fixed or hand-designed |
|---|---|
| Wheel commands | Latch rule (N consecutive; N chosen on validation) |
| When to stop (stop logit) | Arm holds before the latch; wheels stop after it |
| Where to park (B: parking MLP) | PD law; Kp, Kd, lead chosen on validation |
| Arm velocity map M(e, q) | Forward kinematics (robot model); gravity term (optional at the chosen gains) |
| | v3 only: world-bearing feature |
| | Velocity cap and actuator limits |

## 12. Teacher forcing and exposure bias

During training every input x_k comes from states the expert visited. In closed loop the inputs come
from states the policy itself produced. This mismatch is the main limitation of plain behaviour
cloning. 3 of B's 4 fresh-test failures are stop decisions (2 latches never fire, 1 fires at t = 0),
which is consistent with it; the latch makes such an error irreversible.
DAgger-style data aggregation is the natural next step.
