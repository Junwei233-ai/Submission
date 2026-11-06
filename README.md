# Behaviour Cloning for a Mobile Manipulator (COMP0245 2026/27)

We train a policy that maps the robot state (`base_pos`, `base_bearing`, `q_mes`, `qd_mes`) and the
target `TARGET_EE_POS` to the robot commands (`wheel_cmds` for 4 wheels, `torque_arm` for 7 arm joints),
by supervised learning on trajectories recorded from the expert controller in the assignment template.

- **Final method: B-FK.** One GRU drives both the base and the arm. It is helped by a small MLP that
  predicts the parking pose from the target and by an MLP that predicts the end-effector position from
  the joint angles (learned forward kinematics). The end-effector error e in v = M·e comes from this MLP
  **both in training and at run time**; the GRU never sees a URDF-computed e. The GRU has a stop head that
  is latched, and its arm output goes through a fixed joint PD layer without gravity term. **The policy
  uses no URDF, forward kinematics or gravity model**; pinocchio only scores the true error.
- **Variant B (reference).** The same design trained and run with the URDF forward kinematics and gravity
  term. Sections 7.2–7.10 report this variant unless they say otherwise; Section 7.1.1 and
  `results/final_run/fk_run/` hold B-FK.
- **Baseline: v3.** Same GRU, but with a hand-designed "world bearing" input feature instead of the
  learned parking pose.
- **Reference: the expert controller** from the template.

The learners are trained **only on recorded signals** (`inputs`, `actions`; `ee_pos` only as the
offline label of the forward-kinematics MLP, read by no other network). They never read
the expert's internal variables (`phase`, `q_des`, `qd_des`, navigation state) and never call the
expert at run time (Section 7.6). [METHOD.md](METHOD.md) gives the full method, including the maths.

## Quick start: see it work (no data collection or training needed)

Everything below uses only files in this repository: the frozen models in `results/final_run/` and the
scene definitions in `scenes/`. Use the RoboEnv Python environment (Section 2).

```bash
python demo.py                                   # PyBullet window: policy B-FK on 3 fresh test scenes, real time
python demo.py --controller expert --scenes 3    # the template expert on the same scenes, for comparison
python demo.py --scene-id scene_2026_0008        # one scene (the one shown in the report's hold figure)
python demo.py --no-gui --set fresh              # all 300 fresh test scenes headless; prints 288/300
python demo.py --no-gui --set fresh --controller B  # the URDF reference on the same scenes; prints 296/300
```

In the window the red sphere is the target; the status line shows time, stage (driving / reaching) and
the end-effector error, and turns green on success. Press `q` to skip a scene. The terminal prints, for
each scene, the outcome, time, final error and `expert module loaded: False` (the learned policy never
imports the expert's code). `--speed 2` plays twice as fast. Other scene sets: `--set robustness` (with
e.g. `--perturb "--push-base 20"`), `--set high_target`, `far_target`, `low_target`, `wide_start`.
Running the 300 bundled fresh scenes reproduces the frozen evaluation scene by scene on the same platform;
on another platform floating-point differences can flip 1–3 borderline scenes (e.g. 288 vs 289).

To rerun the whole project from scratch (collect data, train, select gains, evaluate), see Section 3.1.

## How to check our results

From quickest to most complete; every step uses only this repository unless noted.

1. **Watch it** (seconds): `python demo.py` (GUI). The terminal shows success, error and
   `expert module loaded: False`.
2. **Re-evaluate the frozen final policy** (about 5 min on 18 cores): `python demo.py --no-gui --set fresh`
   prints 288/300 next to the reported number. Other sets: `--set low_target` (3/40),
   `--set robustness --perturb "--push-base 20"` (53/60), `--controller expert` for the template.
3. **Regenerate every summary, table and figure of the report** (seconds):
   `python summarize_fk.py && python make_report_assets.py`, then `git status` shows no change: the report's
   numbers, tables and PDF figures follow byte for byte from the versioned per-scene results.
4. **Re-run all B-FK evaluations** (about 1 h): `bash run_fk_study.sh L3 | L2 | L1 | hold | robust | ood`
   (Section 3.3). They overwrite `results/final_run/fk_run/<run>/`, so `git diff` shows any difference.
5. **Retrain from scratch** (several hours, needs the expert collection): `bash run_fk_pipeline.sh`
   (Section 3.1).

## Results at a glance

Fresh held-out test set: 300 new scenes (seed 2026), not used for any choice, each controller run once.

| Controller | 1 cm success | 95% Wilson CI | Stays within 1 cm for 2 s after success |
|---|---|---|---|
| **B-FK (ours, final; no robot model in the policy)** | **288 / 300 (96.0%)** | 93.1–97.7% | 277 |
| Variant B (same design, URDF FK and gravity term) | 296 / 300 (98.7%) | 96.6–99.5% | 286 |
| v3 (ours, baseline) | 289 / 300 (96.3%) | 93.6–97.9% | 171 |
| Expert (template controller) | 293 / 300 (97.7%) | 95.3–98.9% | 286 |

B-FK's robustness, out-of-distribution, hold and memory-study results are in Section 7.1.1; Sections
7.2–7.10 give the same studies for the URDF variant B, plus seeds, learning curve and ablations. Every number in this README is generated by the scripts below. The summaries are versioned in
[`results/final_run/`](results/final_run/).

---

## 1. Repository layout

### 1.1 Our code (all written from scratch)

| File | Purpose |
|---|---|
| `collect_data.py` | Samples random scenes and runs the expert collector on each in parallel. Writes one `.npz` trajectory per scene plus `summary.csv` |
| `staged_data.py` | Shared utilities: the **public stop time** (first step after which all recorded wheel commands stay zero), parking pose, whole-trajectory 70/15/15 split |
| `staged_train.py` | Writes the shared `split.json` and trains the parking-pose MLP that variant B uses (target → parking x, y, yaw). Also trains the wheel and stop MLPs of the four-network policy (level L1 of the memory study). `--train-fraction` keeps a nested subset of the training trajectories (learning curve) |
| `staged_fk_train.py` | Trains the forward-kinematics MLP (joint angles → end-effector position in the base frame) on expert runs with a 2 s hold; it replaces the URDF at run time |
| `run_fk_pipeline.sh` | **One command for the final method:** collect, split and parking MLP, torque limit, GRU-B, expert hold runs, forward-kinematics MLP, validation and fresh test |
| `run_fk_study.sh` | Evaluates the frozen B-FK models (L3, L2, L1), the hold test, robustness and out-of-distribution runs with the learned forward kinematics, using only repository files |
| `summarize_fk.py` | Summary of the B-FK runs: Wilson CIs, paired McNemar tests against the expert and variant B, memory mechanisms, hold and compute time (`fk_run/summary.md`) |
| `staged_gru_train.py` | Trains the GRU policy. Options: `--stop-head` (stop logit and latch), `--features bearing\|park` (v3 or variant B inputs), `--arm-output matrix\|torque` (direct-torque ablation), `--fk-model` (end-effector error from the learned forward kinematics instead of the recorded URDF value; B-FK) |
| `staged_policy.py` | NumPy run-time for all policies (no PyTorch needed at run time) and the input feature functions |
| `staged_rollout.py` | Closed-loop simulation of one scene. Base and arm controllers can be chosen independently: learned or expert. Evaluation options: perturbations, `--hold-time`, `--gravity zero` |
| `staged_evaluate.py` | Runs `staged_rollout.py` in parallel over a split or a scene directory and reports success rate, Wilson CI and failure reasons |
| `staged_arm_grid.py` | Grid search of the PD gains Kp, Kd and lead on the validation set, with a selection rule fixed in advance |
| `staged_arm_train.py` | Arm MLP of the four-network policy (level L1 of the memory study; not used by the final method); `--fk-model` as for the GRU |
| `sim_utils.py` | Quaternion to yaw and finger holding, implemented independently so that learned rollouts do not import the expert module |
| `test_sim_utils.py` | Checks `quaternion_yaw` against the template's `quaternion2bearing` on 2000 random quaternions |
| `make_ood_scenes.py` | Generates four out-of-distribution scene families |
| `run_final_pipeline.sh` | Trains v3 and B, grid-searches their gains on validation, checks reproduction on the development test split |
| `report_reproduction.py` | Compares the retrained results with the development-phase numbers |
| `compute_torque_limit.py` | Writes `torque_limit.json`: the largest torque magnitude in the training trajectories, used as the run-time arm torque limit |
| `run_evaluation_suite.sh` | Post-freeze evaluation: fresh test set, robustness, out-of-distribution |
| `summarize_suite.py` | Summary of the suite: Wilson CIs, exact McNemar tests, compute time, model size |
| `run_bc_analyses.sh` | Post-freeze analyses: training seeds, learning curve, gravity-term ablation, holding after success |
| `summarize_analyses.py` | Summary of those analyses |
| `run_memory_study.sh` | Memory study (report Section III-B): retrains the four-network policy (L1) and the GRU without stop head (L2) on the final data and split, evaluates both on validation and the fresh test; also the direct-torque ablation |
| `summarize_memory.py` | Summary of the memory study: success, Wilson CI, paired McNemar against B, failure mechanisms (base never done, wheels restarting, arm drift, creep) from the logged roll-outs |
| `check_stop_label.py` | Verification only: compares the public stop label with the expert's recorded phase (the only script that reads `phase`; nothing is trained from it) |
| `demo.py` | One-command demo and check: plays a frozen controller in the PyBullet GUI, or runs a bundled scene set headless and prints the success rate (Quick start) |
| `staged_direct_train.py` | Plain regressors (L0 of the memory study): ridge and MLP from the 21 raw inputs to the 11 raw actions, no structure |
| `run_direct_baseline.sh` | Trains L0 and evaluates the MLP on validation and the fresh test |
| `analyze_arm_matrix.py` | Interpretability check (after freezing, nothing trained): replays the frozen GRU on B's fresh-test roll-outs and compares its arm matrix M with the arm Jacobian (alignment, task-space gain, null-space share, cosine with J⁺ and Jᵀ); the same measures on the expert's public velocity labels (Section 7.10) |
| `make_report_assets.py` | Builds the report's figures (`report/figures/*.pdf`) and LaTeX tables (`report/tables/*.tex`) from `results/final_run/`, so the report cannot drift from the results |

`scenes/` holds the scene definitions (target, base start pose, initial arm angles; one small JSON per
scene) of the validation set (208), the fresh test set (300), its robustness subset (60) and the four
out-of-distribution families (40 each), so every evaluation can be rerun without the full data.

### 1.2 Template and expert data collection: `robotnik_arm_task/`

| File | Origin | Role |
|---|---|---|
| `robotnik_arm_task.py` | Template, **unmodified** | Expert demo: base drives to a parking pose, then the arm reaches the target |
| `robotnik_arm_control.py` | Template, **unmodified** | Expert controllers (`base_controller`, `arm_controller`) and `check_stop` |
| `robotnik_arm_collect.py` | **Our modified copy** of `robotnik_arm_task.py` | Headless single-scene collector (changes in Section 4) |
| `configs/`, `models/` | Template (RoboEnv) | Robot configuration, URDF and meshes |
| `README.md` | Template, **unmodified** | Template instructions (our collector is documented in Section 4) |

### 1.3 Report: `report/`

`report/main.tex` is the IEEE-format report (compile with `pdflatex main.tex` inside `report/`, or upload
the folder to Overleaf). Its figures and tables are generated by `make_report_assets.py`.

### 1.4 Results snapshot: `results/final_run/`

This is a small versioned copy (about 3 MB) of the final run. The full run (about 23 GB) writes to
`artifacts/final_run/` and is not committed.

| Path | Content |
|---|---|
| `suite/summary.md`, `suite/summary.json` | Fresh test, robustness, out-of-distribution, efficiency |
| `analysis/summary.md`, `analysis/summary.json` | Seeds, learning curve, gravity ablation, holding test |
| `reproduction.md` | Reproduction check against the development-phase numbers |
| `fk_run/fk.npz`, `fk_run/fk.json` | **Frozen forward-kinematics MLP** of B-FK and its accuracy |
| `fk_run/gru_B_fk/`, `fk_run/memory/` | **Frozen B-FK models**: GRU (L3), GRU without stop head (L2), L1 arm MLP, all trained on the learned FK |
| `fk_run/<run>/` | Metrics, configuration and per-scene `results.json` of the B-FK runs written by `run_fk_study.sh` |
| `fk_run/summary.md`, `fk_run/summary.json`, `fk_run/arm_matrix.json` | B-FK summary (`summarize_fk.py`) and arm-matrix analysis |
| `fk_run/reference/` | Per-scene results reused for paired tests: expert runs (unaffected by the FK) and variant B on the fresh test |
| `fk_run/urdf_trained_gru/` | The earlier B-FK whose GRU was trained on URDF e (ablation) |
| `gru_B_park/`, `gru_v3_bearing/` | **Frozen models** (`gru.npz`) and training records (`gru.json`) |
| `base/` | `split.json` (train 973 / validation 208 / test 210) and `policy.npz` (holds the parking MLP) |
| `grid_*/grid.json` | Gain grids and the chosen gains |
| `fresh_split/split.json`, `fresh_test_seed2026/summary.csv` | Fresh test scenes |
| `*/metrics.json`, `*/config.json` | Metrics and configuration of every evaluation run |
| `public_data_provenance.json` | Hashes of the training data (see Section 5) |
| `analysis/hold_trace.csv` | Per-step error of one scene during the hold test (report figure) |
| `direct/` | L0 plain regressors (`mlp.npz`, `linear.npz`, open-loop errors in the `.json`) and their evaluation metrics |
| `memory/summary.md`, `memory/summary.json` | Memory study and direct-torque ablation (Section 7.9) |
| `memory/gru_nostop/`, `memory/fournet_arm/`, `memory/gru_torque/` | L2 model, L1 arm MLP, direct-torque model (L1's park, wheel and stop MLPs are in `base/policy.npz`) |
| `analysis/arm_matrix.json` | What the arm matrix M learned (Section 7.10) |
| `memory/trace_scene_2026_0012.csv` | Per-step wheel command and EE error of the four memory levels in one fresh scene (report figure) |
| `stop_label_check.json` | Public stop label vs the expert's recorded phase on all 1391 demonstrations |

Note: the `labels` text in the metadata of the two frozen models still says "no stop/phase". It is an
outdated description string; the training script's text was corrected after these models were
trained. The models do have a stop head: see `stop_head: true`, `feature_set` and `arm_output: matrix`
in the same metadata.

---

## 2. Requirements

- Python with the RoboEnv packages used in the labs: `pybullet`, `pinocchio`, `simulation_and_control`
- `numpy`, `torch` (training only; the run-time policy is NumPy)
- CPU only. On 20 CPU cores, one GRU training run takes about 30 minutes and one pass over the 208
  validation scenes with 18 parallel workers takes about 10 minutes.

Run every command from the repository root. Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` when
running many simulations in parallel; the shell scripts do this themselves.

## 3. How to run

### 3.1 Final method (B-FK), one command

```bash
PY=/path/to/RoboEnv/python bash run_fk_pipeline.sh artifacts/fk_run 8
```

It collects the demonstrations, trains the parking MLP, runs the expert with a 2 s hold, trains the
forward-kinematics MLP, trains GRU-B on the MLP's end-effector estimate (`staged_gru_train.py --fk-model`),
and evaluates on validation and the fresh test. The gains are fixed at
200/30/0.2 and are not searched, so this step needs no grid. Results go to `artifacts/fk_run/`. The gains
can be rerun with `staged_arm_grid.py` (Section 3.2).

### 3.1.1 Full study with the URDF variant

```bash
PY=/path/to/RoboEnv/python
$PY collect_data.py --n 1500 --seed 42   --workers 6 --output artifacts/final_run/collection          # training data
$PY collect_data.py --n 300  --seed 2026 --workers 6 --output artifacts/final_run/fresh_test_seed2026 # fresh test scenes
SOURCE=artifacts/final_run/collection PY=$PY bash run_final_pipeline.sh artifacts/final_run 9   # train, gain grids, reproduction
PY=$PY bash run_evaluation_suite.sh artifacts/final_run 18 60                                     # fresh test, robustness, OOD
PY=$PY bash run_bc_analyses.sh     artifacts/final_run 18 3                                       # seeds, learning curve, ablations, hold
PY=$PY bash run_memory_study.sh    artifacts/final_run 9                                          # memory study, torque ablation
PY=$PY bash run_direct_baseline.sh artifacts/final_run 5                                          # L0 plain regressors
$PY check_stop_label.py --collection ORIGINAL_COLLECTION_WITH_PHASE --output artifacts/final_run/stop_label_check.json
$PY analyze_arm_matrix.py --rollouts artifacts/final_run/suite/fresh_B/rollouts                  # what M learned
$PY -m unittest test_sim_utils
$PY make_report_assets.py --artifacts artifacts/final_run                                          # report figures and tables
```

| Output | Content |
|---|---|
| `artifacts/final_run/gru_*/gru.npz` | Trained policies |
| `artifacts/final_run/grid_*/grid.json` | Gain grid and chosen gains |
| `artifacts/final_run/reproduction.md` | Reproduction check |
| `artifacts/final_run/suite/summary.md` | Main results (Sections 7.2–7.5) |
| `artifacts/final_run/analysis/summary.md` | Supplementary analyses (Section 7.7) |
| `artifacts/final_run/memory/summary.md` | Memory study and torque ablation (Section 7.9) |
| `.../rollouts/<scene>/rollout.npz`, `result.json` | Per-scene trajectories and outcome of every evaluation |

The scripts refuse to overwrite an existing output directory. If interrupted, rerunning skips
finished steps.

### 3.2 Individual steps

```bash
PY=python; OUT=artifacts/final_run
$PY staged_train.py --source $OUT/collection --output $OUT/base          # split + parking MLP
$PY staged_gru_train.py --model-dir $OUT/base --stop-head --features park    --output $OUT/gru_B_park/gru.npz      # variant B
$PY staged_gru_train.py --model-dir $OUT/base --stop-head --features bearing --output $OUT/gru_v3_bearing/gru.npz  # v3
$PY staged_gru_train.py --model-dir $OUT/base --stop-head --features park --arm-output torque --output $OUT/memory/gru_torque/gru.npz # ablation
$PY staged_gru_train.py --model-dir $OUT/base --features park --output $OUT/memory/gru_nostop/gru.npz   # L2: no stop head
$PY staged_arm_grid.py --model-dir $OUT/base --gru-model $OUT/gru_B_park/gru.npz --limit 0 --workers 9 --output $OUT/grid_B_park
$PY staged_evaluate.py --model-dir $OUT/fresh_split --split test_unfiltered --base gru --arm gru \
    --gru-model $OUT/gru_B_park/gru.npz --gains $OUT/grid_B_park/grid.json --workers 18 --output $OUT/my_eval
$PY staged_evaluate.py --model-dir $OUT/fresh_split --split test_unfiltered --base expert --arm expert \
    --workers 18 --output $OUT/my_eval_expert
# optional flags: --obs-noise 0.01 | --torque-scale 0.8 | --push-ee 30 | --push-base 20 |
#                 --hold-time 2 | --gravity zero | --kp/--kd/--lead | --scenes-dir DIR
$PY staged_rollout.py --scene SCENE.json --base gru --arm gru --gru-model $OUT/gru_B_park/gru.npz \
    --kp 200 --kd 30 --lead 0.2 --output OUT_DIR                          # one scene
```

To run the frozen models without retraining, use `results/final_run/gru_B_park/gru.npz` with gains
200/30/0.2, or `results/final_run/gru_v3_bearing/gru.npz` with gains 100/10/0.1.

### 3.3 Frozen B-FK models

The frozen B-FK models are `results/final_run/fk_run/fk.npz` (forward kinematics), `fk_run/gru_B_fk/`
(L3 = B-FK), `fk_run/memory/gru_nostop_fk/` (L2) and `fk_run/memory/fournet_arm_fk/` (L1 arm; its base MLPs
are in `base/policy.npz`). The GRUs and the L1 arm were trained with `--fk-model results/final_run/fk_run/fk.npz`, using the same
split and settings as the URDF study (the split's `source` must point to the demonstration collection):

```bash
FK=results/final_run/fk_run/fk.npz; B=results/final_run/base; O=artifacts/fk_retrain
$PY staged_gru_train.py --model-dir $B --stop-head --features park --fk-model $FK --output $O/gru_B_fk/gru.npz     # L3 = B-FK
$PY staged_gru_train.py --model-dir $B --features park --fk-model $FK --output $O/gru_nostop_fk/gru.npz            # L2
$PY staged_arm_train.py --model-dir $B --fk-model $FK --output $O/fournet_arm_fk/arm.npz                           # L1 arm
```
No data collection or training is needed to evaluate them. Each task writes to `artifacts/fk_study/` and copies the
metrics and per-scene results to `results/final_run/fk_run/`.

```bash
PY=python bash run_fk_study.sh L3 8        # B-FK: validation and fresh test
PY=python bash run_fk_study.sh L2 8        # L2: validation and fresh test
PY=python bash run_fk_study.sh L1 8        # L1: validation and fresh test
PY=python bash run_fk_study.sh robust 8    # B-FK: five robustness conditions
PY=python bash run_fk_study.sh ood 8       # B-FK: four out-of-distribution families
PY=python bash run_fk_study.sh hold 8      # B-FK: 2 s hold after success
PY=python bash run_fk_study.sh expert 8    # expert on the same scenes (also in fk_run/reference/)
python summarize_fk.py                      # fk_run/summary.md and summary.json
```

All learned levels use gains 200/30/0.2 and no gravity term, so they share one arm layer.

---

## 4. External code and what we changed

| External code | Used for | Modified? |
|---|---|---|
| Template `robotnik_arm_task.py`, `robotnik_arm_control.py`, `configs/`, `models/` | Expert controller, simulation set-up | No |
| Template `robotnik_arm_task.py` → our `robotnik_arm_collect.py` | Expert data collection | **Yes.** Scene parameters (target, base start pose, initial arm angles) come from `--scene`. It runs headless and saves the trajectory (`inputs`, `actions`, `ee_pos`, `phase`, `time`). It records the `check_stop` outcome as `success` / `stalled` / `step_limit`, and additionally requires the final error to be below 1 cm for `success` (Section 8) |
| RoboEnv `simulation_and_control` | PyBullet interface (`SimInterface`, `PinWrapper`, `MotorCommands`) | No |
| pinocchio | B-FK: scoring and recording the end-effector position only. Variant B, v3, L1 and L2 in Section 7.9: also forward kinematics and gravity term g(q) at run time | Called directly |
| PyTorch | Training | Called directly |

The simulation settings in `staged_rollout.py` (ground friction, wheel contact, motor gains, finger
angles) are copied from `robotnik_arm_collect.py`, so evaluation matches collection. The expert's
parking-pose rule (standoff from the target) is only used when the **expert** base runs. Its `--gui`
option (used by `demo.py`) follows the template's GUI mode: `use_gui=True`, real-time pacing and the `q`
key to stop; the target marker, status text and camera placement are ours.

All other `.py` files in the repository root are written from scratch by us.

## 5. Data

- **Scenes:** 1500 random scenes (seed 42).
  - Target x, y ∈ [−1.6, 1.6] m, z ∈ [0.10, 1.30] m.
  - Base start x, y ∈ [−0.6, 0.6] m, any heading.
  - Arm start: the reference posture `[0, 1.57, 0, 1.0, 0, 1.4, 0]` ± 0.15 rad per joint.
  - Recorded at 1 kHz.
- **Demonstrations used:** the expert's successful runs only, 1391 of 1500.
- **Split:** by whole trajectory, train 973 / validation 208 / test 210 (seed 42). Normalisation
  statistics come from the training set only.
- **Expert runs with a hold:** the expert runs the training and validation scenes again and continues
  for 2 s after success (`staged_evaluate.py --hold-time 2`). It succeeds in 967 of 973 training scenes and
  208 of 208 validation scenes, ending a median of 0.6 mm from the target. Only these successful runs
  train the forward-kinematics MLP.
- **Fresh test set:** 300 further scenes (seed 2026) from the same ranges, collected after freezing.
  It is not filtered by expert success.
- **What the learner reads:** `inputs` (base x, y, z, yaw, 7 joint angles, 7 joint velocities, target
  x, y, z), `actions` (4 wheel commands, 7 torques) and `ee_pos` (forward kinematics of the
  measurements). The collector also stores the expert's internal `phase` and `time`. **No training or
  run-time code reads them.** The stop label is derived from the recorded wheel commands alone
  (`staged_data.stop_index`). Checked once with `check_stop_label.py` (verification only): it equals the
  first step of the expert's internal `done` phase on 1390 of 1391 demonstrations and is 1 ms later on the
  other (`results/final_run/stop_label_check.json`).
- **Provenance:** the final run trained on a copy of the collection with the `phase` and `time` fields
  removed; every other array is bit-identical. `results/final_run/public_data_provenance.json` lists the
  SHA-256 of each original and copied file. Training directly on the original collection gives the
  same arrays, so Section 3.1 reproduces the run.

## 6. Method summary

Full details are in [METHOD.md](METHOD.md).

- **Inputs:** 18 features every 10 ms, in the base frame:
  - target position relative to the base and horizontal distance;
  - parking pose relative to the base, predicted from the target by the parking MLP (variant B);
    v3 uses hand-designed target-direction and world-bearing angles instead;
  - end-effector error;
  - 7 joint angles.
- **Network:** GRU (128 hidden units, 60,186 parameters) updated at 100 Hz. Outputs: 4 wheel commands,
  a 7×3 matrix M, and a stop logit. Variant B adds the 17,540-parameter parking MLP.
- **Forward kinematics (B-FK):** an MLP (two hidden layers of 512 units, 268,291 parameters) maps the
  joint angles to the end-effector position in the base frame, learned from recorded positions. On
  validation, its error on the final frame has median 2.0 mm and p95 6.2 mm. The GRU's training features
  use this MLP's estimate too, so training and run time see the same e.
- **Arm:** the desired joint velocity is v = M·e, where e is the end-effector error. A reference is
  integrated, `q_ref ← clip(q_ref + v·dt, q ± lead)`, and tracked by a joint PD layer at 1 kHz:
  `τ = Kp (q_ref − q) + Kd (v − qd)` in B-FK, plus `g(q)` in variant B. Its labels are joint velocities computed by central
  differences of the recorded joint angles. These are observed states, not expert torques.
- **Stop:** the stop logit is trained with BCE on the public stop time and latched after N
  consecutive positive outputs. Before the latch the wheels follow the GRU and the arm holds its start
  pose; after it the wheels stop and the arm reaches.
- **Training:**
  - loss: wheel MSE + arm velocity MSE (only after the stop time) + stop BCE;
  - optimiser: Adam with learning rate 1e-3, batch of 32 whole trajectories;
  - full BPTT with gradient clipping at 1.0;
  - early stopping on validation loss (patience 8, at most 80 epochs).
- **Hyperparameters chosen on validation:** latch length N, chosen open-loop; PD gains Kp, Kd and lead,
  chosen in closed loop on all 208 validation scenes over 12 combinations.
  Chosen gains: variant B 200/30/0.2, v3 100/10/0.1. B-FK reuses B's gains (selected with the URDF
  forward kinematics and gravity term); they were not searched again. B-FK's latch length (N = 1) is
  re-chosen open-loop on validation.

## 7. Evaluation protocol and results

### 7.1 Protocol

- The protocol was fixed before results were seen.
  1. Retrain v3 and B on the same data and split.
  2. Choose each model's gains on validation.
  3. Check reproduction on the development test split.
  4. Freeze the models.
  5. Evaluate once on the fresh test set.
  6. Run robustness and out-of-distribution tests on fresh scenes.
- The development test split had been used three times during development, and variant B was
  designed after looking at its failures. So it is used only for the reproduction check.
- **Success** follows the template's `check_stop` reach condition: base done (for the GRU: latched),
  end-effector error < 1 cm and t > 8 s. The episode limit is 40 s.
- **Statistics:** Wilson 95% intervals; paired comparisons on identical scenes use the exact
  two-sided McNemar test.

### 7.1.1 Final method without the robot model (B-FK)

GRU, L1 arm and L2 trained on the learned forward kinematics' end-effector estimate; run with the same
estimate, no gravity term and gains 200/30/0.2. Summary: `results/final_run/fk_run/summary.md`
(`summarize_fk.py`). All runs on one machine; runs are deterministic there (a second fresh-test run gave
bit-identical results), while across platforms 1–3 borderline scenes can flip.

| | 1 cm success | 95% Wilson CI | 5 cm success | Base latched |
|---|---|---|---|---|
| Validation (208) | 200 | 92.6–98.0% | 206 | 207 |
| Fresh test (300) | **288** | 93.1–97.7% | 296 | 299 |

- **vs expert** (293/300): discordant 6 / 11, p = 0.33. **vs variant B** (296/300): 2 / 10, p = 0.039.
- **Holding 2 s:** 277 of 288 successes stay within 1 cm (expert 286/293, p = 0.09); median end error
  2.3 mm (expert 0.6 mm): the arm converges to where the forward-kinematics MLP places the target.
- **Failures (12):** 1 latch never fires; 1 fires at t ≈ 0 with the end-effector 47 cm away; 8 come no
  closer than 1.1–2.4 cm (the MLP's error range); 2 reach only 7–9 cm with the torque saturated in most steps.
  B-FK solves 10 of the 14 scenes where the expert failed during collection.
- **Memory study (learned FK for every level):** L1 168/208 and 239/300; L2 43/208 and 86/300 (wheels
  restart after "done" in 148/300); L3 = B-FK 200/208 and 288/300. L3 vs L1 52/3, p = 1.5e-12.
- **Robustness (60 scenes; B-FK / expert):** noise 0.003 and 0.01, torque 80%, EE push 30 N: 58 / 60
  each (the 2 failures are the unperturbed failures); base push 20 N: **53** / 38, p = 0.0015.
- **Out of distribution (40 each; B-FK / expert / variant B):** far 40 / 40 / 40; high 17 / 15 / 40;
  low 3 / 30 / 29; wide start 29 / 39 / 38. The learned FK does not extrapolate: at the end of the
  low-target runs its own error has a median of 4.8 cm (high targets 1.8 cm; in distribution 0.2 cm).
- **FK frame sampling** was chosen on validation from two candidates, with the GRU then trained on URDF e
  (commit `d60e55f`): all frames every 5 steps gave 196/208, every 2nd frame after the stop and every 20th
  before gave 198/208 (discordant 2 / 4, not significant); the second is used. The test set was not used.
- **GRU trained on URDF e, run with the learned FK** (the earlier B-FK, `fk_run/urdf_trained_gru/`):
  198/208 and 288/300 on this machine (289/300 on a team member's machine). Retraining on the learned e
  makes training and run time consistent without changing the fresh-test result.
- **What the arm learned** (`analyze_arm_matrix.py --gru-model fk_run/gru_B_fk/gru.npz --fk-model
  fk_run/fk.npz`, `fk_run/arm_matrix.json`): cos(J M e, e) median 0.994; rate 1.49 /s; null-space share
  17%; matrix cosine with J⁺ 0.95, with Jᵀ 0.57.
- **Compute:** 342 µs per 1 kHz step (FK MLP every step); 346,017 learned parameters.

### 7.2 Accuracy: fresh test set (300 scenes)

| | 1 cm success | Expert succeeded during collection (286 scenes) | Expert failed during collection (14 scenes) |
|---|---|---|---|
| **B** | 296 (98.7%, CI 96.6–99.5) | 282 | 14 |
| v3 | 289 (96.3%, CI 93.6–97.9) | 277 | 12 |
| Expert | 293 (97.7%, CI 95.3–98.9) | 286 | 7 |

Paired McNemar tests (discordant scenes, first controller only / second controller only):
- B vs expert: 7 / 4, p = 0.55.
- B vs v3: 10 / 3, p = 0.092.

B fails 4 scenes. In 2 the latch never fires; in 1 it fires at t = 0, with the base 53 cm from the
parking pose (scene 0197); in 1 the arm stops 6 cm from the target (scene 0191).

### 7.3 Robustness (60 fresh scenes; B and expert see the same scenes and perturbations)

| Condition | B | Expert | Paired p |
|---|---|---|---|
| Observation noise σ = 0.003 (m, rad; joint velocities 10σ) | 59 | 60 | 1 |
| Observation noise σ = 0.01 | 59 | 60 | 1 |
| Arm actuators deliver 80% of the torque | 59 | 60 | 1 |
| 30 N push on the end-effector for 0.2 s, 1 s after the base stops | 59 | 60 | 1 |
| 20 N push on the base for 0.2 s at t = 2 s | **56** | 38 | 4e-5 |

- B's single failure in the first four conditions is the same scene. It also fails without any
  perturbation (the latch never fires).
- After the base push, the expert's navigation state machine does not recover before its 25 s timeout,
  and the base ends a median 1.6 m from the parking pose. The GRU re-plans every 10 ms from the current
  pose. The simulated wheels have very low lateral friction (`anisotropicFriction=[1, 0.01, 1]`), so
  20–40 N pushes already cause slides of about a metre.

### 7.4 Generalisation: out-of-distribution scenes (40 scenes per family)

| Family | B | Expert | Paired p |
|---|---|---|---|
| Farther targets, max(\|x\|, \|y\|) ∈ (1.6, 2.2] m | 40 | 40 | 1 |
| Higher targets, z ∈ (1.30, 1.40] m | **40** | 15 | 6e-8 |
| Lower targets, z ∈ [0.03, 0.10) m | 29 | 30 | 1 |
| Wider start, base ±1.2 m and joints ±0.30 rad | 38 | 39 | 1 |

- **Higher targets:** the expert projects the target onto a sphere of 0.85 m around the shoulder and
  stops 1.8–5.8 cm short. B never learned this hard-coded limit and extrapolates.
- **Lower targets:** both controllers reach only about 73%. Near the floor the arm is close to joint
  limits and singular configurations.

### 7.5 Efficiency

| | Compute per 1 kHz control step | Parameters | Model file |
|---|---|---|---|
| B | 87 µs | 60,186 (GRU) + 17,540 (parking MLP) | 616 KiB |
| v3 | 82 µs | 60,186 | 477 KiB |
| Expert (IK + PD + gravity) | 82 µs | — | — |

The time covers features, the GRU update every 10th step (NumPy), and the PD and gravity terms. It is
well inside the 1 ms control period. Training takes about 30 minutes on CPU. Section 7.7 gives the
data efficiency.

### 7.6 Isolation from the expert

`staged_rollout.py` imports `robotnik_arm_control.py` only when `--base expert` or `--arm expert` is
given. Every learned rollout records `expert_module_loaded: false` in its `result.json`, and the run
aborts if the module is loaded. The two helper functions it needed are re-implemented in
`sim_utils.py`.

### 7.7 Supplementary analyses (after freezing; frozen models and gains unchanged)

All validation runs use B's chosen gains 200/30/0.2.

**Training seeds** (GRU seed only; validation, 208 scenes)

| Seed | Success | Best epoch | Latch length |
|---|---|---|---|
| 42 (frozen) | 207 | 78 | 1 |
| 1 | 207 | 52 | 2 |
| 2 | 208 | 78 | 3 |

**Learning curve** (parking MLP and GRU retrained on nested subsets; validation, 208 scenes)

| Training data | Trajectories | Success | Validation loss wheel / arm / stop | Failures: never latched / latched but missed | Median parking offset |
|---|---|---|---|---|---|
| 10% | 97 | 157 | 0.283 / 0.858 / 0.162 | 24 / 27 | 15.2 cm |
| 25% | 243 | 193 | 0.156 / 0.465 / 0.108 | 5 / 10 | 10.8 cm |
| 50% | 486 | 195 | 0.086 / 0.249 / 0.042 | 1 / 12 | 8.7 cm |
| 100% | 973 | 207 | 0.054 / 0.163 / 0.021 | 1 / 0 | 7.5 cm |

With less data, most failures come from the base. At 10%, the stop decision is often never made. At
25–50%, the base parks where the arm cannot reach: the median error of the latched failures is
0.55–0.64 m.

**Gravity term of the PD layer** (`--gravity zero`; validation, paired)

| Gains Kp/Kd/lead | With g(q) | Without g(q) | Discordant (p) |
|---|---|---|---|
| 200/30/0.2 (chosen) | 207 | 207 | 0/0 (1) |
| 50/30/0.1 | 203 | 57 | 147/1 (8e-43) |

The integrated reference acts like an integral term. If Kp × lead exceeds the shoulder's gravity load
(median recorded torque 17.7 N·m), the PD layer holds the arm on its own. So the analytic gravity
model is not needed at the chosen gains.

**Holding after success** (`--hold-time 2`; fresh test, 300 scenes)

| Controller (gains) | Success | Within 1 cm for the whole 2 s | Within 1 cm over the last 1 s | Worst error in the last 1 s, median / p95 |
|---|---|---|---|---|
| **B** (200/30/0.2) | 296 | **286** | 296 | 2.2 / 3.6 mm |
| Expert | 293 | 286 | 293 | 2.4 / 3.4 mm |
| v3 (100/10/0.1) | 289 | 171 | 228 | 5.7 / 19.0 mm |
| Diagnosis: v3 with 100/30/0.1 | 289 | 278 | 289 | 2.3 / 4.1 mm |
| Diagnosis: B with 100/10/0.1 | 296 | 144 | 217 | 6.1 / 23.9 mm |

- B holds as well as the expert (paired 8/8, p = 1). B vs v3: 122/7, p = 3e-28.
- v3's problem is its gains, not its policy. With Kd = 10 both policies oscillate around the target at
  0.5–1.5 Hz (2–28 mm); with Kd = 30 both settle like the expert. The 1 cm success count does not change.
- Cause: when success counts tie, the gain grid picks the shortest mean time. A success test that only
  checks the first entry into 1 cm cannot see the missing damping. We found this on the test set, so it
  is reported as a diagnosis only; v3's gains and results were not changed.

### 7.8 Reproduction check (development test split, 210 scenes)

| | Gains | Retrained | Development run |
|---|---|---|---|
| v3 | own grid (100/10/0.1) | 204 | 205 |
| v3 | 100/30/0.1 | 203 | 204 |
| B | own grid (200/30/0.2) | 208 | 206 |
| Expert | — | 209 | — |

All are within the ±3-scene criterion fixed in advance.

### 7.9 Memory study: what must the policy remember? (report Section III-B)

`run_memory_study.sh` (L1, L2) and `run_direct_baseline.sh` (L0), run after B was frozen. All levels use the final run's 973 training
trajectories, split and scenes; no choice used the fresh set.

| Level | Stage memory | Validation /208 | Fresh /300 | Fresh vs B (only level / only B), p | Base never done (val / fresh) |
|---|---|---|---|---|---|
| L0 plain MLP regressor, 21 raw inputs → 11 raw actions (`staged_direct_train.py`) | none | 0 | 0 | 0 / 296, p = 2e-89 | 49 / 87 |
| L1 four single-frame MLPs (park, wheels, stop, arm) | hand-coded latch only | 189 | 272 | 3 / 27, p = 8e-6 | 11 / 12 |
| L2 B's GRU without stop head | hidden state only | 54 | 83 | 0 / 213, p = 2e-64 | 58 / 78 |
| L3 = B: GRU + stop head + latch | hidden state + latch | 207 | 296 | — | 1 / 2 |

- L0: fits recorded frames well (validation R² 0.93 for wheels, 0.95 for torques; ridge regression
  0.06 / 0.67) but never reaches a target: median closest approach 51 cm on the fresh set, and in the
  median run 91% of its steps hit the actuator limits.
- L2: in 122/208 (validation) and 188/300 (fresh) runs the wheels restart after first falling below
  0.05 rad/s; the arm drifts a median 0.31 rad from its start pose before the base is done (B: 0.04 rad,
  held by design).
- L1: parks a median 9.5 cm from the expert's parking pose on the fresh set (B: 7.4 cm).
- Gains: L1 arm 100/30/0.2 (own grid, with the expert base); L2 uses B's 200/30/0.2 so that the stop
  head is the only difference from B.
- Direct-torque ablation (B's GRU, inputs and stop head; arm head regresses recorded torques, no PD
  layer): 0/208 on validation, median closest approach 13.9 cm; the base latches in 207 runs.

### 7.10 What did the arm learn? (report Section III-E)

`analyze_arm_matrix.py` replays the frozen GRU on the logged states of B's 300 fresh roll-outs and, at
each GRU update after the latch with |e| ≥ 1 cm (84,534 updates), compares the learned matrix M with the
arm's position Jacobian J (base frame, pinocchio; the policy never sees it). The expert row uses the
public velocity labels on the validation demonstrations after the stop time.

| Measure (median) | B | Expert labels |
|---|---|---|
| cos(J M e, e): EE moves straight at the target | 0.992 | 0.994 |
| task-space rate \|J v\| / \|e\| (1/s) | 1.47 | 1.39 |
| share of joint motion in the null space of J | 31% | 6% |
| matrix cosine of M with J⁺ / with Jᵀ | 0.93 / 0.55 | – |

The network has learned a differential-IK map close to the pseudo-inverse from joint-angle differences
alone; it matches the expert in task space more closely than in joint space.

The report also plots, from existing results, the four memory levels in one fresh scene
(`memory/trace_scene_2026_0012.csv`) and open-loop validation wheel loss against closed-loop validation
success (L2's loss, 0.062, lies inside B's seed spread 0.054–0.067, yet it succeeds in 54 vs 207–208 of 208).

## 8. Development history (validation set, 208 scenes)

These are earlier models in the same code base, evaluated on an earlier collection with the same
scene distribution. They motivated the design. The report cites the final-run numbers of Section 7.9
instead, which reproduce the same ordering (L1 189, L2 54, torque 0 vs 195, 88, 2 here).

| Model | Success |
|---|---|
| Expert base + expert arm | 208 |
| Four separate MLPs (parking, wheels, stop, arm), single-frame inputs | 195 |
| GRU v2: no stop head, phases left to the hidden state | 88 |
| GRU v3: stop head and latch | 202–203 |
| GRU with direct torque output (no PD layer) | **2** |

Notes on the template's success test:
- `check_stop` tests "stalled" before "reached". An arm that reaches the target and stops before 8 s
  is reported as unreachable at t = 8 s. In the final collection this happened in 7 scenes, all with
  final errors below 3.1 mm. Our evaluator therefore uses only the reach condition.
- Our collector's `success` label is stricter than the template: it also requires the final error to
  be below 1 cm. In the final collection this excluded 54 template-successful runs, all with low targets
  (z ≤ 0.25 m) and final errors of 1.000–1.004 cm. With the 55 stalled runs, 109 of 1500 were dropped.

## 9. Limitations

- No DAgger-style correction: the policy is trained only on states the expert visited. 3 of B's 4
  fresh-test failures are stop decisions (2 latches never fire, 1 fires at t = 0); a latch makes such a
  wrong decision irreversible.
- Memory study: L1 differs from B in architecture as well as memory, so only the L2–B comparison
  isolates one factor. L2 reuses B's gains instead of its own grid.
- Not fully end-to-end. The joint PD layer, the two-phase structure (park, then reach) and the latch
  rule are fixed, and the gains are selected rather than learned.
- B-FK loses 8 fresh-test scenes against variant B (p = 0.039). The forward-kinematics MLP has a median
  error of 2.0 mm on final frames, but up to 15 mm, and the arm stops where the MLP believes the error is
  zero. It does not extrapolate to unseen postures (low targets 3/40). B-FK's gains were selected with the
  URDF model.
- Success follows the template, which only checks the first entry into 1 cm. Our gain selection rule
  should include holding (Section 7.7).
- Robustness and out-of-distribution tests use 40–60 scenes per condition and one perturbation level.
- Only the GRU seed was varied; hidden size (128), learning rate and the feature set were fixed, not
  swept. There is one parking MLP. Variant B reached its best epoch at 78 of a
  maximum of 80.
- The GRU runs at a fixed 100 Hz; another control rate needs retraining.
