#!/usr/bin/env bash
# Memory study: "what does a BC policy need to remember?" Two models from the development history are
# retrained on the FINAL run's data and split, so that three levels of memory are comparable:
#   L1 no learned memory : four networks (park, wheels, stop MLPs + arm MLP), single-frame inputs; the
#                          stage is held by a hand-coded latch on the stop MLP (staged_train.py, whose
#                          base/policy.npz already holds park, wheels and stop, + staged_arm_train.py).
#   L2 implicit memory   : variant B's GRU, inputs, training and gains, but NO stop head / latch; the
#                          stage must live in the hidden state. Differs from B only in the stop head.
#   L3 explicit latch    : variant B (final run).
# L1 arm gains: the usual grid and rule on all 208 validation scenes (arm with the expert base, its
# original protocol). L2 reuses B's chosen gains so that the stop head is the only difference.
# Evaluation: 208 validation scenes and the fresh 300-scene test (unfiltered). Also runs the direct-torque
# ablation (validation) and summarize_memory.py.
# Usage: PY=python bash run_memory_study.sh [OUT=artifacts/final_run] [WORKERS=9]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
OUT=${1:-artifacts/final_run}
WORKERS=${2:-9}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
M=$OUT/memory
mkdir -p "$M"

run() {  # name, args...
  local name=$1; shift
  [ -f "$M/$name/metrics.json" ] || "$PY" staged_evaluate.py "$@" --workers "$WORKERS" --output "$M/$name" > /dev/null
  echo "done $name"
}

# L1: four networks.
[ -f "$M/fournet_arm/arm.json" ] || "$PY" staged_arm_train.py --model-dir "$OUT/base" \
    --output "$M/fournet_arm/arm.npz" > "$M/train_fournet_arm.log" 2>&1
[ -f "$M/grid_fournet_arm/grid.json" ] || "$PY" staged_arm_grid.py --model-dir "$OUT/base" \
    --arm-model "$M/fournet_arm/arm.npz" --limit 0 --workers "$WORKERS" --output "$M/grid_fournet_arm" > "$M/grid_fournet_arm.log" 2>&1
mkdir -p "$M/fresh_fournet_model"  # fresh split next to the four-network base policy.npz
cp "$OUT/fresh_split/split.json" "$M/fresh_fournet_model/split.json"
ln -sf "$(realpath "$OUT/base/policy.npz")" "$M/fresh_fournet_model/policy.npz"
FOUR=(--base ours --arm ours --arm-model "$M/fournet_arm/arm.npz" --gains "$M/grid_fournet_arm/grid.json")
run val_fournet   --model-dir "$OUT/base"        "${FOUR[@]}" --split validation
run fresh_fournet --model-dir "$M/fresh_fournet_model" "${FOUR[@]}" --split test_unfiltered

# L2: GRU without stop head.
[ -f "$M/gru_nostop/gru.json" ] || "$PY" staged_gru_train.py --model-dir "$OUT/base" --features park \
    --output "$M/gru_nostop/gru.npz" > "$M/train_gru_nostop.log" 2>&1
NOSTOP=(--base gru --arm gru --gru-model "$M/gru_nostop/gru.npz" --gains "$OUT/grid_B_park/grid.json")
run val_gru_nostop   --model-dir "$OUT/base"        "${NOSTOP[@]}" --split validation
run fresh_gru_nostop --model-dir "$OUT/fresh_split" "${NOSTOP[@]}" --split test_unfiltered
# Action-representation ablation: B's GRU, inputs and stop head, but the arm head regresses the recorded
# joint torques directly (no PD layer, so no gains to tune). Validation only.
[ -f "$M/gru_torque/gru.json" ] || "$PY" staged_gru_train.py --model-dir "$OUT/base" --stop-head --features park \
    --arm-output torque --output "$M/gru_torque/gru.npz" > "$M/train_gru_torque.log" 2>&1
run val_gru_torque --model-dir "$OUT/base" --base gru --arm gru --gru-model "$M/gru_torque/gru.npz" --split validation

"$PY" summarize_memory.py --run-dir "$OUT"
echo done
