#!/usr/bin/env bash
# Reproduction protocol (fixed before any result is seen). Order of interpretation:
#   Stage 1  Train GRU v3 (bearing features) and variant B (learned park pose), both with stop head.
#   Stage 2  For each, grid-search Kp/Kd/lead on ALL 208 validation scenes (staged_arm_grid.py rule:
#            max 1 cm successes -> min mean success time -> min torque-clip fraction).
#   Stage 3  BASELINE CHECK: v3 on the original 210-scene test split, with its own chosen gains and
#            with the teammate's first-run gains 100/30/0.1, compared with the reported 205 and 204.
#   Stage 4  Variant B on the same test split with its own chosen gains (reported: 206), and the
#            expert as a reference. This test split was reused during development, so these runs only
#            check reproduction; the fresh test in run_evaluation_suite.sh is the unbiased result.
# Usage: PY=/path/to/python bash run_final_pipeline.sh [OUT=artifacts/final_run] [WORKERS=9]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
OUT=${1:-artifacts/final_run}
WORKERS=${2:-9}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
BASE=$OUT/base

[ -f "$BASE/split.json" ] || "$PY" staged_train.py --source "${SOURCE:?set SOURCE to the collection dir}" --output "$BASE"
"$PY" compute_torque_limit.py --model-dir "$BASE"
for v in v3_bearing:bearing B_park:park; do
  name=${v%%:*}; feat=${v##*:}
  [ -f "$OUT/gru_$name/gru.npz" ] || "$PY" staged_gru_train.py --model-dir "$BASE" --stop-head --features "$feat" \
      --output "$OUT/gru_$name/gru.npz" > "$OUT/train_$name.log" 2>&1 &
done
wait
echo "stage 1 done (training)"

for name in v3_bearing B_park; do
  [ -f "$OUT/grid_$name/grid.json" ] || "$PY" staged_arm_grid.py --model-dir "$BASE" --gru-model "$OUT/gru_$name/gru.npz" \
      --limit 0 --workers "$WORKERS" --output "$OUT/grid_$name" > "$OUT/grid_$name.log" 2>&1 &
done
wait
echo "stage 2 done (validation gain grids)"

evaluate() {  # output name, args...
  local name=$1; shift
  [ -f "$OUT/oldtest/$name/metrics.json" ] || "$PY" staged_evaluate.py --model-dir "$BASE" --split test \
      --workers "$WORKERS" --output "$OUT/oldtest/$name" "$@" > /dev/null
}
mkdir -p "$OUT/oldtest"
GRU_V3=(--base gru --arm gru --gru-model "$OUT/gru_v3_bearing/gru.npz")
GRU_B=(--base gru --arm gru --gru-model "$OUT/gru_B_park/gru.npz")
evaluate v3_own_gains "${GRU_V3[@]}" --gains "$OUT/grid_v3_bearing/grid.json" &
evaluate v3_kp100_kd30_lead0.1 "${GRU_V3[@]}" --kp 100 --kd 30 --lead 0.1 &
wait
echo "stage 3 done (v3 baseline check)"
evaluate B_own_gains "${GRU_B[@]}" --gains "$OUT/grid_B_park/grid.json" &
evaluate expert --base expert --arm expert &
wait
echo "stage 4 done (variant B + expert)"
"$PY" report_reproduction.py --run-dir "$OUT"
echo done
