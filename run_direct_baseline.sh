#!/usr/bin/env bash
# Plain BC baseline (L0): linear and MLP regressors from the 21 raw inputs to the 11 raw actions
# (staged_direct_train.py); the MLP is evaluated in closed loop on 208 validation and 300 fresh scenes.
# Usage: PY=python bash run_direct_baseline.sh [OUT=artifacts/final_run] [WORKERS=5]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
OUT=${1:-artifacts/final_run}
WORKERS=${2:-5}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
D=$OUT/direct
[ -f "$D/mlp.json" ] || "$PY" staged_direct_train.py --model-dir "$OUT/base" --output "$D" > "$OUT/direct_train.log" 2>&1
run() {  # name, args...
  local name=$1; shift
  [ -f "$D/$name/metrics.json" ] || "$PY" staged_evaluate.py "$@" --workers "$WORKERS" --output "$D/$name" > /dev/null
  echo "done $name"
}
# Closed loop for the MLP only; the ridge regressor is reported by its open-loop fit (linear.json).
run val_mlp   --model-dir "$OUT/base"        --base direct --arm direct --direct-model "$D/mlp.npz" --split validation &
run fresh_mlp --model-dir "$OUT/fresh_split" --base direct --arm direct --direct-model "$D/mlp.npz" --split test_unfiltered &
wait
echo done
