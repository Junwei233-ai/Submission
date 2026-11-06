#!/usr/bin/env bash
# One command for the final method, from nothing to the fresh-test result (no URDF in the policy):
#   collect demonstrations -> split, parking MLP -> torque limit -> expert runs with a 2 s hold
#   -> forward kinematics from final poses -> GRU-B on its estimate -> validation and fresh test with
#   fixed gains 200/30/0.2, no gravity term. No recorded end-effector position is read.
# The gains are not searched here. They were selected on validation by staged_arm_grid.py; see METHOD.md.
# Finished steps are skipped when the script is rerun.
# Usage: PY=/path/to/python bash run_fk_pipeline.sh [OUT=artifacts/fk_run] [WORKERS=8]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
OUT=${1:-artifacts/fk_run}
WORKERS=${2:-8}
export OPENBLAS_NUM_THREADS=1
BASE=$OUT/base
GRU=$OUT/gru_B_park/gru.npz
FK=$OUT/fk/fk.npz
POLICY=(--base gru --arm gru --gru-model "$GRU" --fk-model "$FK" --gravity zero --kp 200 --kd 30 --lead 0.2)
mkdir -p "$OUT"

evaluate() {  # name, args...
  local name=$1; shift
  [ -f "$OUT/$name/metrics.json" ] || OMP_NUM_THREADS=1 "$PY" staged_evaluate.py --model-dir "$BASE" "$@" \
      --workers "$WORKERS" --output "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "done $name"
}

[ -f "$OUT/collection/summary.csv" ] || "$PY" collect_data.py --n 1500 --seed 42 --output "$OUT/collection"
[ -f "$BASE/split.json" ] || "$PY" staged_train.py --source "$OUT/collection" --output "$BASE"
"$PY" compute_torque_limit.py --model-dir "$BASE"
evaluate hold_validation --base expert --arm expert --hold-time 2 --split validation
evaluate hold_train      --base expert --arm expert --hold-time 2 --split train
[ -f "$FK" ] || "$PY" staged_fk_train.py --train-rollouts "$OUT/hold_train" --validation-rollouts "$OUT/hold_validation" --output "$FK"
echo "done training forward kinematics"
[ -f "$GRU" ] || "$PY" staged_gru_train.py --model-dir "$BASE" --stop-head --features park --fk-model "$FK" --output "$GRU"
echo "done training GRU-B"

evaluate validation "${POLICY[@]}" --split validation
evaluate fresh_test "${POLICY[@]}" --scenes-dir scenes/fresh_test
"$PY" - "$OUT" <<'EOF'
import json, sys
for name in ("validation", "fresh_test"):
    m = json.load(open(f"{sys.argv[1]}/{name}/metrics.json"))
    print(f"{name}: {m['success_1cm']}/{m['scenes']} within 1 cm, Wilson 95% {m['wilson95_1cm']}")
EOF
