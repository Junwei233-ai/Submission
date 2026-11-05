#!/usr/bin/env bash
# Additional analyses of the frozen final method (variant B), run after run_final_pipeline.sh and
# run_evaluation_suite.sh. None of them changes the frozen model or its gains.
#   1. Seeds: B retrained with GRU seeds 1 and 2 (same split, same park MLP) -> validation.
#   2. Learning curve: B retrained (park/stop MLPs and GRU) on nested 10/25/50% subsets of the
#      training trajectories -> validation. 100% is the frozen model.
#   3. Hold: frozen B, v3 and the expert on the fresh test with 2 s of extra simulation after
#      success, recording the worst EE error while holding; plus a gain-swap diagnosis (v3 with
#      Kd 30, B with v3's 100/10/0.1), added after v3 was seen to oscillate around the target.
#   4. Gravity ablation: frozen B on validation with the gravity term of the PD layer set to zero,
#      at the chosen gains and at 50/30/0.1.
# All validation runs use B's chosen gains (grid_B_park/grid.json); gains are not re-tuned.
# Usage: PY=python bash run_bc_analyses.sh [OUT=artifacts/final_run] [WORKERS=18] [THREADS=3]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
OUT=${1:-artifacts/final_run}
WORKERS=${2:-18}
THREADS=${3:-3}
A=$OUT/analysis
mkdir -p "$A"
SOURCE=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['source'])" "$OUT/base/split.json")
GAINS=(--gains "$OUT/grid_B_park/grid.json")

# --- training (parallel, THREADS torch threads each) ---
train_seed() {
  local seed=$1
  [ -f "$A/seed$seed/gru.npz" ] || OMP_NUM_THREADS=$THREADS "$PY" staged_gru_train.py --model-dir "$OUT/base" \
      --stop-head --features park --seed "$seed" --threads "$THREADS" --output "$A/seed$seed/gru.npz" \
      > "$A/train_seed$seed.log" 2>&1
}
train_fraction() {
  local f=$1 d=$A/frac$1
  [ -f "$d/base/policy.npz" ] || OMP_NUM_THREADS=$THREADS "$PY" staged_train.py --source "$SOURCE" \
      --train-fraction "$f" --output "$d/base" > "$A/train_frac${f}_base.log" 2>&1
  [ -f "$d/gru.npz" ] || OMP_NUM_THREADS=$THREADS "$PY" staged_gru_train.py --model-dir "$d/base" \
      --stop-head --features park --threads "$THREADS" --output "$d/gru.npz" > "$A/train_frac$f.log" 2>&1
}
for s in 1 2; do train_seed $s & done
for f in 0.1 0.25 0.5; do train_fraction $f & done
wait
echo "training done"

validate() {  # name, model, extra args...
  local name=$1 model=$2; shift 2
  [ -f "$A/val_$name/metrics.json" ] || "$PY" staged_evaluate.py --model-dir "$OUT/base" --split validation \
      --base gru --arm gru --gru-model "$model" --workers "$WORKERS" --output "$A/val_$name" "$@" > /dev/null
  echo "done val_$name"
}
for s in 1 2; do validate seed$s "$A/seed$s/gru.npz" "${GAINS[@]}"; done
for f in 0.1 0.25 0.5; do validate frac$f "$A/frac$f/gru.npz" "${GAINS[@]}"; done
validate gravity_zero "$OUT/gru_B_park/gru.npz" "${GAINS[@]}" --gravity zero
validate gravity_zero_kp50_kd30_lead0.1 "$OUT/gru_B_park/gru.npz" --kp 50 --kd 30 --lead 0.1 --gravity zero

hold() {  # name, policy args...
  local name=$1; shift
  [ -f "$A/hold_$name/metrics.json" ] || "$PY" staged_evaluate.py --model-dir "$OUT/fresh_split" \
      --split test_unfiltered --hold-time 2 --workers "$WORKERS" --output "$A/hold_$name" "$@" > /dev/null
  echo "done hold_$name"
}
hold B --base gru --arm gru --gru-model "$OUT/gru_B_park/gru.npz" "${GAINS[@]}"
hold v3 --base gru --arm gru --gru-model "$OUT/gru_v3_bearing/gru.npz" --gains "$OUT/grid_v3_bearing/grid.json"
hold expert --base expert --arm expert
# Diagnosis (after seeing v3 oscillate while holding): swap only the damping / gains between B and v3.
hold diag_v3_kp100_kd30_lead0.1 --base gru --arm gru --gru-model "$OUT/gru_v3_bearing/gru.npz" --kp 100 --kd 30 --lead 0.1
hold diag_B_kp100_kd10_lead0.1 --base gru --arm gru --gru-model "$OUT/gru_B_park/gru.npz" --kp 100 --kd 10 --lead 0.1
"$PY" summarize_analyses.py --run-dir "$OUT"
