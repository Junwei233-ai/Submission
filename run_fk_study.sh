#!/usr/bin/env bash
# Evaluate the models trained by run_fk_train.sh, with learned forward kinematics and no gravity term.
# Uses only files in this repository: no data collection and no training. All levels use gains 200/30/0.2, so the arm
# layer is the same for all of them. Per-scene results are copied to results/final_run/fk_run/ so they can
# be committed and compared scene by scene.
# Usage: PY=/path/to/python bash run_fk_study.sh L3|L2|L1|robust|ood|expert [WORKERS=8]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
TASK=${1:?give L3, L2, L1, robust, ood or expert}
WORKERS=${2:-8}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
R=results/final_run
OUT=artifacts/fk_study
FK=$R/fk_run/fk.npz
ARM=(--gravity zero --fk-model "$FK" --kp 200 --kd 30 --lead 0.2)
L3=(--base gru --arm gru --gru-model "$R/fk_run/gru_B_park/gru.npz" "${ARM[@]}")
L2=(--base gru --arm gru --gru-model "$R/fk_run/gru_nostop/gru.npz" "${ARM[@]}")
L1=(--base ours --arm ours --arm-model "$R/fk_run/fournet_arm/arm.npz" "${ARM[@]}")
EXPERT=(--base expert --arm expert)
mkdir -p "$OUT"

run() {  # name, args...
  local name=$1; shift
  [ -f "$OUT/$name/metrics.json" ] || "$PY" staged_evaluate.py --model-dir "$R/base" "$@" \
      --workers "$WORKERS" --output "$OUT/$name" > "$OUT/$name.log" 2>&1
  mkdir -p "$R/fk_run/$name"
  cp "$OUT/$name/metrics.json" "$OUT/$name/results.json" "$OUT/$name/config.json" "$R/fk_run/$name/"
  "$PY" -c "import json,sys; m=json.load(open(sys.argv[1])); print(sys.argv[2], str(m['success_1cm'])+'/'+str(m['scenes']), m['stop_reasons'])" \
      "$OUT/$name/metrics.json" "$name"
}

conditions() {  # policy name, args...
  local who=$1; shift
  run "robust_noise003_$who"   --scenes-dir scenes/robustness_60 "$@" --obs-noise 0.003
  run "robust_noise010_$who"   --scenes-dir scenes/robustness_60 "$@" --obs-noise 0.01
  run "robust_torque080_$who"  --scenes-dir scenes/robustness_60 "$@" --torque-scale 0.8
  run "robust_pushee30_$who"   --scenes-dir scenes/robustness_60 "$@" --push-ee 30
  run "robust_pushbase20_$who" --scenes-dir scenes/robustness_60 "$@" --push-base 20
}

case "$TASK" in
  L3)     run val_L3 --scenes-dir scenes/validation "${L3[@]}"; run test_L3 --scenes-dir scenes/fresh_test "${L3[@]}" ;;
  L2)     run val_L2 --scenes-dir scenes/validation "${L2[@]}"; run test_L2 --scenes-dir scenes/fresh_test "${L2[@]}" ;;
  L1)     run val_L1 --scenes-dir scenes/validation "${L1[@]}"; run test_L1 --scenes-dir scenes/fresh_test "${L1[@]}" ;;
  robust) conditions L3 "${L3[@]}" ;;
  ood)    for family in far_target high_target low_target wide_start; do
            run "ood_${family}_L3" --scenes-dir "scenes/ood/$family" "${L3[@]}"
          done ;;
  expert) run test_expert --scenes-dir scenes/fresh_test "${EXPERT[@]}"
          conditions expert "${EXPERT[@]}"
          for family in far_target high_target low_target wide_start; do
            run "ood_${family}_expert" --scenes-dir "scenes/ood/$family" "${EXPERT[@]}"
          done ;;
  *)      echo "Unknown task: $TASK"; exit 2 ;;
esac
echo "$TASK done. Results are in $R/fk_run/"
