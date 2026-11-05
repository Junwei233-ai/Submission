#!/usr/bin/env bash
# Evaluation of the FROZEN models (after run_final_pipeline.sh). Nothing here feeds back into
# training or selection. Main method: variant B; baseline: v3; reference: expert. Every condition
# runs on the same scenes for each controller, so comparisons are paired.
#   A. Fresh held-out test: new scenes (collect_data.py --seed 2026) never seen during development.
#   B. Robustness: sensor noise, weaker arm actuators, pushes on the EE and on the base (fresh scenes).
#   C. Generalisation: out-of-distribution targets / start states (make_ood_scenes.py).
# Usage: PY=python [WITH_V3=1] bash run_evaluation_suite.sh [OUT=artifacts/final_run] [WORKERS=9] [N_ROBUST=60]
#   WITH_V3=1 also runs the v3 baseline in B and C (A always includes it).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
OUT=${1:-artifacts/final_run}
WORKERS=${2:-9}
N_ROBUST=${3:-60}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
V3=(--base gru --arm gru --gru-model "$OUT/gru_v3_bearing/gru.npz" --gains "$OUT/grid_v3_bearing/grid.json")
B=(--base gru --arm gru --gru-model "$OUT/gru_B_park/gru.npz" --gains "$OUT/grid_B_park/grid.json")
EXPERT=(--base expert --arm expert)
FRESH=$OUT/fresh_split

# Split file for the fresh scenes: expert successes -> "test", expert failures -> "test_expert_failed".
if [ ! -f "$FRESH/split.json" ]; then
  mkdir -p "$FRESH"
  "$PY" - "$OUT/fresh_test_seed2026" "$FRESH/split.json" <<'EOF'
import csv, json, sys
from pathlib import Path
source = Path(sys.argv[1]).resolve()
rows = sorted(csv.DictReader(open(source / "summary.csv")), key=lambda r: r["id"])
ok = [r["id"] + ".npz" for r in rows if r["success"] == "True"]
bad = [r["id"] + ".npz" for r in rows if r["success"] != "True"]
json.dump(dict(source=str(source), seed=2026, population="fresh held-out scenes, collected after the model was frozen",
               splits=dict(train=[], validation=[], test=ok, test_expert_failed=bad)), open(sys.argv[2], "w"), indent=1)
print(f"fresh test: {len(ok)} expert-successful, {len(bad)} expert-failed")
EOF
fi

run() {  # name, policy args..., -- extra args...
  local name=$1; shift
  [ -d "$OUT/suite/$name" ] || "$PY" staged_evaluate.py "$@" --workers "$WORKERS" --output "$OUT/suite/$name" > /dev/null
  echo "done $name"
}
mkdir -p "$OUT/suite"

# A. fresh held-out test, all scenes (unfiltered): B, v3 and expert
run fresh_B      --model-dir "$FRESH" "${B[@]}"      --split test_unfiltered
run fresh_v3     --model-dir "$FRESH" "${V3[@]}"     --split test_unfiltered
run fresh_expert --model-dir "$FRESH" "${EXPERT[@]}" --split test_unfiltered

each() {  # name, scene args...: run B, expert (and v3 if WITH_V3=1) on the same scenes
  local name=$1; shift
  run "${name}_B"      --model-dir "$FRESH" "${B[@]}"      "$@"
  run "${name}_expert" --model-dir "$FRESH" "${EXPERT[@]}" "$@"
  [ "${WITH_V3:-0}" = 1 ] && run "${name}_v3" --model-dir "$FRESH" "${V3[@]}" "$@" || true
}

# B. robustness on the first N_ROBUST expert-successful fresh scenes
for cond in "noise003 --obs-noise 0.003" "noise010 --obs-noise 0.01" "torque080 --torque-scale 0.8" \
            "pushee30 --push-ee 30" "pushbase20 --push-base 20"; do
  set -- $cond; name=$1; shift
  each "robust_$name" --split test --limit "$N_ROBUST" "$@"
done

# C. out-of-distribution families
[ -d "$OUT/ood_scenes" ] || "$PY" make_ood_scenes.py --output "$OUT/ood_scenes"
for fam in far_target low_target high_target wide_start; do
  each "ood_$fam" --scenes-dir "$OUT/ood_scenes/$fam"
done
"$PY" summarize_suite.py --run-dir "$OUT"
