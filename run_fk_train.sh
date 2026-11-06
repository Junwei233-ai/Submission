#!/usr/bin/env bash
# Retrain the policies with the learned forward kinematics (results/final_run/fk_run/fk.npz).
# The recorded ee_pos is not read. SOURCE is the seed-42 demonstration folder (one .npz per scene).
# Models are written to results/final_run/fk_run/, ready to commit.
# Usage: SOURCE=/path/to/collection PY=/path/to/python bash run_fk_train.sh L3|L2|L1 [THREADS=6]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
TASK=${1:?give L3, L2 or L1}
THREADS=${2:-6}
SOURCE=${SOURCE:?set SOURCE to the seed-42 demonstration folder}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=$THREADS
R=results/final_run
FK=$R/fk_run/fk.npz
BASE=artifacts/fk_train_base
mkdir -p "$BASE"
"$PY" - "$R/base/split.json" "$BASE/split.json" "$SOURCE" <<'EOF'
import json, sys
from pathlib import Path
split = json.load(open(sys.argv[1]))
split["source"] = str(Path(sys.argv[3]).resolve())
missing = [n for part in ("train", "validation") for n in split["splits"][part] if not Path(split["source"], n).exists()]
if missing:
    sys.exit(f"{len(missing)} demonstration files are missing from {split['source']}, for example {missing[0]}")
json.dump(split, open(sys.argv[2], "w"), indent=1)
EOF
cp "$R/base/policy.npz" "$BASE/policy.npz"
case "$TASK" in
  L3) "$PY" staged_gru_train.py --model-dir "$BASE" --stop-head --features park --fk-model "$FK" --threads "$THREADS" \
          --output "$R/fk_run/gru_B_park/gru.npz" ;;
  L2) "$PY" staged_gru_train.py --model-dir "$BASE" --features park --fk-model "$FK" --threads "$THREADS" \
          --output "$R/fk_run/gru_nostop/gru.npz" ;;
  L1) "$PY" staged_arm_train.py --model-dir "$BASE" --fk-model "$FK" --output "$R/fk_run/fournet_arm/arm.npz" ;;
  *)  echo "Unknown task: $TASK"; exit 2 ;;
esac
echo "$TASK trained. Commit results/final_run/fk_run/ and push."
