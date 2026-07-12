#!/bin/bash
#SBATCH -J shohin-apps-code
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/apps_code_%j.out
#SBATCH --open-mode=append
# Build an isolated APPS train-split code SFT pilot. It never touches a frozen
# mix or the live pretraining shards; all source solutions must pass supplied
# I/O tests inside resource-limited subprocesses.
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:-$BASE/artifacts/sft/apps_train_pilot.jsonl}
MAX_SEEN=${MAX_SEEN:-500}
MAX_KEPT=${MAX_KEPT:-250}
[ -e "$OUT" ] && { echo "[apps] refusing to overwrite $OUT" >&2; exit 2; }
echo "[apps] out=$OUT max_seen=$MAX_SEEN max_kept=$MAX_KEPT"
cd "$BASE/pipeline"
"$PY" curate_apps.py \
  --out "$OUT" \
  --evals "$BASE/artifacts/evals" \
  --max-seen "$MAX_SEEN" \
  --max-kept "$MAX_KEPT" \
  --max-tests 3 \
  --timeout 2.0
"$PY" audit_sft_quality.py --data "$OUT" --out "${OUT%.jsonl}.quality.json"
