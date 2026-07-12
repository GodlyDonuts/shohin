#!/bin/bash
#SBATCH -J shohin-text-overlap
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/training_text_overlap_%j.out
#SBATCH --open-mode=append
# Audit every SFT text field against every current eval prompt. This is CPU-only
# and emits a report, never a new training artifact or model checkpoint.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:?set DATA to a frozen SFT JSONL}
OUT=${OUT:?set OUT to a fresh report JSON}
FIELDS=${FIELDS:-"question response completion_prompt"}

[ -s "$DATA" ] || { echo "[text-overlap] data missing: $DATA" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[text-overlap] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE"
# shellcheck disable=SC2206
FIELD_ARGS=($FIELDS)
"$PY" pipeline/audit_training_text_overlap.py --data "$DATA" --evals "$BASE/artifacts/evals" \
  --out "$OUT" --fields "${FIELD_ARGS[@]}" --require-zero
