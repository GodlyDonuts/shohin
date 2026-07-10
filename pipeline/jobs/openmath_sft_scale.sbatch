#!/bin/bash
#SBATCH -J shohin-openmath-sft
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/openmath_sft_%j.out
#SBATCH --open-mode=append
# Build a larger, independently verified OpenMath SFT source. This job is CPU-only
# and writes atomically, so it cannot perturb a running pretrain or an existing
# frozen SFT mix.
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:-$BASE/artifacts/sft/openmath2_600k.jsonl}
TMP=${OUT}.partial
MAX_KEEP=${MAX_KEEP:-600000}

cd "$BASE"
rm -f "$TMP"
"$PY" pipeline/sft_curate.py \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --evals "$BASE/artifacts/evals" \
  --out "$TMP" \
  --max-keep "$MAX_KEEP" \
  --max-sol-tokens 400
"$PY" pipeline/audit_sft_quality.py --data "$TMP" --out "${TMP}.quality.json"
mv "$TMP" "$OUT"
mv "${TMP}.quality.json" "${OUT%.jsonl}.quality.json"
echo "[openmath-sft] committed $OUT"
