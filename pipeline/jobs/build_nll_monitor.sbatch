#!/bin/bash
#SBATCH -J shohin-nll-monitor
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/build_nll_monitor_%j.out
#SBATCH --open-mode=append
# External monitor construction only. It is never a training source.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATASET=${DATASET:-Salesforce/wikitext}
CONFIG=${CONFIG:-wikitext-103-raw-v1}
SPLIT=${SPLIT:-test}
TEXT_COL=${TEXT_COL:-text}
MAX_TOKENS=${MAX_TOKENS:-1000000}
OUT=${OUT:-$BASE/artifacts/monitors/wikitext103_test.jsonl}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$BASE"
"$PY" pipeline/build_nll_monitor.py \
  --dataset "$DATASET" --config "$CONFIG" --split "$SPLIT" --text-col "$TEXT_COL" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --max-tokens "$MAX_TOKENS" --out "$OUT"
