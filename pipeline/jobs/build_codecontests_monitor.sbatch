#!/bin/bash
#SBATCH -J shohin-code-monitor
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/build_code_monitor_%j.out
#SBATCH --open-mode=append
# Build a fixed test-split code monitor only; it is never SFT/pretrain data.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:-$BASE/artifacts/monitors/codecontests_test_python.jsonl}
MAX_TOKENS=${MAX_TOKENS:-500000}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$BASE"
"$PY" pipeline/build_codecontests_monitor.py --out "$OUT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --evals "$BASE/artifacts/evals" \
  --max-tokens "$MAX_TOKENS"
