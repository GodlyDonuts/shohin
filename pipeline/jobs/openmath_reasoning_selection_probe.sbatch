#!/bin/bash
#SBATCH -J shohin-openmath-select
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/openmath_reasoning_selection_%j.out
#SBATCH --open-mode=append
# Read-only yield and quality probe for OpenMathReasoning. It emits no candidate
# JSONL, does not execute code, and cannot affect the active flagship.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
SPLIT=${SPLIT:-cot}
MAX_SEEN=${MAX_SEEN:-50000}
MAX_PROBLEM_TOKENS=${MAX_PROBLEM_TOKENS:-512}
MAX_TRACE_TOKENS=${MAX_TRACE_TOKENS:-512}
MIN_PASS_RATE=${MIN_PASS_RATE:-}
EXCLUDE_USED_IN_KAGGLE=${EXCLUDE_USED_IN_KAGGLE:-0}
OUT=${OUT:?set OUT to a fresh selection-report JSON}

[ ! -e "$OUT" ] || { echo "[openmath-select] refusing existing report $OUT" >&2; exit 2; }
cd "$BASE"
ARGS=(--tokenizer "$BASE/artifacts/shohin-tok-32k.json" --evals "$BASE/artifacts/evals"
      --report "$OUT" --dry-run --split "$SPLIT" --max-seen "$MAX_SEEN"
      --max-problem-tokens "$MAX_PROBLEM_TOKENS" --max-trace-tokens "$MAX_TRACE_TOKENS")
if [ -n "$MIN_PASS_RATE" ]; then
  ARGS+=(--min-pass-rate "$MIN_PASS_RATE")
fi
if [ "$EXCLUDE_USED_IN_KAGGLE" = 1 ]; then
  ARGS+=(--exclude-used-in-kaggle)
fi
"$PY" pipeline/curate_openmath_reasoning.py "${ARGS[@]}"
