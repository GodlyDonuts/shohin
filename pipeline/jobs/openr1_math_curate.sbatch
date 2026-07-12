#!/bin/bash
#SBATCH -J shohin-openr1-curate
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/openr1_math_curate_%j.out
#SBATCH --open-mode=append
# Build an isolated OpenR1 default SFT candidate after schema inspection. The
# output requires a separate quality audit before it can join any frozen mix.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:-$BASE/artifacts/sft/openr1_math_default_90k.jsonl}
MAX_KEEP=${MAX_KEEP:-90000}
MAX_TRACE_TOKENS=${MAX_TRACE_TOKENS:-512}
MIN_NOVEL_WORDS=${MIN_NOVEL_WORDS:-10}
SOURCE_SOLUTION=${SOURCE_SOLUTION:-0}

[ ! -e "$OUT" ] || { echo "[openr1] refusing to overwrite $OUT" >&2; exit 2; }
cd "$BASE"
MODE=()
if [ "$SOURCE_SOLUTION" = 1 ]; then
  MODE+=(--source-solution)
fi
"$PY" pipeline/curate_openr1_math.py \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --evals "$BASE/artifacts/evals" \
  --out "$OUT" \
  --max-keep "$MAX_KEEP" \
  --max-trace-tokens "$MAX_TRACE_TOKENS" \
  --min-novel-words "$MIN_NOVEL_WORDS" "${MODE[@]}"
"$PY" pipeline/audit_sft_quality.py --data "$OUT" --evals "$BASE/artifacts/evals" --out "${OUT}.quality.json"
echo "[openr1] candidate complete; do not mix without review"
