#!/bin/bash
#SBATCH -J shohin-v9-decision
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/v9_decision_%j.out
#SBATCH --open-mode=append
# Read-only decision gate for the broad-plus-memory V9 ablation.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
RAW_BOARD=${RAW_BOARD:?set RAW_BOARD}
CANDIDATE_BOARD=${CANDIDATE_BOARD:?set CANDIDATE_BOARD}
RAW_INTERVIEW=${RAW_INTERVIEW:?set RAW_INTERVIEW}
CANDIDATE_INTERVIEW=${CANDIDATE_INTERVIEW:?set CANDIDATE_INTERVIEW}
MEMORY_DEFAULT=${MEMORY_DEFAULT:?set MEMORY_DEFAULT}
MEMORY_SEMANTIC=${MEMORY_SEMANTIC:?set MEMORY_SEMANTIC}
SEMANTIC_REFERENCE=${SEMANTIC_REFERENCE:?set SEMANTIC_REFERENCE}
OUT=${OUT:?set OUT}

for path in "$RAW_BOARD" "$CANDIDATE_BOARD" "$RAW_INTERVIEW" "$CANDIDATE_INTERVIEW" "$MEMORY_DEFAULT" "$MEMORY_SEMANTIC" "$SEMANTIC_REFERENCE"; do
  [ -s "$path" ] || { echo "[v9-decision] required evaluator artifact missing: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[v9-decision] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" evaluate_v9_memory_mix.py --raw-board "$RAW_BOARD" --candidate-board "$CANDIDATE_BOARD" \
  --raw-interview "$RAW_INTERVIEW" --candidate-interview "$CANDIDATE_INTERVIEW" \
  --memory-default "$MEMORY_DEFAULT" --memory-semantic "$MEMORY_SEMANTIC" \
  --semantic-reference "$SEMANTIC_REFERENCE" --out "$OUT"
