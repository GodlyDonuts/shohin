#!/bin/bash
#SBATCH -J shohin-v8-decision
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/v8_decision_%j.out
#SBATCH --open-mode=append
# Read-only post-evaluation decision. It never writes model weights or submits
# further training; a human/operator must still review an accept_followup.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
RAW_BOARD=${RAW_BOARD:?set RAW_BOARD}
CANDIDATE_BOARD=${CANDIDATE_BOARD:?set CANDIDATE_BOARD}
RAW_INTERVIEW=${RAW_INTERVIEW:?set RAW_INTERVIEW}
CANDIDATE_INTERVIEW=${CANDIDATE_INTERVIEW:?set CANDIDATE_INTERVIEW}
RAW_TRACE=${RAW_TRACE:?set RAW_TRACE}
CANDIDATE_TRACE=${CANDIDATE_TRACE:?set CANDIDATE_TRACE}
OUT=${OUT:?set OUT}

for path in "$RAW_BOARD" "$CANDIDATE_BOARD" "$RAW_INTERVIEW" "$CANDIDATE_INTERVIEW" "$RAW_TRACE" "$CANDIDATE_TRACE"; do
  [ -s "$path" ] || { echo "[v8-decision] required evaluator artifact missing: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[v8-decision] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" evaluate_v8_promotion.py --raw-board "$RAW_BOARD" --candidate-board "$CANDIDATE_BOARD" \
  --raw-interview "$RAW_INTERVIEW" --candidate-interview "$CANDIDATE_INTERVIEW" \
  --raw-trace "$RAW_TRACE" --candidate-trace "$CANDIDATE_TRACE" --out "$OUT"
