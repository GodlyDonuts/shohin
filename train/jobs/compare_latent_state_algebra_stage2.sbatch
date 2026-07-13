#!/bin/bash
#SBATCH -J shohin-lsa-stage2
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/latent_state_algebra_stage2_%j.out
#SBATCH --open-mode=append
# This comparator is CPU-only and is only submitted after stage one advances.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
ANSWER_ONLY=${ANSWER_ONLY:?set answer-only evaluator JSON}
CANDIDATE=${CANDIDATE:?set verified-candidate evaluator JSON}
PAIR_SHUFFLED=${PAIR_SHUFFLED:?set shuffled-pair evaluator JSON}
STATE_PERMUTED=${STATE_PERMUTED:?set permuted-state evaluator JSON}
OUT=${OUT:?set fresh stage-two comparison JSON}
for path in "$ANSWER_ONLY" "$CANDIDATE" "$PAIR_SHUFFLED" "$STATE_PERMUTED"; do
  [ -s "$path" ] || { echo "[lsa-stage2] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[lsa-stage2] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" compare_latent_state_algebra_stage2.py \
  --answer-only "$ANSWER_ONLY" --candidate "$CANDIDATE" --pair-shuffled "$PAIR_SHUFFLED" \
  --state-permuted "$STATE_PERMUTED" --out "$OUT"
