#!/bin/bash
#SBATCH -J shohin-causal-prefix-compare
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/causal_prefix_readback_compare_%j.out
#SBATCH --open-mode=append
# CPU-only locked comparison; prefer Stokes for submission when accessible.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
VERIFIED=${VERIFIED:?set VERIFIED to verified-prefix report}
FINAL_REPLAY=${FINAL_REPLAY:?set FINAL_REPLAY to equal-work final replay report}
LABEL_SHUFFLED=${LABEL_SHUFFLED:?set LABEL_SHUFFLED to shuffled-label report}
OUT=${OUT:?set OUT to a fresh JSON output path}
for path in "$VERIFIED" "$FINAL_REPLAY" "$LABEL_SHUFFLED"; do [ -s "$path" ] || { echo "[causal-prefix-compare] missing input: $path" >&2; exit 2; }; done
[ ! -e "$OUT" ] || { echo "[causal-prefix-compare] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" compare_causal_prefix_readback.py \
  --verified "$VERIFIED" --final-replay "$FINAL_REPLAY" --label-shuffled "$LABEL_SHUFFLED" --out "$OUT"
