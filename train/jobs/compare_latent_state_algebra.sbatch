#!/bin/bash
#SBATCH -J shohin-lsa-compare
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/latent_state_algebra_compare_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CONTROL=${CONTROL:?set CONTROL to the matched answer-only evaluation JSON}
CANDIDATE=${CANDIDATE:?set CANDIDATE to the LSA evaluation JSON}
OUT=${OUT:?set OUT to a fresh comparison JSON}
for path in "$CONTROL" "$CANDIDATE"; do [ -s "$path" ] || { echo "[lsa-compare] missing input: $path" >&2; exit 2; }; done
[ ! -e "$OUT" ] || { echo "[lsa-compare] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" compare_latent_state_algebra.py --control "$CONTROL" --candidate "$CANDIDATE" --out "$OUT"
