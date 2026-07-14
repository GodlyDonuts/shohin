#!/bin/bash
#SBATCH -J shohin-scratch-compare
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/scratch_compare_%j.out
#SBATCH --open-mode=append
# CPU-only locked comparator. Submit on Stokes, not a GPU node.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=${PY:-/usr/bin/python3.11}
RECURRENT=${RECURRENT:?set RECURRENT report}
RESET=${RESET:?set RESET report}
OUT=${OUT:?set OUT to a fresh comparison JSON}
for path in "$RECURRENT" "$RESET"; do
  [ -s "$path" ] || { echo "[scratch-compare] missing report: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[scratch-compare] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" compare_causal_recurrent_scratch.py --recurrent "$RECURRENT" --reset "$RESET" --out "$OUT"
