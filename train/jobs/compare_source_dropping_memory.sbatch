#!/bin/bash
#SBATCH -J shohin-source-memory-compare
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/source_memory_compare_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
M0=${M0:?set M0 to M0 evaluation JSON}
M1=${M1:?set M1 to M1 evaluation JSON}
OUT=${OUT:?set OUT to a fresh comparison JSON}

for path in "$M0" "$M1"; do [ -s "$path" ] || { echo "[source-memory-compare] missing input: $path" >&2; exit 2; }; done
[ ! -e "$OUT" ] || { echo "[source-memory-compare] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" compare_source_dropping_memory.py --m0 "$M0" --m1 "$M1" --out "$OUT"
