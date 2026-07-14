#!/bin/bash
#SBATCH -J shohin-referential-compare
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/referential_slot_compare_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CONTROL=${CONTROL:?set control report}
CANDIDATE=${CANDIDATE:?set candidate report}
OUT=${OUT:?set fresh comparison output}

for path in "$CONTROL" "$CANDIDATE"; do
  [ -s "$path" ] || { echo "[referential-compare] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[referential-compare] refusing existing output: $OUT" >&2; exit 3; }

cd "$BASE/train"
"$PY" compare_referential_slot_microcode.py --control "$CONTROL" --candidate "$CANDIDATE" --out "$OUT"
