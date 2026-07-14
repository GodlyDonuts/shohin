#!/bin/bash
#SBATCH -J shohin-role-equiv-compare
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/role_equiv_compare_%j.out
#SBATCH --open-mode=append
# CPU-only locked comparison. Submit on Stokes after both read-only evaluations.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=/usr/bin/python3.11
CONTROL=${CONTROL:?set control report}
CANDIDATE=${CANDIDATE:?set candidate report}
OUT=${OUT:?set fresh comparison report}
for path in "$CONTROL" "$CANDIDATE"; do
  [ -s "$path" ] || { echo "[role-equiv-compare] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[role-equiv-compare] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" compare_role_equivariant_microcode.py --control "$CONTROL" --candidate "$CANDIDATE" --out "$OUT"
