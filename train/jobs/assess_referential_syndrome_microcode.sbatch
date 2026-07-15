#!/bin/bash
#SBATCH -J shohin-r9c-assess
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/r9c_assess_%j.out
#SBATCH --open-mode=append
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:?set fresh R9c decision output}
[ ! -e "$OUT" ] || { echo "[r9c-assess] refusing existing output: $OUT" >&2; exit 3; }

for board in development fresh; do
  for arm in treatment static no_syndrome shuffled_goal; do
    variable=$(printf '%s_%s' "$board" "$arm" | tr '[:lower:]' '[:upper:]')
    path=${!variable:?set "$variable" report}
    [ -s "$path" ] || { echo "[r9c-assess] missing input: $path" >&2; exit 2; }
  done
done

cd "$BASE/train"
"$PY" assess_referential_syndrome_microcode.py \
  --development-treatment "$DEVELOPMENT_TREATMENT" \
  --development-static "$DEVELOPMENT_STATIC" \
  --development-no-syndrome "$DEVELOPMENT_NO_SYNDROME" \
  --development-shuffled-goal "$DEVELOPMENT_SHUFFLED_GOAL" \
  --fresh-treatment "$FRESH_TREATMENT" \
  --fresh-static "$FRESH_STATIC" \
  --fresh-no-syndrome "$FRESH_NO_SYNDROME" \
  --fresh-shuffled-goal "$FRESH_SHUFFLED_GOAL" \
  --out "$OUT"
