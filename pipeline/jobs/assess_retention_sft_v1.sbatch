#!/bin/bash
#SBATCH -J shohin-retain-assess
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/retention_assess_%j.out
#SBATCH --open-mode=append
# CPU-only evidence collation. It cannot submit or train a successor.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
RAW_PRIMITIVES=${RAW_PRIMITIVES:?}
CANDIDATE_PRIMITIVES=${CANDIDATE_PRIMITIVES:?}
RAW_RG=${RAW_RG:?}
CANDIDATE_RG=${CANDIDATE_RG:?}
MANUAL=${MANUAL:?}
DEEP=${DEEP:?}
OUT=${OUT:?}
for path in "$RAW_PRIMITIVES" "$CANDIDATE_PRIMITIVES" "$RAW_RG" "$CANDIDATE_RG" "$MANUAL" "$DEEP"; do
  [ -s "$path" ] || { echo "[retain-assess] missing evidence: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[retain-assess] refusing existing output" >&2; exit 3; }
cd "$BASE"
"$PY" pipeline/assess_retention_sft_v1.py \
  --raw-primitives "$RAW_PRIMITIVES" --candidate-primitives "$CANDIDATE_PRIMITIVES" \
  --raw-rg "$RAW_RG" --candidate-rg "$CANDIDATE_RG" --manual "$MANUAL" --deep "$DEEP" --out "$OUT"
