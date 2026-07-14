#!/bin/bash
# CPU-only, read-only decision record for the operator-trace candidate.
# It cannot submit or modify a training run.
#SBATCH -J shohin-optrace-assess
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/assess_operator_trace_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
BASELINE_FACTOR=${BASELINE_FACTOR:?set BASELINE_FACTOR}
CANDIDATE_FACTOR=${CANDIDATE_FACTOR:?set CANDIDATE_FACTOR}
BASELINE_PRIMITIVES=${BASELINE_PRIMITIVES:?set BASELINE_PRIMITIVES}
CANDIDATE_PRIMITIVES=${CANDIDATE_PRIMITIVES:?set CANDIDATE_PRIMITIVES}
BASELINE_RG=${BASELINE_RG:?set BASELINE_RG}
CANDIDATE_RG=${CANDIDATE_RG:?set CANDIDATE_RG}
MANUAL=${MANUAL:?set MANUAL}
BASELINE_FIXED_TRACE=${BASELINE_FIXED_TRACE:?set BASELINE_FIXED_TRACE}
CANDIDATE_FIXED_TRACE=${CANDIDATE_FIXED_TRACE:?set CANDIDATE_FIXED_TRACE}
OUT=${OUT:?set OUT to a fresh assessment JSON}

for path in "$BASELINE_FACTOR" "$CANDIDATE_FACTOR" "$BASELINE_PRIMITIVES" "$CANDIDATE_PRIMITIVES" "$BASELINE_RG" "$CANDIDATE_RG" "$MANUAL" "$BASELINE_FIXED_TRACE" "$CANDIDATE_FIXED_TRACE"; do
  [ -s "$path" ] || { echo "[optrace-assess] evidence missing: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[optrace-assess] refusing existing assessment: $OUT" >&2; exit 3; }

cd "$BASE"
"$PY" pipeline/assess_operator_trace_v2.py \
  --baseline-factor "$BASELINE_FACTOR" --candidate-factor "$CANDIDATE_FACTOR" \
  --baseline-primitives "$BASELINE_PRIMITIVES" --candidate-primitives "$CANDIDATE_PRIMITIVES" \
  --baseline-rg "$BASELINE_RG" --candidate-rg "$CANDIDATE_RG" --manual "$MANUAL" \
  --baseline-fixed-trace "$BASELINE_FIXED_TRACE" --candidate-fixed-trace "$CANDIDATE_FIXED_TRACE" \
  --out "$OUT"
echo "[optrace-assess] wrote $OUT"
