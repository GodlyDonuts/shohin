#!/bin/bash
# CPU-only, one-shot FQRB evidence watcher for Stokes. It never submits jobs,
# edits model/data artifacts, or writes an assessment until every bound report
# is complete. The surrounding operator must still review the decision.
set -euo pipefail

BASE=${1:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
MAX_POLLS=${MAX_POLLS:-480}
SLEEP_SECONDS=${SLEEP_SECONDS:-60}
OUT=$BASE/artifacts/eval_history/fqrb_200k_l19_r1_assessment.json
COMBINED=$BASE/artifacts/eval_history/fqrb_200k_l19_r1_combined.json
CORE=$BASE/artifacts/eval_history/fqrb_200k_l19_r1_core_factor.json
MAGNITUDE=$BASE/artifacts/eval_history/fqrb_200k_l19_r1_magnitude_factor.json
MANUAL=$BASE/artifacts/eval_history/manual_capability_raw200k_vs_fqrb_200k_l19_r1.json
CKPT=$BASE/train/fqrb_200k_l19_r1/cra_ep1.pt

if [ -e "$OUT" ]; then
  echo "[fqrb-watch] assessment already exists; refusing to replace it"
  exit 0
fi
for ((poll=1; poll<=MAX_POLLS; poll++)); do
  if [ -s "$COMBINED" ] && [ -s "$CORE" ] && [ -s "$MAGNITUDE" ] && [ -s "$MANUAL" ] && [ -s "$CKPT" ]; then
    echo "[fqrb-watch] all bound reports found at poll=$poll"
    exec "$PY" "$BASE/pipeline/assess_finite_query_residual_basis_v1.py" \
      --combined "$COMBINED" --core "$CORE" --magnitude "$MAGNITUDE" --manual "$MANUAL" \
      --fqrb-checkpoint "$CKPT" --out "$OUT"
  fi
  if (( poll % 30 == 0 )); then
    echo "[fqrb-watch] waiting poll=$poll"
  fi
  sleep "$SLEEP_SECONDS"
done
echo "[fqrb-watch] timed out before all evidence arrived" >&2
exit 1
