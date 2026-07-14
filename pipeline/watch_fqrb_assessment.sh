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
TAXONOMY_OUT=$BASE/artifacts/eval_history/fqrb_200k_l19_r1_failure_taxonomy.json
ECLI_TRAIN=$BASE/artifacts/sft/ephemeral_codebook_fqrb_v1_train.jsonl
ECLI_HELDOUT=$BASE/artifacts/evals/ephemeral_codebook_fqrb_v1_heldout.jsonl
ECLI_AUDIT=$BASE/artifacts/evals/ephemeral_codebook_fqrb_v1_audit.json
TRAIN=$BASE/artifacts/eval_history/fqrb_200k_l19_r1_train_diagnostic.json
COMBINED=$BASE/artifacts/eval_history/fqrb_200k_l19_r1_combined.json
CORE=$BASE/artifacts/eval_history/fqrb_200k_l19_r1_core_factor.json
MAGNITUDE=$BASE/artifacts/eval_history/fqrb_200k_l19_r1_magnitude_factor.json
MANUAL=$BASE/artifacts/eval_history/manual_capability_raw200k_vs_fqrb_200k_l19_r1.json
CKPT=$BASE/train/fqrb_200k_l19_r1/cra_ep1.pt

if [ -e "$OUT" ] || [ -e "$TAXONOMY_OUT" ]; then
  echo "[fqrb-watch] assessment or taxonomy already exists; refusing to replace it"
  exit 0
fi
for ((poll=1; poll<=MAX_POLLS; poll++)); do
  if [ -s "$TRAIN" ] && [ -s "$COMBINED" ] && [ -s "$CORE" ] && [ -s "$MAGNITUDE" ] && [ -s "$MANUAL" ] && [ -s "$CKPT" ]; then
    echo "[fqrb-watch] all bound reports found at poll=$poll"
    "$PY" "$BASE/pipeline/assess_finite_query_residual_basis_v1.py" \
      --combined "$COMBINED" --core "$CORE" --magnitude "$MAGNITUDE" --manual "$MANUAL" \
      --fqrb-checkpoint "$CKPT" --out "$OUT"
    "$PY" "$BASE/pipeline/analyze_finite_query_residual_basis_v1.py" \
      --train "$TRAIN" --combined "$COMBINED" --core "$CORE" --magnitude "$MAGNITUDE" --out "$TAXONOMY_OUT"
    decision=$("$PY" - "$OUT" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))['decision'])
PY
)
    if [ "$decision" != "bounded_fqrb_basis_candidate_magnitude_and_interaction_still_required" ]; then
      echo "[fqrb-watch] decision=$decision; ECLI data remains blocked"
      exit 0
    fi
    if [ -e "$ECLI_TRAIN" ] || [ -e "$ECLI_HELDOUT" ] || [ -e "$ECLI_AUDIT" ]; then
      echo "[fqrb-watch] ECLI output already exists; refusing to replace it"
      exit 0
    fi
    echo "[fqrb-watch] FQRB gate passed; generating conditional ECLI data only"
    exec "$PY" "$BASE/pipeline/generate_ephemeral_codebook_fqrb_v1.py" \
      --train-out "$ECLI_TRAIN" --heldout-out "$ECLI_HELDOUT" --report "$ECLI_AUDIT" --fqrb-assessment "$OUT"
  fi
  if (( poll % 30 == 0 )); then
    echo "[fqrb-watch] waiting poll=$poll"
  fi
  sleep "$SLEEP_SECONDS"
done
echo "[fqrb-watch] timed out before all evidence arrived" >&2
exit 1
