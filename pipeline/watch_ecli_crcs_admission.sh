#!/bin/bash
# One-shot CPU-only CRCS admission watcher. It never submits a GPU job and
# cannot materialize CRCS data unless the immutable ECLI assessment explicitly
# proves bounded late-bound latent use.
set -euo pipefail

BASE=${1:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
MAX_POLLS=${MAX_POLLS:-720}
SLEEP_SECONDS=${SLEEP_SECONDS:-60}
ECLI_ASSESSMENT=$BASE/artifacts/eval_history/ecli_fqrb_200k_l19_r1_assessment.json
TRAIN=$BASE/artifacts/sft/causal_residual_count_sketch_v1_train.jsonl
HELDOUT=$BASE/artifacts/evals/causal_residual_count_sketch_v1_heldout.jsonl
AUDIT=$BASE/artifacts/evals/causal_residual_count_sketch_v1_audit.json
STATUS=$BASE/artifacts/eval_history/crcs_v1_admission.json

mkdir -p "$BASE/artifacts/eval_history"
[ ! -e "$STATUS" ] || { echo "[crcs-watch] status already exists; refusing to repeat"; exit 0; }

write_status() {
  "$PY" - "$STATUS" "$1" "$ECLI_ASSESSMENT" <<'PY'
import hashlib
import json
import os
import sys

path, decision, parent = sys.argv[1:]
payload = {
    'audit': 'causal_residual_count_sketch_v1_admission',
    'decision': decision,
    'claim_boundary': 'CPU-only data admission; no CRCS GPU job was submitted.',
}
if os.path.isfile(parent):
    payload['ecli_assessment_sha256'] = hashlib.sha256(open(parent, 'rb').read()).hexdigest()
json.dump(payload, open(path, 'w'), indent=2, sort_keys=True)
PY
}

for ((poll=1; poll<=MAX_POLLS; poll++)); do
  if [ -s "$ECLI_ASSESSMENT" ]; then
    DECISION=$("$PY" - "$ECLI_ASSESSMENT" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1])).get('decision', 'missing_decision'))
PY
)
    if [ "$DECISION" != bounded_ecli_late_binding_candidate ]; then
      write_status "blocked_ecli_not_admitted:$DECISION"
      echo "[crcs-watch] blocked: ECLI decision=$DECISION"
      exit 0
    fi
    for path in "$TRAIN" "$HELDOUT" "$AUDIT"; do
      [ ! -e "$path" ] || { echo "[crcs-watch] output already exists: $path" >&2; exit 2; }
    done
    echo "[crcs-watch] ECLI admitted CRCS data; building CPU-only curriculum"
    "$PY" "$BASE/pipeline/generate_causal_residual_count_sketch_v1.py" \
      --train-out "$TRAIN" --heldout-out "$HELDOUT" --report "$AUDIT" \
      --ecli-assessment "$ECLI_ASSESSMENT"
    "$PY" - "$STATUS" "$ECLI_ASSESSMENT" "$TRAIN" "$HELDOUT" "$AUDIT" <<'PY'
import hashlib
import json
import sys

path, parent, train, heldout, audit = sys.argv[1:]
digest = lambda item: hashlib.sha256(open(item, 'rb').read()).hexdigest()
report = json.load(open(audit))
required_zero = (
    'bad_train_history_cardinality', 'bad_heldout_history_cardinality',
    'train_heldout_exact_history_hits', 'train_heldout_codebook_hits',
    'train_heldout_semantic_13gram_hits',
)
if report.get('audit') != 'causal_residual_count_sketch_v1' or any(report.get(key) for key in required_zero):
    raise SystemExit('CRCS builder report fails its admission audit')
json.dump({
    'audit': 'causal_residual_count_sketch_v1_admission',
    'decision': 'admitted_cpu_data_only',
    'ecli_assessment_sha256': digest(parent),
    'train_sha256': digest(train),
    'heldout_sha256': digest(heldout),
    'audit_sha256': digest(audit),
    'train_rows': report.get('train_rows'),
    'heldout_rows': report.get('heldout_rows'),
    'claim_boundary': 'CPU-only data admission; no CRCS GPU job was submitted.',
}, open(path, 'w'), indent=2, sort_keys=True)
PY
    echo "[crcs-watch] wrote CPU-only CRCS admission status"
    exit 0
  fi
  if (( poll % 30 == 0 )); then
    echo "[crcs-watch] waiting poll=$poll"
  fi
  sleep "$SLEEP_SECONDS"
done

write_status "blocked_timed_out_waiting_for_ecli_assessment"
echo "[crcs-watch] timed out waiting for ECLI assessment"
