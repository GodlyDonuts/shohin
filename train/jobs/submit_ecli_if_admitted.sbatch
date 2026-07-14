#!/bin/bash
# One-shot CPU gate. It has no GPU request and may submit the isolated ECLI
# chain only after the Stokes watcher has produced hash-bound admitted data.
# A blocked decision is a successful no-op, not a failed training attempt.
#SBATCH -J shohin-ecli-gate
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ecli_gate_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
ASSESSMENT=$BASE/artifacts/eval_history/fqrb_200k_l19_r1_assessment.json
DATA=$BASE/artifacts/sft/ephemeral_codebook_fqrb_v1_train.jsonl
HELDOUT=$BASE/artifacts/evals/ephemeral_codebook_fqrb_v1_heldout.jsonl
AUDIT=$BASE/artifacts/evals/ephemeral_codebook_fqrb_v1_audit.json
INIT=$BASE/train/fqrb_200k_l19_r1/cra_ep1.pt
OUT=$BASE/train/ephemeral_codebook_fqrb_200k_l19_r1
EVAL_OUT=$BASE/artifacts/eval_history/ecli_fqrb_200k_l19_r1_combined.json
ASSESS_OUT=$BASE/artifacts/eval_history/ecli_fqrb_200k_l19_r1_assessment.json
STATUS=$BASE/artifacts/eval_history/ecli_fqrb_200k_l19_r1_admission.json

mkdir -p "$BASE/artifacts/eval_history"
[ ! -e "$STATUS" ] || { echo "[ecli-gate] status already exists; refusing to repeat"; exit 0; }

for poll in $(seq 1 120); do
  if [ -s "$ASSESSMENT" ]; then
    PARENT_DECISION=$("$PY" - "$ASSESSMENT" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1])).get('decision', 'missing_decision'))
PY
)
    if [ "$PARENT_DECISION" != bounded_fqrb_basis_candidate_magnitude_and_interaction_still_required ]; then
      "$PY" - "$STATUS" "$PARENT_DECISION" <<'PY'
import json
import sys
json.dump({
    'audit': 'ecli_fqrb_admission_v1',
    'decision': 'blocked_fqrb_not_admitted:' + sys.argv[2],
    'claim_boundary': 'No GPU job was submitted.',
}, open(sys.argv[1], 'w'), indent=2, sort_keys=True)
PY
      echo "[ecli-gate] blocked: FQRB decision=$PARENT_DECISION"
      exit 0
    fi
    if [ -s "$DATA" ] && [ -s "$HELDOUT" ] && [ -s "$AUDIT" ] && [ -s "$INIT" ]; then
      break
    fi
  fi
  if [ "$poll" = 120 ]; then
    "$PY" - "$STATUS" <<'PY'
import json
import sys
json.dump({
    'audit': 'ecli_fqrb_admission_v1',
    'decision': 'blocked_timed_out_waiting_for_fqrb_admission_or_ecli_data',
    'claim_boundary': 'No GPU job was submitted.',
}, open(sys.argv[1], 'w'), indent=2, sort_keys=True)
PY
    echo "[ecli-gate] timed out waiting for FQRB admission or ECLI data"
    exit 0
  fi
  sleep 60
done

DATA_SHA=$(sha256sum "$DATA" | awk '{print $1}')
HELDOUT_SHA=$(sha256sum "$HELDOUT" | awk '{print $1}')
ASSESSMENT_SHA=$(sha256sum "$ASSESSMENT" | awk '{print $1}')
AUDIT_SHA=$(sha256sum "$AUDIT" | awk '{print $1}')

"$PY" - "$ASSESSMENT" "$AUDIT" "$DATA_SHA" "$HELDOUT_SHA" "$ASSESSMENT_SHA" "$AUDIT_SHA" <<'PY'
import json
import sys

assessment_path, audit_path, data_sha, heldout_sha, assessment_sha, audit_sha = sys.argv[1:]
assessment = json.load(open(assessment_path))
audit = json.load(open(audit_path))
if assessment.get('decision') != 'bounded_fqrb_basis_candidate_magnitude_and_interaction_still_required':
    raise SystemExit('FQRB assessment does not admit ECLI')
if audit.get('audit') != 'ephemeral_codebook_fqrb_v1':
    raise SystemExit('ECLI audit type mismatch')
if audit.get('train_sha256') != data_sha or audit.get('heldout_sha256') != heldout_sha:
    raise SystemExit('ECLI data hashes do not bind the audit')
if audit.get('fqrb_parent_assessment_sha256') != assessment_sha:
    raise SystemExit('ECLI audit does not bind the supplied FQRB assessment')
if not audit_sha:
    raise SystemExit('missing ECLI audit hash')
PY

[ ! -e "$OUT" ] || { echo "[ecli-gate] output exists; refusing to submit"; exit 2; }
[ ! -e "$EVAL_OUT" ] || { echo "[ecli-gate] evaluation output exists; refusing to submit"; exit 2; }
[ ! -e "$ASSESS_OUT" ] || { echo "[ecli-gate] assessment output exists; refusing to submit"; exit 2; }

cd "$BASE"
TRAIN_JOB=$(sbatch --parsable --export=NONE,INIT="$INIT",DATA="$DATA",DATA_SHA="$DATA_SHA",AUDIT="$AUDIT",ASSESSMENT="$ASSESSMENT",OUT="$OUT" "$BASE/train/jobs/train_ephemeral_codebook_fqrb.sbatch")
EVAL_JOB=$(sbatch --parsable --dependency=afterok:"$TRAIN_JOB" --export=NONE,CKPT="$OUT/ecfqrb_ep1.pt",DATA="$HELDOUT",DATA_SHA="$HELDOUT_SHA",OUT="$EVAL_OUT",SPLIT=heldout,GROUPS=500,MAX_NEW=4 "$BASE/train/jobs/eval_ephemeral_codebook_fqrb.sbatch")
ASSESS_JOB=$(sbatch --parsable --dependency=afterok:"$EVAL_JOB" --export=NONE,REPORT="$EVAL_OUT",AUDIT="$AUDIT",OUT="$ASSESS_OUT" "$BASE/train/jobs/assess_ephemeral_codebook_fqrb.sbatch")

"$PY" - "$STATUS" "$TRAIN_JOB" "$EVAL_JOB" "$ASSESS_JOB" "$DATA_SHA" "$HELDOUT_SHA" "$ASSESSMENT_SHA" "$AUDIT_SHA" <<'PY'
import json
import sys
path, train_job, eval_job, assess_job, data_sha, heldout_sha, assessment_sha, audit_sha = sys.argv[1:]
json.dump({
    'audit': 'ecli_fqrb_admission_v1',
    'decision': 'submitted_conditional_ecli_chain',
    'train_job': train_job,
    'eval_job': eval_job,
    'assess_job': assess_job,
    'train_data_sha256': data_sha,
    'heldout_data_sha256': heldout_sha,
    'fqrb_assessment_sha256': assessment_sha,
    'ecli_audit_sha256': audit_sha,
    'claim_boundary': 'Submission is an isolated late-binding test, not a reasoning claim.',
}, open(path, 'w'), indent=2, sort_keys=True)
PY
echo "[ecli-gate] submitted train=$TRAIN_JOB eval=$EVAL_JOB"
