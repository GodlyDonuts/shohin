#!/bin/bash
# One-shot CPU admission gate for the phase-aligned anchor-tape ablation.
# It may submit PAAT only for a control-clean FQRB transport failure. It never
# replaces the original FQRB output or retries a missing primitive generically.
#SBATCH -J shohin-paat-gate
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 02:15:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/paat_gate_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
ASSESSMENT=$BASE/artifacts/eval_history/fqrb_200k_l19_r1_assessment.json
TAXONOMY=$BASE/artifacts/eval_history/fqrb_200k_l19_r1_failure_taxonomy.json
STATUS=$BASE/artifacts/eval_history/fqrb_200k_l19_phase_r1_admission.json
INIT=$BASE/train/flagship_out/best_step200000.pt
DATA=$BASE/artifacts/sft/finite_query_residual_basis_v1_train.jsonl
AUDIT=$BASE/artifacts/evals/finite_query_residual_basis_v1_audit.json
OUT=$BASE/train/fqrb_200k_l19_phase_r1
COMBINED_DATA=$BASE/artifacts/evals/finite_query_residual_basis_v1_heldout.jsonl
CORE_DATA=$BASE/artifacts/evals/finite_query_residual_basis_v1_core_factor.jsonl
MAGNITUDE_DATA=$BASE/artifacts/evals/finite_query_residual_basis_v1_magnitude_factor.jsonl

mkdir -p "$BASE/artifacts/eval_history"
[ ! -e "$STATUS" ] || { echo "[paat-gate] status already exists; refusing to repeat"; exit 0; }

for poll in $(seq 1 120); do
  if [ -s "$ASSESSMENT" ] && [ -s "$TAXONOMY" ]; then
    break
  fi
  if [ "$poll" = 120 ]; then
    "$PY" - "$STATUS" <<'PY'
import json
import sys
json.dump({
    'audit': 'paat_fqrb_admission_v1',
    'decision': 'blocked_missing_fqrb_assessment',
    'claim_boundary': 'No PAAT GPU job was submitted.',
}, open(sys.argv[1], 'w'), indent=2, sort_keys=True)
PY
    echo "[paat-gate] assessment did not arrive before deadline"
    exit 0
  fi
  sleep 60
done

DECISION=$("$PY" - "$ASSESSMENT" "$TAXONOMY" <<'PY'
import json
import sys

assessment = json.load(open(sys.argv[1]))
taxonomy = json.load(open(sys.argv[2]))
gates = taxonomy.get('gates', {})
train = gates.get('train', {})
combined = gates.get('combined', {})
core = gates.get('core', {})
if assessment.get('decision') == 'bounded_fqrb_basis_candidate_magnitude_and_interaction_still_required':
    print('not_needed_fqrb_candidate')
elif train.get('pass') and combined.get('controls') and core.get('controls') and (not combined.get('pass') or not core.get('pass')):
    print('admit_phase_aligned_transport_ablation')
else:
    print('blocked_nontransport_fqrb_failure')
PY
)

if [ "$DECISION" != admit_phase_aligned_transport_ablation ]; then
  "$PY" - "$STATUS" "$DECISION" "$ASSESSMENT" "$TAXONOMY" <<'PY'
import hashlib
import json
import sys
path, decision, assessment_path, taxonomy_path = sys.argv[1:]
digest = lambda item: hashlib.sha256(open(item, 'rb').read()).hexdigest()
json.dump({
    'audit': 'paat_fqrb_admission_v1',
    'decision': decision,
    'fqrb_assessment_sha256': digest(assessment_path),
    'fqrb_taxonomy_sha256': digest(taxonomy_path),
    'claim_boundary': 'No PAAT GPU job was submitted.',
}, open(path, 'w'), indent=2, sort_keys=True)
PY
  echo "[paat-gate] decision=$DECISION"
  exit 0
fi

[ ! -e "$OUT" ] || { echo "[paat-gate] output exists; refusing to submit"; exit 2; }
for report in \
  "$BASE/artifacts/eval_history/fqrb_200k_l19_phase_r1_combined.json" \
  "$BASE/artifacts/eval_history/fqrb_200k_l19_phase_r1_core_factor.json" \
  "$BASE/artifacts/eval_history/fqrb_200k_l19_phase_r1_magnitude_factor.json" \
  "$BASE/artifacts/eval_history/manual_capability_raw200k_vs_fqrb_200k_l19_phase_r1.json"; do
  [ ! -e "$report" ] || { echo "[paat-gate] report exists: $report"; exit 2; }
done

DATA_SHA=$(sha256sum "$DATA" | awk '{print $1}')
cd "$BASE"
TRAIN_JOB=$(sbatch --parsable --export=NONE,INIT="$INIT",DATA="$DATA",AUDIT="$AUDIT",OUT="$OUT",SOURCE_WINDOW=26,BS=4,EPOCHS=1 "$BASE/train/jobs/train_finite_query_residual_basis.sbatch")
COMBINED_JOB=$(sbatch --parsable --dependency=afterok:"$TRAIN_JOB" --export=NONE,CKPT="$OUT/cra_ep1.pt",DATA="$COMBINED_DATA",DATA_SHA=832da1a5264dbd0ec9ccf5b8515276794d545531da5f3dbdacdf4c239452ef97,OUT="$BASE/artifacts/eval_history/fqrb_200k_l19_phase_r1_combined.json",GROUPS=500,SPLIT=heldout,MAX_NEW=6 "$BASE/train/jobs/eval_finite_query_residual_basis.sbatch")
CORE_JOB=$(sbatch --parsable --dependency=afterok:"$TRAIN_JOB" --export=NONE,CKPT="$OUT/cra_ep1.pt",DATA="$CORE_DATA",DATA_SHA=888d18e29740edf4ff1a184ed023debad1715b2610e2e3ec34d871ded4f6abf5,OUT="$BASE/artifacts/eval_history/fqrb_200k_l19_phase_r1_core_factor.json",GROUPS=500,SPLIT=factor_core,MAX_NEW=6 "$BASE/train/jobs/eval_finite_query_residual_basis.sbatch")
MAGNITUDE_JOB=$(sbatch --parsable --dependency=afterok:"$TRAIN_JOB" --export=NONE,CKPT="$OUT/cra_ep1.pt",DATA="$MAGNITUDE_DATA",DATA_SHA=3e530f4e8210aa7faf737c5d8a4605d0e899ab38a3d051f1291317ce3c41b452,OUT="$BASE/artifacts/eval_history/fqrb_200k_l19_phase_r1_magnitude_factor.json",GROUPS=500,SPLIT=factor_magnitude,MAX_NEW=6 "$BASE/train/jobs/eval_finite_query_residual_basis.sbatch")
MANUAL_JOB=$(sbatch --parsable --dependency=afterok:"$TRAIN_JOB" --export=NONE,RAW_CKPT="$INIT",SFT_CKPT="$OUT/cra_ep1.pt",OUT="$BASE/artifacts/eval_history/manual_capability_raw200k_vs_fqrb_200k_l19_phase_r1.json" "$BASE/train/jobs/manual_capability_probe.sbatch")
DEEP_JOB=$(sbatch --parsable --dependency=afterok:"$MANUAL_JOB" --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50 --export=NONE,CKPT="$OUT/cra_ep1.pt",TAG=deep_interaction_fqrb_200k_l19_phase_r1 "$BASE/train/jobs/deep_interaction_audit.sbatch")

"$PY" - "$STATUS" "$TRAIN_JOB" "$COMBINED_JOB" "$CORE_JOB" "$MAGNITUDE_JOB" "$MANUAL_JOB" "$DEEP_JOB" "$DATA_SHA" "$ASSESSMENT" "$TAXONOMY" <<'PY'
import hashlib
import json
import sys
path, train, combined, core, magnitude, manual, deep, data_sha, assessment_path, taxonomy_path = sys.argv[1:]
digest = lambda item: hashlib.sha256(open(item, 'rb').read()).hexdigest()
json.dump({
    'audit': 'paat_fqrb_admission_v1',
    'decision': 'submitted_phase_aligned_transport_ablation',
    'source_window': 26,
    'train_job': train,
    'combined_job': combined,
    'core_job': core,
    'magnitude_job': magnitude,
    'manual_job': manual,
    'deep_interaction_job': deep,
    'train_data_sha256': data_sha,
    'fqrb_assessment_sha256': digest(assessment_path),
    'fqrb_taxonomy_sha256': digest(taxonomy_path),
    'claim_boundary': 'PAAT changes only positional alignment; a result remains a bounded residual-basis test.',
}, open(path, 'w'), indent=2, sort_keys=True)
PY
echo "[paat-gate] submitted train=$TRAIN_JOB combined=$COMBINED_JOB core=$CORE_JOB magnitude=$MAGNITUDE_JOB manual=$MANUAL_JOB deep=$DEEP_JOB"
