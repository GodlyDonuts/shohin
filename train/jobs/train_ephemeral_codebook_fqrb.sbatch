#!/bin/bash
# Conditional isolated ECLI training. Never use before FQRB's causal gates pass.
# Sources reach the suffix only through donor + edited - base residual tapes.
#SBATCH -J shohin-ecli-train
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 05:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ecli_train_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:?set immutable source checkpoint}
DATA=${DATA:?set frozen ECLI train data}
DATA_SHA=${DATA_SHA:?set expected ECLI train SHA-256}
AUDIT=${AUDIT:?set frozen ECLI audit}
ASSESSMENT=${ASSESSMENT:?set admitted FQRB assessment}
OUT=${OUT:?set fresh isolated train/ephemeral_codebook_fqrb_* output}
LAYER=${LAYER:-19}
SOURCE_WINDOW=${SOURCE_WINDOW:--1}
EPOCHS=${EPOCHS:-1}
BS=${BS:-4}
MAX_BATCHES=${MAX_BATCHES:-0}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

[ -s "$INIT" ] || { echo "[ecli-train] init checkpoint missing" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[ecli-train] data missing" >&2; exit 3; }
[ -s "$AUDIT" ] || { echo "[ecli-train] audit missing" >&2; exit 4; }
[ -s "$ASSESSMENT" ] || { echo "[ecli-train] FQRB assessment missing" >&2; exit 4; }
[ ! -e "$OUT" ] || { echo "[ecli-train] refusing existing output" >&2; exit 5; }
case "$OUT" in "$BASE"/train/ephemeral_codebook_fqrb_*) ;; *) echo "[ecli-train] output must remain isolated" >&2; exit 6;; esac
[ "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA" ] || { echo "[ecli-train] data hash mismatch" >&2; exit 7; }

"$PY" - "$DATA" "$AUDIT" "$ASSESSMENT" <<'PY'
import hashlib
import json
import sys

data_path, audit_path, assessment_path = sys.argv[1:]
audit = json.load(open(audit_path))
assessment = json.load(open(assessment_path))
required_zero = (
    'duplicate_train_prompts', 'duplicate_heldout_prompts', 'train_heldout_exact_prompt_hits',
    'train_heldout_exact_source_bundle_hits', 'train_heldout_codebook_hits',
    'train_heldout_semantic_13gram_hits', 'bad_train_group_cardinality', 'bad_heldout_group_cardinality',
)
digest = hashlib.sha256(open(data_path, 'rb').read()).hexdigest()
if audit.get('audit') != 'ephemeral_codebook_fqrb_v1' or audit.get('mechanism') != 'ephemeral_codebook_fqrb_v1':
    raise SystemExit('[ecli-train] audit mechanism mismatch')
if audit.get('train_sha256') != digest or audit.get('train_rows') != 60000 or audit.get('heldout_rows') != 2500:
    raise SystemExit('[ecli-train] audit does not bind a frozen ECLI corpus')
if any(audit.get(field) for field in required_zero):
    raise SystemExit('[ecli-train] overlap/group admission gate failed')
if assessment.get('audit') != 'finite_query_residual_basis_v1_assessment' or assessment.get('decision') != 'bounded_fqrb_basis_candidate_magnitude_and_interaction_still_required':
    raise SystemExit('[ecli-train] FQRB parent decision does not admit ECLI')
if audit.get('fqrb_parent_assessment_sha256') != hashlib.sha256(open(assessment_path, 'rb').read()).hexdigest():
    raise SystemExit('[ecli-train] audit does not bind the supplied FQRB assessment')
print('[ecli-train] hash-bound ECLI admission passed')
PY

cd "$BASE/train"
timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device='cuda', dtype=torch.bfloat16)
PY
ARGS=()
ARGS+=(--source-window "$SOURCE_WINDOW")
if [ "$MAX_BATCHES" -gt 0 ]; then ARGS+=(--max-batches "$MAX_BATCHES"); fi
echo "[ecli-train] init=$INIT data=$DATA layer=$LAYER source_window=$SOURCE_WINDOW out=$OUT epochs=$EPOCHS batch=$BS"
"$PY" train_ephemeral_codebook_fqrb.py \
  --init "$INIT" --data "$DATA" --audit "$AUDIT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --layer "$LAYER" --epochs "$EPOCHS" --batch-size "$BS" "${ARGS[@]}"
test -s "$OUT/ecfqrb_ep1.pt"
echo "[ecli-train] wrote $OUT/ecfqrb_ep1.pt"
