#!/bin/bash
# Isolated Counterfactual Residual Algebra training. Three source worlds can
# influence answers only through donor + edited - base native residual tapes.
# This wrapper cannot read or write a flagship output directory.
#SBATCH -J shohin-cra-train
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/cra_train_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:-$BASE/train/flagship_out/best_step200000.pt}
DATA=${DATA:-$BASE/artifacts/sft/counterfactual_residual_algebra_v1_train.jsonl}
DATA_SHA=e067c2561d397d9e0c44318c5e21d9e50820bd6cd88ccd2513d5b8e2f6e059f5
AUDIT=${AUDIT:-$BASE/artifacts/evals/counterfactual_residual_algebra_v1_audit.json}
OUT=${OUT:?set OUT to a fresh isolated train/cra_* output directory}
LAYER=${LAYER:-19}
TAPE_LEN=${TAPE_LEN:-0}
EPOCHS=${EPOCHS:-1}
BS=${BS:-4}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
MAX_BATCHES=${MAX_BATCHES:-0}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

[ -f "$INIT" ] || { echo "[cra-train] init checkpoint missing" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[cra-train] data missing" >&2; exit 3; }
[ -s "$AUDIT" ] || { echo "[cra-train] audit missing" >&2; exit 4; }
[ ! -e "$OUT" ] || { echo "[cra-train] refusing existing output" >&2; exit 5; }
case "$OUT" in "$BASE"/train/cra_*) ;; *) echo "[cra-train] OUT must remain an isolated train/cra_* directory" >&2; exit 6;; esac
[ "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA" ] || { echo "[cra-train] data hash mismatch" >&2; exit 7; }

"$PY" - "$DATA" "$AUDIT" <<'PY'
import hashlib
import json
import sys

data_path, audit_path = sys.argv[1:]
digest = hashlib.sha256(open(data_path, 'rb').read()).hexdigest()
audit = json.load(open(audit_path))
required = ('duplicate_train_prompts', 'duplicate_heldout_prompts', 'train_heldout_exact_prompt_hits', 'train_heldout_13gram_hits')
if audit.get('audit') != 'counterfactual_residual_algebra_v1' or audit.get('train_sha256') != digest:
    raise SystemExit('[cra-train] audit does not bind the requested train data')
if any(audit.get(field) for field in required):
    raise SystemExit('[cra-train] CRA admission overlap gate failed')
if audit.get('train_rows') != 30000 or audit.get('heldout_rows') != 2000:
    raise SystemExit('[cra-train] unexpected CRA corpus cardinality')
print('[cra-train] hash-bound CRA admission passed')
PY

cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device='cuda', dtype=torch.bfloat16)
PY
then
  echo "[cra-train] CUDA unavailable or timed out after allocation" >&2
  exit 75
fi
ARGS=()
if [ "$TAPE_LEN" -gt 0 ]; then ARGS+=(--tape-len "$TAPE_LEN"); fi
if [ "$MAX_EXAMPLES" -gt 0 ]; then ARGS+=(--max-examples "$MAX_EXAMPLES"); fi
if [ "$MAX_BATCHES" -gt 0 ]; then ARGS+=(--max-batches "$MAX_BATCHES"); fi
echo "[cra-train] init=$INIT data=$DATA layer=$LAYER tape_len=${TAPE_LEN:-auto} out=$OUT epochs=$EPOCHS batch=$BS max_batches=$MAX_BATCHES"
"$PY" train_counterfactual_residual_algebra.py \
  --init "$INIT" --data "$DATA" --audit "$AUDIT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --layer "$LAYER" --epochs "$EPOCHS" --batch-size "$BS" "${ARGS[@]}"
test -s "$OUT/cra_ep1.pt"
echo "[cra-train] wrote $OUT/cra_ep1.pt"
