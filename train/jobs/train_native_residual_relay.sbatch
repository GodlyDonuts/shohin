#!/bin/bash
# Isolated no-parameter Native Residual Relay training. The source can only
# influence answer tokens through a native residual at the requested layer.
# This wrapper never reads or writes the flagship writer output.
#SBATCH -J shohin-nrr-train
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/nrr_train_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:-$BASE/train/flagship_out/best_step200000.pt}
DATA=${DATA:-$BASE/artifacts/sft/native_residual_relay_v1_train.jsonl}
DATA_SHA=bac1e8d041abbfefa892056302a8d78c14abd0d31dd1694e9bc92aefac2fe03c
AUDIT=${AUDIT:-$BASE/artifacts/sft/native_residual_relay_v1.audit.json}
OUT=${OUT:?set OUT to a fresh isolated train/nrr_* output directory}
LAYER=${LAYER:-19}
EPOCHS=${EPOCHS:-1}
BS=${BS:-8}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
MAX_BATCHES=${MAX_BATCHES:-0}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

[ -f "$INIT" ] || { echo "[nrr-train] init checkpoint missing" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[nrr-train] data missing" >&2; exit 3; }
[ -s "$AUDIT" ] || { echo "[nrr-train] audit missing" >&2; exit 4; }
[ ! -e "$OUT" ] || { echo "[nrr-train] refusing existing output" >&2; exit 5; }
case "$OUT" in "$BASE"/train/nrr_*) ;; *) echo "[nrr-train] OUT must remain an isolated train/nrr_* directory" >&2; exit 6;; esac
[ "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA" ] || { echo "[nrr-train] data hash mismatch" >&2; exit 7; }

"$PY" - "$DATA" "$AUDIT" <<'PY'
import hashlib
import json
import sys

data_path, audit_path = sys.argv[1:]
digest = hashlib.sha256(open(data_path, 'rb').read()).hexdigest()
audit = json.load(open(audit_path))
required = ('duplicate_train_prompts', 'duplicate_heldout_prompts', 'train_heldout_exact_prompt_hits', 'train_heldout_13gram_hits')
if audit.get('audit') != 'native_residual_relay_v1' or audit.get('train_sha256') != digest:
    raise SystemExit('[nrr-train] audit does not bind the requested train data')
if any(audit.get(field) for field in required):
    raise SystemExit('[nrr-train] NRR admission overlap gate failed')
if audit.get('train_rows') != 30000 or audit.get('heldout_rows') != 2000:
    raise SystemExit('[nrr-train] unexpected NRR corpus cardinality')
print('[nrr-train] hash-bound NRR admission passed')
PY

cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device='cuda', dtype=torch.bfloat16)
PY
then
  echo "[nrr-train] CUDA unavailable or timed out after allocation" >&2
  exit 75
fi
ARGS=()
if [ "$MAX_EXAMPLES" -gt 0 ]; then ARGS+=(--max-examples "$MAX_EXAMPLES"); fi
if [ "$MAX_BATCHES" -gt 0 ]; then ARGS+=(--max-batches "$MAX_BATCHES"); fi
echo "[nrr-train] init=$INIT data=$DATA layer=$LAYER out=$OUT epochs=$EPOCHS batch=$BS max_batches=$MAX_BATCHES"
"$PY" train_native_residual_relay.py \
  --init "$INIT" --data "$DATA" --audit "$AUDIT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --layer "$LAYER" --epochs "$EPOCHS" --batch-size "$BS" "${ARGS[@]}"
test -s "$OUT/nrr_ep1.pt"
echo "[nrr-train] wrote $OUT/nrr_ep1.pt"
