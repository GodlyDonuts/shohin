#!/bin/bash
# Read-only NRR causal evaluation. It accepts only an isolated NRR checkpoint,
# writes a fresh eval-history report, and never writes a training directory.
#SBATCH -J shohin-nrr-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/nrr_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to an isolated train/nrr_* checkpoint}
DATA=${DATA:-$BASE/artifacts/evals/native_residual_relay_v1_heldout.jsonl}
DATA_SHA=1d8b633713fff41b331e7c2728e9c0aa3ae307a7b99622c526d99d6dc84120f2
OUT=${OUT:?set OUT to a fresh artifacts/eval_history/nrr_*.json path}
ROWS=${ROWS:-500}
MAX_NEW=${MAX_NEW:-12}
SPLIT=${SPLIT:-heldout}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

[ -f "$CKPT" ] || { echo "[nrr-eval] checkpoint missing" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[nrr-eval] held-out data missing" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[nrr-eval] refusing existing output" >&2; exit 4; }
case "$CKPT" in "$BASE"/train/nrr_*/nrr_ep1.pt) ;; *) echo "[nrr-eval] checkpoint must be isolated NRR output" >&2; exit 5;; esac
case "$OUT" in "$BASE"/artifacts/eval_history/nrr_*.json) ;; *) echo "[nrr-eval] OUT must be an NRR eval-history JSON" >&2; exit 6;; esac
[ "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA" ] || { echo "[nrr-eval] held-out hash mismatch" >&2; exit 7; }

cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device='cuda', dtype=torch.bfloat16)
PY
then
  echo "[nrr-eval] CUDA unavailable or timed out after allocation" >&2
  exit 75
fi
echo "[nrr-eval] ckpt=$CKPT data=$DATA split=$SPLIT rows=$ROWS out=$OUT"
"$PY" eval_native_residual_relay.py \
  --ckpt "$CKPT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --split "$SPLIT" --max-examples "$ROWS" --max-new "$MAX_NEW"
test -s "$OUT"
echo "[nrr-eval] wrote $OUT"
