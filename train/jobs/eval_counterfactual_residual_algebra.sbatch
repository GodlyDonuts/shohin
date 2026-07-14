#!/bin/bash
# Read-only CRA evaluation. Only isolated CRA checkpoints and fresh eval-history
# reports are accepted; no training or flagship path is writable.
#SBATCH -J shohin-cra-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/cra_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to an isolated train/cra_* checkpoint}
DATA=${DATA:-$BASE/artifacts/evals/counterfactual_residual_algebra_v1_heldout.jsonl}
DATA_SHA=${DATA_SHA:-f66536b85cfe2fafa71690a0e3a5a2adc8fb5db0dd5b912ad69f8362c927b248}
OUT=${OUT:?set OUT to a fresh artifacts/eval_history/cra_*.json path}
ROWS=${ROWS:-500}
MAX_NEW=${MAX_NEW:-12}
SPLIT=${SPLIT:-heldout}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

[ -f "$CKPT" ] || { echo "[cra-eval] checkpoint missing" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[cra-eval] data missing" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[cra-eval] refusing existing output" >&2; exit 4; }
case "$CKPT" in "$BASE"/train/cra_*/cra_ep1.pt) ;; *) echo "[cra-eval] checkpoint must be isolated CRA output" >&2; exit 5;; esac
case "$OUT" in "$BASE"/artifacts/eval_history/cra_*.json) ;; *) echo "[cra-eval] OUT must be a CRA eval-history JSON" >&2; exit 6;; esac
[ "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA" ] || { echo "[cra-eval] held-out hash mismatch" >&2; exit 7; }

cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device='cuda', dtype=torch.bfloat16)
PY
then
  echo "[cra-eval] CUDA unavailable or timed out after allocation" >&2
  exit 75
fi
echo "[cra-eval] ckpt=$CKPT data=$DATA split=$SPLIT rows=$ROWS out=$OUT"
"$PY" eval_counterfactual_residual_algebra.py \
  --ckpt "$CKPT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --split "$SPLIT" --max-examples "$ROWS" --max-new "$MAX_NEW"
test -s "$OUT"
echo "[cra-eval] wrote $OUT"
