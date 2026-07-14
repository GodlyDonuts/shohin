#!/bin/bash
# Read-only teacher-forced CRA diagnostic. It never writes a checkpoint and its
# NLL margins are explicitly not behavioral capability scores.
#SBATCH -J shohin-cra-nll
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/cra_nll_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT}
DATA=${DATA:-$BASE/artifacts/evals/counterfactual_residual_algebra_v1_heldout.jsonl}
DATA_SHA=${DATA_SHA:-f66536b85cfe2fafa71690a0e3a5a2adc8fb5db0dd5b912ad69f8362c927b248}
OUT=${OUT:?set a fresh CRA NLL output}
ROWS=${ROWS:-500}
SPLIT=${SPLIT:-heldout}
ALLOW_RAW=${ALLOW_RAW:-0}
LAYER=${LAYER:-19}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

[ -f "$CKPT" ] || { echo "[cra-nll] checkpoint missing" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[cra-nll] data missing" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[cra-nll] refusing existing output" >&2; exit 4; }
case "$CKPT" in
  "$BASE"/train/cra_*/cra_ep1.pt) ;;
  "$BASE"/train/flagship_out/best_step200000.pt)
    [ "$ALLOW_RAW" = 1 ] || { echo "[cra-nll] raw checkpoint requires ALLOW_RAW=1" >&2; exit 5; }
    ;;
  *) echo "[cra-nll] checkpoint must be isolated CRA output or explicit raw baseline" >&2; exit 5;;
esac
case "$OUT" in "$BASE"/artifacts/eval_history/cra_*_nll.json) ;; *) echo "[cra-nll] output must be a CRA NLL report" >&2; exit 6;; esac
[ "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA" ] || { echo "[cra-nll] held-out hash mismatch" >&2; exit 7; }

cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device='cuda', dtype=torch.bfloat16)
PY
then
  echo "[cra-nll] CUDA unavailable or timed out after allocation" >&2
  exit 75
fi
RAW_ARGS=()
if [ "$ALLOW_RAW" = 1 ]; then RAW_ARGS=(--allow-raw --layer "$LAYER"); fi
echo "[cra-nll] ckpt=$CKPT rows=$ROWS out=$OUT allow_raw=$ALLOW_RAW"
"$PY" eval_counterfactual_residual_algebra_nll.py \
  --ckpt "$CKPT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --split "$SPLIT" --max-examples "$ROWS" "${RAW_ARGS[@]}"
test -s "$OUT"
echo "[cra-nll] wrote $OUT"
