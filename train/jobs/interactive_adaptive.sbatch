#!/bin/bash
#SBATCH -J shohin-interact-adaptive
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/interactive_adaptive_%j.out
#SBATCH --open-mode=append
# Read-only multi-turn capability diagnostic. It must use a preserved checkpoint,
# never a rotating flagship checkpoint path.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to a preserved named checkpoint}
TAG=${TAG:-interactive_adaptive_v1}
OUT=$BASE/artifacts/eval_history/${TAG}_${SLURM_JOB_ID}.json

[ -f "$CKPT" ] || { echo "[adaptive] checkpoint missing: $CKPT" >&2; exit 2; }
cd "$BASE/train"
if ! "$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
then
  echo "[adaptive] CUDA unavailable after allocation" >&2
  exit 75
fi
"$PY" interactive_adaptive.py --ckpt "$CKPT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT"
echo "[adaptive] wrote $OUT"
