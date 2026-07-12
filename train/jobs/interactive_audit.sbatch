#!/bin/bash
#SBATCH -J shohin-interact
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/interactive_audit_%j.out
#SBATCH --open-mode=append
# Read-only qualitative comparison. It never writes into a training output directory.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
RAW=${RAW:-$BASE/train/flagship_out/best_step166250.pt}
SFT=${SFT:-$BASE/train/sft_v2_120k/sft_ep1.pt}
TAG=${TAG:-interactive_v1}
OUT=$BASE/artifacts/eval_history/${TAG}_${SLURM_JOB_ID}.json

for path in "$RAW" "$SFT"; do
  if [ ! -f "$path" ]; then
    echo "[interactive-audit] checkpoint missing: $path" >&2
    exit 2
  fi
done

cd $BASE/train
if ! "$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
then
  echo "[interactive-audit] CUDA unavailable after allocation" >&2
  exit 75
fi

"$PY" interactive_audit.py \
  --ckpt "$RAW" "$SFT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT"
echo "[interactive-audit] wrote $OUT"
