#!/bin/bash
#SBATCH -J shohin-capability-matrix
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/capability_matrix_%j.out
#SBATCH --open-mode=append
# Isolated direct-capability audit. It never writes to any training directory.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
RAW=${RAW:-$BASE/train/flagship_out/ckpt_0168000.pt}
SFT=${SFT:-$BASE/train/sft_v2_120k/sft_ep1.pt}
OUT=${OUT:-$BASE/artifacts/eval_history/capability_matrix_v1_${SLURM_JOB_ID}.json}

[ -f "$RAW" ] || { echo "[capability-matrix] raw checkpoint missing: $RAW" >&2; exit 2; }
[ -f "$SFT" ] || { echo "[capability-matrix] SFT checkpoint missing: $SFT" >&2; exit 3; }

cd "$BASE/train"
if ! "$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
then
  echo "[capability-matrix] CUDA unavailable after allocation" >&2
  exit 75
fi

echo "[capability-matrix] raw=$RAW sft=$SFT out=$OUT"
"$PY" capability_matrix.py --ckpt "$RAW" "$SFT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT"
