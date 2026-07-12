#!/bin/bash
#SBATCH -J shohin-generalize
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:35:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/generalization_interview_%j.out
#SBATCH --open-mode=append
# Isolated, transcript-only checkpoint interview. Never writes a training path.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to a preserved named checkpoint}
TAG=${TAG:-generalization_interview}
OUT=$BASE/artifacts/eval_history/${TAG}_${SLURM_JOB_ID}.json
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

[ -f "$CKPT" ] || { echo "[generalization] checkpoint missing: $CKPT" >&2; exit 2; }
cd "$BASE/train"
if ! "$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
then
  echo "[generalization] CUDA unavailable after allocation" >&2
  exit 75
fi

"$PY" generalization_interview.py \
  --ckpt "$CKPT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT"
echo "[generalization] wrote $OUT"
