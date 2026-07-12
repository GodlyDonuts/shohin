#!/bin/bash
#SBATCH -J shohin-pretrain-nll
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:35:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/eval_nll_%j.out
#SBATCH --open-mode=append
# Isolated pretraining-quality measurement. Never writes training data or checkpoints.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to a preserved named checkpoint}
INPUTS=${INPUTS:?set INPUTS to space-separated label=JSONL monitor inputs}
TAG=${TAG:-pretrain_nll}
MAX_SEQUENCES_PER_INPUT=${MAX_SEQUENCES_PER_INPUT:-256}
OUT=$BASE/artifacts/monitor_history/${TAG}_${SLURM_JOB_ID}.json
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

[ -f "$CKPT" ] || { echo "[pretrain-nll] checkpoint missing: $CKPT" >&2; exit 2; }
cd "$BASE/train"
if ! "$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
then
  echo "[pretrain-nll] CUDA unavailable after allocation" >&2
  exit 75
fi

ARGS=()
for spec in $INPUTS; do
  ARGS+=(--input "$spec")
done
"$PY" eval_nll.py \
  --ckpt "$CKPT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --max-sequences-per-input "$MAX_SEQUENCES_PER_INPUT" \
  --out "$OUT" \
  "${ARGS[@]}"
echo "[pretrain-nll] wrote $OUT"
