#!/bin/bash
#SBATCH -J shohin-recur-infer
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:50:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/recurrent_inference_%j.out
#SBATCH --open-mode=append
# Matched test-time latent-depth probe. It loads immutable weights twice with
# n_loop=1 and n_loop=2; it never writes a training checkpoint or promotion.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:-$BASE/train/flagship_out/best_step180000.pt}
TAG=${TAG:-raw180_recur_infer}
HIST=$BASE/artifacts/eval_history
MATRIX_N=${MATRIX_N:-4}
MAX_NEW=${MAX_NEW:-64}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

[ -f "$CKPT" ] || { echo "[recur-infer] checkpoint missing: $CKPT" >&2; exit 2; }
cd "$BASE/train"
if ! "$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
then
  echo "[recur-infer] CUDA unavailable after allocation" >&2
  exit 75
fi

mkdir -p "$HIST"
for LOOP in 1 2; do
  MATRIX=$HIST/${TAG}_loop${LOOP}_matrix_${SLURM_JOB_ID}.json
  INTERVIEW=$HIST/${TAG}_loop${LOOP}_interview_${SLURM_JOB_ID}.json
  "$PY" capability_matrix.py --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
    --out "$MATRIX" --n-per-family "$MATRIX_N" --modes qa cot --max-new "$MAX_NEW" --n-loop "$LOOP"
  "$PY" generalization_interview.py --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
    --out "$INTERVIEW" --max-new "$MAX_NEW" --n-loop "$LOOP"
  echo "[recur-infer] loop=$LOOP matrix=$MATRIX interview=$INTERVIEW"
done
