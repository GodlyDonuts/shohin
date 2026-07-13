#!/bin/bash
#SBATCH -J shohin-think-trace
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:25:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/thinking_trace_%j.out
#SBATCH --open-mode=append
# Transcript-only visible-reasoning audit. It never writes a training path.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to an immutable checkpoint}
TAG=${TAG:-thinking_trace}
OUT=$BASE/artifacts/eval_history/${TAG}_${SLURM_JOB_ID}.json
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

[ -f "$CKPT" ] || { echo "[thinking-trace] checkpoint missing: $CKPT" >&2; exit 2; }
cd "$BASE/train"
if ! "$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
then
  echo "[thinking-trace] CUDA unavailable after allocation" >&2
  exit 75
fi

"$PY" thinking_trace_audit.py --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT"
echo "[thinking-trace] wrote $OUT"
