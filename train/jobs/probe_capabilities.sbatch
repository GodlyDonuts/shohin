#!/bin/bash
#SBATCH -J shohin-probe
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/probe_%j.out
#SBATCH --open-mode=append
# Qualitative probe. It reads a checkpoint and writes a transcript only.
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT}
TAG=${TAG:-$(basename "$CKPT" .pt)}
OUT=$BASE/artifacts/eval_history/${TAG}_qualitative_${SLURM_JOB_ID}.json

cd "$BASE/train"
for i in $(seq 1 30); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1 || true)
  [ "${U:-9999}" -lt 3000 ] 2>/dev/null && break
  sleep 10
done
"$PY" probe_capabilities.py --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT"
echo "[probe] wrote $OUT"
