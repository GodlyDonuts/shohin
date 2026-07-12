#!/bin/bash
#SBATCH -J shohin-rg-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/eval_rg_%j.out
#SBATCH --open-mode=append
# Evaluate a checkpoint on Reasoning-Gym's generator-held-out development set.
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to a model checkpoint}
N=${N:-400}
TAG=${TAG:-$(basename "$CKPT" .pt)}
DATA=${DATA:-$BASE/artifacts/rg_v2/rg_eval.jsonl}
OUT=$BASE/artifacts/eval_history/${TAG}_rg_heldout_${SLURM_JOB_ID}.json

cd "$BASE/train"
for i in $(seq 1 30); do
  # Some freshly allocated shared nodes transiently return NVML exit 6. Treat
  # that as busy/unready rather than aborting before the evaluator starts.
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1 || true)
  [ "${U:-9999}" -lt 3000 ] 2>/dev/null && break
  echo "[rg-eval] GPU not ready (max ${U:-unknown} MiB), retry $i/30"
  sleep 10
done
"$PY" eval_rg.py \
  --ckpt "$CKPT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --data "$DATA" \
  --n "$N" \
  --out-json "$OUT"
echo "[rg-eval] wrote $OUT"
