#!/bin/bash
#SBATCH -J shohin-sembridge-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/semantic_bridge_eval_%j.out
#SBATCH --open-mode=append
# Isolated held-out bridge evaluator. It never writes a training checkpoint.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to an immutable checkpoint}
TAG=${TAG:-semantic_bridge_eval}
DATA=${DATA:-$BASE/artifacts/evals/semantic_bridge_v1_heldout.jsonl}
OUT=$BASE/artifacts/eval_history/${TAG}_${SLURM_JOB_ID}.json
PER_FAMILY=${PER_FAMILY:-100}
SEED=${SEED:-20260712}
MAX_NEW=${MAX_NEW:-128}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

[ -f "$CKPT" ] || { echo "[semantic-bridge] checkpoint missing: $CKPT" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[semantic-bridge] held-out data missing: $DATA" >&2; exit 2; }
cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device="cuda", dtype=torch.bfloat16)
PY
then
  echo "[semantic-bridge] CUDA unavailable or timed out after allocation" >&2
  exit 75
fi

"$PY" eval_semantic_bridge.py --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --data "$DATA" --out "$OUT" --per-family "$PER_FAMILY" --seed "$SEED" --max-new "$MAX_NEW"
echo "[semantic-bridge] wrote $OUT"
