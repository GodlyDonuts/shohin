#!/bin/bash
# Isolated CBC gate. It reads a checkpoint and a held-out protocol and writes
# one unique result; it never trains or shares a flagship output path.
#SBATCH -J shohin-cbc-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/counterfactual_bisimulation_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to an immutable checkpoint}
EPISODES=${EPISODES:-$BASE/artifacts/evals/counterfactual_bisimulation_v1_heldout.jsonl}
TAG=${TAG:-cbc_eval}
OUT=$BASE/artifacts/eval_history/${TAG}_${SLURM_JOB_ID}.json
PER_REGIME=${PER_REGIME:-10}
SEED=${SEED:-20260713}
MAX_NEW=${MAX_NEW:-48}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

[ -f "$CKPT" ] || { echo "[cbc-eval] checkpoint missing: $CKPT" >&2; exit 2; }
[ -s "$EPISODES" ] || { echo "[cbc-eval] held-out episodes missing: $EPISODES" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[cbc-eval] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device="cuda", dtype=torch.bfloat16)
PY
then
  echo "[cbc-eval] CUDA unavailable or timed out after allocation" >&2
  exit 75
fi
"$PY" eval_counterfactual_bisimulation.py \
  --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --episodes "$EPISODES" --out "$OUT" --per-regime "$PER_REGIME" \
  --seed "$SEED" --max-new "$MAX_NEW"
echo "[cbc-eval] wrote $OUT"
