#!/bin/bash
#SBATCH -J shohin-semcapsule-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/semantic_capsule_eval_%j.out
#SBATCH --open-mode=append
# Read-only closed-loop context-scaling evaluation. It never writes weights.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to an immutable checkpoint}
EPISODES=${EPISODES:?set EPISODES to held-out semantic-capsule episodes}
TAG=${TAG:-semantic_capsule_eval}
OUT=${OUT:-$BASE/artifacts/eval_history/${TAG}_${SLURM_JOB_ID}.json}

[ -f "$CKPT" ] || { echo "[semantic-capsule] checkpoint missing: $CKPT" >&2; exit 2; }
[ -s "$EPISODES" ] || { echo "[semantic-capsule] episodes missing: $EPISODES" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[semantic-capsule] refusing existing output: $OUT" >&2; exit 3; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device="cuda", dtype=torch.bfloat16)
PY
then
  echo "[semantic-capsule] CUDA unavailable or timed out after allocation" >&2
  exit 75
fi
"$PY" eval_semantic_capsule.py --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --episodes "$EPISODES" --out "$OUT" --per-regime "${PER_REGIME:-100}"
