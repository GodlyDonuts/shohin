#!/bin/bash
#SBATCH -J shohin-semcapsule-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
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
cd "$BASE/train"
"$PY" - <<'PY'
import sys
import torch
try:
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[semantic-capsule] CUDA unavailable: {exc}", file=sys.stderr)
    sys.exit(75)
PY
"$PY" eval_semantic_capsule.py --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --episodes "$EPISODES" --out "$OUT" --per-regime "${PER_REGIME:-100}"
