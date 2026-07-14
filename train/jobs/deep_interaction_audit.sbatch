#!/bin/bash
#SBATCH -J shohin-deep-interact
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc36,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/deep_interaction_%j.out
#SBATCH --open-mode=append
# Read-only live-checkpoint interview. It never writes to a training output path.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to a preserved named checkpoint}
TAG=${TAG:-deep_interaction_v1}
OUT=$BASE/artifacts/eval_history/${TAG}_${SLURM_JOB_ID}.json

[ -f "$CKPT" ] || { echo "[deep-interaction] checkpoint missing: $CKPT" >&2; exit 2; }
cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    probe = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    del probe
except Exception as exc:
    print(f"[deep-interaction] CUDA preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
PY
then
  echo "[deep-interaction] CUDA unavailable after allocation" >&2
  exit 75
fi

"$PY" deep_interaction_audit.py --ckpt "$CKPT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT"
echo "[deep-interaction] wrote $OUT"
