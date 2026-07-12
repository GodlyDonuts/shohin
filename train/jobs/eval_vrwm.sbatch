#!/bin/bash
#SBATCH -J shohin-vrwm-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/vrwm_eval_%j.out
#SBATCH --open-mode=append
#
# Isolated VRWM baseline/candidate evaluation. It reads a pinned checkpoint and
# writes only a new JSON report. It never alters pretraining or selects/corrects
# model states during a rollout.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to a preserved raw or isolated SFT checkpoint}
EPISODES=${EPISODES:?set EPISODES to a frozen VRWM episode JSONL}
OUT=${OUT:?set OUT to a fresh VRWM report path}
PER_SPLIT=${PER_SPLIT:-10}

[ -f "$CKPT" ] || { echo "[vrwm-eval] checkpoint missing: $CKPT" >&2; exit 2; }
[ -f "$EPISODES" ] || { echo "[vrwm-eval] episodes missing: $EPISODES" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[vrwm-eval] refusing existing output: $OUT" >&2; exit 4; }
cd "$BASE/train"
for attempt in 1 2 3; do
  if "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[vrwm-eval] CUDA preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
PY
  then
    break
  fi
  [ "$attempt" = 3 ] && exit 75
  sleep 30
done
"$PY" eval_vrwm.py --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --episodes "$EPISODES" --out "$OUT" --per-split "$PER_SPLIT" --max-new "${MAX_NEW:-32}"
