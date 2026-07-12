#!/bin/bash
#SBATCH -J shohin-state-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/state_protocol_eval_%j.out
#SBATCH --open-mode=append
# Isolated typed-state gate. It never writes a training output directory.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to a raw or isolated SFT checkpoint}
DATA=${DATA:?set DATA to held-out state-protocol JSONL}
OUT=${OUT:?set OUT to a fresh JSON report}
PER_CONTRACT_FAMILY=${PER_CONTRACT_FAMILY:-20}

[ -f "$CKPT" ] || { echo "[state-eval] missing checkpoint: $CKPT" >&2; exit 2; }
[ -f "$DATA" ] || { echo "[state-eval] missing data: $DATA" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[state-eval] refusing existing output: $OUT" >&2; exit 4; }
cd "$BASE/train"
for i in $(seq 1 30); do
  if "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[state-eval] CUDA not ready: {exc}", file=sys.stderr)
    sys.exit(75)
PY
  then
    break
  fi
  sleep 10
done
if ! "$PY" - <<'PY'
import sys
import torch
try:
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[state-eval] CUDA unavailable after retry: {exc}", file=sys.stderr)
    sys.exit(75)
PY
then
  exit 75
fi
"$PY" eval_state_protocol.py --ckpt "$CKPT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --data "$DATA" --out "$OUT" \
  --per-contract-family "$PER_CONTRACT_FAMILY" --max-new "${MAX_NEW:-96}"
