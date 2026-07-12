#!/bin/bash
#SBATCH -J shohin-verifier-rollouts
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/verifier_rollouts_%j.out
#SBATCH --open-mode=append
# Generate train-only or held-out student candidates. The caller must set every
# path explicitly; this job never writes to a flagship or SFT checkpoint folder.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT}
DATA=${DATA:?set DATA}
OUT=${OUT:?set OUT}
N=${N:-2000}
K=${K:-8}
TAG=${TAG:-verifier-rollouts}
ANSWER_MODE=${ANSWER_MODE:-gsm8k}

[ -f "$CKPT" ] || { echo "[verifier-rollouts] missing checkpoint: $CKPT" >&2; exit 2; }
[ -f "$DATA" ] || { echo "[verifier-rollouts] missing data: $DATA" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[verifier-rollouts] refusing existing output: $OUT" >&2; exit 4; }
cd "$BASE/train"
for attempt in 1 2 3; do
  rm -f "${OUT}.partial"
  if ! "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[verifier-rollouts] CUDA not ready: {exc}", file=sys.stderr)
    sys.exit(75)
PY
  then
    echo "[verifier-rollouts] CUDA readiness attempt $attempt failed; retrying" >&2
    sleep 30
    continue
  fi
  if "$PY" generate_verifier_rollouts.py --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
      --data "$DATA" --out "$OUT" --n "$N" --k "$K" --answer-mode "$ANSWER_MODE"; then
    break
  fi
  echo "[verifier-rollouts] generation attempt $attempt failed; retrying" >&2
  rm -f "${OUT}.partial"
  sleep 30
done
[ -e "$OUT" ] || { echo "[verifier-rollouts] exhausted retries without output" >&2; exit 75; }
echo "[verifier-rollouts] tag=$TAG wrote $OUT"
