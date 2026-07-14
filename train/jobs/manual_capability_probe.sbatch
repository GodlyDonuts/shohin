#!/bin/bash
#SBATCH -J shohin-manual-probe
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 0:35:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/manual_capability_probe_%j.out
# Run only as an isolated diagnostic.  It never writes a training artifact.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
RAW_CKPT=${RAW_CKPT:?set RAW_CKPT to a preserved raw checkpoint}
SFT_CKPT=${SFT_CKPT:?set SFT_CKPT to an isolated SFT checkpoint}
OUT=${OUT:?set OUT to an isolated JSON output}

for path in "$RAW_CKPT" "$SFT_CKPT"; do
  [ -f "$path" ] || { echo "[manual-probe] checkpoint missing: $path" >&2; exit 2; }
done
cd "$BASE/train"
for preflight in 1 2 3; do
  if timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[manual-probe] CUDA preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
PY
  then
    break
  fi
  if [ "$preflight" = 3 ]; then
    echo "[manual-probe] CUDA unavailable or timed out after allocation" >&2
    exit 75
  fi
  echo "[manual-probe] CUDA preflight $preflight failed; waiting before retry" >&2
  sleep 30
done
echo "[manual-probe] raw=$RAW_CKPT sft=$SFT_CKPT out=$OUT"
for attempt in 1 2 3; do
  if "$PY" manual_capability_probe.py \
    --ckpt "$RAW_CKPT" "$SFT_CKPT" \
    --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
    --out "$OUT"; then
    exit 0
  fi
  status=$?
  # The probe writes its artifact only after both models finish, so retrying an
  # empty attempt cannot replace a valid transcript. Stop if that invariant
  # ever changes instead of deleting an existing result.
  if [ -e "$OUT" ]; then
    echo "[manual-probe] failed after creating $OUT; refusing to retry" >&2
    exit "$status"
  fi
  echo "[manual-probe] attempt $attempt failed before output; retrying CUDA allocation" >&2
  sleep 30
done
exit "$status"
