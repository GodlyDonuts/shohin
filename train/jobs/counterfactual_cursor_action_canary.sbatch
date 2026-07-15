#!/bin/bash
#SBATCH -J shohin-r12-ccaa
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/r12_ccaa_train_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:-$BASE/train/flagship_out/best_step260000.pt}
CANARY=${CANARY:-$BASE/artifacts/r12/counterfactual_cursor_action_canary_v1.json}
AUDIT=${AUDIT:-$BASE/artifacts/r12/counterfactual_cursor_action_canary_v1.audit.json}
TOKENIZER=${TOKENIZER:-$BASE/artifacts/shohin-tok-32k.json}
OUT=${OUT:-$BASE/train/r12_cursor_action_canary_v1}

for path in "$CKPT" "$CANARY" "$AUDIT" "$TOKENIZER"; do
  [ -s "$path" ] || { echo "[ccaa-job] missing input: $path" >&2; exit 2; }
  [ ! -L "$path" ] || { echo "[ccaa-job] symlink input rejected: $path" >&2; exit 3; }
  [ ! -w "$path" ] || { echo "[ccaa-job] writable input rejected: $path" >&2; exit 6; }
done
[ ! -e "$OUT" ] || { echo "[ccaa-job] refusing existing output: $OUT" >&2; exit 4; }
case "$OUT" in "$BASE"/train/r12_cursor_action_canary_v1*) ;;
  *) echo "[ccaa-job] invalid isolated output path: $OUT" >&2; exit 5;;
esac

cd "$BASE/train"
CUDA_PREFLIGHT_TIMEOUT=${CUDA_PREFLIGHT_TIMEOUT:-300}
echo "[ccaa-job] CUDA preflight timeout=${CUDA_PREFLIGHT_TIMEOUT}s"
timeout --kill-after=10s "${CUDA_PREFLIGHT_TIMEOUT}s" "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    value = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    value.add_(1)
    torch.cuda.synchronize()
except Exception as exc:
    print(f"[ccaa-job] CUDA preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
print("[ccaa-job] CUDA preflight passed")
PY

"$PY" train_counterfactual_cursor_action.py \
  --base "$CKPT" \
  --canary "$CANARY" \
  --audit "$AUDIT" \
  --tokenizer "$TOKENIZER" \
  --out "$OUT" \
  --cache-batch-size 64
