#!/bin/bash
# Read-only raw-checkpoint confirmation of the frozen source-scheduled renderer.
# This job cannot write the flagship output or training data.
#SBATCH -J shohin-scheduled-confirm
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc34,evc36,evc43,evc44,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/scheduled_reasoning_confirm_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to an immutable raw checkpoint}
BOARD=${BOARD:?set BOARD to the frozen confirmation JSON}
OUT=${OUT:?set OUT to a fresh result path}

[ -f "$CKPT" ] || { echo "[scheduled-confirm] missing checkpoint" >&2; exit 2; }
[ -s "$BOARD" ] || { echo "[scheduled-confirm] missing board" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[scheduled-confirm] refusing existing output" >&2; exit 4; }
[ ! -w "$BOARD" ] || { echo "[scheduled-confirm] board must be read-only" >&2; exit 5; }

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('[scheduled-confirm] CUDA unavailable')
torch.empty(1, device='cuda')
print('[scheduled-confirm] CUDA preflight passed', flush=True)
PY

cd "$BASE/train"
"$PY" eval_scheduled_reasoning_confirmation.py \
  --ckpt "$CKPT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --board "$BOARD" \
  --out "$OUT"
[ ! -w "$OUT" ] || { echo "[scheduled-confirm] output remained writable" >&2; exit 6; }
echo "[scheduled-confirm] done"
