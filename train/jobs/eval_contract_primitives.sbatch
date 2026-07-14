#!/bin/bash
#SBATCH -J shohin-contract-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc36,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/contract_eval_%j.out
#SBATCH --open-mode=append
# Exact-prompt heldout gate for the contract-diverse primitive curriculum.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to a pinned raw or isolated SFT checkpoint}
DATA=${DATA:?set DATA to contract-heldout JSONL}
OUT=${OUT:?set OUT to a fresh JSON report}

[ -f "$CKPT" ] || { echo "[contract-eval] missing checkpoint: $CKPT" >&2; exit 2; }
[ -f "$DATA" ] || { echo "[contract-eval] missing data: $DATA" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[contract-eval] refusing existing output: $OUT" >&2; exit 4; }
cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    probe = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    del probe
except Exception as exc:
    print(f"[contract-eval] CUDA preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
PY
then
  echo "[contract-eval] CUDA unavailable after allocation" >&2
  exit 75
fi
"$PY" eval_contract_primitives.py --ckpt "$CKPT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --data "$DATA" --out "$OUT" \
  --n "${N:-0}" --per-contract-family "${PER_CONTRACT_FAMILY:-0}" --max-new "${MAX_NEW:-96}"
