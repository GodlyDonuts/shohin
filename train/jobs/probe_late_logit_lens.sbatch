#!/bin/bash
#SBATCH -J shohin-late-lens
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc36,evc43,evc45,evc49,evc50
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/late_lens_%j.out
#SBATCH --open-mode=append
# Read-only measurement. It neither trains nor writes a model output.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to an immutable or isolated checkpoint}
OUT=${OUT:?set OUT to a fresh JSON report}
[ -s "$CKPT" ] || { echo "[late-lens] checkpoint missing" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[late-lens] refusing existing output" >&2; exit 3; }
cd "$BASE/train"
"$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[late-lens] CUDA preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
PY
"$PY" probe_late_logit_lens.py --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT"
