#!/bin/bash
#SBATCH -J shohin-source-memory-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/source_memory_eval_%j.out
#SBATCH --open-mode=append
# Held-out source-removal evaluation. It only reads an isolated checkpoint.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to an isolated source-memory checkpoint}
DATA=${DATA:?set DATA to the held-out source-memory JSONL}
OUT=${OUT:?set OUT to a fresh JSON result path}

for path in "$CKPT" "$DATA"; do [ -s "$path" ] || { echo "[source-memory-eval] missing input: $path" >&2; exit 2; }; done
[ ! -e "$OUT" ] || { echo "[source-memory-eval] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" - <<'PY'
import sys
import torch
try:
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[source-memory-eval] CUDA unavailable: {exc}", file=sys.stderr)
    sys.exit(75)
PY
"$PY" eval_source_dropping_memory.py --ckpt "$CKPT" --data "$DATA" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT" \
  --per-chunk-regime "${PER_CHUNK_REGIME:-16}" --max-new "${MAX_NEW:-24}" --seed "${SEED:-20260714}"
