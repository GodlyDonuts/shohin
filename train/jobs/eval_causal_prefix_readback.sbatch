#!/bin/bash
#SBATCH -J shohin-causal-prefix-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/causal_prefix_readback_eval_%j.out
#SBATCH --open-mode=append
# Read-only held-out semantic readback evaluator for an isolated checkpoint.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to an isolated causal-prefix-readback checkpoint}
DATA=${DATA:?set DATA to held-out LSA r2 JSONL}
OUT=${OUT:?set OUT to a fresh JSON output path}
for path in "$CKPT" "$DATA"; do [ -s "$path" ] || { echo "[causal-prefix-eval] missing input: $path" >&2; exit 2; }; done
[ ! -e "$OUT" ] || { echo "[causal-prefix-eval] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" - <<'PY'
import sys
import torch
try:
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[causal-prefix-eval] CUDA unavailable: {exc}", file=sys.stderr)
    sys.exit(75)
PY
args=(
  --ckpt "$CKPT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT"
  --per-chunk-regime "${PER_CHUNK_REGIME:-16}" --max-new "${MAX_NEW:-12}" --seed "${SEED:-20260717}"
)
[ "${ALL_HELDOUT:-0}" = 1 ] && args+=(--all-heldout)
"$PY" eval_causal_prefix_readback.py "${args[@]}"
