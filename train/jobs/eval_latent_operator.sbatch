#!/bin/bash
#SBATCH -J shohin-latent-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/latent_eval_%j.out
#SBATCH --open-mode=append
# Read-only held-out evaluation for an immutable latent or control checkpoint.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:?set CKPT to an immutable latent/control checkpoint}
DATA=${DATA:-$BASE/artifacts/evals/latent_operator_v1_heldout.jsonl}
OUT=${OUT:?set OUT to a fresh report path}
LATENT_STEPS=${LATENT_STEPS:-"0 1 2 4 8"}
PER_DEPTH=${PER_DEPTH:-200}

for path in "$CKPT" "$DATA"; do
  [ -s "$path" ] || { echo "[latent-eval] required input missing: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[latent-eval] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" - <<'PY'
import sys
import torch
try:
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[latent-eval] CUDA unavailable: {exc}", file=sys.stderr)
    sys.exit(75)
PY
"$PY" eval_latent_operator.py --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --data "$DATA" --out "$OUT" --per-depth "$PER_DEPTH" --latent-steps $LATENT_STEPS
