#!/bin/bash
#SBATCH -J shohin-scratch-nll
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc36,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/scratch_nll_%j.out
#SBATCH --open-mode=append
# Read-only intervention screen for one frozen-base scratch adapter.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
BASE_CKPT=${BASE_CKPT:?set BASE_CKPT to the adapter-bound immutable checkpoint}
ADAPTER=${ADAPTER:?set ADAPTER to a scratch_adapter_ep1.pt}
DATA=${DATA:-$BASE/artifacts/evals/latent_operator_eval_slices_v2_64.jsonl}
OUT=${OUT:?set OUT to a fresh eval-history JSON}
for path in "$BASE_CKPT" "$ADAPTER" "$DATA"; do
  [ -s "$path" ] || { echo "[scratch-nll] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[scratch-nll] refusing existing output: $OUT" >&2; exit 3; }
case "$OUT" in "$BASE"/artifacts/eval_history/scratch_*.json) ;; *) echo "[scratch-nll] invalid OUT path" >&2; exit 4;; esac
cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
except Exception as exc:
    print(f"[scratch-nll] CUDA preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
PY
then
  exit 75
fi
"$PY" eval_causal_recurrent_scratch_nll.py \
  --base "$BASE_CKPT" --adapter "$ADAPTER" --data "$DATA" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT" \
  --per-depth "${PER_DEPTH:-16}" --batch-size "${BS:-8}" --seed "${SEED:-20260714}"
