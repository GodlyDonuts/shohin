#!/bin/bash
#SBATCH -J shohin-isq-r7
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/isq_r7_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
BASE_CKPT=${BASE_CKPT:?set immutable base checkpoint}
DATA=${DATA:?set admitted development data}
ADMISSION=${ADMISSION:?set admission report}
OUT=${OUT:?set fresh R7 canary report}

for path in "$BASE_CKPT" "$DATA" "$ADMISSION"; do
  [ -s "$path" ] || { echo "[isq-r7] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[isq-r7] refusing existing output: $OUT" >&2; exit 3; }
case "$OUT" in "$BASE"/artifacts/eval_history/counterfactual_influence_r7_*.json) ;; *) echo "[isq-r7] invalid output" >&2; exit 4;; esac

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4

cd "$BASE/train"
timeout --kill-after=10s 300s "$PY" -u - <<'PY'
import sys
print("[isq-r7] python-start", flush=True)
import torch
print(f"[isq-r7] torch-imported version={torch.__version__}", flush=True)
try:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.empty(1024, device="cuda", dtype=torch.bfloat16); torch.cuda.synchronize()
    print("[isq-r7] CUDA preflight passed", flush=True)
except Exception as exc:
    print("[isq-r7] CUDA preflight failed: {}".format(exc), file=sys.stderr)
    sys.exit(75)
PY

"$PY" eval_counterfactual_influence_quotient.py \
  --base "$BASE_CKPT" --data "$DATA" --admission "$ADMISSION" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT" \
  --layers 5,11,17,23,29 --channels 2 --limit-per-opcode 12 \
  --regimes language_ood,full_ood --seed 20260714
