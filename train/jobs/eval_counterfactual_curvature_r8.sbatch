#!/bin/bash
#SBATCH -J shohin-ccb-r8
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ccb_r8_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
BASE_CKPT=${BASE_CKPT:?set immutable base checkpoint}
DATA=${DATA:?set admitted development data}
ADMISSION=${ADMISSION:?set admission report}
R7_REPORT=${R7_REPORT:?set frozen R7 report}
OUT=${OUT:?set fresh R8 canary report}

for path in "$BASE_CKPT" "$DATA" "$ADMISSION" "$R7_REPORT"; do
  [ -s "$path" ] || { echo "[ccb-r8] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[ccb-r8] refusing existing output: $OUT" >&2; exit 3; }
case "$OUT" in "$BASE"/artifacts/eval_history/counterfactual_curvature_r8_*.json) ;; *) echo "[ccb-r8] invalid output" >&2; exit 4;; esac

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4

cd "$BASE/train"
timeout --kill-after=10s 300s "$PY" -u - <<'PY'
import sys
print("[ccb-r8] python-start", flush=True)
import torch
print(f"[ccb-r8] torch-imported version={torch.__version__}", flush=True)
try:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.empty(1024, device="cuda", dtype=torch.bfloat16); torch.cuda.synchronize()
    print("[ccb-r8] CUDA preflight passed", flush=True)
except Exception as exc:
    print("[ccb-r8] CUDA preflight failed: {}".format(exc), file=sys.stderr)
    sys.exit(75)
PY

"$PY" eval_counterfactual_curvature_r8.py \
  --base "$BASE_CKPT" --data "$DATA" --admission "$ADMISSION" --r7-report "$R7_REPORT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT" \
  --layers 5,11,17,23,29 --pairs 2 --limit-per-opcode 12 \
  --regimes language_ood,full_ood --seed 20260714
