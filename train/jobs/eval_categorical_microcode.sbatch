#!/bin/bash
#SBATCH -J shohin-microcode-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/microcode_eval_%j.out
#SBATCH --open-mode=append
# Read-only Causal Microcode Bottleneck held-out evaluation.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
BASE_CKPT=${BASE_CKPT:?set BASE_CKPT}
ADAPTER=${ADAPTER:?set ADAPTER}
DATA=${DATA:-$BASE/artifacts/evals/latent_operator_eval_slices_v2_64.jsonl}
CMB_ADMISSION=${CMB_ADMISSION:-$BASE/artifacts/sft/categorical_microcode_v1.admission.json}
OUT=${OUT:?set OUT to a fresh eval-history JSON}
for path in "$BASE_CKPT" "$ADAPTER" "$DATA" "$CMB_ADMISSION"; do
  [ -s "$path" ] || { echo "[microcode-eval] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[microcode-eval] refusing existing output: $OUT" >&2; exit 3; }
case "$OUT" in "$BASE"/artifacts/eval_history/microcode_*.json) ;; *) echo "[microcode-eval] invalid OUT path" >&2; exit 4;; esac
cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys, torch
try:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
except Exception as exc:
    print("[microcode-eval] CUDA preflight failed: {}".format(exc), file=sys.stderr)
    sys.exit(75)
PY
then
  exit 75
fi
"$PY" eval_categorical_microcode.py --base "$BASE_CKPT" --adapter "$ADAPTER" \
  --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --admission "$CMB_ADMISSION" \
  --out "$OUT" --batch-size "${BS:-16}"
