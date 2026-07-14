#!/bin/bash
#SBATCH -J shohin-effect-r6-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/future_effect_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
BASE_CKPT=${BASE_CKPT:?set immutable base checkpoint}
ADAPTER=${ADAPTER:?set immutable R6 adapter}
DATA=${DATA:?set admitted evaluation JSONL}
ADMISSION=${ADMISSION:?set structural admission}
LABEL_ADMISSION=${LABEL_ADMISSION:-$BASE/artifacts/sft/referential_slot_labels_v4.admission.json}
EVAL_LABEL_ADMISSION=${EVAL_LABEL_ADMISSION:-$LABEL_ADMISSION}
OUT=${OUT:?set fresh R6 evaluation report}

for path in "$BASE_CKPT" "$ADAPTER" "$DATA" "$ADMISSION" "$LABEL_ADMISSION" "$EVAL_LABEL_ADMISSION"; do
  [ -s "$path" ] || { echo "[effect-r6-eval] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[effect-r6-eval] refusing existing output: $OUT" >&2; exit 3; }
case "$OUT" in "$BASE"/artifacts/eval_history/future_effect_r6_*.json) ;; *) echo "[effect-r6-eval] invalid OUT path" >&2; exit 4;; esac

cd "$BASE/train"
timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys, torch
try:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.empty(1024,device="cuda",dtype=torch.bfloat16); torch.cuda.synchronize()
except Exception as exc:
    print("[effect-r6-eval] CUDA preflight failed: {}".format(exc),file=sys.stderr); sys.exit(75)
PY

"$PY" eval_future_effect_compiler.py \
  --base "$BASE_CKPT" --adapter "$ADAPTER" --data "$DATA" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --admission "$ADMISSION" \
  --label-admission "$LABEL_ADMISSION" --evaluation-label-admission "$EVAL_LABEL_ADMISSION" \
  --out "$OUT" --batch-size "${BATCH_SIZE:-16}" --latent-steps "${LATENT_STEPS:-3}" \
  --plausible-count "${PLAUSIBLE_COUNT:-64}" --max-value "${MAX_VALUE:-99}" \
  --seed "${SEED:-20260714}"
