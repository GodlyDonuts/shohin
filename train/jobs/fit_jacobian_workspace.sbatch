#!/bin/bash
#SBATCH -J shohin-jacobian-workspace
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/jacobian_workspace_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:-$BASE/train/flagship_out/best_step200000.pt}
TOKENIZER=${TOKENIZER:-$BASE/artifacts/shohin-tok-32k.json}
PROMPTS=${PROMPTS:-$BASE/artifacts/monitors/wikitext103_test.jsonl}
OUT=${OUT:?set a fresh Jacobian output}

for path in "$INIT" "$TOKENIZER" "$PROMPTS"; do
  [ -s "$path" ] || { echo "[jacobian-workspace] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[jacobian-workspace] refusing existing output" >&2; exit 3; }
case "$OUT" in "$BASE"/artifacts/diagnostics/jacobian_workspace_*) ;;
  *) echo "[jacobian-workspace] invalid output path" >&2; exit 4;;
esac

cd "$BASE/train"
timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys, torch
try:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    x=torch.randn(128,128,device="cuda",dtype=torch.bfloat16)
    (x@x).sum().item()
except Exception as exc:
    print("[jacobian-workspace] CUDA preflight failed: {}".format(exc),file=sys.stderr)
    sys.exit(75)
PY

"$PY" jacobian_workspace.py \
  --base "$INIT" --tokenizer "$TOKENIZER" --prompts "$PROMPTS" --out "$OUT" \
  --source-layers "${SOURCE_LAYERS:-5,9,13,17,21,25,28}" \
  --target-layer "${TARGET_LAYER:--1}" --prompt-count "${PROMPT_COUNT:-8}" \
  --max-seq-len "${MAX_SEQ_LEN:-128}" --skip-first "${SKIP_FIRST:-16}" \
  --dim-batch "${DIM_BATCH:-8}" --seed "${SEED:-20260714}"
