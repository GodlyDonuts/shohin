#!/bin/bash
#SBATCH -J shohin-code-format
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/code_prompt_audit_%j.out
#SBATCH --open-mode=append
# Compare the official raw-completion code prompt against the SFT instruction
# contract. This is diagnostic only and writes no model or training artifact.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:-$BASE/train/sft_v2_120k/sft_ep1.pt}
N=${N:-40}

[ -f "$CKPT" ] || { echo "[code-format] checkpoint missing: $CKPT" >&2; exit 2; }
cd "$BASE/train"
if ! "$PY" - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 75)
PY
then
  echo "[code-format] CUDA unavailable after allocation" >&2
  exit 75
fi

for task in humaneval mbpp; do
  for style in completion instruction; do
    "$PY" eval_code.py --ckpt "$CKPT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
      --task "$task" --data "$BASE/artifacts/evals/${task}_full.jsonl" --n "$N" --k 1 \
      --prompt-style "$style"
  done
done
