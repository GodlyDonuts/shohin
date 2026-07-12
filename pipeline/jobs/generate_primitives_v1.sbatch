#!/bin/bash
#SBATCH -J shohin-primitives
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/primitives_%j.out
#SBATCH --open-mode=append
# Generate an isolated, solver-verified primitive-reasoning SFT source and a
# disjoint held-out split. It never touches frozen v4, public evals, or pretrain.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
TRAIN_OUT=${TRAIN_OUT:-$BASE/artifacts/sft/primitives_v1_train.jsonl}
EVAL_OUT=${EVAL_OUT:-$BASE/artifacts/evals/primitives_v1_heldout.jsonl}

cd "$BASE"
"$BASE/miniforge3/bin/python" pipeline/generate_primitives_v1.py \
  --train-out "$TRAIN_OUT" --eval-out "$EVAL_OUT"
"$BASE/miniforge3/bin/python" pipeline/audit_sft_quality.py \
  --data "$TRAIN_OUT" --out "${TRAIN_OUT%.jsonl}.quality.json"
