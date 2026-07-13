#!/bin/bash
#SBATCH -J shohin-sembridge
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/semantic_bridge_%j.out
#SBATCH --open-mode=append
# CPU-only generation of an isolated, solver-verified bridge curriculum.
# It cannot alter frozen mixes, checkpoints, or active pretraining inputs.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
TRAIN=${TRAIN:-$BASE/artifacts/sft/semantic_bridge_v1_train.jsonl}
EVAL=${EVAL:-$BASE/artifacts/evals/semantic_bridge_v1_heldout.jsonl}
TRAIN_PER_FAMILY=${TRAIN_PER_FAMILY:-40000}
EVAL_PER_FAMILY=${EVAL_PER_FAMILY:-1000}

[ ! -e "$TRAIN" ] || { echo "[semantic-bridge] refusing existing train output: $TRAIN" >&2; exit 3; }
[ ! -e "$EVAL" ] || { echo "[semantic-bridge] refusing existing held-out output: $EVAL" >&2; exit 3; }
cd "$BASE"
"$PY" pipeline/generate_semantic_bridge_v1.py \
  --train-out "$TRAIN" --eval-out "$EVAL" \
  --train-per-family "$TRAIN_PER_FAMILY" --eval-per-family "$EVAL_PER_FAMILY"
