#!/bin/bash
#SBATCH -J shohin-latent-operator
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/latent_operator_%j.out
#SBATCH --open-mode=append
# CPU-only creation of an answer-only continuous-latent research corpus.
# It cannot modify pretraining, existing SFT data, or evaluation artifacts.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
TRAIN=${TRAIN:-$BASE/artifacts/sft/latent_operator_v1_train.jsonl}
HELDOUT=${HELDOUT:-$BASE/artifacts/evals/latent_operator_v1_heldout.jsonl}
AUDIT=${AUDIT:-$BASE/artifacts/sft/latent_operator_v1.audit.r1.json}
TRAIN_ROWS=${TRAIN_ROWS:-96000}
EVAL_PER_DEPTH=${EVAL_PER_DEPTH:-600}

for path in "$TRAIN" "$HELDOUT" "$AUDIT"; do
  [ ! -e "$path" ] || { echo "[latent-operator] refusing existing artifact: $path" >&2; exit 3; }
done
cd "$BASE/pipeline"
"$PY" generate_latent_operator_v1.py --train-out "$TRAIN" --eval-out "$HELDOUT" \
  --train-rows "$TRAIN_ROWS" --eval-per-depth "$EVAL_PER_DEPTH"
"$PY" audit_latent_operator_v1.py --train "$TRAIN" --heldout "$HELDOUT" --out "$AUDIT"
