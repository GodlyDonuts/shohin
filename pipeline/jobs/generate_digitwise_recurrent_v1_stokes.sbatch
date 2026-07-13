#!/bin/bash
#SBATCH -J shohin-digitwise-v1
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/digitwise_recurrent_v1_%j.out
#SBATCH --open-mode=append
# CPU-only, Stokes-targeted construction of an isolated digitwise recurrence
# curriculum. It cannot modify the active pretraining stream or any checkpoint.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=${PY:-/usr/bin/python3.12}
TRAIN=${TRAIN:-$BASE/artifacts/sft/digitwise_recurrent_v1_train.jsonl}
HELDOUT=${HELDOUT:-$BASE/artifacts/evals/digitwise_recurrent_v1_heldout.jsonl}
REPORT=${REPORT:-$BASE/artifacts/sft/digitwise_recurrent_v1_train.report.json}
TRAIN_EPISODES=${TRAIN_EPISODES:-40000}
HELDOUT_PER_REGIME=${HELDOUT_PER_REGIME:-300}

for path in "$TRAIN" "$HELDOUT" "$REPORT"; do
  [ ! -e "$path" ] || { echo "[digitwise] refusing existing artifact: $path" >&2; exit 3; }
done
cd "$BASE"
"$PY" pipeline/generate_digitwise_recurrent_v1.py \
  --train-out "$TRAIN" \
  --heldout-out "$HELDOUT" \
  --report "$REPORT" \
  --train-episodes "$TRAIN_EPISODES" \
  --heldout-per-regime "$HELDOUT_PER_REGIME"
test -s "$TRAIN"
test -s "$HELDOUT"
test -s "$REPORT"
echo "[digitwise] committed immutable CPU candidate"
