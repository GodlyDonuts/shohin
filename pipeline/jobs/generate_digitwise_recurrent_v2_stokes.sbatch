#!/bin/bash
# Corrected immutable DRS candidate. v1 remains rejected and untouched.
#SBATCH -J shohin-digitwise-v2
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/digitwise_recurrent_v2_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=${PY:-/usr/bin/python3.12}
TRAIN=${TRAIN:-$BASE/artifacts/sft/digitwise_recurrent_v2_train.jsonl}
HELDOUT=${HELDOUT:-$BASE/artifacts/evals/digitwise_recurrent_v2_heldout.jsonl}
REPORT=${REPORT:-$BASE/artifacts/sft/digitwise_recurrent_v2_train.report.json}
TRAIN_EPISODES=${TRAIN_EPISODES:-40000}
HELDOUT_PER_REGIME=${HELDOUT_PER_REGIME:-300}

for path in "$TRAIN" "$HELDOUT" "$REPORT"; do
  [ ! -e "$path" ] || { echo "[digitwise-v2] refusing existing artifact: $path" >&2; exit 3; }
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
echo "[digitwise-v2] committed immutable CPU candidate"
