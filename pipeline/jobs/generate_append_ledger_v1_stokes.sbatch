#!/bin/bash
# CPU-only candidate: model-authored short deltas followed by block compaction.
# No flagship or GPU path may reference it until its independent audit passes.
#SBATCH -J shohin-ledger-v1
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/append_ledger_v1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=${PY:-/usr/bin/python3.12}
TRAIN=${TRAIN:-$BASE/artifacts/sft/append_ledger_v1_train.jsonl}
HELDOUT=${HELDOUT:-$BASE/artifacts/evals/append_ledger_v1_heldout.jsonl}
REPORT=${REPORT:-$BASE/artifacts/sft/append_ledger_v1_train.report.json}
TRAIN_EPISODES=${TRAIN_EPISODES:-24000}
HELDOUT_PER_REGIME=${HELDOUT_PER_REGIME:-200}
BLOCK_SIZE=${BLOCK_SIZE:-4}

for path in "$TRAIN" "$HELDOUT" "$REPORT"; do
  [ ! -e "$path" ] || { echo "[append-ledger] refusing existing artifact: $path" >&2; exit 3; }
done
cd "$BASE"
"$PY" pipeline/generate_append_ledger_v1.py \
  --train-out "$TRAIN" \
  --heldout-out "$HELDOUT" \
  --report "$REPORT" \
  --train-episodes "$TRAIN_EPISODES" \
  --heldout-per-regime "$HELDOUT_PER_REGIME" \
  --block-size "$BLOCK_SIZE"
test -s "$TRAIN"
test -s "$HELDOUT"
test -s "$REPORT"
echo "[append-ledger] committed immutable CPU candidate"
