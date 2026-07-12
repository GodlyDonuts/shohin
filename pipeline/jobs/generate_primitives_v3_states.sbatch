#!/bin/bash
#SBATCH -J shohin-state-data
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/state_protocol_data_%j.out
#SBATCH --open-mode=append
# CPU-only source build. It does not write a frozen SFT mix or touch pretraining.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
TRAIN=${TRAIN:-$BASE/artifacts/sft/primitives_v3_state_train.jsonl}
HELDOUT=${HELDOUT:-$BASE/artifacts/evals/primitives_v3_state_heldout.jsonl}
TRAIN_PER_FAMILY=${TRAIN_PER_FAMILY:-15000}
EVAL_PER_FAMILY=${EVAL_PER_FAMILY:-500}
EVALS=$BASE/artifacts/evals

[ ! -e "$TRAIN" ] || { echo "[state-data] refusing existing train output: $TRAIN" >&2; exit 2; }
[ ! -e "$HELDOUT" ] || { echo "[state-data] refusing existing heldout output: $HELDOUT" >&2; exit 3; }
cd "$BASE"
"$PY" pipeline/generate_primitives_v3_states.py \
  --train-out "$TRAIN" --eval-out "$HELDOUT" \
  --train-per-family "$TRAIN_PER_FAMILY" --eval-per-family "$EVAL_PER_FAMILY"
"$PY" pipeline/audit_sft_quality.py --data "$TRAIN" --evals "$EVALS" \
  --out "${TRAIN%.jsonl}.quality.json"
"$PY" pipeline/audit_sft_quality.py --data "$HELDOUT" --evals "$EVALS" \
  --out "${HELDOUT%.jsonl}.quality.json"
echo "[state-data] done train=$TRAIN heldout=$HELDOUT"
