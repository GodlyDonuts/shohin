#!/bin/bash
#SBATCH -J shohin-certified-ledger
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/certified_ledger_%j.out
#SBATCH --open-mode=append
# CPU-only build and independent admission of a future source-free ledger task.
# It cannot write a checkpoint, the flagship data stream, or an SFT mix.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
TRAIN=${TRAIN:?set TRAIN to a fresh certified-ledger train JSONL}
EVAL=${EVAL:?set EVAL to a fresh certified-ledger held-out JSONL}
AUDIT=${AUDIT:?set AUDIT to a fresh certified-ledger audit JSON}
TRAIN_EPISODES=${TRAIN_EPISODES:-16000}
EVAL_EPISODES_PER_CHUNK=${EVAL_EPISODES_PER_CHUNK:-128}
SEED=${SEED:-20260715}
TAG_SCHEME=${TAG_SCHEME:-wide_v1}

for path in "$TRAIN" "$EVAL" "$AUDIT"; do
  [ ! -e "$path" ] || { echo "[certified-ledger] refusing existing output: $path" >&2; exit 3; }
done
cd "$BASE/pipeline"
"$PY" generate_certified_latent_ledger_v1.py \
  --train-out "$TRAIN" --eval-out "$EVAL" --train-episodes "$TRAIN_EPISODES" \
  --eval-episodes-per-chunk "$EVAL_EPISODES_PER_CHUNK" --seed "$SEED" --tag-scheme "$TAG_SCHEME"
"$PY" audit_certified_latent_ledger_v1.py --train "$TRAIN" --eval "$EVAL" --out "$AUDIT"
