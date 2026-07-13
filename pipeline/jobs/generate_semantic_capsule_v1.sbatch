#!/bin/bash
#SBATCH -J shohin-semcapsule
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/semantic_capsule_%j.out
#SBATCH --open-mode=append
# CPU-only generation of an isolated semantic-capsule curriculum and held-out
# controller episodes. It cannot modify the active pretraining stream.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
TRAIN=${TRAIN:-$BASE/artifacts/sft/semantic_capsule_v1_train.jsonl}
HELDOUT=${HELDOUT:-$BASE/artifacts/evals/semantic_capsule_v1_heldout.jsonl}
TRAIN_PER_DOMAIN=${TRAIN_PER_DOMAIN:-15000}
HELDOUT_PER_DOMAIN=${HELDOUT_PER_DOMAIN:-1000}

[ ! -e "$TRAIN" ] || { echo "[semantic-capsule] refusing existing train artifact: $TRAIN" >&2; exit 3; }
[ ! -e "$HELDOUT" ] || { echo "[semantic-capsule] refusing existing held-out artifact: $HELDOUT" >&2; exit 3; }
cd "$BASE"
"$PY" pipeline/generate_semantic_capsule_v1.py --train-out "$TRAIN" --heldout-out "$HELDOUT" \
  --train-per-domain "$TRAIN_PER_DOMAIN" --heldout-per-domain "$HELDOUT_PER_DOMAIN"
