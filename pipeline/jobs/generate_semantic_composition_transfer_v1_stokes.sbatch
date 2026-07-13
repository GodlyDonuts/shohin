#!/bin/bash
# CPU-only evaluation-suite builder. It does not generate SFT data or reserve a GPU.
#SBATCH -J shohin-semcomp-build
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/semantic_composition_transfer_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=/usr/bin/python3.12
OUT=${OUT:-$BASE/artifacts/evals/semantic_composition_transfer_v1.jsonl}
AUDIT=${AUDIT:-$BASE/artifacts/evals/semantic_composition_transfer_v1.audit.json}
TRAIN=${TRAIN:-$BASE/artifacts/sft/semantic_bridge_v1_train.jsonl}

[ -s "$TRAIN" ] || { echo "[semcomp] training reference missing: $TRAIN" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[semcomp] refusing existing suite: $OUT" >&2; exit 3; }
[ ! -e "$AUDIT" ] || { echo "[semcomp] refusing existing audit: $AUDIT" >&2; exit 3; }
cd "$BASE"
"$PY" pipeline/generate_semantic_composition_transfer_v1.py --out "$OUT" --per-family 100
"$PY" pipeline/audit_semantic_composition_transfer_v1.py \
  --suite "$OUT" --training-reference "$TRAIN" --out "$AUDIT"
echo "[semcomp] admitted suite=$OUT audit=$AUDIT"
