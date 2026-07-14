#!/bin/bash
#SBATCH -J shohin-slot-label-audit
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/slot_label_audit_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
TRAIN=${TRAIN:-$BASE/artifacts/sft/role_equivariant_microcode_v3.jsonl}
EVAL=${EVAL:-$BASE/artifacts/evals/latent_operator_eval_slices_v2_64.jsonl}
MANUAL=${MANUAL:-$BASE/artifacts/evals/categorical_microcode_manual_v1.jsonl}
OUT=${OUT:-$BASE/artifacts/sft/referential_slot_labels_v4.admission.json}

for path in "$TRAIN" "$EVAL" "$MANUAL" "$BASE/artifacts/shohin-tok-32k.json"; do
  [ -s "$path" ] || { echo "[slot-label-audit] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[slot-label-audit] refusing existing output: $OUT" >&2; exit 3; }

cd "$BASE"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" pipeline/audit_referential_slot_labels.py \
  --train "$TRAIN" --eval "$EVAL" --manual "$MANUAL" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT"
