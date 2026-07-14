#!/bin/bash
#SBATCH -J shohin-argument-admit
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/argument_graph_admit_%j.out
#SBATCH --open-mode=append
# CPU-only immutable admission. Submit on Stokes after the fresh board exists.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
TRAIN=${TRAIN:-$BASE/artifacts/sft/role_equivariant_microcode_v3.jsonl}
EVAL=${EVAL:-$BASE/artifacts/evals/referential_argument_graph_v5_fresh.jsonl}
MANUAL=${MANUAL:-$BASE/artifacts/evals/categorical_microcode_manual_v1.jsonl}
STRUCTURAL=${STRUCTURAL:-$BASE/artifacts/evals/referential_argument_graph_v5_fresh.admission.json}
LABELS=${LABELS:-$BASE/artifacts/evals/referential_argument_graph_v5_fresh.label_admission.json}
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json

for path in "$TRAIN" "$EVAL" "$MANUAL" "$TOKENIZER"; do
  [ -s "$path" ] || { echo "[argument-admit] missing input: $path" >&2; exit 2; }
done
for path in "$STRUCTURAL" "$LABELS"; do
  [ ! -e "$path" ] || { echo "[argument-admit] refusing existing output: $path" >&2; exit 3; }
done

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$BASE"
"$PY" pipeline/audit_role_equivariant_microcode_v3.py \
  --train "$TRAIN" --eval "$EVAL" --tokenizer "$TOKENIZER" --out "$STRUCTURAL" --programs 48000
"$PY" pipeline/audit_referential_slot_labels.py \
  --train "$TRAIN" --eval "$EVAL" --manual "$MANUAL" --tokenizer "$TOKENIZER" \
  --out "$LABELS" --eval-rows 896
