#!/bin/bash
#SBATCH -J shohin-microcode-admit
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/microcode_admit_%j.out
#SBATCH --open-mode=append
# CPU-only immutable lexer/executor admission. Submit on Stokes.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
TRAIN=${TRAIN:-$BASE/artifacts/sft/latent_operator_v1_train.jsonl}
EVAL=${EVAL:-$BASE/artifacts/evals/latent_operator_eval_slices_v2_64.jsonl}
TOKENIZER=${TOKENIZER:-$BASE/artifacts/shohin-tok-32k.json}
OUT=${OUT:-$BASE/artifacts/sft/categorical_microcode_v1.admission.json}

for path in "$TRAIN" "$EVAL" "$TOKENIZER"; do
  [ -s "$path" ] || { echo "[microcode-admit] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[microcode-admit] refusing existing report: $OUT" >&2; exit 3; }

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$BASE"
"$PY" pipeline/audit_categorical_microcode_v1.py \
  --train "$TRAIN" --eval "$EVAL" --tokenizer "$TOKENIZER" --out "$OUT"
