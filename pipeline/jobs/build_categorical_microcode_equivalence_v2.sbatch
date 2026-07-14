#!/bin/bash
#SBATCH -J shohin-microcode-equiv-build
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/microcode_equiv_build_%j.out
#SBATCH --open-mode=append
# CPU-only paired-language build and admission. Submit on Stokes.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
SOURCE=${SOURCE:-$BASE/artifacts/sft/latent_operator_v1_train.jsonl}
DATA=${DATA:-$BASE/artifacts/sft/categorical_microcode_equivalence_v2.jsonl}
BUILD=${BUILD:-$BASE/artifacts/sft/categorical_microcode_equivalence_v2.build.json}
ADMISSION=${ADMISSION:-$BASE/artifacts/sft/categorical_microcode_equivalence_v2.admission.json}
QUALITY=${QUALITY:-$BASE/artifacts/sft/categorical_microcode_equivalence_v2.quality.json}
FULL_TEXT=${FULL_TEXT:-$BASE/artifacts/sft/categorical_microcode_equivalence_v2.full_text.json}
CONTRACTS=${CONTRACTS:-$BASE/artifacts/sft/categorical_microcode_equivalence_v2.contracts.json}
EVAL=$BASE/artifacts/evals/latent_operator_eval_slices_v2_64.jsonl
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json

for path in "$SOURCE" "$EVAL" "$TOKENIZER"; do
  [ -s "$path" ] || { echo "[microcode-equiv-build] missing input: $path" >&2; exit 2; }
done
for path in "$DATA" "$BUILD" "$ADMISSION" "$QUALITY" "$FULL_TEXT" "$CONTRACTS"; do
  [ ! -e "$path" ] || { echo "[microcode-equiv-build] refusing existing output: $path" >&2; exit 3; }
done

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$BASE"
"$PY" pipeline/generate_categorical_microcode_equivalence_v2.py \
  --source "$SOURCE" --out "$DATA" --report "$BUILD" --pairs 48000
"$PY" pipeline/audit_categorical_microcode_equivalence_v2.py \
  --train "$DATA" --eval "$EVAL" --tokenizer "$TOKENIZER" --out "$ADMISSION"
"$PY" pipeline/audit_sft_quality.py --data "$DATA" --evals "$BASE/artifacts/evals" --out "$QUALITY"
"$PY" pipeline/audit_training_text_overlap.py --data "$DATA" --evals "$BASE/artifacts/evals" \
  --out "$FULL_TEXT" --fields question response --require-zero
"$PY" pipeline/audit_response_contracts.py --data "$DATA" --out "$CONTRACTS"
