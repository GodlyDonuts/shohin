#!/bin/bash
#SBATCH -J shohin-role-equiv-build
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/role_equiv_build_%j.out
#SBATCH --open-mode=append
# CPU-only construction and immutable admission. Submit on Stokes.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
SOURCE=${SOURCE:-$BASE/artifacts/sft/latent_operator_v1_train.jsonl}
DATA=${DATA:-$BASE/artifacts/sft/role_equivariant_microcode_v3.jsonl}
BUILD=${BUILD:-$BASE/artifacts/sft/role_equivariant_microcode_v3.build.json}
ADMISSION=${ADMISSION:-$BASE/artifacts/sft/role_equivariant_microcode_v3.admission.json}
QUALITY=${QUALITY:-$BASE/artifacts/sft/role_equivariant_microcode_v3.quality.json}
FULL_TEXT=${FULL_TEXT:-$BASE/artifacts/sft/role_equivariant_microcode_v3.full_text.json}
CONTRACTS=${CONTRACTS:-$BASE/artifacts/sft/role_equivariant_microcode_v3.contracts.json}
EVAL=$BASE/artifacts/evals/latent_operator_eval_slices_v2_64.jsonl
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
PROGRAMS=${PROGRAMS:-48000}

for path in "$SOURCE" "$EVAL" "$TOKENIZER"; do
  [ -s "$path" ] || { echo "[role-equiv-build] missing input: $path" >&2; exit 2; }
done
for path in "$DATA" "$BUILD" "$ADMISSION" "$QUALITY" "$FULL_TEXT" "$CONTRACTS"; do
  [ ! -e "$path" ] || { echo "[role-equiv-build] refusing existing output: $path" >&2; exit 3; }
done

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$BASE"
"$PY" pipeline/generate_role_equivariant_microcode_v3.py \
  --source "$SOURCE" --out "$DATA" --report "$BUILD" --programs "$PROGRAMS"
"$PY" pipeline/audit_role_equivariant_microcode_v3.py \
  --train "$DATA" --eval "$EVAL" --tokenizer "$TOKENIZER" --out "$ADMISSION" --programs "$PROGRAMS"
"$PY" pipeline/audit_sft_quality.py --data "$DATA" --evals "$BASE/artifacts/evals" --out "$QUALITY"
"$PY" pipeline/audit_role_equivariant_full_text_v3.py --data "$DATA" --evals "$BASE/artifacts/evals" \
  --out "$FULL_TEXT" --programs "$PROGRAMS"
"$PY" pipeline/audit_response_contracts.py --data "$DATA" --out "$CONTRACTS"
