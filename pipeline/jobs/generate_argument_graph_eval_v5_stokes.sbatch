#!/bin/bash
#SBATCH -J shohin-argument-v5-data
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/argument_graph_v5_data_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
TRAIN=${TRAIN:-$BASE/artifacts/sft/role_equivariant_microcode_v3.jsonl}
DEVELOPMENT=${DEVELOPMENT:-$BASE/artifacts/evals/latent_operator_eval_slices_v2_64.jsonl}
OUT=${OUT:-$BASE/artifacts/evals/referential_argument_graph_v5_fresh.jsonl}
REPORT=${REPORT:-$BASE/artifacts/evals/referential_argument_graph_v5_fresh.report.json}

for path in "$TRAIN" "$DEVELOPMENT"; do
  [ -s "$path" ] || { echo "[argument-v5-data] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[argument-v5-data] refusing existing output: $OUT" >&2; exit 3; }
[ ! -e "$REPORT" ] || { echo "[argument-v5-data] refusing existing report: $REPORT" >&2; exit 3; }

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$BASE"
"$PY" pipeline/generate_argument_graph_eval_v5.py --train "$TRAIN" \
  --development "$DEVELOPMENT" --out "$OUT" --report "$REPORT"
