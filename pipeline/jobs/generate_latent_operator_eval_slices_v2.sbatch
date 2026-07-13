#!/bin/bash
#SBATCH -J shohin-latent-slices
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/latent_slices_%j.out
#SBATCH --open-mode=append
# CPU-only evaluation-data build. It cannot write a checkpoint or training mix.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
TRAIN=${TRAIN:-$BASE/artifacts/sft/latent_operator_v1_train.jsonl}
OUT=${OUT:?set OUT to a fresh diagnostic-evaluation JSONL}
REPORT=${REPORT:?set REPORT to a fresh diagnostic-evaluation report}
PER_DEPTH=${PER_DEPTH:-64}

[ -s "$TRAIN" ] || { echo "[latent-slices] missing train data: $TRAIN" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[latent-slices] refusing existing output: $OUT" >&2; exit 3; }
[ ! -e "$REPORT" ] || { echo "[latent-slices] refusing existing report: $REPORT" >&2; exit 3; }
cd "$BASE/pipeline"
"$PY" generate_latent_operator_eval_slices_v2.py --train "$TRAIN" --out "$OUT" --report "$REPORT" \
  --per-depth "$PER_DEPTH"
