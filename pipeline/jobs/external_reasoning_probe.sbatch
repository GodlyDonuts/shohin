#!/bin/bash
#SBATCH -J shohin-source-probe
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/external_reasoning_probe_%j.out
#SBATCH --open-mode=append
# Read-only intake gate for external reasoning sources. It writes only a small
# schema/provenance report and cannot create train data or touch the flagship.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATASET=${DATASET:?set DATASET to the Hugging Face dataset name}
CONFIG=${CONFIG:-}
SPLIT=${SPLIT:-train}
MAX_ROWS=${MAX_ROWS:-64}
SLUG=${SLUG:?set SLUG to a stable report name}
OUT=${OUT:-$BASE/artifacts/source_probes/${SLUG}.json}

[ ! -e "$OUT" ] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
cd "$BASE"
ARGS=(--dataset "$DATASET" --split "$SPLIT" --max-rows "$MAX_ROWS"
      --evals-dir "$BASE/artifacts/evals" --out "$OUT")
if [ -n "$CONFIG" ]; then
  ARGS+=(--config "$CONFIG")
fi
"$PY" pipeline/probe_reasoning_source.py "${ARGS[@]}"
