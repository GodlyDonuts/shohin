#!/bin/bash
#SBATCH -J shohin-source-memory-data
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/source_memory_data_%j.out
#SBATCH --open-mode=append
# CPU-only source-memory data build. It cannot write training checkpoints.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
TRAIN=${TRAIN:?set TRAIN to a fresh source-memory train JSONL}
EVAL=${EVAL:?set EVAL to a fresh source-memory eval JSONL}
AUDIT=${AUDIT:?set AUDIT to a fresh source-memory audit JSON}

for path in "$TRAIN" "$EVAL" "$AUDIT"; do
  [ ! -e "$path" ] || { echo "[source-memory-data] refusing existing output: $path" >&2; exit 3; }
done
cd "$BASE/pipeline"
"$PY" generate_source_memory_packet_v1.py --train-out "$TRAIN" --eval-out "$EVAL"
"$PY" audit_source_memory_packet_v1.py --train "$TRAIN" --eval "$EVAL" --out "$AUDIT"
