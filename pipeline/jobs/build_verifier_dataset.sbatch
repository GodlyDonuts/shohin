#!/bin/bash
#SBATCH -J shohin-verifier-data
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/verifier_data_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
IN=${IN:?set IN to train-only rollout JSONL}
OUT=${OUT:?set OUT to a fresh verifier JSONL}
cd "$BASE"
"$PY" pipeline/build_verifier_dataset.py --input "$IN" --out "$OUT"
"$PY" pipeline/audit_sft_quality.py --data "$OUT" --out "${OUT%.jsonl}.quality.json"
