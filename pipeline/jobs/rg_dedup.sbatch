#!/bin/bash
#SBATCH -J shohin-rg-dedup
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/rg_dedup_%j.out
#SBATCH --open-mode=append
# Build a new deduplicated SFT derivative from immutable verified RG traces.
# It never mutates the source corpus or a frozen SFT mix.
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
IN=${IN:?set IN to rg_traces_train.jsonl}
OUT=${OUT:?set OUT to a new rg_traces_sft JSONL path}
[ -e "$OUT" ] && { echo "[rg-dedup] refusing to overwrite $OUT" >&2; exit 2; }
cd "$BASE/pipeline"
"$PY" prepare_rg_sft.py --input "$IN" --out "$OUT"
"$PY" audit_sft_quality.py --data "$OUT" --out "${OUT%.jsonl}.quality.json"
