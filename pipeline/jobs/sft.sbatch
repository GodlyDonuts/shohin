#!/bin/bash
#SBATCH -J shohin-sft
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_%j.out
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export HF_TOKEN=$(cat "$BASE/.hf_token" 2>/dev/null || true)
cd "$BASE/pipeline"
$PY sft_curate.py --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
    --out "$BASE/artifacts/sft/openmath2_concise.jsonl" \
    --max-keep 300000 --max-sol-tokens 400
echo "[job] sft done."
