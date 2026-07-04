#!/bin/bash
#SBATCH -J shohin-decontam
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/decontam_%j.out
# Drop eval-contaminated items from the SFT corpus. Chain after the SFT job.
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
cd "$BASE/pipeline"
$PY decontam_filter.py --grams "$BASE/artifacts/evals/evalgrams.pkl" \
    --in "$BASE/artifacts/sft/openmath2_concise.jsonl" \
    --field problem --also-field solution \
    --out "$BASE/artifacts/sft/openmath2_concise.clean.jsonl"
echo "[job] decontam-filter done."
