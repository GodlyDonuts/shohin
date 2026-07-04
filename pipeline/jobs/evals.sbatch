#!/bin/bash
#SBATCH -J shohin-evals
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/evals_%j.out
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export HF_TOKEN=$(cat "$BASE/.hf_token" 2>/dev/null || true)
cd "$BASE/pipeline"
$PY fetch_evals.py --out-dir "$BASE/artifacts/evals"
echo "[job] evals done."
