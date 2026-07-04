#!/bin/bash
#SBATCH -J shohin-rg
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/rg_%j.out
# Generate the verified, held-out-disciplined Reasoning-Gym procedural corpus.
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
cd "$BASE/pipeline"
$PY gen_reasoning_gym.py --out-dir "$BASE/artifacts/rg" --per-family 300 --eval-per-family 80
echo "[job] rg corpus done."
