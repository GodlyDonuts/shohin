#!/bin/bash
#SBATCH -J shohin-upload
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/upload_%j.out
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export HF_TOKEN=$(cat "$BASE/.hf_token")
$PY $BASE/pipeline/upload_hf.py
