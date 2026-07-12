#!/bin/bash
#SBATCH -J shohin-gsm8k-train
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/gsm8k_train_%j.out
#SBATCH --open-mode=append
# Build a train-only labelled source for the isolated verifier experiment.  This
# never alters public eval data, frozen SFT data, or the active flagship run.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
OUT=${OUT:-$BASE/artifacts/problems/gsm8k_train.jsonl}
EVAL=${EVAL:-$BASE/artifacts/evals/gsm8k.jsonl}

cd "$BASE"
"$BASE/miniforge3/bin/python" pipeline/fetch_gsm8k_train.py --out "$OUT" --eval "$EVAL"
