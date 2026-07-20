#!/bin/bash
#SBATCH --job-name=sd-cst-atomic
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --exclude=evc28,evc31,evc44
#SBATCH --output=logs/sd_cst_atomic_%j.out

set -euo pipefail

BASE=${BASE:?set BASE to the isolated SD-CST v1.1 canary source}
PYTHON=${PYTHON:?set PYTHON to the Newton training interpreter}
OUT=${OUT:?set OUT to a new atomic canary report}
SEED_START=${SEED_START:?set the first fresh canary seed}
SEEDS=${SEEDS:-64}

cd "$BASE"
mkdir -p logs "$(dirname "$OUT")"
export PYTHONPATH="$BASE/pipeline:$BASE/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=$((SEED_START % 4294967296))
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$PYTHON" train/atomic_optimizer_canary.py \
  --seed-start "$SEED_START" \
  --seeds "$SEEDS" \
  --out "$OUT"

sha256sum "$OUT"
