#!/bin/bash
#SBATCH --job-name=shohin-s6-law
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/s6_contextual_affine_law_%j.out

set -euo pipefail

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PYTHON=${PYTHON:-$BASE/miniforge3/bin/python}
DATA=${DATA:?set DATA to the frozen S6 development board}
OUTDIR=${OUTDIR:?set OUTDIR to a new isolated S6 output directory}
TRAIN_SEED=${TRAIN_SEED:?set the frozen post-commit S6 training seed}

cd "$BASE"
export PYTHONPATH="$BASE/pipeline:$BASE/train${PYTHONPATH:+:$PYTHONPATH}"
# CPython accepts only unsigned 32-bit PYTHONHASHSEED values. Keep the full
# frozen seed for model/data RNGs and derive the interpreter seed explicitly.
export PYTHONHASHSEED=$((TRAIN_SEED % 4294967296))

if [ -e "$OUTDIR" ]; then
  echo "refusing existing S6 output directory: $OUTDIR" >&2
  exit 2
fi
mkdir -p "$OUTDIR"

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("S6 H100 job has no CUDA")
probe = torch.ones(1, device="cuda")
print({"cuda": True, "device": torch.cuda.get_device_name(), "probe": probe.item()})
PY

"$PYTHON" train/train_s6_contextual_affine_law.py \
  --data-dir "$DATA" \
  --out "$OUTDIR/checkpoint.pt" \
  --device cuda \
  --seed "$TRAIN_SEED" \
  --updates 4000 \
  --batch-size 256 \
  --learning-rate 5e-4 \
  --weight-decay 0.01

"$PYTHON" train/eval_s6_contextual_affine_law.py \
  --data-dir "$DATA" \
  --checkpoint "$OUTDIR/checkpoint.pt" \
  --out "$OUTDIR/evaluation.json" \
  --device cuda

"$PYTHON" train/assess_s6_contextual_affine_law.py \
  --evaluation "$OUTDIR/evaluation.json" \
  --out "$OUTDIR/assessment.json"

sha256sum "$OUTDIR/checkpoint.pt" "$OUTDIR/evaluation.json" "$OUTDIR/assessment.json"
