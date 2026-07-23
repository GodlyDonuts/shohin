#!/bin/bash
#SBATCH --job-name=shohin-qerarm-dev
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/qerarm_dev_%j.out

set -euo pipefail

BASE=${BASE:?set BASE to an isolated clean source checkout}
SOURCE_COMMIT=${SOURCE_COMMIT:?set SOURCE_COMMIT to the frozen Git commit}
OUTDIR=${OUTDIR:?set OUTDIR to a new isolated output directory}
TRAIN_SEED=${TRAIN_SEED:?set the train-only optimizer seed}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

if [ -e "$OUTDIR" ]; then
  echo "refusing existing QERARM output: $OUTDIR" >&2
  exit 2
fi
if [ "$(git -C "$BASE" rev-parse HEAD)" != "$SOURCE_COMMIT" ]; then
  echo "QERARM source commit differs" >&2
  exit 2
fi
if [ -n "$(git -C "$BASE" status --porcelain --untracked-files=no)" ]; then
  echo "QERARM tracked source is dirty" >&2
  exit 2
fi

mkdir -p "$OUTDIR"
cd "$BASE"
export OMP_NUM_THREADS=2
export PYTHONHASHSEED=$((TRAIN_SEED % 4294967296))
export PYTHONPATH="$BASE/train:$BASE/pipeline${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("QERARM canary has no CUDA")
print(
    {
        "cuda": True,
        "device": torch.cuda.get_device_name(),
        "count": torch.cuda.device_count(),
    },
    flush=True,
)
PY

sha256sum \
  train/equivariant_relation_register_machine.py \
  train/train_fixed_point_register_machine.py \
  pipeline/contrastive_fixed_point_board.py

"$PYTHON" train/train_fixed_point_register_machine.py \
  --out "$OUTDIR/checkpoint.pt" \
  --report "$OUTDIR/report.json" \
  --seed "$TRAIN_SEED" \
  --train-rows "${TRAIN_ROWS:-8192}" \
  --development-rows "${DEVELOPMENT_ROWS:-1024}" \
  --updates "${UPDATES:-5000}" \
  --warmup "${WARMUP:-200}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --learning-rate "${LEARNING_RATE:-0.0003}" \
  --controller-width "${CONTROLLER_WIDTH:-512}" \
  --controller-layers "${CONTROLLER_LAYERS:-3}" \
  --maximum-steps "${MAXIMUM_STEPS:-24}" \
  --log-every "${LOG_EVERY:-100}" \
  --device cuda

sha256sum "$OUTDIR/checkpoint.pt" "$OUTDIR/report.json"
