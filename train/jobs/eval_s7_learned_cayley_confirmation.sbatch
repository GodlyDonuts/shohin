#!/bin/bash
#SBATCH --job-name=shohin-s7-confirm
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/s7_learned_cayley_confirmation_%j.out

set -euo pipefail

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PYTHON=${PYTHON:-$BASE/miniforge3/bin/python}
DATA=${DATA:?set DATA to the frozen S7 board}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to the frozen S7 checkpoint}
DEVELOPMENT_ASSESSMENT=${DEVELOPMENT_ASSESSMENT:?set frozen development assessment}
OUTDIR=${OUTDIR:?set OUTDIR to a new isolated confirmation output directory}

cd "$BASE"
export PYTHONPATH="$BASE/pipeline:$BASE/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=20260719

if [ -e "$OUTDIR" ]; then
  echo "refusing existing S7 confirmation output directory: $OUTDIR" >&2
  exit 2
fi
mkdir -p "$OUTDIR"

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("S7 confirmation job has no CUDA")
probe = torch.ones(1, device="cuda")
print({"cuda": True, "device": torch.cuda.get_device_name(), "probe": probe.item()})
PY

"$PYTHON" train/eval_s7_learned_cayley_confirmation.py \
  --data-dir "$DATA" \
  --checkpoint "$CHECKPOINT" \
  --development-assessment "$DEVELOPMENT_ASSESSMENT" \
  --out "$OUTDIR/evaluation.json" \
  --device cuda

"$PYTHON" train/assess_s7_learned_cayley_confirmation.py \
  --evaluation "$OUTDIR/evaluation.json" \
  --out "$OUTDIR/assessment.json"

sha256sum "$OUTDIR/evaluation.json" "$OUTDIR/assessment.json"
