#!/bin/bash
#SBATCH --job-name=shohin-s8-nil-graph
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/s8_nil_linked_graph_%j.out

set -euo pipefail

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PYTHON=${PYTHON:-$BASE/miniforge3/bin/python}
DATA=${DATA:?set DATA to the frozen S8 board}
TOKENIZER=${TOKENIZER:?set TOKENIZER to the frozen tokenizer}
BASE_CHECKPOINT=${BASE_CHECKPOINT:?set BASE_CHECKPOINT to protected 300k}
INITIALIZER=${INITIALIZER:?set INITIALIZER to confirmed S4 parser}
OUTDIR=${OUTDIR:?set OUTDIR to a new isolated S8 output directory}
TRAIN_SEED=${TRAIN_SEED:?set frozen S8 training seed}

cd "$BASE"
export PYTHONPATH="$BASE/pipeline:$BASE/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=$((TRAIN_SEED % 4294967296))
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

if [ -e "$OUTDIR" ]; then
  echo "refusing existing S8 output directory: $OUTDIR" >&2
  exit 2
fi
mkdir -p "$OUTDIR"

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("S8 job has no CUDA")
print({"cuda": True, "device": torch.cuda.get_device_name()})
PY

"$PYTHON" train/train_s8_nil_linked_graph.py \
  --base "$BASE_CHECKPOINT" \
  --initializer "$INITIALIZER" \
  --data-dir "$DATA" \
  --tokenizer "$TOKENIZER" \
  --out "$OUTDIR/checkpoint.pt" \
  --seed "$TRAIN_SEED"

"$PYTHON" train/eval_s8_nil_linked_graph.py \
  --base "$BASE_CHECKPOINT" \
  --checkpoint "$OUTDIR/checkpoint.pt" \
  --data-dir "$DATA" \
  --tokenizer "$TOKENIZER" \
  --out "$OUTDIR/evaluation.json"

"$PYTHON" train/assess_s8_nil_linked_graph.py \
  --evaluation "$OUTDIR/evaluation.json" \
  --out "$OUTDIR/assessment.json"

sha256sum \
  "$OUTDIR/checkpoint.pt" \
  "$OUTDIR/evaluation.json" \
  "$OUTDIR/assessment.json"
