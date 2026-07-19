#!/bin/bash
#SBATCH --job-name=shohin-s9-oqrc
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/s9_occurrence_quotient_%j.out

set -euo pipefail

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PYTHON=${PYTHON:-$BASE/miniforge3/bin/python}
DATA=${DATA:?set DATA to the frozen S9 board}
TOKENIZER=${TOKENIZER:?set TOKENIZER to the frozen tokenizer}
BASE_CHECKPOINT=${BASE_CHECKPOINT:?set BASE_CHECKPOINT to protected 300k}
INITIALIZER=${INITIALIZER:?set INITIALIZER to closed S8.1 checkpoint}
OUTDIR=${OUTDIR:?set OUTDIR to a new isolated S9 output directory}
TRAIN_SEED=${TRAIN_SEED:?set frozen S9 training seed}
SOURCE_COMMIT=${SOURCE_COMMIT:?set frozen S9 source commit}

cd "$BASE"
export PYTHONPATH="$BASE/pipeline:$BASE/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=$((TRAIN_SEED % 4294967296))
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

if [ -e "$OUTDIR" ]; then
  echo "refusing existing S9 output directory: $OUTDIR" >&2
  exit 2
fi
mkdir -p "$OUTDIR"

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("S9 job has no CUDA")
major, _ = torch.cuda.get_device_capability()
if major < 8:
    raise SystemExit("S9 bf16 job requires Ampere or newer")
print({"cuda": True, "device": torch.cuda.get_device_name(), "bf16": torch.cuda.is_bf16_supported()})
PY

"$PYTHON" train/train_s9_occurrence_quotient.py \
  --base "$BASE_CHECKPOINT" \
  --initializer "$INITIALIZER" \
  --data-dir "$DATA" \
  --tokenizer "$TOKENIZER" \
  --out "$OUTDIR/checkpoint.pt" \
  --seed "$TRAIN_SEED" \
  --source-commit "$SOURCE_COMMIT"

"$PYTHON" train/eval_s9_occurrence_quotient.py \
  --base "$BASE_CHECKPOINT" \
  --checkpoint "$OUTDIR/checkpoint.pt" \
  --data-dir "$DATA" \
  --tokenizer "$TOKENIZER" \
  --out "$OUTDIR/evaluation.json"

"$PYTHON" train/assess_s9_occurrence_quotient.py \
  --evaluation "$OUTDIR/evaluation.json" \
  --out "$OUTDIR/assessment.json"

sha256sum "$OUTDIR/checkpoint.pt" "$OUTDIR/evaluation.json" "$OUTDIR/assessment.json"
