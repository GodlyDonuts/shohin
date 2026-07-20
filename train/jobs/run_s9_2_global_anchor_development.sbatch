#!/bin/bash
#SBATCH --job-name=shohin-s92-anchor
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --exclude=evc28,evc31,evc44
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/s9_2_global_anchor_%j.out

set -euo pipefail

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PYTHON=${PYTHON:-$BASE/miniforge3/bin/python}
DATA=${DATA:?set DATA to the frozen S9.2 board}
TOKENIZER=${TOKENIZER:?set TOKENIZER to the frozen tokenizer}
BASE_CHECKPOINT=${BASE_CHECKPOINT:?set BASE_CHECKPOINT to protected 300k}
INITIALIZER=${INITIALIZER:?set INITIALIZER to closed S8.1 checkpoint}
OUTDIR=${OUTDIR:?set OUTDIR to a new isolated S9.2 output directory}
TRAIN_SEED=${TRAIN_SEED:?set frozen S9.2 training seed}
SOURCE_COMMIT=${SOURCE_COMMIT:?set frozen S9.2 source commit}

cd "$BASE"
export PYTHONPATH="$BASE/pipeline:$BASE/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=$((TRAIN_SEED % 4294967296))
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

SOURCE_PATHS=(
  pipeline/build_s8_nil_linked_law_graph_board.py
  pipeline/s8_nil_linked_law_graph.py
  pipeline/s9_occurrence_quotient.py
  pipeline/s9_occurrence_quotient_falsifier.py
  train/model.py
  train/s7_learned_cayley_generator.py
  train/s8_nil_linked_graph_compiler.py
  train/s9_occurrence_quotient_compiler.py
  train/s9_1_alpha_closed_compiler.py
  train/s9_2_global_anchor_compiler.py
  train/train_s8_nil_linked_graph.py
  train/train_s9_2_global_anchor.py
  train/eval_s9_2_global_anchor.py
  train/assess_s9_2_global_anchor.py
  train/jobs/run_s9_2_global_anchor_development.sbatch
)
git cat-file -e "$SOURCE_COMMIT^{commit}"
git merge-base --is-ancestor "$SOURCE_COMMIT" HEAD
git diff --quiet "$SOURCE_COMMIT" -- "${SOURCE_PATHS[@]}"

if [ -e "$OUTDIR" ]; then
  echo "refusing existing S9.2 output directory: $OUTDIR" >&2
  exit 2
fi
mkdir -p "$OUTDIR"

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("S9.2 job has no CUDA")
major, _ = torch.cuda.get_device_capability()
if major < 8 or not torch.cuda.is_bf16_supported():
    raise SystemExit("S9.2 requires a bf16-capable Ampere-or-newer GPU")
print({"cuda": True, "device": torch.cuda.get_device_name(), "bf16": True})
PY

"$PYTHON" train/train_s9_2_global_anchor.py \
  --base "$BASE_CHECKPOINT" \
  --initializer "$INITIALIZER" \
  --data-dir "$DATA" \
  --tokenizer "$TOKENIZER" \
  --out "$OUTDIR/checkpoint.pt" \
  --seed "$TRAIN_SEED" \
  --source-commit "$SOURCE_COMMIT"

"$PYTHON" train/eval_s9_2_global_anchor.py \
  --base "$BASE_CHECKPOINT" \
  --checkpoint "$OUTDIR/checkpoint.pt" \
  --data-dir "$DATA" \
  --tokenizer "$TOKENIZER" \
  --out "$OUTDIR/evaluation.json"

"$PYTHON" train/assess_s9_2_global_anchor.py \
  --evaluation "$OUTDIR/evaluation.json" \
  --out "$OUTDIR/assessment.json"

sha256sum "$OUTDIR/checkpoint.pt" "$OUTDIR/evaluation.json" "$OUTDIR/assessment.json"
LEDGER="$BASE/artifacts/r12/access_ledgers/s9_2_$(sha256sum "$DATA/report.json" | awk '{print $1}').development.json"
sha256sum "$LEDGER"
