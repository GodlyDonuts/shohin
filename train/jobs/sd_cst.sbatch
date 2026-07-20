#!/bin/bash
#SBATCH --job-name=shohin-sd-cst
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=evc28,evc31,evc44
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/sd_cst_%j.out

set -euo pipefail

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PYTHON=${PYTHON:-$BASE/miniforge3/bin/python}
DATA=${DATA:?set DATA to the frozen admitted SD-CST board}
TOKENIZER=${TOKENIZER:?set TOKENIZER to the frozen tokenizer}
BASE_CHECKPOINT=${BASE_CHECKPOINT:?set BASE_CHECKPOINT to the protected Shohin base}
OUTDIR=${OUTDIR:?set OUTDIR to a new isolated SD-CST output directory}
TRAIN_SEED=${TRAIN_SEED:?set the frozen SD-CST training seed}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the frozen SD-CST source commit}

cd "$BASE"
export PYTHONPATH="$BASE/pipeline:$BASE/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=$((TRAIN_SEED % 4294967296))
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

SOURCE_PATHS=(
  R12_SD_CST_PREREG.md
  pipeline/assess_sd_cst.py
  pipeline/audit_sd_cst_board.py
  pipeline/build_sd_cst_board.py
  train/model.py
  train/sd_cst.py
  train/train_sd_cst.py
  train/jobs/sd_cst.sbatch
)
git cat-file -e "$SOURCE_COMMIT^{commit}"
git merge-base --is-ancestor "$SOURCE_COMMIT" HEAD
git diff --quiet "$SOURCE_COMMIT" -- "${SOURCE_PATHS[@]}"

for path in "$DATA/report.json" "$DATA/train.jsonl" \
  "$DATA/development.jsonl" "$TOKENIZER" "$BASE_CHECKPOINT"; do
  [ -f "$path" ] || { echo "missing SD-CST input: $path" >&2; exit 2; }
done
[ ! -e "$OUTDIR" ] || { echo "refusing existing SD-CST output: $OUTDIR" >&2; exit 2; }
mkdir -p "$OUTDIR"

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("SD-CST job has no CUDA")
major, _ = torch.cuda.get_device_capability()
if major < 8 or not torch.cuda.is_bf16_supported():
    raise SystemExit("SD-CST requires a bf16-capable Ampere-or-newer GPU")
print({"cuda": True, "device": torch.cuda.get_device_name(), "bf16": True})
PY

"$PYTHON" train/train_sd_cst.py train \
  --base "$BASE_CHECKPOINT" \
  --data-dir "$DATA" \
  --tokenizer "$TOKENIZER" \
  --out "$OUTDIR/checkpoint.pt" \
  --repo-root "$BASE" \
  --source-commit "$SOURCE_COMMIT" \
  --seed "$TRAIN_SEED"

"$PYTHON" train/train_sd_cst.py gate-config \
  --checkpoint "$OUTDIR/checkpoint.pt" \
  --data-dir "$DATA" \
  --out "$OUTDIR/gate_config.json" \
  --repo-root "$BASE" \
  --source-commit "$SOURCE_COMMIT"

"$PYTHON" train/train_sd_cst.py development \
  --base "$BASE_CHECKPOINT" \
  --checkpoint "$OUTDIR/checkpoint.pt" \
  --data-dir "$DATA" \
  --tokenizer "$TOKENIZER" \
  --out "$OUTDIR/development.json" \
  --repo-root "$BASE" \
  --source-commit "$SOURCE_COMMIT" \
  --access-ledger-dir "$BASE/artifacts/r12/access_ledgers"

"$PYTHON" pipeline/assess_sd_cst.py \
  --eval "$OUTDIR/development.json" \
  --config "$OUTDIR/gate_config.json" \
  --out "$OUTDIR/assessment.json"

sha256sum "$OUTDIR/checkpoint.pt" "$OUTDIR/gate_config.json" \
  "$OUTDIR/development.json" "$OUTDIR/assessment.json"
echo "[sd-cst] train/development complete; sealed confirmation was never addressed"
