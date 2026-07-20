#!/bin/bash
#SBATCH --job-name=sd-cst-hier-bind
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=evc28,evc31,evc44
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/sd_cst_hierarchical_binding_pilot_%j.out

set -euo pipefail

CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to the exact committed source capsule}
DATA=${DATA:?set DATA to the consumed frozen SD-CST v1.1 board}
PARENT=${PARENT:?set PARENT to the exact byte-compiler checkpoint}
OUTDIR=${OUTDIR:?set OUTDIR to a new isolated training-only output directory}
PILOT_SEED=${PILOT_SEED:?set the post-source-commit pilot seed}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact pilot source commit}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

SOURCE_PATHS=(
  R12_SD_CST_HIERARCHICAL_BINDING_PILOT_PREREG.md
  train/sd_cst.py
  train/sd_cst_byte_addressed.py
  train/sd_cst_binding_bus.py
  train/pilot_sd_cst_byte_addressed.py
  train/pilot_sd_cst_binding_bus.py
  train/pilot_sd_cst_hierarchical_binding.py
  train/jobs/sd_cst_hierarchical_binding_pilot.sbatch
)

cd "$CODE_ROOT"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
git diff --quiet
git diff --cached --quiet
for path in "${SOURCE_PATHS[@]}"; do
  git cat-file -e "$SOURCE_COMMIT:$path"
done
for path in "$DATA/report.json" "$DATA/train.jsonl" "$PARENT"; do
  test -f "$path" || { echo "missing training-only input: $path" >&2; exit 2; }
done
test ! -e "$OUTDIR" || { echo "refusing existing output: $OUTDIR" >&2; exit 2; }

export PYTHONPATH="$CODE_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=$((PILOT_SEED % 4294967296))
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("hierarchical binding pilot has no CUDA")
major, _ = torch.cuda.get_device_capability()
if major < 8 or not torch.cuda.is_bf16_supported():
    raise SystemExit("hierarchical binding pilot requires a bf16-capable GPU")
print({"cuda": True, "device": torch.cuda.get_device_name(), "bf16": True})
PY

"$PYTHON" train/pilot_sd_cst_hierarchical_binding.py \
  --data-dir "$DATA" \
  --parent-checkpoint "$PARENT" \
  --out-dir "$OUTDIR" \
  --seed "$PILOT_SEED"

sha256sum "$OUTDIR/compiler.pt" "$OUTDIR/report.json"
echo "[sd-cst-hierarchical-binding] complete; no scored split was addressed"
