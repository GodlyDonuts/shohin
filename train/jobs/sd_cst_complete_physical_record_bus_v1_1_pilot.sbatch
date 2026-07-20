#!/bin/bash
#SBATCH --job-name=sd-cst-local-v1-1
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --exclude=evc26,evc28,evc31,evc44
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/sd_cst_complete_physical_record_bus_v1_1_%j.out

set -euo pipefail

CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to the exact committed source capsule}
DATA=${DATA:?set DATA to the consumed projected-v2 board}
JOINT_CHECKPOINT=${JOINT_CHECKPOINT:?set the exact joint checkpoint}
PHYSICAL_CHECKPOINT=${PHYSICAL_CHECKPOINT:?set the exact physical-record checkpoint}
V1_CHECKPOINT=${V1_CHECKPOINT:?set the exact complete-local v1 checkpoint}
OUTDIR=${OUTDIR:?set OUTDIR to a new isolated training-only output directory}
PILOT_SEED=${PILOT_SEED:?set the post-source-commit pilot seed}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact post-prereg source commit}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

SOURCE_PATHS=(
  R12_SD_CST_COMPLETE_PHYSICAL_RECORD_BUS_V1_1_PREREG.md
  train/pilot_sd_cst_complete_physical_record_bus_v1_1.py
  train/pilot_sd_cst_complete_physical_record_bus.py
  train/pilot_sd_cst_physical_record_bus.py
  train/pilot_sd_cst_renderer_orbit.py
  train/sd_cst_complete_physical_record_bus_v1_1.py
  train/sd_cst_complete_physical_record_bus.py
  train/sd_cst_physical_record_bus.py
  train/test_sd_cst_complete_physical_record_bus_v1_1.py
  train/jobs/sd_cst_complete_physical_record_bus_v1_1_pilot.sbatch
)

cd "$CODE_ROOT"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
git diff --quiet
git diff --cached --quiet
for path in "${SOURCE_PATHS[@]}"; do
  git cat-file -e "$SOURCE_COMMIT:$path"
done
for path in "$DATA/train.jsonl" "$JOINT_CHECKPOINT" "$PHYSICAL_CHECKPOINT" "$V1_CHECKPOINT"; do
  test -f "$path" || { echo "missing v1.1 input: $path" >&2; exit 2; }
done
test ! -e "$OUTDIR" || { echo "refusing existing v1.1 output: $OUTDIR" >&2; exit 2; }

export PYTHONPATH="$CODE_ROOT/pipeline:$CODE_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=$((PILOT_SEED % 4294967296))
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$PYTHON" - <<'PY'
import torch
from pilot_sd_cst_byte_addressed import BASE_PARAMETERS, MOTOR_PARAMETERS, READER_PARAMETERS
from sd_cst_complete_physical_record_bus_v1_1 import CompletePhysicalRecordBusCompilerV1_1

if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("v1.1 pilot requires bf16 CUDA")
major, _ = torch.cuda.get_device_capability()
if major < 8:
    raise SystemExit("v1.1 pilot requires Ampere-or-newer GPU")
model = CompletePhysicalRecordBusCompilerV1_1()
complete = BASE_PARAMETERS + model.parameter_count() + MOTOR_PARAMETERS + READER_PARAMETERS
if complete != 191_675_285 or complete >= 200_000_000:
    raise SystemExit("v1.1 parameter certificate differs")
allocation = torch.empty((1024, 1024), device="cuda", dtype=torch.bfloat16)
print({
    "cuda": True,
    "device": torch.cuda.get_device_name(),
    "bf16": True,
    "allocation": int(allocation.numel()),
    "complete_parameters": complete,
})
PY

"$PYTHON" train/pilot_sd_cst_complete_physical_record_bus_v1_1.py \
  --train-jsonl "$DATA/train.jsonl" \
  --joint-checkpoint "$JOINT_CHECKPOINT" \
  --physical-checkpoint "$PHYSICAL_CHECKPOINT" \
  --v1-checkpoint "$V1_CHECKPOINT" \
  --out-dir "$OUTDIR" \
  --seed "$PILOT_SEED"

sha256sum "$OUTDIR/compiler.pt" "$OUTDIR/report.json"
echo "[sd-cst-complete-physical-record-bus-v1.1] complete; no scored split was addressed"
