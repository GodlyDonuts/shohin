#!/bin/bash
#SBATCH --job-name=sd-cst-v1-1-audit
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=evc26,evc28,evc31,evc44
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/sd_cst_complete_physical_record_bus_v1_1_audit_%j.out

set -euo pipefail

CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to the exact committed source capsule}
DATA=${DATA:?set DATA to the consumed projected-v2 board}
JOINT_CHECKPOINT=${JOINT_CHECKPOINT:?set the exact joint checkpoint}
PHYSICAL_CHECKPOINT=${PHYSICAL_CHECKPOINT:?set the exact physical-record checkpoint}
V1_CHECKPOINT=${V1_CHECKPOINT:?set the exact complete-local v1 checkpoint}
ENDPOINT=${ENDPOINT:?set the exact v1.1 endpoint}
OUT=${OUT:?set OUT to a new audit report path}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact audit source commit}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

cd "$CODE_ROOT"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
git diff --quiet
git diff --cached --quiet
test ! -e "$OUT" || { echo "refusing existing six-slot audit: $OUT" >&2; exit 2; }

export PYTHONPATH="$CODE_ROOT/pipeline:$CODE_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$PYTHON" train/audit_sd_cst_complete_physical_record_bus_v1_1.py \
  --train-jsonl "$DATA/train.jsonl" \
  --joint-checkpoint "$JOINT_CHECKPOINT" \
  --physical-checkpoint "$PHYSICAL_CHECKPOINT" \
  --v1-checkpoint "$V1_CHECKPOINT" \
  --endpoint "$ENDPOINT" \
  --out "$OUT"

sha256sum "$OUT"
echo "[sd-cst-complete-physical-record-bus-v1.1-audit] complete; no scored split was addressed"
