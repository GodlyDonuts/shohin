#!/bin/bash
#SBATCH --job-name=sd-cst-joint-audit
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=evc26,evc28,evc31,evc44
#SBATCH --chdir=/lustre/fs1/home/sa305415/shohin
#SBATCH --output=logs/sd_cst_renderer_native_joint_audit_%j.out

set -euo pipefail

CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to the exact committed source capsule}
DATA=${DATA:?set DATA to the consumed projected-v2 board}
ORBIT_CHECKPOINT=${ORBIT_CHECKPOINT:?set ORBIT_CHECKPOINT to exact v1.2 checkpoint}
JOINT_CHECKPOINT=${JOINT_CHECKPOINT:?set JOINT_CHECKPOINT to exact joint checkpoint}
OUT=${OUT:?set OUT to a new post-hoc diagnostic report path}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact source commit}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

cd "$CODE_ROOT"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
git diff --quiet
git diff --cached --quiet
for path in \
  train/audit_sd_cst_renderer_native_joint.py \
  train/sd_cst_renderer_native_program.py \
  train/jobs/sd_cst_renderer_native_joint_audit.sbatch; do
  git cat-file -e "$SOURCE_COMMIT:$path"
done
for path in "$DATA/train.jsonl" "$ORBIT_CHECKPOINT" "$JOINT_CHECKPOINT"; do
  test -f "$path" || { echo "missing joint diagnostic input: $path" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "refusing existing joint diagnostic output: $OUT" >&2; exit 2; }

export PYTHONPATH="$CODE_ROOT/pipeline:$CODE_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("joint diagnostic requires bf16 CUDA")
print({"cuda": True, "device": torch.cuda.get_device_name(), "bf16": True})
PY

"$PYTHON" train/audit_sd_cst_renderer_native_joint.py \
  --train-jsonl "$DATA/train.jsonl" \
  --orbit-checkpoint "$ORBIT_CHECKPOINT" \
  --joint-checkpoint "$JOINT_CHECKPOINT" \
  --out "$OUT"

sha256sum "$OUT"
echo "[sd-cst-renderer-native-joint-audit] complete; no scored split was addressed"
