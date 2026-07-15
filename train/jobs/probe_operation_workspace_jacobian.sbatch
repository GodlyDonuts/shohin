#!/bin/bash
# Isolated, read-only raw-260k diagnostic. This file never submits another job,
# writes a checkpoint, invokes training, or shares a production output path.
#SBATCH -J shohin-op-jlens
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/operation_workspace_jacobian_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=${CKPT:-$BASE/train/flagship_out/ckpt_0260000.pt}
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
PROBE=$BASE/train/probe_operation_workspace_jacobian.py
PREREG=$BASE/R12_OPERATION_WORKSPACE_JACOBIAN_PREREG.md
OUT=${OUT:?set OUT to a fresh isolated operation-workspace result JSON}

CHECKPOINT_SHA256=91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d
TOKENIZER_SHA256=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
PROBE_SHA256=c7a8b1626604c8b7c2eeb4fc065fcc7076c84c0e98cb83064f39bbecffd5c95c
PREREG_SHA256=7ba947ee2cfb161448e840931bcd525fe4d55bc216ccf40f4e10e8dba7352eb7

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTHONHASHSEED=0

check_sha256() {
  expected=$1
  path=$2
  [ -s "$path" ] || { echo "[operation-jacobian] missing input: $path" >&2; exit 2; }
  observed=$(sha256sum "$path" | awk '{print $1}')
  [ "$observed" = "$expected" ] || {
    echo "[operation-jacobian] SHA-256 mismatch for $path: $observed" >&2
    exit 3
  }
}

check_sha256 "$CHECKPOINT_SHA256" "$CKPT"
check_sha256 "$TOKENIZER_SHA256" "$TOKENIZER"
check_sha256 "$PROBE_SHA256" "$PROBE"
check_sha256 "$PREREG_SHA256" "$PREREG"
[ ! -e "$OUT" ] || { echo "[operation-jacobian] refusing existing output: $OUT" >&2; exit 4; }
case "$OUT" in
  "$BASE"/artifacts/diagnostics/operation_workspace_jacobian_*.json) ;;
  *) echo "[operation-jacobian] OUT must be a fresh isolated diagnostic JSON" >&2; exit 5 ;;
esac

cd "$BASE/train"
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[operation-jacobian] CUDA unavailable after allocation")
tensor = torch.empty(1, device="cuda")
if tensor.device.type != "cuda":
    raise SystemExit("[operation-jacobian] CUDA allocation did not reach the GPU")
PY

echo "[operation-jacobian] ckpt=$CKPT out=$OUT"
"$PY" "$PROBE" \
  --ckpt "$CKPT" \
  --tokenizer "$TOKENIZER" \
  --out "$OUT" \
  --device cuda
echo "[operation-jacobian] complete"
