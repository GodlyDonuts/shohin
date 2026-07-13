#!/bin/bash
# Real H100 allocation gate for isolated experiments. This never touches a
# checkpoint, model-output directory, or training data.
#SBATCH -J shohin-cuda-preflight
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/cuda_h100_preflight_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=2

echo "[cuda-preflight] host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PY" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("[cuda-preflight] CUDA unavailable after Slurm allocation")
device = torch.device("cuda")
x = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
if not torch.isfinite(y).all().item():
    raise SystemExit("[cuda-preflight] non-finite CUDA matmul")
print(
    "[cuda-preflight] pass "
    f"device={torch.cuda.get_device_name(device)} "
    f"allocated={torch.cuda.memory_allocated(device)}"
)
PY
