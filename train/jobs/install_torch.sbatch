#!/bin/bash
#SBATCH -J shohin-torch
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 00:40:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/torch_%j.out
# One-time: install CUDA PyTorch into the shared miniforge env. CPU-only, single core.
set -uo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
$PY -m pip install --no-input torch --index-url https://download.pytorch.org/whl/cu124
$PY -c "import torch;print('TORCH',torch.__version__,'cuda_build',torch.version.cuda)"
echo "[job] torch install done."
