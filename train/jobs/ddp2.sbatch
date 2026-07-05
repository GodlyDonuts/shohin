#!/bin/bash
#SBATCH -J shohin-ddp2
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ddp2_%j.out
# Dual-H100 DDP run. Params via env (SIZE/STEPS/BS/ACC/WARMUP/CKPT/OUT), defaults verify DDP.
set -uo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE/train
SHARDS="$BASE/artifacts/shards/finemath4 $BASE/artifacts/shards/openwebmath $BASE/artifacts/shards/code_python"
$BASE/miniforge3/bin/torchrun --standalone --nproc_per_node=2 train.py \
  --size ${SIZE:-mame} --shard-dirs $SHARDS \
  --steps ${STEPS:-400} --batch-size ${BS:-16} --grad-accum ${ACC:-1} \
  --warmup ${WARMUP:-50} --log-every 20 --ckpt-every ${CKPT:-0} --out $BASE/train/${OUT:-ddp2_out}
echo "[job] ddp2 done."
