#!/bin/bash
#SBATCH -J shohin-smoke
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/smoke_%j.out
# Verify the whole pipeline on 1 H100: torch+CUDA, model fwd/bwd, Muon, shard loader, loss decreases.
set -uo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
cd $BASE/train
$PY train.py --size tiny \
  --shard-dirs $BASE/artifacts/shards/finemath4 $BASE/artifacts/shards/openwebmath $BASE/artifacts/shards/code_python \
  --steps 60 --batch-size 8 --warmup 10 --log-every 5 --ckpt-every 0 --out $BASE/train/smoke_out
echo "[job] smoke done."
