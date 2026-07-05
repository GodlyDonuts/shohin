#!/bin/bash
#SBATCH -J shohin-flagship
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 3-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/flagship_%j.out
#SBATCH --open-mode=append
# Shohin-135M baseline pretrain on the decontaminated math+code core.
# Resumable + checkpointed -> scale to more GPUs (NG=) or recover from preemption without loss.
set -uo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE/train
SHARDS="$BASE/artifacts/shards/finemath4 $BASE/artifacts/shards/openwebmath $BASE/artifacts/shards/code_python"
NG=${NG:-1}
ARGS="--size shohin --shard-dirs $SHARDS --steps ${STEPS:-60000} --batch-size ${BS:-16} \
  --grad-accum ${ACC:-16} --warmup 2000 --lr-muon 0.02 --lr-adam 3e-3 \
  --log-every 10 --ckpt-every ${CKPT:-1000} --out $BASE/train/flagship_out --resume --compile"
if [ "$NG" = "1" ]; then
  $BASE/miniforge3/bin/python train.py $ARGS          # plain python: clean torch.compile on 1 GPU
else
  $BASE/miniforge3/bin/torchrun --standalone --nproc_per_node=$NG train.py $ARGS
fi
echo "[job] flagship block done (resubmit with --resume to continue)."
