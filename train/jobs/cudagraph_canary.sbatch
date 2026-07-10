#!/bin/bash
#SBATCH -J shohin-cudagraph
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/cudagraph_canary_%j.out
#SBATCH --open-mode=append
# Whole-update CUDA graph speed canary. It reads a preserved checkpoint but writes no checkpoint and
# does not share the flagship output directory.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE/train"

SHARDS="$BASE/artifacts/shards/finemath4 $BASE/artifacts/shards/openwebmath $BASE/artifacts/shards/code_python $BASE/artifacts/shards/finemath3"
SRC_CKPT=${SRC_CKPT:-$BASE/train/flagship_out/best_step110000.pt}
OUT_DIR=${OUT_DIR:-$BASE/artifacts/cudagraph_canary/${SLURM_JOB_ID}}
BS=${BS:-32}
ACC=${ACC:-8}
STEPS=${STEPS:-80}

[ -f "$SRC_CKPT" ] || { echo "[cudagraph] missing source checkpoint: $SRC_CKPT" >&2; exit 2; }
echo "[cudagraph] node=${SLURMD_NODENAME:-?} src=$SRC_CKPT bs=$BS acc=$ACC steps=$STEPS"
nvidia-smi -L

$BASE/miniforge3/bin/python cudagraph_canary.py \
  --checkpoint "$SRC_CKPT" --shard-dirs $SHARDS --out "$OUT_DIR" \
  --batch-size "$BS" --grad-accum "$ACC" --timed-steps "$STEPS"
