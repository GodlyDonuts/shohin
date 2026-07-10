#!/bin/bash
#SBATCH -J shohin-step-profile
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/profile_step_%j.out
#SBATCH --open-mode=append
# Profile the exact compiled BS32/ACC8 pretraining update on an isolated H100. This writes no
# training checkpoints and never shares the flagship output directory.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE/train"

SHARDS="$BASE/artifacts/shards/finemath4 $BASE/artifacts/shards/openwebmath $BASE/artifacts/shards/code_python $BASE/artifacts/shards/finemath3"
SRC_CKPT=${SRC_CKPT:-$BASE/train/flagship_out/best_step110000.pt}
OUT_DIR=${OUT_DIR:-$BASE/artifacts/profiles/step_${SLURM_JOB_ID}}
BS=${BS:-32}
ACC=${ACC:-8}
COMPILE_MODE=${COMPILE_MODE:-default}

[ -f "$SRC_CKPT" ] || { echo "[profile] missing source checkpoint: $SRC_CKPT" >&2; exit 2; }
echo "[profile] node=${SLURMD_NODENAME:-?} src=$SRC_CKPT bs=$BS acc=$ACC compile_mode=$COMPILE_MODE"
nvidia-smi -L

$BASE/miniforge3/bin/python profile_step.py \
  --checkpoint "$SRC_CKPT" --shard-dirs $SHARDS --out "$OUT_DIR" \
  --batch-size "$BS" --grad-accum "$ACC" --compile-mode "$COMPILE_MODE"
echo "[profile] complete: $OUT_DIR"
