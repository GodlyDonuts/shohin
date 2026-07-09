#!/bin/bash
#SBATCH -J shohin-bs-canary
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/microbatch_canary_%j.out
#SBATCH --open-mode=append
# Single-H100 side branch to test larger per-GPU microbatches without touching
# the live flagship output directory. Keep NG*BS*ACC fixed unless deliberately
# testing a global-batch change.
set -uo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE/train"

SHARDS="$BASE/artifacts/shards/finemath4 $BASE/artifacts/shards/openwebmath $BASE/artifacts/shards/code_python $BASE/artifacts/shards/finemath3"
SRC_CKPT=${SRC_CKPT:-$BASE/train/flagship_out/ckpt_0100000.pt}
OUT_DIR=${OUT_DIR:-$BASE/train/microbatch_canary_100k_bs32_acc8}
NG=${NG:-1}
BS=${BS:-32}
ACC=${ACC:-8}
STEPS=${STEPS:-100300}
LRTOTAL=${LRTOTAL:-300000}
WARMUP=${WARMUP:-2000}
LRMUON=${LRMUON:-0.005}
LRADAM=${LRADAM:-1e-3}
DSEED=${DSEED:-777}
GNORMMULT=${GNORMMULT:-8}
LOGEVERY=${LOGEVERY:-10}
CKPT=${CKPT:-0}

mkdir -p "$OUT_DIR"
if ! ls "$OUT_DIR"/ckpt_[0-9]*.pt >/dev/null 2>&1; then
  echo "[mb-canary] seeding $OUT_DIR from $SRC_CKPT"
  cp "$SRC_CKPT" "$OUT_DIR/$(basename "$SRC_CKPT")"
fi

echo "[mb-canary] node=${SLURMD_NODENAME:-?} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[mb-canary] src=$SRC_CKPT out=$OUT_DIR ng=$NG bs=$BS acc=$ACC steps=$STEPS lr_total=$LRTOTAL compile=${COMPILE:-1}"
echo "[mb-canary] effective batch tokens/update = $((NG * BS * ACC * 2048))"
nvidia-smi -L 2>&1 | head

EXTRA=""
if [ "${COMPILE:-1}" = "1" ]; then
  EXTRA="$EXTRA --compile"
fi
if [ -n "${NOMUON:-}" ]; then
  EXTRA="$EXTRA --no-muon"
fi

RUN="$BASE/miniforge3/bin/python train.py"
if [ "$NG" != "1" ]; then
  RUN="$BASE/miniforge3/bin/torchrun --standalone --nproc_per_node=$NG train.py"
fi
ARGS="--size shohin --shard-dirs $SHARDS --steps $STEPS --batch-size $BS --grad-accum $ACC \
  --lr-total-steps $LRTOTAL --warmup $WARMUP --lr-muon $LRMUON --lr-adam $LRADAM --log-every $LOGEVERY --ckpt-every $CKPT \
  --data-seed $DSEED --gnorm-mult $GNORMMULT --out $OUT_DIR --resume $EXTRA"

$RUN $ARGS
echo "[mb-canary] microbatch canary done."
