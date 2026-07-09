#!/bin/bash
#SBATCH -J shohin-ddp2-canary
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ddp2_canary_%j.out
#SBATCH --open-mode=append
# Short two-H100 flagship branch. Default is a safe 60k fresh-opt replay with the SAME global batch
# as the single-GPU run: 1*16*16 == 2*16*8 sequences/update. This tests speed/stability without
# touching train/flagship_out or interrupting the live single-GPU flagship.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE/train

SHARDS="$BASE/artifacts/shards/finemath4 $BASE/artifacts/shards/openwebmath $BASE/artifacts/shards/code_python $BASE/artifacts/shards/finemath3"
# A numbered live checkpoint may have aged out of trainer retention. Prefer a preserved
# best_step checkpoint and copy it under a ckpt_ name because train.py resumes only that glob.
SRC_CKPT=${SRC_CKPT:-$BASE/train/flagship_out/best_step100000.pt}
OUT_DIR=${OUT_DIR:-$BASE/train/ddp2_canary_60000}
NG=${NG:-2}
BS=${BS:-16}
ACC=${ACC:-8}
STEPS=${STEPS:-100240}
LRTOTAL=${LRTOTAL:-300000}
WARMUP=${WARMUP:-2000}
LRMUON=${LRMUON:-0.005}
LRADAM=${LRADAM:-1e-3}
DSEED=${DSEED:-777}
GNORMMULT=${GNORMMULT:-8}
LOGEVERY=${LOGEVERY:-10}
CKPT=${CKPT:-1000}
MAX_RETRIES=${MAX_RETRIES:-2}

mkdir -p "$OUT_DIR"
if ! ls "$OUT_DIR"/ckpt_[0-9]*.pt >/dev/null 2>&1; then
  [ -f "$SRC_CKPT" ] || { echo "[canary] missing source checkpoint: $SRC_CKPT" >&2; exit 2; }
  SEED_STEP=$(basename "$SRC_CKPT" | tr -cd '0-9')
  [ -n "$SEED_STEP" ] || { echo "[canary] cannot infer step from: $SRC_CKPT" >&2; exit 2; }
  SEED_PATH="$OUT_DIR/ckpt_$(printf '%07d' "$((10#$SEED_STEP))").pt"
  echo "[canary] seeding $OUT_DIR from $SRC_CKPT"
  cp "$SRC_CKPT" "$SEED_PATH"
fi

EXTRA=""
if [ "${FRESH_OPT:-1}" = "1" ]; then
  EXTRA="$EXTRA --fresh-opt"
fi
if [ "${COMPILE:-1}" = "1" ]; then
  EXTRA="$EXTRA --compile"
fi
if [ -n "${NOMUON:-}" ]; then
  EXTRA="$EXTRA --no-muon"
fi

echo "[canary] node=${SLURMD_NODENAME:-?} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[canary] src=$SRC_CKPT out=$OUT_DIR ng=$NG bs=$BS acc=$ACC steps=$STEPS lr_total=$LRTOTAL warmup=$WARMUP fresh_opt=${FRESH_OPT:-1} compile=${COMPILE:-1}"
echo "[canary] effective batch sequences/update = $((NG * BS * ACC))"
nvidia-smi -L 2>&1 | head

RUN="$BASE/miniforge3/bin/torchrun --standalone --nproc_per_node=$NG train.py"
ARGS="--size shohin --shard-dirs $SHARDS --steps $STEPS --batch-size $BS --grad-accum $ACC \
  --lr-total-steps $LRTOTAL --warmup $WARMUP --lr-muon $LRMUON --lr-adam $LRADAM --log-every $LOGEVERY --ckpt-every $CKPT \
  --data-seed $DSEED --gnorm-mult $GNORMMULT --out $OUT_DIR --resume $EXTRA"

for att in $(seq 1 "$MAX_RETRIES"); do
  $RUN $ARGS && break
  echo "[canary] attempt $att failed; retry in 30s"
  sleep 30
done
echo "[canary] ddp2 canary block done."
