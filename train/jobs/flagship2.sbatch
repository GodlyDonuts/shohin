#!/bin/bash
#SBATCH -J shohin-flagship2
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 3-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/flagship2_%j.out
#SBATCH --open-mode=append
# Two-H100 flagship continuation. This preserves the 1-GPU global batch by default:
# 1*BS16*ACC16 == 2*BS16*ACC8 sequences/update. Use after a clean DDP canary only.
# Keep CPU modest so the GPU request, not CPU hoarding, controls scheduling.
set -uo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE/train

SHARDS="$BASE/artifacts/shards/finemath4 $BASE/artifacts/shards/openwebmath $BASE/artifacts/shards/code_python $BASE/artifacts/shards/finemath3"
NG=${NG:-2}
BS=${BS:-16}
ACC=${ACC:-8}
STEPS=${STEPS:-300000}
WARMUP=${WARMUP:-2000}
LRMUON=${LRMUON:-0.005}
LRADAM=${LRADAM:-1e-3}
DSEED=${DSEED:-777}
GNORMMULT=${GNORMMULT:-8}
LOGEVERY=${LOGEVERY:-10}
CKPT=${CKPT:-1000}

ARGS="--size shohin --shard-dirs $SHARDS --steps $STEPS --batch-size $BS \
  --grad-accum $ACC --warmup $WARMUP --lr-muon $LRMUON --lr-adam $LRADAM \
  --log-every $LOGEVERY --ckpt-every $CKPT --data-seed $DSEED --gnorm-mult $GNORMMULT \
  ${FRESH_OPT:+--fresh-opt} ${NOMUON:+--no-muon} --out $BASE/train/flagship_out --resume --compile"

for i in $(seq 1 30); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1)
  [ "${U:-9999}" -lt 3000 ] 2>/dev/null && break
  echo "[job2] GPU busy (max ${U:-?} MiB used), waiting..."
  sleep 10
done

echo "[job2] node=${SLURMD_NODENAME:-?} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-?} SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[job2] ng=$NG bs=$BS acc=$ACC effective_batch_sequences=$((NG * BS * ACC)) steps=$STEPS lr_muon=$LRMUON lr_adam=$LRADAM dseed=$DSEED fresh_opt=${FRESH_OPT:-0}"
nvidia-smi -L 2>&1 | head

RUN="$BASE/miniforge3/bin/torchrun --standalone --nproc_per_node=$NG train.py"
for att in 1 2 3 4; do
  $RUN $ARGS && break
  echo "[job2] attempt $att failed; retry in 30s"
  sleep 30
done

MYLOG=$BASE/logs/flagship2_${SLURM_JOB_ID}.out
BAD=$BASE/train/bad_nodes.txt
if grep -qE "^step " "$MYLOG" 2>/dev/null; then
  : > "$BAD"
else
  echo "${SLURMD_NODENAME:-unknown}" >> "$BAD"
  EXCL=$(sort -u "$BAD" | paste -sd, -)
  if [ "${AUTO_REQUEUE:-1}" != "1" ]; then
    echo "[job2] no training progress on ${SLURMD_NODENAME}; AUTO_REQUEUE disabled."
  elif [ "$(sort -u "$BAD" | wc -l)" -lt 12 ]; then
    echo "[job2] no training progress on ${SLURMD_NODENAME}; auto-requeue excluding: $EXCL"
    sbatch --exclude="$EXCL" "$BASE/train/jobs/flagship2.sbatch"
  else
    echo "[job2] too many bad nodes; stopping auto-requeue."
  fi
fi
echo "[job2] flagship2 block done."
