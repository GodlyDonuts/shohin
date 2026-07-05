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
# Wait for the allocated GPU to be free (avoid 'device busy' race after a just-cancelled job).
for i in $(seq 1 24); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  [ "${U:-9999}" -lt 3000 ] 2>/dev/null && break
  echo "[job] GPU busy (${U:-?} MiB used), waiting..."; sleep 10
done
if [ "$NG" = "1" ]; then
  for att in 1 2 3; do   # retry belt-and-suspenders for transient CUDA-busy
    $BASE/miniforge3/bin/python train.py $ARGS && break
    echo "[job] attempt $att failed; retry in 30s"; sleep 30
  done
else
  $BASE/miniforge3/bin/torchrun --standalone --nproc_per_node=$NG train.py $ARGS
fi

# Self-heal: if this run logged no training step, it failed at CUDA init on this node -> record the
# node and auto-resubmit excluding all known-bad nodes (bounded), so it hops to a clean GPU on its own.
MYLOG=$BASE/logs/flagship_${SLURM_JOB_ID}.out
BAD=$BASE/train/bad_nodes.txt
if grep -qE "^step " "$MYLOG" 2>/dev/null; then
  : > "$BAD"   # trained fine -> clear the bad-node list
else
  echo "${SLURMD_NODENAME:-unknown}" >> "$BAD"
  EXCL=$(sort -u "$BAD" | paste -sd, -)
  if [ "$(sort -u "$BAD" | wc -l)" -lt 12 ]; then
    echo "[job] no training progress on ${SLURMD_NODENAME}; auto-requeue excluding: $EXCL"
    sbatch --exclude="$EXCL" "$BASE/train/jobs/flagship.sbatch"
  else
    echo "[job] too many bad nodes; stopping auto-requeue."
  fi
fi
echo "[job] flagship block done."
