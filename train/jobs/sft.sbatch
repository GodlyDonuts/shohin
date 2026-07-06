#!/bin/bash
#SBATCH -J shohin-sft
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_%j.out
#SBATCH --open-mode=append
# Reasoning SFT: completion-masked fine-tune of the pretrained model on the verified concise-CoT set.
# INIT defaults to the newest pretrain checkpoint; override with INIT=<path>. Single-GPU (2xH100 is
# broken on shared nodes). After it finishes, eval sft_out/sft_ep*.pt with eval_suite.py + eval_code.py.
set -uo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE/train
INIT=${INIT:-$(ls -t $BASE/train/flagship_out/ckpt_[0-9]*.pt 2>/dev/null | head -1)}
# Verified + eval-aligned math (openmath2.jsonl) + procedural-CoT diversity (rgym.jsonl). NOT the
# older *_concise_2M sets (not verified / not eval-aligned) — quality over quantity for a 135M first SFT.
DATA=${DATA:-$BASE/artifacts/sft/openmath2.jsonl $BASE/artifacts/sft/rgym.jsonl}
echo "[sft-job] init=$INIT data=$DATA"
[ -z "$INIT" ] && { echo "[sft-job] no init checkpoint found"; exit 1; }

# Wait for the allocated GPU(s) to be free (avoid the device-busy race after a just-freed job).
for i in $(seq 1 30); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1)
  [ "${U:-9999}" -lt 3000 ] 2>/dev/null && break
  echo "[sft-job] GPU busy (max ${U:-?} MiB), waiting..."; sleep 10
done

ARGS="--init $INIT --data $DATA --tokenizer $BASE/artifacts/shohin-tok-32k.json \
  --epochs ${EPOCHS:-3} --batch-size ${BS:-16} --lr-muon ${LRMUON:-2e-3} --lr-adam ${LRADAM:-5e-4} \
  --compile --out $BASE/train/sft_out"
for att in 1 2 3; do
  $BASE/miniforge3/bin/python sft.py $ARGS && break
  echo "[sft-job] attempt $att failed (CUDA-busy race?); retry in 30s"; sleep 30
done
echo "[sft-job] done. checkpoints in $BASE/train/sft_out/"
