#!/bin/bash
#SBATCH -J shohin-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/eval_%j.out
# Full reasoning-benchmark board on one checkpoint. CKPT defaults to the newest SFT epoch (or set
# CKPT=<path> to eval a base/pretrain ckpt). GSM8K uses self-consistency (maj@k) for the headline
# number; the rest are pass@1. sft_ep*.pt / ckpt_*.pt both carry {model,cfg} so eval loads directly.
set -uo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE/train
CKPT=${CKPT:-$(ls -t $BASE/train/sft_out/sft_ep*.pt 2>/dev/null | head -1)}
[ -z "$CKPT" ] && CKPT=$(ls -t $BASE/train/flagship_out/ckpt_[0-9]*.pt 2>/dev/null | head -1)
TOK=$BASE/artifacts/shohin-tok-32k.json
E=$BASE/artifacts/evals
PY=$BASE/miniforge3/bin/python
echo "[eval-job] ckpt=$CKPT"
for i in $(seq 1 30); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1)
  [ "${U:-9999}" -lt 3000 ] 2>/dev/null && break; sleep 10
done
echo "===== SHOHIN BENCHMARK BOARD  ckpt=$(basename $CKPT) ====="
$PY eval_suite.py --ckpt $CKPT --tokenizer $TOK --task gsm8k    --data $E/gsm8k.jsonl    --n ${N:-200} --k ${K:-8} --temp 0.8
$PY eval_suite.py --ckpt $CKPT --tokenizer $TOK --task gsm8k    --data $E/gsm8k.jsonl    --n ${N:-200} --k 1
$PY eval_suite.py --ckpt $CKPT --tokenizer $TOK --task math500  --data $E/math500.jsonl  --n ${N:-200} --k 1
$PY eval_code.py  --ckpt $CKPT --tokenizer $TOK --task humaneval --data $E/humaneval_full.jsonl --n 164 --k 1
$PY eval_code.py  --ckpt $CKPT --tokenizer $TOK --task mbpp      --data $E/mbpp_full.jsonl      --n ${N:-200} --k 1
echo "===== board done ====="
