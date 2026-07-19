#!/bin/bash
#SBATCH -J shohin-rlpc-islands
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/referential_literal_pointer_islands_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE"

CKPT=${CKPT:-$BASE/train/flagship_out/ckpt_0300000.pt}
DATA_DIR=${DATA_DIR:-$BASE/artifacts/r12/referential_literal_pointer_corpus_v1_1}
OUT_DIR=${OUT_DIR:-$BASE/train/referential_literal_pointer_compiler_v1_3_islands}
TRAIN=$DATA_DIR/train.jsonl
DEVELOPMENT=$DATA_DIR/development.jsonl
REPORT=$DATA_DIR/report.json
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
ADAPTER=$OUT_DIR/codon.pt
DEVELOPMENT_RESULT=$OUT_DIR/development.json

for path in "$CKPT" "$TRAIN" "$DEVELOPMENT" "$REPORT" "$TOKENIZER"; do
  [ -f "$path" ] || { echo "missing required input: $path" >&2; exit 2; }
done
[ ! -e "$ADAPTER" ] || { echo "refusing existing adapter: $ADAPTER" >&2; exit 2; }
[ ! -e "$DEVELOPMENT_RESULT" ] || { echo "refusing existing result: $DEVELOPMENT_RESULT" >&2; exit 2; }
mkdir -p "$OUT_DIR"

echo "[rlpc-islands] node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

$PYTHON train/train_referential_literal_pointer_compiler.py \
  --base "$CKPT" --data "$TRAIN" --report "$REPORT" --tokenizer "$TOKENIZER" --out "$ADAPTER" \
  --layer 19 --width 256 --heads 8 --decoder-layers 2 --ff 1024 \
  --encoder-layers 4 --role-supervision --role-weight 0.5 --separate-kind-decoder \
  --batch-size 64 --epochs 1 --lr 0.001 --warmup 50 --clip 1.0 --kind-weight 1.0 \
  --seed 2026071805

$PYTHON train/eval_referential_literal_pointer_compiler.py \
  --base "$CKPT" --adapter "$ADAPTER" --data "$DEVELOPMENT" --report "$REPORT" \
  --tokenizer "$TOKENIZER" --split development --out "$DEVELOPMENT_RESULT" --batch-size 64

echo "[rlpc-islands] exposed-development diagnostic complete; confirmation remains sealed"
