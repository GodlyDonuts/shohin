#!/bin/bash
#SBATCH -J shohin-rlpc-factor
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/referential_literal_pointer_factorized_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE"

ARM=${ARM:?set ARM to islands, ordinary, structured, or free_slots}
SEED=${SEED:-2026071810}
CKPT=${CKPT:-$BASE/train/flagship_out/ckpt_0300000.pt}
DATA_DIR=${DATA_DIR:-$BASE/artifacts/r12/referential_literal_pointer_factorized_v1}
OUT_DIR=${OUT_DIR:-$BASE/train/referential_literal_pointer_factorized_${ARM}_${SEED}}
TRAIN=$DATA_DIR/train.jsonl
COMPOSITIONAL=$DATA_DIR/development_compositional.jsonl
LEXICAL_OOD=$DATA_DIR/development_lexical_ood.jsonl
REPORT=$DATA_DIR/development_report.json
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
ADAPTER=$OUT_DIR/compiler.pt
COMPOSITIONAL_RESULT=$OUT_DIR/development_compositional.json
LEXICAL_RESULT=$OUT_DIR/development_lexical_ood.json

for path in "$CKPT" "$TRAIN" "$COMPOSITIONAL" "$LEXICAL_OOD" "$REPORT" "$TOKENIZER"; do
  [ -f "$path" ] || { echo "missing required input: $path" >&2; exit 2; }
done
for path in "$ADAPTER" "$COMPOSITIONAL_RESULT" "$LEXICAL_RESULT"; do
  [ ! -e "$path" ] || { echo "refusing existing output: $path" >&2; exit 2; }
done

case "$ARM" in
  islands)
    ARM_ARGS=(
      --layer 19 --width 256 --heads 8 --decoder-layers 2 --ff 1024
      --encoder-layers 4 --role-supervision --role-weight 0.5
      --separate-kind-decoder
    )
    ;;
  ordinary)
    ARM_ARGS=(
      --layer 19 --width 384 --heads 8 --decoder-layers 2 --ff 1408
      --encoder-layers 5 --role-supervision --role-weight 0.5
      --ordinary-tagger
    )
    ;;
  structured)
    ARM_ARGS=(
      --layer 19 --width 256 --heads 8 --decoder-layers 2 --ff 1024
      --encoder-layers 4 --role-supervision --role-weight 0.5
    )
    ;;
  free_slots)
    ARM_ARGS=(--layer 19 --width 256 --heads 8 --decoder-layers 2 --ff 1024)
    ;;
  *)
    echo "unknown ARM=$ARM" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_DIR"
echo "[rlpc-factorized] arm=$ARM seed=$SEED node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

"$PYTHON" train/train_referential_literal_pointer_compiler.py \
  --base "$CKPT" --data "$TRAIN" --report "$REPORT" --tokenizer "$TOKENIZER" \
  --out "$ADAPTER" "${ARM_ARGS[@]}" \
  --batch-size 64 --epochs 1 --lr 0.001 --warmup 50 --clip 1.0 \
  --kind-weight 1.0 --seed "$SEED"

"$PYTHON" train/eval_referential_literal_pointer_compiler.py \
  --base "$CKPT" --adapter "$ADAPTER" --data "$COMPOSITIONAL" --report "$REPORT" \
  --tokenizer "$TOKENIZER" --split development_compositional \
  --out "$COMPOSITIONAL_RESULT" --batch-size 64

"$PYTHON" train/eval_referential_literal_pointer_compiler.py \
  --base "$CKPT" --adapter "$ADAPTER" --data "$LEXICAL_OOD" --report "$REPORT" \
  --tokenizer "$TOKENIZER" --split development_lexical_ood \
  --out "$LEXICAL_RESULT" --batch-size 64

echo "[rlpc-factorized] development diagnostics complete; confirmation path was never referenced"
