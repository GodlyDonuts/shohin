#!/bin/bash
#SBATCH -J shohin-rgde
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/referential_gather_delete_executor_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE"

ARM=${ARM:?set ARM to tied, untied, tied_gold, or source_retained}
SEED=${SEED:-2026071901}
CKPT=${CKPT:-$BASE/train/flagship_out/ckpt_0300000.pt}
COMPILER=${COMPILER:-$BASE/train/referential_literal_pointer_factorized_ordinary_2026071810/compiler.pt}
EXPECTED_COMPILER_SHA256=747a559b827c6d114943c091b9dea5b4b90cef7af13aa5003b8435c092d24991
DATA_DIR=${DATA_DIR:-$BASE/artifacts/r12/referential_literal_pointer_factorized_v1}
TRAIN=$DATA_DIR/train.jsonl
DEVELOPMENT=$DATA_DIR/development_compositional.jsonl
REPORT=$DATA_DIR/development_report.json
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
OUT_DIR=${OUT_DIR:-$BASE/train/referential_gather_delete_executor_${ARM}_${SEED}}
EXECUTOR=$OUT_DIR/executor.pt

for path in "$CKPT" "$COMPILER" "$TRAIN" "$DEVELOPMENT" "$REPORT" "$TOKENIZER"; do
  [ -f "$path" ] || { echo "missing required input: $path" >&2; exit 2; }
done
[ "$(sha256sum "$COMPILER" | awk '{print $1}')" = "$EXPECTED_COMPILER_SHA256" ] || {
  echo "qualified compiler hash mismatch" >&2
  exit 2
}
[ ! -e "$OUT_DIR" ] || { echo "refusing existing output directory: $OUT_DIR" >&2; exit 2; }

case "$ARM" in
  tied)
    TRAIN_ARM=tied
    TRAIN_ORACLE=none
    ;;
  untied)
    TRAIN_ARM=untied
    TRAIN_ORACLE=none
    ;;
  tied_gold)
    TRAIN_ARM=tied
    TRAIN_ORACLE=full
    ;;
  source_retained)
    TRAIN_ARM=source_retained
    TRAIN_ORACLE=none
    ;;
  *)
    echo "unknown ARM=$ARM" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_DIR"
echo "[rgde] arm=$ARM seed=$SEED node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

"$PYTHON" train/train_referential_gather_delete_executor.py \
  --base "$CKPT" --compiler "$COMPILER" \
  --expected-compiler-sha256 "$EXPECTED_COMPILER_SHA256" \
  --data "$TRAIN" --report "$REPORT" --tokenizer "$TOKENIZER" \
  --out "$EXECUTOR" --arm "$TRAIN_ARM" --packet-oracle "$TRAIN_ORACLE" \
  --width 192 --batch-size 64 --epochs 1 --lr 0.001 --warmup 50 \
  --clip 1.0 --seed "$SEED"

evaluate() {
  local suffix=$1
  shift
  "$PYTHON" train/eval_referential_gather_delete_executor.py \
    --base "$CKPT" --compiler "$COMPILER" --executor "$EXECUTOR" \
    --data "$DEVELOPMENT" --report "$REPORT" --tokenizer "$TOKENIZER" \
    --split development_compositional --out "$OUT_DIR/development_${suffix}.json" \
    --batch-size 64 "$@"
}

case "$ARM" in
  tied)
    evaluate predicted --packet-oracle none
    evaluate predicted_shuffled --packet-oracle none --shuffle-operations
    evaluate query_shuffled --packet-oracle none --shuffle-query
    evaluate gold_packet --packet-oracle full
    ;;
  untied)
    evaluate predicted --packet-oracle none
    ;;
  tied_gold)
    evaluate gold_packet --packet-oracle full
    ;;
  source_retained)
    evaluate source_retained
    ;;
esac

echo "[rgde] Stage-B development diagnostics complete; confirmation was never referenced"
