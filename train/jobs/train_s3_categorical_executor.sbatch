#!/bin/bash
#SBATCH -J shohin-s3-reg
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/s3_categorical_executor_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python
DATA=$BASE/artifacts/r12/referential_literal_pointer_factorized_v1
DEPTH=$BASE/artifacts/r12/rgde_relational_development_v1
OUT_DIR=${OUT_DIR:-$BASE/train/s3_categorical_executor_2026071903}
EXECUTOR=$OUT_DIR/executor.pt
CKPT=$BASE/train/flagship_out/ckpt_0300000.pt
COMPILER=$BASE/train/referential_literal_pointer_factorized_ordinary_2026071810/compiler.pt
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
REPORT=$DATA/development_report.json
EXPECTED_COMPILER_SHA256=747a559b827c6d114943c091b9dea5b4b90cef7af13aa5003b8435c092d24991
cd "$BASE"
for path in "$CKPT" "$COMPILER" "$TOKENIZER" "$REPORT" "$DATA/train.jsonl" \
  "$DATA/development_compositional.jsonl" "$DATA/development_lexical_ood.jsonl" \
  "$DEPTH/development_relational.jsonl" "$DEPTH/report.json"; do
  [ -f "$path" ] || { echo "missing S3 input: $path" >&2; exit 2; }
done
[ ! -e "$OUT_DIR" ] || { echo "refusing existing S3 output: $OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR"
nvidia-smi -L

"$PYTHON" train/train_s3_categorical_executor.py \
  --base "$CKPT" --compiler "$COMPILER" \
  --expected-compiler-sha256 "$EXPECTED_COMPILER_SHA256" \
  --data "$DATA/train.jsonl" --report "$REPORT" --tokenizer "$TOKENIZER" \
  --out "$EXECUTOR" --width 192 --batch-size 64 --epochs 1 \
  --lr 0.001 --warmup 50 --clip 1.0 --seed 2026071903

evaluate_two() {
  local split=$1
  local mode=$2
  local suffix=$3
  "$PYTHON" train/eval_s3_categorical_executor.py \
    --base "$CKPT" --compiler "$COMPILER" --executor "$EXECUTOR" \
    --data "$DATA/${split}.jsonl" --report "$REPORT" --tokenizer "$TOKENIZER" \
    --split "$split" --identity-mode "$mode" \
    --out "$OUT_DIR/${suffix}.json" --batch-size 64
}

evaluate_depth() {
  local mode=$1
  "$PYTHON" train/eval_s3_categorical_depth.py \
    --base "$CKPT" --compiler "$COMPILER" --executor "$EXECUTOR" \
    --board "$DEPTH/development_relational.jsonl" --report "$DEPTH/report.json" \
    --tokenizer "$TOKENIZER" --identity-mode "$mode" \
    --out "$OUT_DIR/depth_${mode}.json" --batch-size 64
}

for mode in mean ordered gold; do
  evaluate_two development_compositional "$mode" "comp_${mode}"
done
evaluate_two development_lexical_ood mean lexical_mean
for mode in mean ordered gold; do
  evaluate_depth "$mode"
done

"$PYTHON" train/assess_s3_categorical_executor.py \
  --executor "$EXECUTOR" \
  --comp-mean "$OUT_DIR/comp_mean.json" \
  --comp-ordered "$OUT_DIR/comp_ordered.json" \
  --comp-gold "$OUT_DIR/comp_gold.json" \
  --lexical-mean "$OUT_DIR/lexical_mean.json" \
  --depth-mean "$OUT_DIR/depth_mean.json" \
  --depth-ordered "$OUT_DIR/depth_ordered.json" \
  --depth-gold "$OUT_DIR/depth_gold.json" \
  --out "$OUT_DIR/assessment.json"

echo "[s3-register] public development complete; confirmation was never referenced"
