#!/bin/bash
#SBATCH -J shohin-s3-anchor
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/s3_pointer_anchor_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python
DATA=$BASE/artifacts/r12/referential_literal_pointer_factorized_v1
DEPTH=$BASE/artifacts/r12/rgde_relational_development_v1
OUT_DIR=${OUT_DIR:-$BASE/train/s3_pointer_anchor_2026071903}
EXECUTOR=$BASE/train/s3_equivariant_executor_2026071903/executor.pt
LEXICON=$BASE/train/s3_lexical_action_2026071903/kind_lexicon.json
CKPT=$BASE/train/flagship_out/ckpt_0300000.pt
COMPILER=$BASE/train/referential_literal_pointer_factorized_ordinary_2026071810/compiler.pt
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
REPORT=$DATA/development_report.json
EXPECTED_EXECUTOR_SHA256=39e77e4355b31314de1be5c8349d029c8d25027f29f4607da2b98672683e9830
EXPECTED_LEXICON_SHA256=dda061ccc4e3ba5ba4d0df0186fae01e3ab09b1feaa319701e320533b7ac3189
cd "$BASE"
for path in "$CKPT" "$COMPILER" "$TOKENIZER" "$REPORT" "$EXECUTOR" "$LEXICON" \
  "$DATA/development_compositional.jsonl" "$DATA/development_lexical_ood.jsonl" \
  "$DEPTH/development_relational.jsonl" "$DEPTH/report.json"; do
  [ -f "$path" ] || { echo "missing pointer-anchor input: $path" >&2; exit 2; }
done
[ "$(sha256sum "$EXECUTOR" | awk '{print $1}')" = "$EXPECTED_EXECUTOR_SHA256" ] || exit 2
[ "$(sha256sum "$LEXICON" | awk '{print $1}')" = "$EXPECTED_LEXICON_SHA256" ] || exit 2
[ ! -e "$OUT_DIR" ] || { echo "refusing existing pointer-anchor output: $OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR"
nvidia-smi -L

evaluate_two() {
  local split=$1 mode=$2 suffix=$3
  "$PYTHON" train/eval_s3_categorical_executor.py \
    --base "$CKPT" --compiler "$COMPILER" --executor "$EXECUTOR" \
    --data "$DATA/${split}.jsonl" --report "$REPORT" --tokenizer "$TOKENIZER" \
    --split "$split" --identity-mode "$mode" --closed-action \
    --kind-lexicon "$LEXICON" --kind-decoder pointer_anchor \
    --out "$OUT_DIR/${suffix}.json" --batch-size 64
}

evaluate_depth() {
  local mode=$1
  "$PYTHON" train/eval_s3_categorical_depth.py \
    --base "$CKPT" --compiler "$COMPILER" --executor "$EXECUTOR" \
    --board "$DEPTH/development_relational.jsonl" --report "$DEPTH/report.json" \
    --tokenizer "$TOKENIZER" --identity-mode "$mode" --closed-action \
    --kind-lexicon "$LEXICON" --kind-decoder pointer_anchor \
    --out "$OUT_DIR/depth_${mode}.json" --batch-size 64
}

for mode in mean ordered gold; do
  evaluate_two development_compositional "$mode" "comp_${mode}"
done
evaluate_two development_lexical_ood mean lexical_mean
for mode in mean ordered gold; do
  evaluate_depth "$mode"
done

"$PYTHON" train/assess_s3_lexical_action.py \
  --executor "$EXECUTOR" --lexicon "$LEXICON" \
  --kind-protocol training_lexicon_pointer_anchor_v1 \
  --comp-mean "$OUT_DIR/comp_mean.json" \
  --comp-ordered "$OUT_DIR/comp_ordered.json" \
  --comp-gold "$OUT_DIR/comp_gold.json" \
  --lexical-mean "$OUT_DIR/lexical_mean.json" \
  --depth-mean "$OUT_DIR/depth_mean.json" \
  --depth-ordered "$OUT_DIR/depth_ordered.json" \
  --depth-gold "$OUT_DIR/depth_gold.json" \
  --out "$OUT_DIR/assessment.json"

echo "[s3-anchor] zero-fit public development complete; confirmation not referenced"
