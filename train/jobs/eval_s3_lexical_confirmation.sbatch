#!/bin/bash
#SBATCH -J shohin-s3-confirm
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/s3_lexical_confirmation_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python
BOARD_DIR=${BOARD_DIR:?set BOARD_DIR to fresh S3 confirmation directory}
OUT_DIR=$BOARD_DIR/evaluation
EXECUTOR=$BASE/train/s3_equivariant_executor_2026071903/executor.pt
LEXICON=$BASE/train/s3_lexical_action_2026071903/kind_lexicon.json
CKPT=$BASE/train/flagship_out/ckpt_0300000.pt
COMPILER=$BASE/train/referential_literal_pointer_factorized_ordinary_2026071810/compiler.pt
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
BOARD=$BOARD_DIR/confirmation_depth.jsonl
REPORT=$BOARD_DIR/report.json
EXPECTED_EXECUTOR_SHA256=39e77e4355b31314de1be5c8349d029c8d25027f29f4607da2b98672683e9830
EXPECTED_LEXICON_SHA256=dda061ccc4e3ba5ba4d0df0186fae01e3ab09b1feaa319701e320533b7ac3189
cd "$BASE"
for path in "$CKPT" "$COMPILER" "$EXECUTOR" "$LEXICON" "$TOKENIZER" "$BOARD" "$REPORT"; do
  [ -f "$path" ] || { echo "missing S3 confirmation input: $path" >&2; exit 2; }
done
[ "$(sha256sum "$EXECUTOR" | awk '{print $1}')" = "$EXPECTED_EXECUTOR_SHA256" ] || exit 2
[ "$(sha256sum "$LEXICON" | awk '{print $1}')" = "$EXPECTED_LEXICON_SHA256" ] || exit 2
[ ! -e "$OUT_DIR" ] || { echo "refusing existing S3 confirmation output" >&2; exit 2; }
mkdir -p "$OUT_DIR"
nvidia-smi -L

evaluate() {
  local name=$1 mode=$2 intervention=$3
  "$PYTHON" train/eval_s3_lexical_confirmation.py \
    --base "$CKPT" --compiler "$COMPILER" --executor "$EXECUTOR" \
    --lexicon "$LEXICON" --board "$BOARD" --report "$REPORT" \
    --tokenizer "$TOKENIZER" --identity-mode "$mode" \
    --intervention "$intervention" --out "$OUT_DIR/${name}.json" --batch-size 64
}

evaluate ordered ordered none
evaluate mean mean none
evaluate gold gold none
evaluate operation ordered operations
evaluate query ordered query
echo "[s3-confirm] one-shot evaluation complete"
