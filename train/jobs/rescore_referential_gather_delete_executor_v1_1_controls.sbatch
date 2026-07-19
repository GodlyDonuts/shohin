#!/bin/bash
#SBATCH -J shohin-rgde11-ctl
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/referential_gather_delete_executor_v1_1_controls_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python
CKPT=$BASE/train/flagship_out/ckpt_0300000.pt
COMPILER=$BASE/train/referential_literal_pointer_factorized_ordinary_2026071810/compiler.pt
EXECUTOR=$BASE/train/referential_gather_delete_executor_v1_1_tied_2026071902/executor.pt
DATA_DIR=$BASE/artifacts/r12/referential_literal_pointer_factorized_v1
DEVELOPMENT=$DATA_DIR/development_compositional.jsonl
REPORT=$DATA_DIR/development_report.json
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
OUT_DIR=$BASE/train/referential_gather_delete_executor_v1_1_tied_2026071902

cd "$BASE"
for path in "$CKPT" "$COMPILER" "$EXECUTOR" "$DEVELOPMENT" "$REPORT" "$TOKENIZER"; do
  [ -f "$path" ] || { echo "missing required input: $path" >&2; exit 2; }
done
for path in \
  "$OUT_DIR/development_operation_semantic_derangement.json" \
  "$OUT_DIR/development_query_semantic_derangement.json"; do
  [ ! -e "$path" ] || { echo "refusing existing output: $path" >&2; exit 2; }
done

nvidia-smi -L
"$PYTHON" train/eval_referential_gather_delete_executor.py \
  --base "$CKPT" --compiler "$COMPILER" --executor "$EXECUTOR" \
  --data "$DEVELOPMENT" --report "$REPORT" --tokenizer "$TOKENIZER" \
  --split development_compositional \
  --out "$OUT_DIR/development_operation_semantic_derangement.json" \
  --batch-size 64 --packet-oracle none --shuffle-operations
"$PYTHON" train/eval_referential_gather_delete_executor.py \
  --base "$CKPT" --compiler "$COMPILER" --executor "$EXECUTOR" \
  --data "$DEVELOPMENT" --report "$REPORT" --tokenizer "$TOKENIZER" \
  --split development_compositional \
  --out "$OUT_DIR/development_query_semantic_derangement.json" \
  --batch-size 64 --packet-oracle none --shuffle-query

echo "[rgde-v1.1-controls] both semantic derangements complete"
