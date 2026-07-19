#!/bin/bash
#SBATCH -J shohin-rgde-depth
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/rgde_depth_confirmation_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python
BOARD_DIR=${BOARD_DIR:?set BOARD_DIR to the fresh depth-confirmation directory}
CKPT=$BASE/train/flagship_out/ckpt_0300000.pt
COMPILER=$BASE/train/referential_literal_pointer_factorized_ordinary_2026071810/compiler.pt
EXECUTOR=$BASE/train/referential_gather_delete_executor_v1_1_tied_2026071902/executor.pt
EXPECTED_EXECUTOR_SHA256=adb6323202f6d25280f3a1cfd34a5b88fbc876331643726e38db389ead746b74
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
BOARD=$BOARD_DIR/confirmation_depth.jsonl
REPORT=$BOARD_DIR/report.json
OUT_DIR=$BOARD_DIR/evaluation

cd "$BASE"
for path in "$CKPT" "$COMPILER" "$EXECUTOR" "$TOKENIZER" "$BOARD" "$REPORT"; do
  [ -f "$path" ] || { echo "missing required input: $path" >&2; exit 2; }
done
[ "$(sha256sum "$EXECUTOR" | awk '{print $1}')" = "$EXPECTED_EXECUTOR_SHA256" ] || {
  echo "frozen tied executor hash mismatch" >&2
  exit 2
}
[ ! -e "$OUT_DIR" ] || { echo "refusing existing evaluation directory: $OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR"
nvidia-smi -L

evaluate() {
  local name=$1
  shift
  "$PYTHON" train/eval_rgde_depth_confirmation.py \
    --base "$CKPT" --compiler "$COMPILER" --executor "$EXECUTOR" \
    --board "$BOARD" --report "$REPORT" --tokenizer "$TOKENIZER" \
    --out "$OUT_DIR/${name}.json" --batch-size 64 "$@"
}

evaluate predicted --packet-oracle none --intervention none
evaluate gold_packet --packet-oracle full --intervention none
evaluate operation_derangement --packet-oracle none --intervention operations
evaluate query_derangement --packet-oracle none --intervention query

echo "[rgde-depth] one-shot confirmation evaluations complete"
