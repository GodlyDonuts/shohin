#!/bin/bash
#SBATCH -J shohin-s4-v2-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/s4_v2_eval_%j.out
#SBATCH --open-mode=append

set -euo pipefail
BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PYTHON=$BASE/miniforge3/bin/python
DATA=${DATA:?set DATA to the frozen fresh-board directory}
ARM=${ARM:-treatment}
PARSER=${PARSER:-$BASE/train/s4_event_relative_${ARM}_2026071905/parser.pt}
OUT=${OUT:-$BASE/artifacts/r12/s4_event_relative_${ARM}_2026071905.eval.json}
for path in \
  "$BASE/train/flagship_out/ckpt_0300000.pt" "$PARSER" \
  "$BASE/train/s3_equivariant_executor_2026071903/executor.pt" \
  "$DATA/development.jsonl" "$DATA/report.json" \
  "$BASE/artifacts/shohin-tok-32k.json"; do
  test -s "$path" || { echo "missing $path" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "refusing existing $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")" "$BASE/logs"
cd "$BASE"
"$PYTHON" train/eval_s4_event_relative_pointer.py \
  --base train/flagship_out/ckpt_0300000.pt --parser "$PARSER" \
  --s3-executor train/s3_equivariant_executor_2026071903/executor.pt \
  --data "$DATA/development.jsonl" --report "$DATA/report.json" \
  --tokenizer artifacts/shohin-tok-32k.json --out "$OUT" --batch-size 64
