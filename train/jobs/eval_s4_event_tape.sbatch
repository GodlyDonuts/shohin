#!/bin/bash
#SBATCH --job-name=shohin-s4-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/s4_eval_%j.out

set -euo pipefail
BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
DATA=$BASE/artifacts/r12/s4_self_delimiting_event_tape_10818821031317486155
ARM=${ARM:-treatment}
PARSER=${PARSER:-$BASE/train/s4_event_tape_${ARM}_2026071904/parser.pt}
OUT=${OUT:-$BASE/artifacts/r12/s4_event_tape_${ARM}_2026071904.eval.json}
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
srun python3 train/eval_s4_event_tape.py \
  --base train/flagship_out/ckpt_0300000.pt --parser "$PARSER" \
  --s3-executor train/s3_equivariant_executor_2026071903/executor.pt \
  --data "$DATA/development.jsonl" --report "$DATA/report.json" \
  --tokenizer artifacts/shohin-tok-32k.json --out "$OUT" --batch-size 64
