#!/bin/bash
#SBATCH --job-name=shohin-s4-parser
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/s4_parser_%j.out

set -euo pipefail
BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
DATA=$BASE/artifacts/r12/s4_self_delimiting_event_tape_10818821031317486155
ARM=${ARM:-treatment}
OUT=${OUT:-$BASE/train/s4_event_tape_${ARM}_2026071904/parser.pt}
EXTRA=()
if [[ "$ARM" == "shuffled" ]]; then
  EXTRA+=(--shuffle-supervision)
elif [[ "$ARM" != "treatment" ]]; then
  echo "invalid S4 arm $ARM" >&2
  exit 2
fi
for path in \
  "$BASE/train/flagship_out/ckpt_0300000.pt" \
  "$BASE/train/referential_literal_pointer_factorized_ordinary_2026071810/compiler.pt" \
  "$DATA/train.jsonl" "$DATA/report.json" \
  "$BASE/artifacts/shohin-tok-32k.json"; do
  test -s "$path" || { echo "missing $path" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "refusing existing $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")" "$BASE/logs"
cd "$BASE"
srun python3 train/train_s4_event_tape.py \
  --base train/flagship_out/ckpt_0300000.pt \
  --initializer train/referential_literal_pointer_factorized_ordinary_2026071810/compiler.pt \
  --data "$DATA/train.jsonl" --report "$DATA/report.json" \
  --tokenizer artifacts/shohin-tok-32k.json --out "$OUT" \
  --batch-size 64 --epochs 1 --seed 2026071904 "${EXTRA[@]}"
