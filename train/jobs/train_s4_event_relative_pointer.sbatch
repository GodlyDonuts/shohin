#!/bin/bash
#SBATCH -J shohin-s4-v2-train
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/s4_v2_train_%j.out
#SBATCH --open-mode=append

set -euo pipefail
BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PYTHON=$BASE/miniforge3/bin/python
DATA=$BASE/artifacts/r12/s4_self_delimiting_event_tape_10818821031317486155
ARM=${ARM:-treatment}
OUT=${OUT:-$BASE/train/s4_event_relative_${ARM}_2026071905/parser.pt}
EXTRA=()
if [[ "$ARM" == "shuffled" ]]; then
  EXTRA+=(--shuffle-supervision)
elif [[ "$ARM" != "treatment" ]]; then
  echo "invalid S4 v2 arm $ARM" >&2
  exit 2
fi
for path in \
  "$BASE/train/flagship_out/ckpt_0300000.pt" \
  "$BASE/train/s4_event_tape_treatment_2026071904/parser.pt" \
  "$DATA/train.jsonl" "$DATA/report.json" \
  "$BASE/artifacts/shohin-tok-32k.json"; do
  test -s "$path" || { echo "missing $path" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "refusing existing $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")" "$BASE/logs"
cd "$BASE"
"$PYTHON" train/train_s4_event_relative_pointer.py \
  --base train/flagship_out/ckpt_0300000.pt \
  --v1-parser train/s4_event_tape_treatment_2026071904/parser.pt \
  --data "$DATA/train.jsonl" --report "$DATA/report.json" \
  --tokenizer artifacts/shohin-tok-32k.json --out "$OUT" \
  --batch-size 64 --epochs 1 --seed 2026071905 "${EXTRA[@]}"
