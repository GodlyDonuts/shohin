#!/bin/bash
#SBATCH -J shohin-s5-confirm
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/s5_confirm_%j.out
#SBATCH --open-mode=append

set -euo pipefail
BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PYTHON=$BASE/miniforge3/bin/python
DATA=${DATA:?set DATA to the frozen S5 confirmation board directory}
TAG=${TAG:-2026071912}
FIT=$BASE/train/s5_learned_generator_2026071911
EVAL=$BASE/artifacts/r12/s5_learned_generator_confirmation_${TAG}.eval.json
ASSESS=$BASE/artifacts/r12/s5_learned_generator_confirmation_${TAG}.assessment.json
for path in \
  "$BASE/train/flagship_out/ckpt_0300000.pt" \
  "$BASE/train/s4_event_tape_treatment_2026071904/parser.pt" \
  "$BASE/artifacts/r12/s4_hard_island_soft_interface_v5.promoted.json" \
  "$FIT/treatment.pt" "$FIT/shuffled.pt" "$FIT/report.json" \
  "$DATA/confirmation.jsonl" "$DATA/report.json" \
  "$BASE/artifacts/shohin-tok-32k.json"; do
  test -s "$path" || { echo "missing $path" >&2; exit 2; }
done
for path in "$EVAL" "$ASSESS"; do
  test ! -e "$path" || { echo "refusing existing $path" >&2; exit 2; }
done
cd "$BASE"
"$PYTHON" train/eval_s5_learned_generator_executor.py \
  --base train/flagship_out/ckpt_0300000.pt \
  --parser train/s4_event_tape_treatment_2026071904/parser.pt \
  --promoted artifacts/r12/s4_hard_island_soft_interface_v5.promoted.json \
  --treatment-generator "$FIT/treatment.pt" \
  --shuffled-generator "$FIT/shuffled.pt" \
  --fit-report "$FIT/report.json" \
  --data "$DATA/confirmation.jsonl" --report "$DATA/report.json" \
  --tokenizer artifacts/shohin-tok-32k.json --out "$EVAL" --batch-size 64 \
  --board-role confirmation
"$PYTHON" train/assess_s5_learned_generator_confirmation.py \
  --evaluation "$EVAL" --out "$ASSESS"
