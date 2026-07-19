#!/bin/bash
#SBATCH -J shohin-s4-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/s4_eval_%j.out
#SBATCH --open-mode=append

set -euo pipefail
BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PYTHON=$BASE/miniforge3/bin/python
DATA=${DATA:-$BASE/artifacts/r12/s4_self_delimiting_event_tape_10818821031317486155}
ARM=${ARM:-treatment}
PARSER=${PARSER:-$BASE/train/s4_event_tape_${ARM}_2026071904/parser.pt}
OUT=${OUT:-$BASE/artifacts/r12/s4_event_tape_${ARM}_2026071904.eval.json}
STRUCTURAL_LEXICON=${STRUCTURAL_LEXICON:-}
for path in \
  "$BASE/train/flagship_out/ckpt_0300000.pt" "$PARSER" \
  "$BASE/train/s3_equivariant_executor_2026071903/executor.pt" \
  "$DATA/development.jsonl" "$DATA/report.json" \
  "$BASE/artifacts/shohin-tok-32k.json"; do
  test -s "$path" || { echo "missing $path" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "refusing existing $OUT" >&2; exit 2; }
EXTRA_ARGS=()
if [[ -n "$STRUCTURAL_LEXICON" ]]; then
  test -s "$STRUCTURAL_LEXICON" || { echo "missing $STRUCTURAL_LEXICON" >&2; exit 2; }
  EXTRA_ARGS+=(--structural-lexicon "$STRUCTURAL_LEXICON")
fi
mkdir -p "$(dirname "$OUT")" "$BASE/logs"
cd "$BASE"
"$PYTHON" train/eval_s4_event_tape.py \
  --base train/flagship_out/ckpt_0300000.pt --parser "$PARSER" \
  --s3-executor train/s3_equivariant_executor_2026071903/executor.pt \
  --data "$DATA/development.jsonl" --report "$DATA/report.json" \
  --tokenizer artifacts/shohin-tok-32k.json --out "$OUT" --batch-size 64 \
  "${EXTRA_ARGS[@]}"
