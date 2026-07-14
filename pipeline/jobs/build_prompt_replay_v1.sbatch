#!/bin/bash
#SBATCH -J shohin-prompt-replay
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/prompt_replay_%j.out
#SBATCH --open-mode=append
# Build a frozen answer-free behavior-retention set. This CPU-only job must not
# modify a frozen SFT mix, a held-out set, or a training output.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:?set OUT to a fresh prompt replay JSONL}
AUDIT=${AUDIT:?set AUDIT to a fresh replay audit JSON}
SOURCE=(
  "$BASE/artifacts/sft/sft_mix_reasoning_v4.jsonl"
  "$BASE/artifacts/sft/primitives_v1_train.jsonl"
)
EXCLUDE=(
  "$BASE/artifacts/evals/primitives_v1_heldout.jsonl"
  "$BASE/artifacts/rg_v2/rg_eval.jsonl"
)

[ ! -e "$OUT" ] || { echo "[prompt-replay] refusing existing output" >&2; exit 2; }
[ ! -e "$AUDIT" ] || { echo "[prompt-replay] refusing existing audit" >&2; exit 3; }
for path in "${SOURCE[@]}" "${EXCLUDE[@]}"; do
  [ -s "$path" ] || { echo "[prompt-replay] missing source: $path" >&2; exit 4; }
done
cd "$BASE"
"$PY" pipeline/build_prompt_replay_v1.py --source "${SOURCE[@]}" --exclude "${EXCLUDE[@]}" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT" --audit "$AUDIT" --limit 4096 --max-tokens 128
