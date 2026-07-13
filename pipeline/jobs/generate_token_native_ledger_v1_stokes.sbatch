#!/bin/bash
# CPU-only build of the token-native delta-ledger candidate. It must remain
# isolated from pretraining and H100 work until its independent audit passes.
#SBATCH -J shohin-tnl-v1
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/token_native_ledger_v1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=${BASE:-/home/sa305415/shohin}
PY=${PY:-/usr/bin/python3.12}
TRAIN=${TRAIN:-$BASE/artifacts/sft/token_native_ledger_v1_train.jsonl}
HELDOUT=${HELDOUT:-$BASE/artifacts/evals/token_native_ledger_v1_heldout.jsonl}
REPORT=${REPORT:-$BASE/artifacts/sft/token_native_ledger_v1_train.report.json}
VARIANTS=${VARIANTS:-8}
HELDOUT_PER_REGIME=${HELDOUT_PER_REGIME:-300}

for path in "$TRAIN" "$HELDOUT" "$REPORT"; do
  [ ! -e "$path" ] || { echo "[tnl-build] refusing existing artifact: $path" >&2; exit 3; }
done
cd "$BASE"
"$PY" pipeline/generate_token_native_ledger_v1.py \
  --train-out "$TRAIN" \
  --heldout-out "$HELDOUT" \
  --report "$REPORT" \
  --variants "$VARIANTS" \
  --heldout-per-regime "$HELDOUT_PER_REGIME"
test -s "$TRAIN"
test -s "$HELDOUT"
test -s "$REPORT"
echo "[tnl-build] committed immutable CPU candidate"
