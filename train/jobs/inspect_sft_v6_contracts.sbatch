#!/bin/bash
#SBATCH -J shohin-sft-v6-audit
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v6_audit_%j.out
#SBATCH --open-mode=append
# Read-only capacity/replay gate for contract-diverse primitive SFT.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=(
  "$BASE/artifacts/sft/sft_mix_reasoning_v4.jsonl"
  "$BASE/artifacts/sft/primitives_v2_contract_train.jsonl"
)
OUT=${OUT:-$BASE/artifacts/sft/sft_mix_reasoning_v6_contracts.packing.json}
WEIGHTS="math=0.45 procedural=0.15 code=0.05 teacher=0.05 contracts=0.30"

for data in "${DATA[@]}"; do
  [ -s "$data" ] || { echo "[sft-v6-audit] data missing: $data" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[sft-v6-audit] refusing to overwrite $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" inspect_sft_packing.py --data "${DATA[@]}" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --weights $WEIGHTS
echo "[sft-v6-audit] wrote $OUT"
