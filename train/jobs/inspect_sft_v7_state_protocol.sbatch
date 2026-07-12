#!/bin/bash
#SBATCH -J shohin-sft-v7-audit
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v7_audit_%j.out
#SBATCH --open-mode=append
# Read-only capacity/replay gate for typed compact-state SFT.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
STATE_DATA=${STATE_DATA:-$BASE/artifacts/sft/primitives_v3_state_train.jsonl}
DATA=("$BASE/artifacts/sft/sft_mix_reasoning_v4.jsonl" "$STATE_DATA")
OUT=${OUT:-$BASE/artifacts/sft/sft_mix_reasoning_v7_state_protocol_revised.packing.json}
# Keep scarce code and teacher groups under two replays per epoch while the
# state protocol remains large enough to test its semantic gate.
WEIGHTS="math=0.432 procedural=0.18 code=0.028 teacher=0.04 state_protocol=0.32"

for data in "${DATA[@]}"; do
  [ -s "$data" ] || { echo "[sft-v7-audit] data missing: $data" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[sft-v7-audit] refusing to overwrite $OUT" >&2; exit 3; }
cd "$BASE/train"
"$PY" inspect_sft_packing.py --data "${DATA[@]}" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --weights $WEIGHTS
echo "[sft-v7-audit] wrote $OUT"
