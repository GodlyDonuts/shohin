#!/bin/bash
#SBATCH -J shohin-sft-v7
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v7_%j.out
#SBATCH --open-mode=append
# Isolated SFT candidate for a typed, semantically checkable compact-state protocol.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
STATE_DATA=${STATE_DATA:-$BASE/artifacts/sft/primitives_v3_state_train.jsonl}
DATA=("$BASE/artifacts/sft/sft_mix_reasoning_v4.jsonl" "$STATE_DATA")
INIT=${INIT:?set INIT to a preserved raw flagship checkpoint}
OUT=${OUT:?set OUT to a fresh isolated directory}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
LRMUON=${LRMUON:-0.0015}
LRADAM=${LRADAM:-0.00035}
# Derived from the V7 packing audit: code and teacher both remain below two
# replays per epoch, while the state protocol receives enough mass to test it.
WEIGHTS="math=0.432 procedural=0.18 code=0.028 teacher=0.04 state_protocol=0.32"

for data in "${DATA[@]}"; do
  [ -s "$data" ] || { echo "[sft-v7] data missing: $data" >&2; exit 2; }
done
[ -f "$INIT" ] || { echo "[sft-v7] init missing: $INIT" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[sft-v7] refusing to overwrite $OUT" >&2; exit 4; }
cd "$BASE/train"
for i in $(seq 1 30); do
  if "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[sft-v7] CUDA not ready: {exc}", file=sys.stderr)
    sys.exit(75)
PY
  then
    break
  fi
  sleep 10
done
if ! "$PY" - <<'PY'
import sys
import torch
try:
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[sft-v7] CUDA unavailable after retry: {exc}", file=sys.stderr)
    sys.exit(75)
PY
then
  exit 75
fi
echo "[sft-v7] init=$INIT data=${DATA[*]} out=$OUT epochs=$EPOCHS weights=$WEIGHTS"
"$PY" sft.py --init "$INIT" --data "${DATA[@]}" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --epochs "$EPOCHS" --batch-size "$BS" --lr-muon "$LRMUON" --lr-adam "$LRADAM" \
  --group-field training_group --prompt-override-field completion_prompt \
  --sample-weights $WEIGHTS --compile --out "$OUT"
echo "[sft-v7] done"
