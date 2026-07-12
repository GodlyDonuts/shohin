#!/bin/bash
#SBATCH -J shohin-sft-verifier
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_verifier_%j.out
#SBATCH --open-mode=append
# Train an isolated binary verifier. It never writes to the flagship or a
# general-reasoning SFT path; held-out selector performance is its only gate.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:?set INIT to a preserved raw pretrain checkpoint}
DATA=${DATA:?set DATA to a frozen balanced verifier JSONL}
OUT=${OUT:?set OUT to a fresh isolated verifier output directory}
EPOCHS=${EPOCHS:-2}
PACKLEN=${PACKLEN:-512}

[ -f "$INIT" ] || { echo "[sft-verifier] init missing: $INIT" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[sft-verifier] data missing: $DATA" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[sft-verifier] output exists: $OUT" >&2; exit 4; }
cd "$BASE/train"
for attempt in 1 2 3; do
  if "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[sft-verifier] CUDA unavailable: {exc}", file=sys.stderr)
    sys.exit(75)
PY
  then
    break
  fi
  [ "$attempt" = 3 ] && exit 75
  sleep 30
done
echo "[sft-verifier] init=$INIT data=$DATA out=$OUT epochs=$EPOCHS pack=$PACKLEN"
"$PY" sft.py --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --epochs "$EPOCHS" --batch-size 16 --pack-len "$PACKLEN" \
  --lr-muon 0.001 --lr-adam 0.00025 --group-field training_group \
  --sample-weights verifier_correct=0.5 verifier_incorrect=0.5 --compile --out "$OUT"
