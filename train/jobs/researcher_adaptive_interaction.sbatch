#!/bin/bash
#SBATCH -J researcher-adaptive
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc37,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 0:35:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/researcher_adaptive_%j.out
#SBATCH -e /lustre/fs1/home/sa305415/shohin/logs/researcher_adaptive_%j.err
set -euo pipefail
umask 077

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
RAW200=${RAW200:?set RAW200}
DRS=${DRS:?set DRS}
RAW300=${RAW300:?set RAW300}
OUT=${OUT:?set OUT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set SOURCE_COMMIT}

for path in "$RAW200" "$DRS" "$RAW300" "$BASE/artifacts/shohin-tok-32k.json"; do
  [ -f "$path" ] || { echo "[adaptive-interaction] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[adaptive-interaction] refusing existing output: $OUT" >&2; exit 2; }

cd "$BASE/train"
for attempt in 1 2 3; do
  if timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("[adaptive-interaction] CUDA unavailable", file=sys.stderr)
    raise SystemExit(75)
torch.empty(1, device="cuda")
print(torch.cuda.get_device_name(0), flush=True)
PY
  then
    break
  fi
  status=$?
  if [[ "$attempt" -eq 3 ]]; then
    echo "[adaptive-interaction] CUDA preflight failed after three attempts" >&2
    exit "$status"
  fi
  sleep 20
done

"$PY" researcher_adaptive_interaction.py \
  --model "raw200=$RAW200" \
  --model "drs_r3_200=$DRS" \
  --model "raw300=$RAW300" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --source-commit "$SOURCE_COMMIT" \
  --device cuda \
  --max-new 64 \
  --out "$OUT"

chmod 0444 "$OUT"
sha256sum "$OUT"
