#!/bin/bash
#SBATCH -J shohin-ssc-halt
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc23,evc25,evc26,evc27,evc31,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ssc_halt_first_%j.out
#SBATCH --open-mode=append
set -euo pipefail
echo "[job] start $(date) host=$(hostname)"
export PYTHONUNBUFFERED=1
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4

# Raw-260k milestone used for SSC confirmation (immutable)
CKPT=${CKPT:-$BASE/train/flagship_out/best_step260000.model.pt}
# fallbacks
if [ ! -f "$CKPT" ]; then CKPT=$BASE/train/flagship_out/best_step200000.pt; fi
if [ ! -f "$CKPT" ]; then CKPT=$BASE/train/flagship_out/best_step200000.model.pt; fi
BOARD=$BASE/artifacts/evals/source_scheduled_reasoning_confirmation_v1.json
OUT=${OUT:-$BASE/artifacts/r12/ssc_halt_first_live/decision.json}
TOK=$BASE/artifacts/shohin-tok-32k.json

mkdir -p "$(dirname "$OUT")" "$BASE/logs"
[ -f "$CKPT" ] || { echo missing ckpt; ls $BASE/train/flagship_out/*200* $BASE/train/flagship_out/*260* 2>/dev/null | head; exit 2; }
[ ! -e "$OUT" ] || { echo refusing out; exit 4; }

for attempt in 1 2 3 4 5 6 7 8; do
  echo "[job] cuda attempt $attempt on $(hostname)"
  if timeout 180 "$PY" -u - <<'CUDAPY'
import torch, sys
assert torch.cuda.is_available()
torch.empty(1, device="cuda", dtype=torch.bfloat16)
print(torch.cuda.get_device_name(0), flush=True)
CUDAPY
  then echo cuda_ok; break; fi
  sleep 10
  [ "$attempt" -eq 8 ] && exit 12
done

cd "$BASE/train"
echo "ckpt=$CKPT"
"$PY" eval_ssc_halt_first_live.py --ckpt "$CKPT" --tokenizer "$TOK" --board "$BOARD" --out "$OUT" --device cuda
echo done
