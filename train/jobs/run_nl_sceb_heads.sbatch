#!/bin/bash
#SBATCH -J shohin-nl-sceb
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc23,evc25,evc26,evc27,evc31,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/nl_sceb_%j.out
set -euo pipefail
echo "[job] start $(date) host=$(hostname)"
export PYTHONUNBUFFERED=1
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=$BASE/train/flagship_out/best_step200000.pt
BOARD=$BASE/artifacts/evals/source_scheduled_reasoning_confirmation_v1.json
OUT=$BASE/train/nl_sceb_heads_r1
TOK=$BASE/artifacts/shohin-tok-32k.json
mkdir -p "$BASE/logs"
[ ! -e "$OUT" ] || { echo refusing; exit 4; }
for attempt in 1 2 3 4 5 6 7 8; do
  echo "[job] cuda attempt $attempt"
  if timeout 180 "$PY" -u - <<'CUDAPY'
import torch,sys
assert torch.cuda.is_available()
torch.empty(1,device="cuda",dtype=torch.bfloat16)
print(torch.cuda.get_device_name(0),flush=True)
CUDAPY
  then echo cuda_ok; break; fi
  sleep 10
  [ "$attempt" -eq 8 ] && exit 12
done
cd $BASE/train
"$PY" -u train_nl_sceb_heads.py --init "$INIT" --board "$BOARD" --tokenizer "$TOK" --out "$OUT" --epochs 8 --device cuda
echo done
