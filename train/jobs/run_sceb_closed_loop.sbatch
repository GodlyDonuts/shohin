#!/bin/bash
#SBATCH -J shohin-sceb-loop
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc23,evc25,evc26,evc27,evc31,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sceb_closed_loop_%j.out
#SBATCH --open-mode=append
set -euo pipefail
echo "[job] start $(date) host=$(hostname)"
export PYTHONUNBUFFERED=1
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4

CKPT=$BASE/train/sft_typed_controller_v1_200k_r1/sft_ep1.pt
HEADS=$BASE/train/discrete_controller_heads_r1/heads_ep4.pt
HELD=$BASE/artifacts/r12/typed_controller_v1/heldout.jsonl
OUT=$BASE/artifacts/r12/sceb_closed_loop_v1/decision.json
TOK=$BASE/artifacts/shohin-tok-32k.json

mkdir -p "$(dirname "$OUT")" "$BASE/logs"
[ -f "$HEADS" ] || { echo missing heads; exit 2; }
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
"$PY" -u eval_sceb_closed_loop.py \
  --ckpt "$CKPT" --heads "$HEADS" --tokenizer "$TOK" --heldout "$HELD" --out "$OUT" --device cuda
echo done
