#!/bin/bash
#SBATCH -J shohin-rgde-rel
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/rgde_relational_identity_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python
BOARD=${BOARD:-$BASE/artifacts/r12/rgde_relational_development_v1/development_relational.jsonl}
REPORT=${REPORT:-$BASE/artifacts/r12/rgde_relational_development_v1/report.json}
OUT=${OUT:-$BASE/artifacts/r12/rgde_relational_identity_probe_v1.json}
cd "$BASE"
[ ! -e "$OUT" ] || { echo "refusing existing output: $OUT" >&2; exit 2; }
nvidia-smi -L
"$PYTHON" train/probe_rgde_relational_identity.py \
  --base "$BASE/train/flagship_out/ckpt_0300000.pt" \
  --compiler "$BASE/train/referential_literal_pointer_factorized_ordinary_2026071810/compiler.pt" \
  --board "$BOARD" --report "$REPORT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --batch-size 64
