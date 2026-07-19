#!/bin/bash
#SBATCH -J shohin-rgde-cons
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/rgde_consumer_transport_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python
BOARD=$BASE/artifacts/r12/rgde_relational_development_v1/development_relational.jsonl
REPORT=$BASE/artifacts/r12/rgde_relational_development_v1/report.json
EXECUTOR=$BASE/train/referential_gather_delete_executor_v1_1_tied_2026071902/executor.pt
OUT=${OUT:-$BASE/artifacts/r12/rgde_consumer_transport_diagnostic_v1.json}
cd "$BASE"
[ ! -e "$OUT" ] || { echo "refusing existing output: $OUT" >&2; exit 2; }
[ "$(sha256sum "$EXECUTOR" | awk '{print $1}')" = \
  "adb6323202f6d25280f3a1cfd34a5b88fbc876331643726e38db389ead746b74" ] || {
  echo "frozen tied executor hash mismatch" >&2
  exit 2
}
nvidia-smi -L
"$PYTHON" train/diagnose_rgde_consumer_transport.py \
  --base "$BASE/train/flagship_out/ckpt_0300000.pt" \
  --compiler "$BASE/train/referential_literal_pointer_factorized_ordinary_2026071810/compiler.pt" \
  --executor "$EXECUTOR" --board "$BOARD" --report "$REPORT" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --batch-size 64
