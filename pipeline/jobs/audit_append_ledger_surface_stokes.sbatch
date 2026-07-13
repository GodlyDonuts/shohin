#!/bin/bash
# CPU-only packing/length audit for the frozen Append-only Delta Ledger corpus.
# This is intentionally Stokes-friendly: it imports only the Rust tokenizers
# package, does not import Torch, and cannot allocate or reference a GPU.
#SBATCH --job-name=shohin-adl-surface
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sa305415/shohin/logs/adl_surface_%j.out

set -euo pipefail
BASE=${BASE:-/home/sa305415/shohin}
PY=${PY:-$HOME/.venvs/shohin-surface/bin/python}
DATA=${DATA:-$BASE/artifacts/sft/append_ledger_v1_train.jsonl}
OUT=${OUT:-$BASE/artifacts/sft/append_ledger_v1_train.surface_audit.json}

cd "$BASE"
test -x "$PY"
test -f "$DATA"
test -f "$BASE/artifacts/shohin-tok-32k.json"
"$PY" pipeline/audit_sft_surface.py \
  --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT" \
  --pack-len 2048 --prompt-field completion_prompt --response-field response \
  --by kind width prompt_style
