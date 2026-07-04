#!/bin/bash
#SBATCH -J shohin-tok
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/tok_%j.out
# Stream a 2 GB mixed sample, train the 32k BPE, and run the fertility A/B vs SmolLM2.
set -euo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export HF_TOKEN=$(cat "$BASE/.hf_token" 2>/dev/null || true)
export HF_HUB_ENABLE_HF_TRANSFER=0
cd "$BASE/pipeline"
GB="${SHOHIN_TOK_GB:-2}"

echo "[job] streaming ${GB} GB sample ..."
$PY stream_sample.py --out "$BASE/tok_sample.txt" --gb "$GB"

mkdir -p "$BASE/artifacts"
echo "[job] training tokenizer ..."
$PY train_tokenizer.py --sample "$BASE/tok_sample.txt" --out "$BASE/artifacts/shohin-tok-32k.json"
echo "[job] done."
