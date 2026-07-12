#!/bin/bash
#SBATCH -J shohin-shard-scan
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/shard_scan_%j.out
#SBATCH --open-mode=append
# Full corpus-admission scan. Set SHARD_DIR explicitly; the corpus is read-only
# and the JSON audit record is written outside the source directory.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
SHARD_DIR=${SHARD_DIR:?set SHARD_DIR to a committed shard directory}
DIVERGE_TOKEN=${DIVERGE_TOKEN:-0}
REPORT=${REPORT:-$BASE/artifacts/shard_scans/$(basename "$SHARD_DIR").json}

[ -s "$SHARD_DIR/manifest.json" ] || { echo "[shard-scan] missing manifest: $SHARD_DIR" >&2; exit 2; }
cd "$BASE"
"$BASE/miniforge3/bin/python" pipeline/scan_shards.py \
  --shard-dirs "$SHARD_DIR" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --seed 20260712 --diverge-token "$DIVERGE_TOKEN" --out "$REPORT"
[ -s "$REPORT" ] || { echo "[shard-scan] report missing: $REPORT" >&2; exit 3; }
