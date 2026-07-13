#!/bin/bash
# Isolated bridge-only SFT from the immutable raw 200k checkpoint.  This is a
# primitive learnability experiment, not a broad-reasoning promotion path.
#SBATCH -J shohin-sft-v10a-bridge
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v10a_bridge_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:-$BASE/train/flagship_out/best_step200000.pt}
DATA=${DATA:-$BASE/artifacts/sft/semantic_bridge_v1_train.jsonl}
QUALITY=${QUALITY:-$BASE/artifacts/sft/semantic_bridge_v1_train.quality.r1.json}
PACKING=${PACKING:-$BASE/artifacts/sft/semantic_bridge_v1_train.packing.r1.json}
CONTRACTS=${CONTRACTS:-$BASE/artifacts/sft/semantic_bridge_v1_train.response_contracts.r1.json}
OUT=${OUT:?set OUT to a fresh isolated output directory}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
LRMUON=${LRMUON:-0.0012}
LRADAM=${LRADAM:-0.0003}

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -f "$INIT" ] || { echo "[v10a] init missing: $INIT" >&2; exit 2; }
[ -s "$DATA" ] || { echo "[v10a] data missing: $DATA" >&2; exit 2; }
[ -f "$QUALITY" ] || { echo "[v10a] quality report missing: $QUALITY" >&2; exit 2; }
[ -f "$PACKING" ] || { echo "[v10a] packing report missing: $PACKING" >&2; exit 2; }
[ -f "$CONTRACTS" ] || { echo "[v10a] response-contract report missing: $CONTRACTS" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[v10a] refusing existing output: $OUT" >&2; exit 3; }

"$PY" - "$DATA" "$QUALITY" "$PACKING" "$CONTRACTS" <<'PY'
import hashlib
import json
import sys

data, quality_path, packing_path, contracts_path = sys.argv[1:]
digest = hashlib.sha256()
with open(data, "rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
data_sha = digest.hexdigest()
quality = json.load(open(quality_path))
packing = json.load(open(packing_path))
contracts = json.load(open(contracts_path))
if quality.get("data") != data or quality.get("data_sha256") != data_sha:
    raise SystemExit("[v10a] quality report does not bind current data")
if quality.get("valid_rows") != 200000 or quality.get("malformed_rows") or quality.get("duplicate_normalized_questions"):
    raise SystemExit("[v10a] semantic-bridge quality gate failed")
if quality.get("eval_overlap", {}).get("exact_prompt_hits") or quality.get("eval_overlap", {}).get("ngram_prompt_hits"):
    raise SystemExit("[v10a] semantic-bridge overlap gate failed")
if packing.get("data_sha256", {}).get(data) != data_sha or packing.get("packed_sequences") != 8432:
    raise SystemExit("[v10a] semantic-bridge packing gate failed")
if packing.get("group_counts") != {"semantic_bridge": 8432}:
    raise SystemExit("[v10a] semantic-bridge group gate failed")
group = contracts.get("groups", {}).get("semantic_bridge", {})
if contracts.get("data_sha256") != data_sha or group.get("rows") != 200000:
    raise SystemExit("[v10a] response contract does not bind current data")
if group.get("starts_think_rate") != 1.0 or group.get("answer_marker_rate") != 1.0:
    raise SystemExit("[v10a] bridge response contract failed")
print("[v10a] data admitted", json.dumps({"sha256": data_sha, "packs": packing["packed_sequences"]}, sort_keys=True))
PY

cd "$BASE/train"
for preflight in 1 2 3; do
  if timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device="cuda", dtype=torch.bfloat16)
PY
  then
    break
  fi
  if [ "$preflight" = 3 ]; then
    echo "[v10a] CUDA unavailable or timed out after allocation" >&2
    exit 75
  fi
  echo "[v10a] CUDA preflight $preflight failed; retrying" >&2
  sleep 30
done

echo "[v10a] init=$INIT data=$DATA out=$OUT epochs=$EPOCHS bs=$BS"
"$PY" sft.py \
  --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --epochs "$EPOCHS" --batch-size "$BS" --lr-muon "$LRMUON" --lr-adam "$LRADAM" \
  --warmup 50 --pack-len 2048 --group-field training_group --compile --out "$OUT"
[ -s "$OUT/sft_ep${EPOCHS}.pt" ] || { echo "[v10a] expected final checkpoint missing" >&2; exit 4; }
echo "[v10a] done checkpoint=$OUT/sft_ep${EPOCHS}.pt"
