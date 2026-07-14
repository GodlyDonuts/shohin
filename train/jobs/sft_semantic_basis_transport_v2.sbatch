#!/bin/bash
# One isolated learnability ablation for the exact-carrier semantic-basis task.
# It cannot change the flagship, pretraining shards, or any existing SFT output.
#SBATCH -J shohin-sft-sbasis-v2
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_semantic_basis_v2_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:-$BASE/train/flagship_out/best_step200000.pt}
DATA=${DATA:-$BASE/artifacts/sft/semantic_basis_transport_v2_train.jsonl}
QUALITY=${QUALITY:-$BASE/artifacts/sft/semantic_basis_transport_v2_train.quality.json}
FULL_TEXT=${FULL_TEXT:-$BASE/artifacts/sft/semantic_basis_transport_v2_train.training_text_overlap.json}
CONTRACTS=${CONTRACTS:-$BASE/artifacts/sft/semantic_basis_transport_v2_train.response_contracts.json}
PACKING=${PACKING:-$BASE/artifacts/sft/semantic_basis_transport_v2_train.packing_qa.json}
OUT=${OUT:?set OUT to a fresh isolated output directory}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
LRMUON=${LRMUON:-0.0012}
LRADAM=${LRADAM:-0.0003}

export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for path in "$INIT" "$DATA" "$QUALITY" "$FULL_TEXT" "$CONTRACTS" "$PACKING"; do
  [ -s "$path" ] || { echo "[sbasis-sft] required input missing: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[sbasis-sft] refusing existing output: $OUT" >&2; exit 3; }

"$PY" - "$DATA" "$QUALITY" "$FULL_TEXT" "$CONTRACTS" "$PACKING" <<'PY'
import hashlib
import json
import sys

data, quality_path, text_path, contracts_path, packing_path = sys.argv[1:]
digest = hashlib.sha256()
with open(data, "rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
data_sha = digest.hexdigest()
quality, text, contracts, packing = (json.load(open(path)) for path in (quality_path, text_path, contracts_path, packing_path))
if quality.get("data_sha256") != data_sha or quality.get("valid_rows") != 150000:
    raise SystemExit("[sbasis-sft] quality report does not bind current 150k data")
if quality.get("malformed_rows") or quality.get("missing_question_or_response") or quality.get("duplicate_normalized_questions"):
    raise SystemExit("[sbasis-sft] structural quality gate failed")
if any(quality.get("eval_overlap", {}).get(key) for key in ("exact_prompt_hits", "ngram_prompt_hits")):
    raise SystemExit("[sbasis-sft] public eval overlap gate failed")
if text.get("data_sha256") != data_sha or text.get("malformed_json_rows"):
    raise SystemExit("[sbasis-sft] full-text report does not bind current data")
if any(text.get("overlap", {}).get(key) for key in ("exact_rows", "ngram_rows")):
    raise SystemExit("[sbasis-sft] full-text overlap gate failed")
group = contracts.get("groups", {}).get("semantic_basis_transport_exact_carrier", {})
if contracts.get("data_sha256") != data_sha or contracts.get("malformed_rows") or contracts.get("missing_response_rows"):
    raise SystemExit("[sbasis-sft] response contract does not bind current data")
if group.get("rows") != 150000 or group.get("think_marker_rate") != 0.0 or group.get("answer_marker_rate") != 0.0:
    raise SystemExit("[sbasis-sft] exact-carrier response contract failed")
if group.get("short_response_rate") != 1.0:
    raise SystemExit("[sbasis-sft] response length contract failed")
if data_sha not in set(packing.get("data_sha256", {}).values()):
    raise SystemExit("[sbasis-sft] packing report does not bind current data")
if packing.get("pack_len") != 2048 or packing.get("group_counts") != {"semantic_basis_transport_exact_carrier": 4016}:
    raise SystemExit("[sbasis-sft] packing coverage gate failed")
if packing.get("requested_weights") != {"semantic_basis_transport_exact_carrier": 1.0}:
    raise SystemExit("[sbasis-sft] packing weight gate failed")
print("[sbasis-sft] data admitted", json.dumps({"sha256": data_sha, "packs": 4016}, sort_keys=True))
PY

cd "$BASE/train"
for preflight in 1 2 3; do
  if timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device="cuda", dtype=torch.bfloat16)
PY
  then
    break
  fi
  if [ "$preflight" = 3 ]; then
    echo "[sbasis-sft] CUDA unavailable or timed out after allocation" >&2
    exit 75
  fi
  echo "[sbasis-sft] CUDA preflight $preflight failed; retrying" >&2
  sleep 30
done

echo "[sbasis-sft] init=$INIT data=$DATA out=$OUT epochs=$EPOCHS bs=$BS"
"$PY" sft.py \
  --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --epochs "$EPOCHS" --batch-size "$BS" --lr-muon "$LRMUON" --lr-adam "$LRADAM" \
  --warmup 50 --pack-len 2048 --group-field training_group --compile --out "$OUT"
[ -s "$OUT/sft_ep${EPOCHS}.pt" ] || { echo "[sbasis-sft] expected final checkpoint missing" >&2; exit 4; }
echo "[sbasis-sft] done checkpoint=$OUT/sft_ep${EPOCHS}.pt"
