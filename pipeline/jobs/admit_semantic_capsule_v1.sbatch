#!/bin/bash
#SBATCH -J shohin-semcapsule-admit
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/semantic_capsule_admit_%j.out
#SBATCH --open-mode=append
# Read-only admission gates for semantic-capsule data. This does not train,
# modify the flagship, or construct a mixed SFT dataset.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:-$BASE/artifacts/sft/semantic_capsule_v1_train.jsonl}
EVALS=${EVALS:-$BASE/artifacts/evals}
PROTOCOL=${PROTOCOL:-$BASE/artifacts/sft/semantic_capsule_v1_train.protocol_audit.r1.json}
QUALITY=${QUALITY:-$BASE/artifacts/sft/semantic_capsule_v1_train.quality.r1.json}
FULL_TEXT=${FULL_TEXT:-$BASE/artifacts/sft/semantic_capsule_v1_train.training_text_overlap.r1.json}
CONTRACTS=${CONTRACTS:-$BASE/artifacts/sft/semantic_capsule_v1_train.response_contracts.r1.json}
PACKING=${PACKING:-$BASE/artifacts/sft/semantic_capsule_v1_train.packing.r1.json}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

[ -s "$DATA" ] || { echo "[semantic-capsule-admit] missing data: $DATA" >&2; exit 2; }
[ -s "$PROTOCOL" ] || { echo "[semantic-capsule-admit] missing protocol audit: $PROTOCOL" >&2; exit 2; }
for out in "$QUALITY" "$FULL_TEXT" "$CONTRACTS" "$PACKING"; do
  [ ! -e "$out" ] || { echo "[semantic-capsule-admit] refusing existing output: $out" >&2; exit 3; }
done

cd "$BASE"
"$PY" pipeline/audit_sft_quality.py --data "$DATA" --evals "$EVALS" --out "$QUALITY"
"$PY" pipeline/audit_training_text_overlap.py --data "$DATA" --evals "$EVALS" --out "$FULL_TEXT" \
  --fields question response --require-zero
"$PY" pipeline/audit_response_contracts.py --data "$DATA" --out "$CONTRACTS"
cd "$BASE/train"
"$PY" inspect_sft_packing.py --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$PACKING" --pack-len 2048 --group-field training_group --weights semantic_capsule=1
"$PY" - "$DATA" "$PROTOCOL" "$QUALITY" "$FULL_TEXT" "$CONTRACTS" "$PACKING" <<'PY'
import hashlib
import json
import os
import sys

data, protocol_path, quality_path, text_path, contracts_path, packing_path = map(os.path.realpath, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

digest = sha256(data)
protocol, quality, text, contracts, packing = (
    json.load(open(path)) for path in (protocol_path, quality_path, text_path, contracts_path, packing_path)
)
if protocol.get("data_sha256") != digest:
    raise SystemExit("[semantic-capsule-admit] protocol audit is bound to another data file")
if protocol.get("malformed_train_rows") or protocol.get("duplicate_normalized_train_questions") or protocol.get("invalid_heldout_episodes"):
    raise SystemExit("[semantic-capsule-admit] protocol structural gate failed")
if protocol.get("overlap", {}).get("exact_prompt_hits") or protocol.get("overlap", {}).get("ngram13_hits"):
    raise SystemExit("[semantic-capsule-admit] all-controller-prompt overlap gate failed")
if quality.get("data_sha256") != digest or quality.get("malformed_rows") or quality.get("missing_question_or_response") or quality.get("duplicate_normalized_questions"):
    raise SystemExit("[semantic-capsule-admit] quality gate failed")
if quality.get("eval_overlap", {}).get("exact_prompt_hits") or quality.get("eval_overlap", {}).get("ngram_prompt_hits"):
    raise SystemExit("[semantic-capsule-admit] prompt overlap gate failed")
if text.get("data_sha256") != digest or text.get("malformed_rows") or text.get("malformed_json_rows"):
    raise SystemExit("[semantic-capsule-admit] full-text structural gate failed")
if text.get("overlap", {}).get("exact_rows") or text.get("overlap", {}).get("ngram_rows"):
    raise SystemExit("[semantic-capsule-admit] full-text overlap gate failed")
if contracts.get("data_sha256") != digest or contracts.get("malformed_rows") or contracts.get("missing_response_rows"):
    raise SystemExit("[semantic-capsule-admit] response-contract gate failed")
groups = contracts.get("groups", {})
if set(groups) != {"semantic_capsule"} or groups["semantic_capsule"].get("starts_think_rate") != 1.0:
    raise SystemExit("[semantic-capsule-admit] trace response contract failed")
if packing.get("data_sha256", {}).get(data) != digest or packing.get("pack_len") != 2048:
    raise SystemExit("[semantic-capsule-admit] packing digest or length mismatch")
if set(packing.get("group_counts", {})) != {"semantic_capsule"} or packing.get("packed_sequences", 0) < 20_000:
    raise SystemExit("[semantic-capsule-admit] insufficient exact packed capacity")
print("[semantic-capsule-admit] passed rows={} packs={}".format(quality["valid_rows"], packing["packed_sequences"]))
PY
