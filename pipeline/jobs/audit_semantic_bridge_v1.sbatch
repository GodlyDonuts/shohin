#!/bin/bash
#SBATCH -J shohin-sembridge-audit
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/semantic_bridge_audit_%j.out
#SBATCH --open-mode=append
# Read-only admission audit for semantic bridge data. This does not build a
# mixed SFT file and cannot modify active pretraining.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:-$BASE/artifacts/sft/semantic_bridge_v1_train.jsonl}
EVALS=${EVALS:-$BASE/artifacts/evals}
QUALITY=${QUALITY:-$BASE/artifacts/sft/semantic_bridge_v1_train.quality.r1.json}
FULL_TEXT=${FULL_TEXT:-$BASE/artifacts/sft/semantic_bridge_v1_train.training_text_overlap.r1.json}
CONTRACTS=${CONTRACTS:-$BASE/artifacts/sft/semantic_bridge_v1_train.response_contracts.r1.json}
PACKING=${PACKING:-$BASE/artifacts/sft/semantic_bridge_v1_train.packing.r1.json}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

[ -s "$DATA" ] || { echo "[semantic-bridge-audit] missing data: $DATA" >&2; exit 2; }
for out in "$QUALITY" "$FULL_TEXT" "$CONTRACTS" "$PACKING"; do
  [ ! -e "$out" ] || { echo "[semantic-bridge-audit] refusing existing report: $out" >&2; exit 3; }
done
cd "$BASE"
"$PY" pipeline/audit_sft_quality.py --data "$DATA" --evals "$EVALS" --out "$QUALITY"
"$PY" pipeline/audit_training_text_overlap.py --data "$DATA" --evals "$EVALS" --out "$FULL_TEXT" \
  --fields question response --require-zero
"$PY" pipeline/audit_response_contracts.py --data "$DATA" --out "$CONTRACTS"
cd "$BASE/train"
"$PY" inspect_sft_packing.py --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$PACKING" --pack-len 2048 --group-field training_group --weights semantic_bridge=1
"$PY" - "$DATA" "$QUALITY" "$FULL_TEXT" "$CONTRACTS" "$PACKING" <<'PY'
import hashlib
import json
import os
import sys

data, quality_path, full_text_path, contracts_path, packing_path = map(os.path.realpath, sys.argv[1:])
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
digest = sha256(data)
quality, full_text, contracts, packing = (json.load(open(path)) for path in (quality_path, full_text_path, contracts_path, packing_path))
if quality.get("data_sha256") != digest or quality.get("malformed_rows") or quality.get("missing_question_or_response") or quality.get("duplicate_normalized_questions"):
    raise SystemExit("[semantic-bridge-audit] quality gate failed")
if quality.get("eval_overlap", {}).get("exact_prompt_hits") or quality.get("eval_overlap", {}).get("ngram_prompt_hits"):
    raise SystemExit("[semantic-bridge-audit] prompt overlap gate failed")
if full_text.get("data_sha256") != digest or full_text.get("overlap", {}).get("exact_rows") or full_text.get("overlap", {}).get("ngram_rows"):
    raise SystemExit("[semantic-bridge-audit] full-text overlap gate failed")
if contracts.get("data_sha256") != digest or contracts.get("malformed_rows") or contracts.get("missing_response_rows"):
    raise SystemExit("[semantic-bridge-audit] response-contract gate failed")
if set(contracts.get("groups", {})) != {"semantic_bridge"} or contracts["groups"]["semantic_bridge"].get("starts_think_rate") != 1.0:
    raise SystemExit("[semantic-bridge-audit] bridge trace contract failed")
if packing.get("data_sha256", {}).get(data) != digest or packing.get("pack_len") != 2048 or packing.get("packed_sequences", 0) < 6_000:
    raise SystemExit("[semantic-bridge-audit] insufficient exact packing")
print(f"[semantic-bridge-audit] passed rows={quality['valid_rows']} packs={packing['packed_sequences']}")
PY
