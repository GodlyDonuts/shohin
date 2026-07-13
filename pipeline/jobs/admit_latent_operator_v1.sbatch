#!/bin/bash
#SBATCH -J shohin-latent-admit
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/latent_operator_admit_%j.out
#SBATCH --open-mode=append
# Read-only generic admission after the solver/held-out protocol audit.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:-$BASE/artifacts/sft/latent_operator_v1_train.jsonl}
EVALS=${EVALS:-$BASE/artifacts/evals}
AUDIT=${AUDIT:-$BASE/artifacts/sft/latent_operator_v1.audit.r1.json}
QUALITY=${QUALITY:-$BASE/artifacts/sft/latent_operator_v1.quality.r1.json}
FULL_TEXT=${FULL_TEXT:-$BASE/artifacts/sft/latent_operator_v1.training_text_overlap.r1.json}
CONTRACTS=${CONTRACTS:-$BASE/artifacts/sft/latent_operator_v1.response_contracts.r1.json}

for path in "$DATA" "$AUDIT"; do
  [ -s "$path" ] || { echo "[latent-admit] required input missing: $path" >&2; exit 2; }
done
for path in "$QUALITY" "$FULL_TEXT" "$CONTRACTS"; do
  [ ! -e "$path" ] || { echo "[latent-admit] refusing existing report: $path" >&2; exit 3; }
done
cd "$BASE"
"$PY" pipeline/audit_sft_quality.py --data "$DATA" --evals "$EVALS" --out "$QUALITY"
"$PY" pipeline/audit_training_text_overlap.py --data "$DATA" --evals "$EVALS" --out "$FULL_TEXT" \
  --fields question response --require-zero
"$PY" pipeline/audit_response_contracts.py --data "$DATA" --out "$CONTRACTS"
"$PY" - "$DATA" "$AUDIT" "$QUALITY" "$FULL_TEXT" "$CONTRACTS" <<'PY'
import hashlib
import json
import os
import sys

data, audit_path, quality_path, text_path, contracts_path = map(os.path.realpath, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

digest = sha256(data)
audit, quality, text, contracts = (json.load(open(path)) for path in
                                   (audit_path, quality_path, text_path, contracts_path))
if audit.get("train_sha256") != digest:
    raise SystemExit("[latent-admit] solver audit does not bind the training data")
if any(audit.get(key) for key in ("invalid_train_rows", "invalid_heldout_rows",
                                  "duplicate_train_questions", "duplicate_heldout_questions")):
    raise SystemExit("[latent-admit] solver/structural gate failed")
if any(audit.get("overlap", {}).get(key) for key in ("exact_prompt_hits", "ngram13_hits")):
    raise SystemExit("[latent-admit] held-out overlap gate failed")
if quality.get("data_sha256") != digest or quality.get("malformed_rows") or quality.get("missing_question_or_response") or quality.get("duplicate_normalized_questions"):
    raise SystemExit("[latent-admit] quality report rejected data")
if any(quality.get("eval_overlap", {}).get(key) for key in ("exact_prompt_hits", "ngram_prompt_hits")):
    raise SystemExit("[latent-admit] public-eval overlap gate failed")
if text.get("data_sha256") != digest or text.get("malformed_rows") or text.get("malformed_json_rows"):
    raise SystemExit("[latent-admit] full-text report rejected data")
if any(text.get("overlap", {}).get(key) for key in ("exact_rows", "ngram_rows")):
    raise SystemExit("[latent-admit] full-text overlap gate failed")
if contracts.get("data_sha256") != digest or contracts.get("malformed_rows") or contracts.get("missing_response_rows"):
    raise SystemExit("[latent-admit] response contract rejected data")
groups = contracts.get("groups", {})
if set(groups) != {"latent_operator"}:
    raise SystemExit("[latent-admit] unexpected response groups")
contract = groups["latent_operator"]
if contract.get("think_marker_rate") != 0.0 or contract.get("state_marker_rate") != 0.0:
    raise SystemExit("[latent-admit] answer-only protocol was contaminated by serialized reasoning/state")
if contract.get("answer_marker_rate") != 1.0:
    raise SystemExit("[latent-admit] answer contract is incomplete")
print("[latent-admit] passed rows={} sha256={}".format(quality["valid_rows"], digest))
PY
