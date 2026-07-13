#!/bin/bash
#SBATCH -J shohin-latent-train
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/latent_train_%j.out
#SBATCH --open-mode=append
# Isolated continuous-latent experiment. It cannot write to flagship paths.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:?set INIT to an immutable pretraining checkpoint}
DATA=${DATA:-$BASE/artifacts/sft/latent_operator_v1_train.jsonl}
AUDIT=${AUDIT:-$BASE/artifacts/sft/latent_operator_v1.audit.r1.json}
QUALITY=${QUALITY:-$BASE/artifacts/sft/latent_operator_v1.quality.r1.json}
FULL_TEXT=${FULL_TEXT:-$BASE/artifacts/sft/latent_operator_v1.training_text_overlap.r1.json}
CONTRACTS=${CONTRACTS:-$BASE/artifacts/sft/latent_operator_v1.response_contracts.r1.json}
OUT=${OUT:?set OUT to a fresh isolated output directory}
MAX_LATENT_STEPS=${MAX_LATENT_STEPS:-4}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}

for path in "$INIT" "$DATA" "$AUDIT" "$QUALITY" "$FULL_TEXT" "$CONTRACTS"; do
  [ -s "$path" ] || { echo "[latent-train] required input missing: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[latent-train] refusing existing output: $OUT" >&2; exit 3; }
case "$MAX_LATENT_STEPS" in 0|1|2|4) ;; *) echo "[latent-train] unsupported max latent steps: $MAX_LATENT_STEPS" >&2; exit 4;; esac
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
if audit.get("train_sha256") != digest or any(audit.get(key) for key in ("invalid_train_rows", "invalid_heldout_rows", "duplicate_train_questions", "duplicate_heldout_questions")):
    raise SystemExit("[latent-train] solver audit is invalid")
if any(audit.get("overlap", {}).get(key) for key in ("exact_prompt_hits", "ngram13_hits")):
    raise SystemExit("[latent-train] held-out overlap gate failed")
if quality.get("data_sha256") != digest or quality.get("malformed_rows") or quality.get("missing_question_or_response") or quality.get("duplicate_normalized_questions"):
    raise SystemExit("[latent-train] quality gate failed")
if any(quality.get("eval_overlap", {}).get(key) for key in ("exact_prompt_hits", "ngram_prompt_hits")):
    raise SystemExit("[latent-train] public-eval overlap gate failed")
if text.get("data_sha256") != digest or any(text.get("overlap", {}).get(key) for key in ("exact_rows", "ngram_rows")):
    raise SystemExit("[latent-train] full-text overlap gate failed")
group = contracts.get("groups", {}).get("latent_operator", {})
if contracts.get("data_sha256") != digest or set(contracts.get("groups", {})) != {"latent_operator"}:
    raise SystemExit("[latent-train] response contract does not bind data")
if group.get("think_marker_rate") != 0.0 or group.get("state_marker_rate") != 0.0 or group.get("answer_marker_rate") != 1.0:
    raise SystemExit("[latent-train] response contract is not answer-only")
print("[latent-train] immutable input gate passed sha256={}".format(digest))
PY
cd "$BASE/train"
"$PY" - <<'PY'
import sys
import torch
try:
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[latent-train] CUDA unavailable: {exc}", file=sys.stderr)
    sys.exit(75)
PY
"$PY" latent_rollout_train.py --init "$INIT" --data "$DATA" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" --out "$OUT" \
  --epochs "${EPOCHS:-1}" --batch-size "${BS:-16}" --max-latent-steps "$MAX_LATENT_STEPS" \
  --max-examples "$MAX_EXAMPLES" --seed "${SEED:-20260713}"
