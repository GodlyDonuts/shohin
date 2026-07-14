#!/bin/bash
#SBATCH -J shohin-scratch-train
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc36,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/scratch_train_%j.out
#SBATCH --open-mode=append
# Frozen-base recurrent-scratch experiment. Never writes a flagship path.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:?set INIT to an immutable base checkpoint}
DATA=${DATA:-$BASE/artifacts/sft/latent_operator_v1_train.jsonl}
AUDIT=${AUDIT:-$BASE/artifacts/sft/latent_operator_v1.audit.r1.json}
QUALITY=${QUALITY:-$BASE/artifacts/sft/latent_operator_v1.quality.r1.json}
FULL_TEXT=${FULL_TEXT:-$BASE/artifacts/sft/latent_operator_v1.training_text_overlap.r1.json}
CONTRACTS=${CONTRACTS:-$BASE/artifacts/sft/latent_operator_v1.response_contracts.r1.json}
OUT=${OUT:?set OUT to a fresh isolated output directory}
MODE=${MODE:?set MODE to recurrent or reset}

for path in "$INIT" "$DATA" "$AUDIT" "$QUALITY" "$FULL_TEXT" "$CONTRACTS"; do
  [ -s "$path" ] || { echo "[scratch-train] required input missing: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[scratch-train] refusing existing output: $OUT" >&2; exit 3; }
case "$MODE" in recurrent|reset) ;; *) echo "[scratch-train] invalid MODE=$MODE" >&2; exit 4;; esac
case "$OUT" in "$BASE"/train/scratch_*) ;; *) echo "[scratch-train] OUT must be a scratch_* research path" >&2; exit 5;; esac

"$PY" - "$DATA" "$AUDIT" "$QUALITY" "$FULL_TEXT" "$CONTRACTS" <<'PY'
import hashlib
import json
import os
import sys

data, audit_path, quality_path, text_path, contracts_path = map(os.path.realpath, sys.argv[1:])
def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()
sha = digest(data)
audit, quality, text, contracts = (json.load(open(path)) for path in
                                   (audit_path, quality_path, text_path, contracts_path))
if audit.get("train_sha256") != sha or any(audit.get(key) for key in (
    "invalid_train_rows", "invalid_heldout_rows", "duplicate_train_questions", "duplicate_heldout_questions",
)):
    raise SystemExit("[scratch-train] solver audit is invalid")
if any(audit.get("overlap", {}).get(key) for key in ("exact_prompt_hits", "ngram13_hits")):
    raise SystemExit("[scratch-train] held-out overlap gate failed")
if quality.get("data_sha256") != sha or quality.get("malformed_rows") or quality.get("missing_question_or_response") or quality.get("duplicate_normalized_questions"):
    raise SystemExit("[scratch-train] quality gate failed")
if any(quality.get("eval_overlap", {}).get(key) for key in ("exact_prompt_hits", "ngram_prompt_hits")):
    raise SystemExit("[scratch-train] public-eval overlap gate failed")
if text.get("data_sha256") != sha or any(text.get("overlap", {}).get(key) for key in ("exact_rows", "ngram_rows")):
    raise SystemExit("[scratch-train] full-text overlap gate failed")
group = contracts.get("groups", {}).get("latent_operator", {})
if contracts.get("data_sha256") != sha or set(contracts.get("groups", {})) != {"latent_operator"}:
    raise SystemExit("[scratch-train] response contract does not bind data")
if group.get("think_marker_rate") != 0.0 or group.get("state_marker_rate") != 0.0 or group.get("answer_marker_rate") != 1.0:
    raise SystemExit("[scratch-train] data is not answer-only")
print("[scratch-train] immutable input gate passed sha256={}".format(sha))
PY

cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
except Exception as exc:
    print(f"[scratch-train] CUDA preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
PY
then
  exit 75
fi
"$PY" train_causal_recurrent_scratch.py \
  --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$OUT" --mode "$MODE" --layer "${LAYER:-19}" --slots "${SLOTS:-4}" \
  --width "${WIDTH:-96}" --steps "${STEPS:-4}" --workspace-topk "${WORKSPACE_TOPK:-8}" \
  --workspace-temperature "${WORKSPACE_TEMPERATURE:-0.2}" --epochs "${EPOCHS:-1}" \
  --batch-size "${BS:-8}" --max-examples "${MAX_EXAMPLES:-8192}" \
  --lr "${LR:-0.001}" --seed "${SEED:-20260714}"
