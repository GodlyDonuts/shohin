#!/bin/bash
# CPU-only construction and independent admission of a conditional CWI corpus.
# It cannot allocate a GPU, write a model, or alter the flagship or live shards.
# CWI remains unsubmitted for SFT until STRR first proves state transport.
#SBATCH -J shohin-cwi-build
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/counterfactual_workspace_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-/usr/bin/python3.12}
FACTOR_TRAIN=${FACTOR_TRAIN:-$BASE/artifacts/sft/digitwise_factor_v1_train.jsonl}
FACTOR_HELDOUT=${FACTOR_HELDOUT:-$BASE/artifacts/evals/digitwise_factor_v1_heldout.jsonl}
TRAIN=${TRAIN:-$BASE/artifacts/sft/counterfactual_workspace_v1_train.jsonl}
HELDOUT=${HELDOUT:-$BASE/artifacts/evals/counterfactual_workspace_v1_heldout.jsonl}
BUILD=${BUILD:-$BASE/artifacts/sft/counterfactual_workspace_v1_train.build.json}
AUDIT=${AUDIT:-$BASE/artifacts/sft/counterfactual_workspace_v1_train.protocol_audit.json}

for path in "$FACTOR_TRAIN" "$FACTOR_HELDOUT"; do
  [ -s "$path" ] || { echo "[cwi-build] required factor input missing: $path" >&2; exit 2; }
done
for path in "$TRAIN" "$HELDOUT" "$BUILD" "$AUDIT"; do
  [ ! -e "$path" ] || { echo "[cwi-build] refusing existing artifact: $path" >&2; exit 3; }
done
cd "$BASE"
"$PY" pipeline/build_counterfactual_workspace_v1.py \
  --factor-train "$FACTOR_TRAIN" --factor-heldout "$FACTOR_HELDOUT" \
  --train-out "$TRAIN" --heldout-out "$HELDOUT" --report "$BUILD"
"$PY" pipeline/audit_counterfactual_workspace_v1.py \
  --train "$TRAIN" --heldout "$HELDOUT" --out "$AUDIT"
"$PY" - "$AUDIT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
for key in (
    "invalid_train_rows",
    "duplicate_train_identities",
    "duplicate_normalized_train_prompts",
    "invalid_heldout_rows",
    "duplicate_heldout_identities",
    "duplicate_normalized_heldout_prompts",
    "missing_legal_local_contexts",
    "invalid_heldout_world_pairs",
    "train_heldout_exact_prompt_hits",
    "train_heldout_13gram_hits",
):
    if report.get(key) != 0:
        raise SystemExit("[cwi-build] admission failed: {}={}".format(key, report.get(key)))
if report.get("covered_legal_local_contexts") != report.get("required_local_contexts"):
    raise SystemExit("[cwi-build] local-context coverage is incomplete")
if set(report.get("train_foil_counts", ())) != {"legal", "carry", "result_digit", "program_counter", "tape"}:
    raise SystemExit("[cwi-build] train foil coverage is incomplete")
if report.get("valid_train_rows", 0) < 600000 or report.get("valid_heldout_rows", 0) < 50000:
    raise SystemExit("[cwi-build] insufficient CWI candidate capacity")
print("[cwi-build] admitted rows={} heldout={} pairs={}".format(
    report["valid_train_rows"], report["valid_heldout_rows"], report["heldout_world_pairs"]
))
PY
echo "[cwi-build] committed conditional CPU candidate"
