#!/bin/bash
# Freeze a broad anchored operator-trace SFT candidate on CPU only.  This job
# cannot write a checkpoint or alter the flagship's data stream.
#SBATCH -J shohin-optrace-anchor
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/optrace_anchor_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=${PY:-/usr/bin/python3.11}
BROAD=$BASE/artifacts/sft/sft_mix_reasoning_v4.jsonl
PRIMITIVES=$BASE/artifacts/sft/primitives_v1_train.jsonl
OPERATOR=$BASE/artifacts/sft/operator_trace_contrast_v2_train.jsonl
OUT=${OUT:-$BASE/artifacts/sft/operator_trace_v2_broad_anchor_v1.jsonl}
MIX=${MIX:-$BASE/artifacts/sft/operator_trace_v2_broad_anchor_v1.mix.json}
QUALITY=${QUALITY:-$BASE/artifacts/sft/operator_trace_v2_broad_anchor_v1.quality.json}
TEXT_OVERLAP=${TEXT_OVERLAP:-$BASE/artifacts/sft/operator_trace_v2_broad_anchor_v1.training_text_overlap.json}
TRACE_OVERLAP=${TRACE_OVERLAP:-$BASE/artifacts/sft/operator_trace_v2_broad_anchor_v1.trace_overlap.json}
OOD_OVERLAP=${OOD_OVERLAP:-$BASE/artifacts/sft/operator_trace_v2_broad_anchor_v1.ood_overlap.json}
FACTOR_OVERLAP=${FACTOR_OVERLAP:-$BASE/artifacts/sft/operator_trace_v2_broad_anchor_v1.factor_wf_overlap.json}
DIRECTNESS=${DIRECTNESS:-$BASE/artifacts/sft/operator_trace_v2_broad_anchor_v1.directness.json}
EXCLUDE_CONTRACTS=${EXCLUDE_CONTRACTS:-}
REQUIRED_OPERATOR_MIN_ROWS=${REQUIRED_OPERATOR_MIN_ROWS:-200000}

for path in "$BROAD" "$PRIMITIVES" "$OPERATOR"; do
  [ -s "$path" ] || { echo "[optrace-anchor] missing input: $path" >&2; exit 2; }
done
for path in "$OUT" "$MIX" "$QUALITY" "$TEXT_OVERLAP" "$TRACE_OVERLAP" "$OOD_OVERLAP" "$FACTOR_OVERLAP" "$DIRECTNESS"; do
  [ ! -e "$path" ] || { echo "[optrace-anchor] refusing existing output: $path" >&2; exit 3; }
done

cd "$BASE"
BUILD_ARGS=(--inputs "$BROAD" "$PRIMITIVES" "$OPERATOR" --out "$OUT" --report "$MIX" --eval-glob "$BASE"/artifacts/evals/*.jsonl)
if [ -n "$EXCLUDE_CONTRACTS" ]; then
  IFS=',' read -r -a CONTRACTS <<< "$EXCLUDE_CONTRACTS"
  for contract in "${CONTRACTS[@]}"; do
    BUILD_ARGS+=(--exclude-contract "$contract")
  done
fi
"$PY" pipeline/build_sft_multisource_mix.py \
  "${BUILD_ARGS[@]}"
"$PY" pipeline/audit_sft_quality.py --data "$OUT" --evals "$BASE/artifacts/evals" --out "$QUALITY"
"$PY" pipeline/audit_training_text_overlap.py --data "$OUT" --evals "$BASE/artifacts/evals" \
  --out "$TEXT_OVERLAP" --require-zero
"$PY" pipeline/audit_generalization_overlap.py --data "$OUT" \
  --case-source "$BASE/train/thinking_trace_audit.py" --out "$TRACE_OVERLAP"
"$PY" pipeline/audit_generalization_overlap.py --data "$OUT" \
  --cases-json "$BASE/artifacts/evals/trace_state_ood_v2.json" --out "$OOD_OVERLAP"
"$PY" pipeline/audit_generalization_overlap.py --data "$OUT" \
  --cases-json "$BASE/artifacts/evals/operator_trace_contrast_v2_eval.json" \
  --case-regimes wording full --out "$FACTOR_OVERLAP"
"$PY" pipeline/audit_operator_trace_directness.py --data "$OUT" --out "$DIRECTNESS" \
  --require-only-contract direct
"$PY" - "$OUT" "$MIX" "$QUALITY" "$TEXT_OVERLAP" "$TRACE_OVERLAP" "$OOD_OVERLAP" "$FACTOR_OVERLAP" "$DIRECTNESS" "$REQUIRED_OPERATOR_MIN_ROWS" <<'PY'
import hashlib
import json
import os
import sys

data, mix_path, quality_path, text_path, trace_path, ood_path, factor_path, directness_path = map(os.path.realpath, sys.argv[1:9])
required_operator_min_rows = int(sys.argv[9])

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

digest = sha256(data)
mix, quality, text, trace, ood, factor, directness = (json.load(open(path)) for path in (
    mix_path, quality_path, text_path, trace_path, ood_path, factor_path, directness_path
))
if mix.get("out") != data or mix.get("out_sha256") != digest:
    raise SystemExit("[optrace-anchor] mix report does not bind data")
if any(mix.get(key, 1) for key in (
    "malformed_json_rows", "missing_question_response_or_group", "duplicate_normalized_questions_dropped"
)):
    raise SystemExit("[optrace-anchor] structural mix gate failed")
if not mix.get("eval_filter", {}).get("enabled"):
    raise SystemExit("[optrace-anchor] hard public-eval filter was not enabled")
if quality.get("data_sha256") != digest or any(quality.get(key, 1) for key in (
    "malformed_rows", "missing_question_or_response", "duplicate_normalized_questions"
)):
    raise SystemExit("[optrace-anchor] quality report does not bind a clean candidate")
if any(quality.get("eval_overlap", {}).get(key, 1) for key in ("exact_prompt_hits", "ngram_prompt_hits")):
    raise SystemExit("[optrace-anchor] public-eval overlap remains")
if text.get("data_sha256") != digest or text.get("overlap", {}).get("exact_rows") or text.get("overlap", {}).get("ngram_rows"):
    raise SystemExit("[optrace-anchor] training-text overlap remains")
for name, report, cases in (("trace", trace, 12), ("ood", ood, 72), ("factor", factor, 600)):
    if report.get("data_sha256") != digest or report.get("cases") != cases:
        raise SystemExit(f"[optrace-anchor] {name} audit does not bind the expected case suite")
    if report.get("malformed_rows") or report.get("missing_question") or report.get("exact_prompt_hits") or report.get("ngram_prompt_hits"):
        raise SystemExit(f"[optrace-anchor] {name} held-out overlap remains")
if directness.get("data_sha256") != digest or directness.get("operator_rows", 0) < required_operator_min_rows:
    raise SystemExit("[optrace-anchor] directness audit does not bind enough operator rows")
if directness.get("contract_rows") != {"direct": directness.get("operator_rows")}:
    raise SystemExit("[optrace-anchor] non-direct operator rows remain")
if any(directness.get("pair_marker_rows", {}).values()):
    raise SystemExit("[optrace-anchor] paired-answer response grammar remains")
groups = mix.get("training_group_rows", {})
required = {"math", "procedural", "code", "teacher", "primitives", "operator_trace_contrast"}
if set(groups) != required or groups["operator_trace_contrast"] < required_operator_min_rows:
    raise SystemExit("[optrace-anchor] broad/primitive/operator group coverage is incomplete")
print(json.dumps({
    "admitted": True,
    "data_sha256": digest,
    "rows": quality["valid_rows"],
    "eval_filter_drops": mix["eval_filter"],
    "groups": groups,
    "excluded_contract_rows": mix.get("excluded_contract_rows", {}),
    "directness": directness,
}, sort_keys=True))
PY
echo "[optrace-anchor] admitted frozen broad anchor: $OUT"
