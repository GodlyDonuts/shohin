#!/bin/bash
# Build and independently admit two matched counterfactual-reflection arms on
# CPU only. This job cannot write a checkpoint, submit an SFT, or touch the
# flagship data stream.
#SBATCH -J shohin-counterfact-data
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/counterfact_data_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=${PY:-/usr/bin/python3.11}
TOKPY=${TOKPY:-$BASE/miniforge3/bin/python}
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
EVALS=$BASE/artifacts/evals
PER_FAMILY=${PER_FAMILY:-10000}
PREFIX=${PREFIX:-$BASE/artifacts/sft/operator_counterfactual_reflection_v1}
REFLECTION=${REFLECTION:-${PREFIX}_reflection.jsonl}
NEUTRAL=${NEUTRAL:-${PREFIX}_neutral.jsonl}
BUILD_REPORT=${BUILD_REPORT:-${PREFIX}.build.json}
AUDIT=${AUDIT:-${PREFIX}.audit.json}

for path in "$TOKENIZER" "$EVALS" "$BASE/train/thinking_trace_audit.py" \
  "$EVALS/trace_state_ood_v2.json" "$EVALS/operator_trace_contrast_v2_eval.json"; do
  [ -s "$path" ] || { echo "[counterfact-data] missing input: $path" >&2; exit 2; }
done
[ -x "$TOKPY" ] || { echo "[counterfact-data] tokenizer Python missing: $TOKPY" >&2; exit 2; }

REPORTS=(
  "${PREFIX}_reflection.quality.json" "${PREFIX}_neutral.quality.json"
  "${PREFIX}_reflection.training_text_overlap.json" "${PREFIX}_neutral.training_text_overlap.json"
  "${PREFIX}_reflection.trace_overlap.json" "${PREFIX}_neutral.trace_overlap.json"
  "${PREFIX}_reflection.ood_overlap.json" "${PREFIX}_neutral.ood_overlap.json"
  "${PREFIX}_reflection.factor_overlap.json" "${PREFIX}_neutral.factor_overlap.json"
)
for path in "$REFLECTION" "$NEUTRAL" "$BUILD_REPORT" "$AUDIT" "${REPORTS[@]}"; do
  [ ! -e "$path" ] || { echo "[counterfact-data] refusing existing output: $path" >&2; exit 3; }
done

cd "$BASE"
"$PY" pipeline/generate_operator_counterfactual_reflection_v1.py \
  --reflection-out "$REFLECTION" --neutral-out "$NEUTRAL" --report-out "$BUILD_REPORT" \
  --per-family "$PER_FAMILY"
"$TOKPY" pipeline/audit_operator_counterfactual_reflection_v1.py \
  --reflection "$REFLECTION" --neutral "$NEUTRAL" --tokenizer "$TOKENIZER" --out "$AUDIT"

for arm in reflection neutral; do
  if [ "$arm" = reflection ]; then
    data=$REFLECTION
  else
    data=$NEUTRAL
  fi
  "$PY" pipeline/audit_sft_quality.py --data "$data" --evals "$EVALS" \
    --out "${PREFIX}_${arm}.quality.json"
  "$PY" pipeline/audit_training_text_overlap.py --data "$data" --evals "$EVALS" \
    --out "${PREFIX}_${arm}.training_text_overlap.json" --require-zero
  "$PY" pipeline/audit_generalization_overlap.py --data "$data" \
    --case-source "$BASE/train/thinking_trace_audit.py" --out "${PREFIX}_${arm}.trace_overlap.json"
  "$PY" pipeline/audit_generalization_overlap.py --data "$data" \
    --cases-json "$EVALS/trace_state_ood_v2.json" --out "${PREFIX}_${arm}.ood_overlap.json"
  "$PY" pipeline/audit_generalization_overlap.py --data "$data" \
    --cases-json "$EVALS/operator_trace_contrast_v2_eval.json" \
    --out "${PREFIX}_${arm}.factor_overlap.json"
done

"$PY" - "$REFLECTION" "$NEUTRAL" "$BUILD_REPORT" "$AUDIT" "${REPORTS[@]}" <<'PY'
import hashlib
import json
import os
import sys

reflection, neutral, build_path, audit_path, *reports = map(os.path.realpath, sys.argv[1:])
(
    reflection_quality, neutral_quality,
    reflection_text, neutral_text,
    reflection_trace, neutral_trace,
    reflection_ood, neutral_ood,
    reflection_factor, neutral_factor,
) = reports

def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

reflection_sha = digest(reflection)
neutral_sha = digest(neutral)
build = json.load(open(build_path))
audit = json.load(open(audit_path))
if build.get("reflection_sha256") != reflection_sha or build.get("neutral_sha256") != neutral_sha:
    raise SystemExit("[counterfact-data] build report does not bind both arms")
if audit.get("reflection_sha256") != reflection_sha or audit.get("neutral_sha256") != neutral_sha:
    raise SystemExit("[counterfact-data] semantic audit does not bind both arms")
if audit.get("matched_pairs") != build.get("rows_per_arm") or not audit.get("supervised_response_token_delta", {}).get("all_pairs_equal"):
    raise SystemExit("[counterfact-data] matched-response contract failed")
if not audit.get("prompt_token_delta_histogram"):
    raise SystemExit("[counterfact-data] prompt-token delta was not measured")
for path in (reflection_quality, neutral_quality):
    report = json.load(open(path))
    is_reflection = "_reflection." in os.path.basename(path)
    expected = reflection_sha if is_reflection else neutral_sha
    if report.get("data_sha256") != expected:
        raise SystemExit(f"[counterfact-data] report does not bind its arm: {path}")
    if report.get("malformed_rows") or report.get("missing_question_or_response") or report.get("duplicate_normalized_questions"):
        raise SystemExit(f"[counterfact-data] quality gate failed: {path}")
    public = report.get("eval_overlap", {})
    if public.get("exact_prompt_hits") or public.get("ngram_prompt_hits"):
        raise SystemExit(f"[counterfact-data] public-eval overlap remains: {path}")
for path in (reflection_text, neutral_text):
    report = json.load(open(path))
    is_reflection = "_reflection." in os.path.basename(path)
    expected = reflection_sha if is_reflection else neutral_sha
    if report.get("data_sha256") != expected or report.get("malformed_json_rows") or report.get("missing_fields"):
        raise SystemExit(f"[counterfact-data] full-text audit failed: {path}")
    overlap = report.get("overlap", {})
    if overlap.get("exact_rows") or overlap.get("ngram_rows"):
        raise SystemExit(f"[counterfact-data] training-text eval overlap remains: {path}")
for path, expected_cases in (
    (reflection_trace, 12), (neutral_trace, 12),
    (reflection_ood, 72), (neutral_ood, 72),
    (reflection_factor, 900), (neutral_factor, 900),
):
    report = json.load(open(path))
    is_reflection = "_reflection." in os.path.basename(path)
    expected = reflection_sha if is_reflection else neutral_sha
    if report.get("data_sha256") != expected or report.get("cases") != expected_cases:
        raise SystemExit(f"[counterfact-data] held-out audit does not bind expected suite: {path}")
    if report.get("malformed_rows") or report.get("missing_question") or report.get("exact_prompt_hits") or report.get("ngram_prompt_hits"):
        raise SystemExit(f"[counterfact-data] held-out overlap remains: {path}")
print(json.dumps({
    "admitted": True,
    "reflection_sha256": reflection_sha,
    "neutral_sha256": neutral_sha,
    "rows_per_arm": build["rows_per_arm"],
    "prompt_token_delta_histogram": audit["prompt_token_delta_histogram"],
}, sort_keys=True))
PY
echo "[counterfact-data] admitted matched auxiliary arms: $REFLECTION $NEUTRAL"
