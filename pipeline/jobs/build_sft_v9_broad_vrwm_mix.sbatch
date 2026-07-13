#!/bin/bash
#SBATCH -J shohin-sft-v9-mix
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v9_mix_%j.out
#SBATCH --open-mode=append
# Build an auditable broad-anchor + low-share VRWM candidate. It never changes
# the flagship, frozen V8 data, or VRWM source data.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
V8=$BASE/artifacts/sft/sft_mix_reasoning_v8_candidate_r2.jsonl
VRWM=$BASE/artifacts/sft/vrwm_v2_scratch_r4.jsonl
OUT=${OUT:-$BASE/artifacts/sft/sft_mix_reasoning_v9_broad_vrwm_candidate.jsonl}
MIX_REPORT=${MIX_REPORT:-$BASE/artifacts/sft/sft_mix_reasoning_v9_broad_vrwm_candidate.mix.r1.json}
QUALITY=${QUALITY:-$BASE/artifacts/sft/sft_mix_reasoning_v9_broad_vrwm_candidate.quality.r1.json}
FULL_TEXT=${FULL_TEXT:-$BASE/artifacts/sft/sft_mix_reasoning_v9_broad_vrwm_candidate.training_text_overlap.r1.json}
PACKING=${PACKING:-$BASE/artifacts/sft/sft_mix_reasoning_v9_broad_vrwm_candidate.packing.r1.json}
WEIGHTS=(math=0.40 procedural=0.42 code=0.08 teacher=0.05 vrwm=0.05)

for path in "$V8" "$VRWM"; do
  [ -s "$path" ] || { echo "[v9-mix] required input missing: $path" >&2; exit 2; }
done
for path in "$OUT" "$MIX_REPORT" "$QUALITY" "$FULL_TEXT" "$PACKING"; do
  [ ! -e "$path" ] || { echo "[v9-mix] refusing existing output: $path" >&2; exit 3; }
done

cd "$BASE"
"$PY" pipeline/build_sft_multisource_mix.py --inputs "$V8" "$VRWM" --out "$OUT" --report "$MIX_REPORT"
"$PY" pipeline/audit_sft_quality.py --data "$OUT" --evals "$BASE/artifacts/evals" --out "$QUALITY"
"$PY" pipeline/audit_training_text_overlap.py --data "$OUT" --evals "$BASE/artifacts/evals" \
  --out "$FULL_TEXT" --require-zero
cd "$BASE/train"
"$PY" inspect_sft_packing.py --data "$OUT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$PACKING" --weights "${WEIGHTS[@]}"
"$PY" - "$MIX_REPORT" "$QUALITY" "$FULL_TEXT" "$PACKING" <<'PY'
import json
import sys

mix, quality, text, packing = (json.load(open(path)) for path in sys.argv[1:])
if mix["malformed_json_rows"] or mix["missing_question_response_or_group"] or mix["duplicate_normalized_questions_dropped"]:
    raise SystemExit("[v9-mix] cross-source structural gate failed")
if quality["malformed_rows"] or quality["missing_question_or_response"] or quality["duplicate_normalized_questions"]:
    raise SystemExit("[v9-mix] quality gate failed")
if quality["eval_overlap"]["exact_prompt_hits"] or quality["eval_overlap"]["ngram_prompt_hits"]:
    raise SystemExit("[v9-mix] prompt overlap gate failed")
if text["overlap"]["exact_rows"] or text["overlap"]["ngram_rows"]:
    raise SystemExit("[v9-mix] full-text overlap gate failed")
expected = {"math": 0.40, "procedural": 0.42, "code": 0.08, "teacher": 0.05, "vrwm": 0.05}
if {key: float(value) for key, value in packing["requested_weights"].items()} != expected:
    raise SystemExit("[v9-mix] weight binding failed")
if set(packing["group_counts"]) != set(expected) or packing["packed_sequences"] < 90_000:
    raise SystemExit("[v9-mix] insufficient packed capacity")
if any(float(value) > 3.0 for value in packing["repeat_factor_per_epoch"].values()):
    raise SystemExit("[v9-mix] unsafe replay factor")
if float(packing["repeat_factor_per_epoch"]["vrwm"]) > 0.30:
    raise SystemExit("[v9-mix] VRWM share is too concentrated")
print("[v9-mix] candidate admitted", json.dumps({
    "rows": quality["valid_rows"], "packs": packing["packed_sequences"],
    "replay": packing["repeat_factor_per_epoch"],
}, sort_keys=True))
PY
