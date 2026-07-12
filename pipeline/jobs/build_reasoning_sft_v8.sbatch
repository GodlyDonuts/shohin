#!/bin/bash
#SBATCH -J shohin-sft-mix-v8
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_mix_v8_%j.out
#SBATCH --open-mode=append
# Freeze a broader, source-balanced future SFT candidate. This job only writes
# a fresh immutable data snapshot; it never changes the flagship or starts SFT.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:-$BASE/artifacts/sft/sft_mix_reasoning_v8_candidate.jsonl}
REPORT=${REPORT:-$BASE/artifacts/sft/sft_mix_reasoning_v8_candidate.report.json}
QUALITY=${QUALITY:-$BASE/artifacts/sft/sft_mix_reasoning_v8_candidate.quality.json}
MIN_CODE_ROWS=${MIN_CODE_ROWS:-5000}

INPUTS=(
  "$BASE/artifacts/sft/openmath2_600k.jsonl"
  "$BASE/artifacts/sft/openr1_math_default_solution_filtered_full.jsonl"
  "$BASE/artifacts/rg_v4/rg_traces_sft.jsonl"
  "$BASE/artifacts/sft/code_completion_v1.jsonl"
  "$BASE/artifacts/sft/code_contests_train_v2.jsonl"
  "$BASE/artifacts/sft/hy3_reasoning_claude.jsonl"
  "$BASE/artifacts/sft/hy3_reasoning_glm.jsonl"
  "$BASE/artifacts/sft/hy3_reasoning_nemotron.jsonl"
  "$BASE/artifacts/sft/hy3_reasoning.jsonl"
  "$BASE/artifacts/sft/hy3_gsm8k.jsonl"
  "$BASE/artifacts/sft/hy3_arc_challenge.jsonl"
)

[ ! -e "$OUT" ] || { echo "[sft-v8] refusing to overwrite $OUT" >&2; exit 2; }
for path in "${INPUTS[@]}"; do
  [ -s "$path" ] || { echo "[sft-v8] required input missing or empty: $path" >&2; exit 3; }
done

cd "$BASE"
"$PY" pipeline/build_sft_mix.py --inputs "${INPUTS[@]}" \
  --out "$OUT" --report "$REPORT" --max-openmath 600000
"$PY" pipeline/audit_sft_quality.py --data "$OUT" --out "$QUALITY" \
  --evals "$BASE/artifacts/evals"
"$PY" - "$QUALITY" "$MIN_CODE_ROWS" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
minimum_code = int(sys.argv[2])
if report["malformed_rows"] or report["missing_question_or_response"]:
    raise SystemExit(f"[sft-v8] malformed or incomplete rows: {report}")
if report["duplicate_normalized_questions"]:
    raise SystemExit(f"[sft-v8] duplicate prompts after freezing: {report}")
overlap = report.get("eval_overlap", {})
if overlap.get("exact_prompt_hits") or overlap.get("ngram_prompt_hits"):
    raise SystemExit(f"[sft-v8] eval overlap after freezing: {overlap}")
groups = report.get("training_group_counts", {})
for group in ("math", "procedural", "code"):
    if groups.get(group, 0) <= 0:
        raise SystemExit(f"[sft-v8] required group absent: {group}; got {groups}")
if groups["code"] < minimum_code:
    raise SystemExit(f"[sft-v8] insufficient verified code rows: {groups['code']} < {minimum_code}")
print(f"[sft-v8] quality gate passed: {groups}")
PY
echo "[sft-v8] committed $OUT"
