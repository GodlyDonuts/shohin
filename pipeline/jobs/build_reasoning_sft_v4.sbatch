#!/bin/bash
#SBATCH -J shohin-sft-mix-v4
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_mix_v4_%j.out
#SBATCH --open-mode=append
# Freeze a source-balanced, decontaminated SFT candidate. This job only builds
# immutable data artifacts; a separate SFT/eval chain must still earn promotion.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=$BASE/artifacts/sft/sft_mix_reasoning_v4.jsonl
REPORT=$BASE/artifacts/sft/sft_mix_reasoning_v4.report.json
QUALITY=$BASE/artifacts/sft/sft_mix_reasoning_v4.quality.json
MIN_CODE_ROWS=${MIN_CODE_ROWS:-1000}

INPUTS=(
  "$BASE/artifacts/sft/openmath2_600k.jsonl"
  "$BASE/artifacts/rg_v4/rg_traces_sft.jsonl"
  "$BASE/artifacts/sft/code_completion_v1.jsonl"
  "$BASE/artifacts/sft/hy3_reasoning_claude.jsonl"
  "$BASE/artifacts/sft/hy3_reasoning_glm.jsonl"
  "$BASE/artifacts/sft/hy3_reasoning_nemotron.jsonl"
  "$BASE/artifacts/sft/hy3_reasoning.jsonl"
  "$BASE/artifacts/sft/hy3_gsm8k.jsonl"
  "$BASE/artifacts/sft/hy3_arc_challenge.jsonl"
)

[ ! -e "$OUT" ] || { echo "[sft-v4] refusing to overwrite $OUT" >&2; exit 2; }
for path in "${INPUTS[@]}"; do
  [ -s "$path" ] || { echo "[sft-v4] required input missing or empty: $path" >&2; exit 3; }
done

cd "$BASE"
"$PY" pipeline/build_sft_mix.py --inputs "${INPUTS[@]}" \
  --out "$OUT" --report "$REPORT" --max-openmath 600000
"$PY" pipeline/audit_sft_quality.py --data "$OUT" --out "$QUALITY"
"$PY" - "$QUALITY" "$MIN_CODE_ROWS" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
minimum_code = int(sys.argv[2])
if report["malformed_rows"] or report["missing_question_or_response"] or report["duplicate_normalized_questions"]:
    raise SystemExit(f"[sft-v4] quality failure: {report}")
groups = report.get("training_group_counts", {})
for group in ("math", "procedural", "code"):
    if groups.get(group, 0) <= 0:
        raise SystemExit(f"[sft-v4] required group absent: {group}; got {groups}")
if groups["code"] < minimum_code:
    raise SystemExit(f"[sft-v4] insufficient verified code rows: {groups['code']} < {minimum_code}")
print(f"[sft-v4] quality gate passed: {groups}")
PY
echo "[sft-v4] committed $OUT"
