#!/bin/bash
#SBATCH -J shohin-verifier-data
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/verifier_data_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
IN=${IN:?set IN to one or more space-separated train-only rollout JSONLs}
OUT=${OUT:?set OUT to a fresh verifier JSONL}
BALANCE_CLASSES=${BALANCE_CLASSES:-0}
cd "$BASE"
read -r -a IN_PATHS <<< "$IN"
[ "${#IN_PATHS[@]}" -gt 0 ] || { echo "[verifier-data] no input paths" >&2; exit 2; }
for path in "${IN_PATHS[@]}"; do
  [ -s "$path" ] || { echo "[verifier-data] input missing: $path" >&2; exit 2; }
done
ARGS=(--input "${IN_PATHS[@]}" --out "$OUT")
[ "$BALANCE_CLASSES" = 1 ] && ARGS+=(--balance-classes)
"$PY" pipeline/build_verifier_dataset.py "${ARGS[@]}"
REPORT="${OUT%.jsonl}.quality.json"
"$PY" pipeline/audit_sft_quality.py --data "$OUT" --out "$REPORT" --evals "$BASE/artifacts/evals"
"$PY" - "$REPORT" "$BALANCE_CLASSES" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
bad = [key for key in (
    "malformed_rows", "missing_question_or_response", "duplicate_normalized_questions",
) if report.get(key)]
overlap = report.get("eval_overlap", {})
if report.get("valid_rows", 0) == 0 or bad or overlap.get("exact_prompt_hits") or overlap.get("ngram_prompt_hits"):
    raise SystemExit(f"[verifier-data] quality gate failed: rows={report.get('valid_rows')} bad={bad} overlap={overlap}")
print(f"[verifier-data] quality gate passed: {report['valid_rows']} rows")
PY
