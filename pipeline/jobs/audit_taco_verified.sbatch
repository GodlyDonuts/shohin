#!/bin/bash
#SBATCH -J shohin-taco-full-audit
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 06:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/taco_full_audit_%j.out
#SBATCH --open-mode=append
# Re-execute all available bounded tests for a frozen pilot. This is an
# admission gate, not a training-data writer and never mutates its input.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
IN=${IN:?set IN to the initial TACO pilot JSONL}
OUT=${OUT:?set OUT to a fresh full-verified JSONL}
REPORT=${REPORT:-${OUT%.jsonl}.quality.json}

[ -s "$IN" ] || { echo "[taco-full-audit] input missing: $IN" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[taco-full-audit] output exists: $OUT" >&2; exit 3; }
[ ! -e "$OUT.partial" ] || { echo "[taco-full-audit] stale partial: $OUT.partial" >&2; exit 4; }
cd "$BASE"
"$PY" pipeline/audit_taco_verified.py --input "$IN" --out "$OUT"
"$PY" pipeline/audit_sft_quality.py --data "$OUT" --evals "$BASE/artifacts/evals" --out "$REPORT"
"$PY" - "$REPORT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
bad = [key for key in (
    "malformed_rows", "missing_question_or_response", "duplicate_normalized_questions",
) if report.get(key)]
overlap = report.get("eval_overlap", {})
if report.get("valid_rows", 0) == 0 or bad or overlap.get("exact_prompt_hits") or overlap.get("ngram_prompt_hits"):
    raise SystemExit(f"[taco-full-audit] quality gate failed: rows={report.get('valid_rows')} bad={bad} overlap={overlap}")
print(f"[taco-full-audit] quality gate passed: {report['valid_rows']} rows")
PY
