#!/bin/bash
#SBATCH -J shohin-taco-pilot
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/taco_verified_pilot_%j.out
#SBATCH --open-mode=append
# Build a small, isolated execution-verified code source. It must pass the
# generic full-evaluation audit before a larger scale run or any SFT inclusion.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:-$BASE/artifacts/sft/taco_verified_pilot.jsonl}
REPORT=${REPORT:-$BASE/artifacts/sft/taco_verified_pilot.quality.json}

[ ! -e "$OUT" ] || { echo "[taco-pilot] output exists: $OUT" >&2; exit 2; }
[ ! -e "$OUT.partial" ] || { echo "[taco-pilot] stale partial: $OUT.partial" >&2; exit 3; }
cd "$BASE"
"$PY" pipeline/curate_taco_verified.py --out "$OUT" --evals "$BASE/artifacts/evals" \
  --max-seen "${MAX_SEEN:-1000}" --max-kept "${MAX_KEPT:-250}" --max-tests "${MAX_TESTS:-3}"
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
    raise SystemExit(f"[taco-pilot] quality gate failed: rows={report.get('valid_rows')} bad={bad} overlap={overlap}")
print(f"[taco-pilot] quality gate passed: {report['valid_rows']} rows")
PY
