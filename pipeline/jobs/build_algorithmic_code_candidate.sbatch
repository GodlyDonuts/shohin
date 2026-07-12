#!/bin/bash
#SBATCH -J shohin-algocode-candidate
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/algocode_candidate_%j.out
#SBATCH --open-mode=append
# Freeze one all-test-verified algorithmic-code source into a standalone SFT
# candidate. It never creates a mixed SFT corpus or starts a GPU training job.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
IN=${IN:?set IN to an all-test-verified algorithmic-code JSONL}
OUT=${OUT:?set OUT to a fresh frozen candidate JSONL}
REPORT=${REPORT:-${OUT%.jsonl}.report.json}
QUALITY=${QUALITY:-${OUT%.jsonl}.quality.json}
MIN_ROWS=${MIN_ROWS:-1000}

[ -s "$IN" ] || { echo "[algocode] input missing: $IN" >&2; exit 2; }
for path in "$OUT" "$REPORT" "$QUALITY"; do
  [ ! -e "$path" ] || { echo "[algocode] refusing existing output: $path" >&2; exit 3; }
done
cd "$BASE"
"$PY" pipeline/build_sft_mix.py --inputs "$IN" --out "$OUT" --report "$REPORT"
"$PY" pipeline/audit_sft_quality.py --data "$OUT" --out "$QUALITY" --evals "$BASE/artifacts/evals"
"$PY" - "$REPORT" "$QUALITY" "$MIN_ROWS" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
quality = json.load(open(sys.argv[2]))
minimum = int(sys.argv[3])
groups = report.get("kept_by_training_group", {})
bad = [key for key in (
    "malformed_rows", "missing_question_or_response", "duplicate_normalized_questions",
) if quality.get(key)]
overlap = quality.get("eval_overlap", {})
if bad or overlap.get("exact_prompt_hits") or overlap.get("ngram_prompt_hits"):
    raise SystemExit(f"[algocode] quality gate failed: bad={bad} overlap={overlap}")
if groups.get("algorithmic_code", 0) < minimum:
    raise SystemExit(f"[algocode] insufficient algorithmic rows: {groups}")
if set(groups) - {"algorithmic_code"}:
    raise SystemExit(f"[algocode] unexpected training groups: {groups}")
print(f"[algocode] quality gate passed: {groups}")
PY
