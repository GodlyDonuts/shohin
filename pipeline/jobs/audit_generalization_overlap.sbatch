#!/bin/bash
#SBATCH -J shohin-generalize-overlap
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/generalization_overlap_%j.out
#SBATCH --open-mode=append
# Read-only lexical-overlap gate for a direct capability interview.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:?set DATA to a frozen candidate JSONL}
CASE_SOURCE=${CASE_SOURCE:-$BASE/train/generalization_interview.py}
OUT=${OUT:?set OUT to a fresh overlap report}

[ -s "$DATA" ] || { echo "[generalize-overlap] data missing: $DATA" >&2; exit 2; }
[ -s "$CASE_SOURCE" ] || { echo "[generalize-overlap] case source missing: $CASE_SOURCE" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[generalize-overlap] refusing existing report: $OUT" >&2; exit 3; }
cd "$BASE"
"$PY" pipeline/audit_generalization_overlap.py --data "$DATA" --case-source "$CASE_SOURCE" --out "$OUT"
"$PY" - "$OUT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
if int(report.get("cases", 0)) != 8 or not report.get("data_sha256") or not report.get("case_source_sha256"):
    raise SystemExit(f"[generalize-overlap] malformed report: {report}")
if report.get("malformed_rows") or report.get("missing_question"):
    raise SystemExit(f"[generalize-overlap] invalid candidate rows: {report}")
if report.get("exact_prompt_hits") or report.get("ngram_prompt_hits"):
    raise SystemExit(f"[generalize-overlap] held-out interview overlap: {report['examples']}")
print(f"[generalize-overlap] passed: {report['valid_rows']} rows")
PY
