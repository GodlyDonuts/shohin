#!/bin/bash
#SBATCH -J shohin-sft-v8-quality
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v8_quality_%j.out
#SBATCH --open-mode=append
# Re-audit the frozen V8 data snapshot with a hash-bound quality report. This
# is read-only and cannot alter the data or launch SFT.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:-$BASE/artifacts/sft/sft_mix_reasoning_v8_candidate_r2.jsonl}
OUT=${OUT:-$BASE/artifacts/sft/sft_mix_reasoning_v8_candidate_r2.quality.r3.json}

[ -s "$DATA" ] || { echo "[sft-v8-quality] data missing: $DATA" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[sft-v8-quality] refusing existing report: $OUT" >&2; exit 3; }
cd "$BASE"
"$PY" pipeline/audit_sft_quality.py --data "$DATA" --out "$OUT" --evals "$BASE/artifacts/evals"
"$PY" - "$OUT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
bad = [key for key in (
    "malformed_rows", "missing_question_or_response", "duplicate_normalized_questions",
) if report.get(key)]
overlap = report.get("eval_overlap", {})
if not report.get("data_sha256") or int(report.get("valid_rows", 0)) < 600_000 or bad:
    raise SystemExit(f"[sft-v8-quality] structural gate failed: rows={report.get('valid_rows')} bad={bad}")
if overlap.get("exact_prompt_hits") or overlap.get("ngram_prompt_hits"):
    raise SystemExit(f"[sft-v8-quality] eval-overlap gate failed: {overlap}")
print(f"[sft-v8-quality] hash-bound quality gate passed: {report['valid_rows']} rows")
PY
