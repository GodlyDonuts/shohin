#!/bin/bash
#SBATCH -J shohin-verifier-bank
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/verifier_bank_%j.out
#SBATCH --open-mode=append
# Freeze a balanced train-only RG bank. It is not an SFT mix and uses no GPU.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INPUT=${INPUT:-$BASE/artifacts/rg_v4/rg_traces_train.jsonl}
OUT=${OUT:?set OUT to a fresh verifier-bank JSONL}
PER_FAMILY=${PER_FAMILY:-80}

[ -s "$INPUT" ] || { echo "[verifier-bank] input missing: $INPUT" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[verifier-bank] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE"
"$PY" pipeline/sample_verifier_bank.py --input "$INPUT" --out "$OUT" --per-family "$PER_FAMILY"
"$PY" pipeline/audit_sft_quality.py --data "$OUT" --out "${OUT%.jsonl}.quality.json" \
  --evals "$BASE/artifacts/evals"
"$PY" - "${OUT%.jsonl}.quality.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
if report["malformed_rows"] or report["missing_question_or_response"]:
    raise SystemExit(f"[verifier-bank] malformed data: {report}")
if report["duplicate_normalized_questions"]:
    raise SystemExit(f"[verifier-bank] duplicate questions: {report}")
overlap = report.get("eval_overlap", {})
if overlap.get("exact_prompt_hits") or overlap.get("ngram_prompt_hits"):
    raise SystemExit(f"[verifier-bank] eval overlap: {overlap}")
print(f"[verifier-bank] quality gate passed: {report['valid_rows']} rows")
PY
