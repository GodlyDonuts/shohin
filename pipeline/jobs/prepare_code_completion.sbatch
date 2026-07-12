#!/bin/bash
#SBATCH -J shohin-code-completion
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/code_completion_%j.out
#SBATCH --open-mode=append
# Produce an immutable code-completion training input from already verified
# sources. It never modifies a source file, frozen mix, or model checkpoint.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=$BASE/artifacts/sft/code_completion_v1.jsonl
QUALITY=$BASE/artifacts/sft/code_completion_v1.quality.json
INPUTS=(
  "$BASE/artifacts/sft/code.jsonl"
  "$BASE/artifacts/sft/apps_train_v3.jsonl"
  "$BASE/artifacts/sft/code_contests_train_v1.jsonl"
)

[ ! -e "$OUT" ] || { echo "[code-completion] refusing to overwrite $OUT" >&2; exit 2; }
for path in "${INPUTS[@]}"; do
  [ -s "$path" ] || { echo "[code-completion] required input missing or empty: $path" >&2; exit 3; }
done

cd "$BASE"
"$PY" pipeline/prepare_code_completion.py --inputs "${INPUTS[@]}" --out "$OUT"
"$PY" pipeline/audit_sft_quality.py --data "$OUT" --out "$QUALITY"
"$PY" - "$QUALITY" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
if report["malformed_rows"] or report["missing_question_or_response"] or report["duplicate_normalized_questions"]:
    raise SystemExit(f"[code-completion] quality failure: {report}")
if report["valid_rows"] < 1000:
    raise SystemExit(f"[code-completion] insufficient rows: {report['valid_rows']}")
print(f"[code-completion] quality gate passed: {report['valid_rows']} rows")
PY
