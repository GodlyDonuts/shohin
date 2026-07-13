#!/bin/bash
# Read-only audit for the corrected immutable DRS candidate.
#SBATCH -J shohin-digitwise-v2-audit
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/digitwise_recurrent_v2_audit_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=${PY:-/usr/bin/python3.12}
TRAIN=${TRAIN:-$BASE/artifacts/sft/digitwise_recurrent_v2_train.jsonl}
HELDOUT=${HELDOUT:-$BASE/artifacts/evals/digitwise_recurrent_v2_heldout.jsonl}
OUT=${OUT:-$BASE/artifacts/sft/digitwise_recurrent_v2_train.protocol_audit.json}

[ -s "$TRAIN" ] || { echo "[digitwise-v2-audit] missing train data" >&2; exit 2; }
[ -s "$HELDOUT" ] || { echo "[digitwise-v2-audit] missing heldout episodes" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[digitwise-v2-audit] refusing existing report: $OUT" >&2; exit 3; }
cd "$BASE"
"$PY" pipeline/audit_digitwise_recurrent_v1.py --data "$TRAIN" --episodes "$HELDOUT" --out "$OUT"
"$PY" - "$OUT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
if report["invalid_rows_or_episodes"] or report["duplicate_normalized_train_questions"]:
    raise SystemExit("[digitwise-v2-audit] structural gate failed")
overlap = report["overlap"]
if overlap["exact_prompt_hits"] or overlap["ngram13_hits"]:
    raise SystemExit("[digitwise-v2-audit] heldout overlap gate failed")
required = {"fit_w4", "fit_w6", "value_ood_w4", "value_ood_w6", "width_ood_w8"}
if set(report["regimes"]) != required or report["counterfactual_pairs"] != report["heldout_episodes"]:
    raise SystemExit("[digitwise-v2-audit] missing regimes or interventions")
print("[digitwise-v2-audit] passed")
PY
