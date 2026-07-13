#!/bin/bash
# Read-only full admission audit for append-ledger v1.
#SBATCH -J shohin-ledger-v1-audit
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/append_ledger_v1_audit_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=${PY:-/usr/bin/python3.12}
TRAIN=${TRAIN:-$BASE/artifacts/sft/append_ledger_v1_train.jsonl}
HELDOUT=${HELDOUT:-$BASE/artifacts/evals/append_ledger_v1_heldout.jsonl}
OUT=${OUT:-$BASE/artifacts/sft/append_ledger_v1_train.protocol_audit.json}

[ -s "$TRAIN" ] || { echo "[append-ledger-audit] missing train data" >&2; exit 2; }
[ -s "$HELDOUT" ] || { echo "[append-ledger-audit] missing heldout episodes" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[append-ledger-audit] refusing existing report" >&2; exit 3; }
cd "$BASE"
"$PY" pipeline/audit_append_ledger_v1.py --data "$TRAIN" --episodes "$HELDOUT" --out "$OUT"
"$PY" - "$OUT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
if report["invalid_rows_or_episodes"] or report["duplicate_normalized_train_questions"]:
    raise SystemExit("[append-ledger-audit] structural gate failed")
if report["overlap"]["exact_prompt_hits"] or report["overlap"]["ngram13_hits"]:
    raise SystemExit("[append-ledger-audit] heldout overlap gate failed")
expected = {"fit_w8", "fit_w16", "value_ood_w8", "value_ood_w16", "width_ood_w32"}
if set(report["regimes"]) != expected or report["counterfactual_pairs"] != report["heldout_episodes"]:
    raise SystemExit("[append-ledger-audit] missing regimes or counterfactuals")
print("[append-ledger-audit] passed")
PY
