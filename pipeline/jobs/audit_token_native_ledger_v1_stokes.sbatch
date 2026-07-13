#!/bin/bash
# Read-only independent admission for the token-native delta-ledger candidate.
#SBATCH -J shohin-tnl-v1-audit
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/token_native_ledger_v1_audit_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=${BASE:-/home/sa305415/shohin}
PY=${PY:-/usr/bin/python3}
TRAIN=${TRAIN:-$BASE/artifacts/sft/token_native_ledger_v1_train.jsonl}
HELDOUT=${HELDOUT:-$BASE/artifacts/evals/token_native_ledger_v1_heldout.jsonl}
OUT=${OUT:-$BASE/artifacts/sft/token_native_ledger_v1_train.protocol_audit.json}

[ -s "$TRAIN" ] || { echo "[tnl-audit] missing train data" >&2; exit 2; }
[ -s "$HELDOUT" ] || { echo "[tnl-audit] missing heldout episodes" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[tnl-audit] refusing existing output: $OUT" >&2; exit 3; }
cd "$BASE"
"$PY" pipeline/audit_token_native_ledger_v1.py --data "$TRAIN" --episodes "$HELDOUT" --out "$OUT"
"$PY" - "$OUT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
for key in (
    "invalid_train_rows",
    "duplicate_normalized_train_prompts",
    "invalid_heldout_episodes",
    "counterfactual_mismatches",
    "missing_local_contexts",
    "train_heldout_exact_prompt_hits",
    "train_heldout_13gram_hits",
):
    if report.get(key) != 0:
        raise SystemExit("[tnl-audit] admission gate failed: {}".format(key))
if set(report.get("heldout_regimes", ())) != {"recombine_w4", "recombine_w6", "width_ood_w8"}:
    raise SystemExit("[tnl-audit] heldout regimes incomplete")
if report.get("required_local_contexts") != report.get("covered_local_contexts"):
    raise SystemExit("[tnl-audit] local context basis incomplete")
print("[tnl-audit] passed")
PY
