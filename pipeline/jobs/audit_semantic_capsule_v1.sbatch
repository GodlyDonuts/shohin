#!/bin/bash
#SBATCH -J shohin-semcapsule-audit
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/semantic_capsule_audit_%j.out
#SBATCH --open-mode=append
# Read-only semantic-capsule protocol audit. It cannot write model weights.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:-$BASE/artifacts/sft/semantic_capsule_v1_train.jsonl}
EPISODES=${EPISODES:-$BASE/artifacts/evals/semantic_capsule_v1_heldout.jsonl}
OUT=${OUT:-$BASE/artifacts/sft/semantic_capsule_v1_train.protocol_audit.r1.json}

[ -s "$DATA" ] || { echo "[semantic-capsule-audit] missing train data: $DATA" >&2; exit 2; }
[ -s "$EPISODES" ] || { echo "[semantic-capsule-audit] missing held-out episodes: $EPISODES" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "[semantic-capsule-audit] refusing existing report: $OUT" >&2; exit 3; }
cd "$BASE"
"$PY" pipeline/audit_semantic_capsule_v1.py --data "$DATA" --episodes "$EPISODES" --out "$OUT"
"$PY" - "$OUT" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
if report["malformed_train_rows"] or report["duplicate_normalized_train_questions"] or report["invalid_heldout_episodes"]:
    raise SystemExit("[semantic-capsule-audit] structural gate failed")
if report["overlap"]["exact_prompt_hits"] or report["overlap"]["ngram13_hits"]:
    raise SystemExit("[semantic-capsule-audit] train-heldout overlap gate failed")
if set(report["regimes"]) != {"semantic_len4", "semantic_len8", "semantic_len12"}:
    raise SystemExit("[semantic-capsule-audit] missing held-out regimes")
print("[semantic-capsule-audit] passed")
PY
