#!/bin/bash
# CPU-only admission of a conditional semantic-basis transport corpus.
# This cannot allocate a GPU, write a model, or alter the flagship. It is
# submitted only after the V10A result establishes the required failure mode.
#SBATCH -J shohin-semantic-basis-build
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/semantic_basis_transport_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-/usr/bin/python3.12}
TRAIN=${TRAIN:-$BASE/artifacts/sft/semantic_basis_transport_v1_train.jsonl}
HELDOUT=${HELDOUT:-$BASE/artifacts/evals/semantic_basis_transport_v1_heldout.jsonl}
AUDIT=${AUDIT:-$BASE/artifacts/sft/semantic_basis_transport_v1.audit.json}
TRAIN_EPISODES=${TRAIN_EPISODES:-30000}
HELDOUT_EPISODES=${HELDOUT_EPISODES:-1000}

for path in "$TRAIN" "$HELDOUT" "$AUDIT"; do
  [ ! -e "$path" ] || { echo "[semantic-basis] refusing existing artifact: $path" >&2; exit 3; }
done
cd "$BASE"
"$PY" pipeline/generate_semantic_basis_transport_v1.py \
  --train-out "$TRAIN" --heldout-out "$HELDOUT" \
  --train-episodes "$TRAIN_EPISODES" --heldout-episodes "$HELDOUT_EPISODES"
"$PY" pipeline/audit_semantic_basis_transport_v1.py \
  --train "$TRAIN" --heldout "$HELDOUT" --out "$AUDIT"
"$PY" - "$AUDIT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
for split in ("train_report", "heldout_report"):
    values = report[split]
    if values["invalid_rows"] or values["duplicate_normalized_questions"] or values["bad_episodes"]:
        raise SystemExit("[semantic-basis] {} admission failed".format(split))
if report["cross_split_exact_prompt_hits"] or report["cross_split_ngram13_hits"]:
    raise SystemExit("[semantic-basis] split-leakage admission failed")
if report["train_report"]["rows"] != 150000 or report["heldout_report"]["rows"] != 5000:
    raise SystemExit("[semantic-basis] unexpected candidate capacity")
print("[semantic-basis] admitted train_rows={} heldout_rows={}".format(
    report["train_report"]["rows"], report["heldout_report"]["rows"]
))
PY
echo "[semantic-basis] conditional candidate committed"
