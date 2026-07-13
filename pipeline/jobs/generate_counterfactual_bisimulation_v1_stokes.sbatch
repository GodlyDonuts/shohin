#!/bin/bash
# CPU-only construction and independent admission of the Counterfactual
# Bisimulation Compiler candidate. It cannot allocate a GPU, write a model,
# or modify the flagship corpus.
#SBATCH -J shohin-cbc-build
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/counterfactual_bisimulation_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-/usr/bin/python3.12}
TRAIN=${TRAIN:-$BASE/artifacts/sft/counterfactual_bisimulation_v1_train.jsonl}
HELDOUT=${HELDOUT:-$BASE/artifacts/evals/counterfactual_bisimulation_v1_heldout.jsonl}
GENERATED=${GENERATED:-$BASE/artifacts/sft/counterfactual_bisimulation_v1_train.report.json}
AUDIT=${AUDIT:-$BASE/artifacts/sft/counterfactual_bisimulation_v1_train.protocol_audit.json}
TRAIN_PER_DOMAIN=${TRAIN_PER_DOMAIN:-4000}
HELDOUT_PER_DOMAIN=${HELDOUT_PER_DOMAIN:-200}

for path in "$TRAIN" "$HELDOUT" "$GENERATED" "$AUDIT"; do
  [ ! -e "$path" ] || { echo "[cbc-build] refusing existing artifact: $path" >&2; exit 3; }
done
cd "$BASE"
"$PY" pipeline/generate_counterfactual_bisimulation_v1.py \
  --train-out "$TRAIN" --heldout-out "$HELDOUT" --report "$GENERATED" \
  --train-per-domain "$TRAIN_PER_DOMAIN" --heldout-per-domain "$HELDOUT_PER_DOMAIN"
"$PY" pipeline/audit_counterfactual_bisimulation_v1.py \
  --train "$TRAIN" --heldout "$HELDOUT" --report "$AUDIT"
"$PY" - "$AUDIT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
for key in (
    "invalid_train_rows",
    "invalid_heldout_episodes",
    "train_heldout_exact_prompt_hits",
    "train_heldout_13gram_hits",
):
    if report.get(key) != 0:
        raise SystemExit("[cbc-build] admission failed: {}={}".format(key, report.get(key)))
if set(report.get("train_kinds", ())) != {"compile_a", "compile_b", "update", "inverse_delta", "readout_sum"}:
    raise SystemExit("[cbc-build] incomplete CBC train task coverage")
if set(report.get("heldout_regimes", ())) != {"cbc_len4", "cbc_len8", "cbc_len12"}:
    raise SystemExit("[cbc-build] incomplete held-out regimes")
if report.get("valid_train_rows", 0) < 100000 or report.get("valid_heldout_episodes", 0) < 300:
    raise SystemExit("[cbc-build] insufficient candidate capacity")
print("[cbc-build] admitted rows={} heldout={}".format(
    report["valid_train_rows"], report["valid_heldout_episodes"]
))
PY
echo "[cbc-build] committed immutable CPU candidate"
