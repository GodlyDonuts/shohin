#!/bin/bash
#SBATCH -J shohin-sft-v10-mix
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v10_mix_%j.out
#SBATCH --open-mode=append
# Build an auditable broad-anchor + 10% semantic-bridge candidate. This is
# data-only and cannot modify a training checkpoint or active shard stream.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
V8=$BASE/artifacts/sft/sft_mix_reasoning_v8_candidate_r2.jsonl
BRIDGE=$BASE/artifacts/sft/semantic_bridge_v1_train.jsonl
OUT=${OUT:-$BASE/artifacts/sft/sft_mix_reasoning_v10_broad_bridge_candidate.jsonl}
MIX_REPORT=${MIX_REPORT:-$BASE/artifacts/sft/sft_mix_reasoning_v10_broad_bridge_candidate.mix.r1.json}
QUALITY=${QUALITY:-$BASE/artifacts/sft/sft_mix_reasoning_v10_broad_bridge_candidate.quality.r1.json}
FULL_TEXT=${FULL_TEXT:-$BASE/artifacts/sft/sft_mix_reasoning_v10_broad_bridge_candidate.training_text_overlap.r1.json}
CONTRACTS=${CONTRACTS:-$BASE/artifacts/sft/sft_mix_reasoning_v10_broad_bridge_candidate.response_contracts.r1.json}
PACKING=${PACKING:-$BASE/artifacts/sft/sft_mix_reasoning_v10_broad_bridge_candidate.packing.r1.json}
WEIGHTS=(math=0.38 procedural=0.39 code=0.08 teacher=0.05 semantic_bridge=0.10)

for path in "$V8" "$BRIDGE"; do
  [ -s "$path" ] || { echo "[v10-mix] required input missing: $path" >&2; exit 2; }
done
for path in "$OUT" "$MIX_REPORT" "$QUALITY" "$FULL_TEXT" "$CONTRACTS" "$PACKING"; do
  [ ! -e "$path" ] || { echo "[v10-mix] refusing existing output: $path" >&2; exit 3; }
done

cd "$BASE"
"$PY" pipeline/build_sft_multisource_mix.py --inputs "$V8" "$BRIDGE" --out "$OUT" --report "$MIX_REPORT"
"$PY" pipeline/audit_sft_quality.py --data "$OUT" --evals "$BASE/artifacts/evals" --out "$QUALITY"
"$PY" pipeline/audit_training_text_overlap.py --data "$OUT" --evals "$BASE/artifacts/evals" \
  --out "$FULL_TEXT" --require-zero
"$PY" pipeline/audit_response_contracts.py --data "$OUT" --out "$CONTRACTS"
cd "$BASE/train"
"$PY" inspect_sft_packing.py --data "$OUT" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --out "$PACKING" --weights "${WEIGHTS[@]}"
"$PY" - "$OUT" "$MIX_REPORT" "$QUALITY" "$FULL_TEXT" "$CONTRACTS" "$PACKING" <<'PY'
import hashlib
import json
import os
import sys

data, mix_path, quality_path, full_text_path, contracts_path, packing_path = map(os.path.realpath, sys.argv[1:])
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
digest = sha256(data)
mix, quality, full_text, contracts, packing = (json.load(open(path)) for path in (mix_path, quality_path, full_text_path, contracts_path, packing_path))
expected = {"math": 0.38, "procedural": 0.39, "code": 0.08, "teacher": 0.05, "semantic_bridge": 0.10}
if mix.get("out") != data or mix.get("out_sha256") != digest:
    raise SystemExit("[v10-mix] mix report does not bind data")
if mix.get("malformed_json_rows") or mix.get("missing_question_response_or_group") or mix.get("duplicate_normalized_questions_dropped"):
    raise SystemExit("[v10-mix] cross-source structural gate failed")
if quality.get("data_sha256") != digest or quality.get("malformed_rows") or quality.get("missing_question_or_response") or quality.get("duplicate_normalized_questions"):
    raise SystemExit("[v10-mix] quality gate failed")
if quality.get("eval_overlap", {}).get("exact_prompt_hits") or quality.get("eval_overlap", {}).get("ngram_prompt_hits"):
    raise SystemExit("[v10-mix] prompt overlap gate failed")
if full_text.get("data_sha256") != digest or full_text.get("overlap", {}).get("exact_rows") or full_text.get("overlap", {}).get("ngram_rows"):
    raise SystemExit("[v10-mix] full-text overlap gate failed")
if contracts.get("data_sha256") != digest or contracts.get("malformed_rows") or contracts.get("missing_response_rows"):
    raise SystemExit("[v10-mix] response-contract gate failed")
groups = contracts.get("groups", {})
if set(groups) != set(expected) or groups["semantic_bridge"].get("starts_think_rate") != 1.0:
    raise SystemExit("[v10-mix] bridge response contract failed")
if {key: float(value) for key, value in packing.get("requested_weights", {}).items()} != expected:
    raise SystemExit("[v10-mix] weights do not bind report")
if set(packing.get("group_counts", {})) != set(expected) or packing.get("packed_sequences", 0) < 80_000:
    raise SystemExit("[v10-mix] insufficient packed coverage")
if any(float(value) > 3.0 for value in packing.get("repeat_factor_per_epoch", {}).values()):
    raise SystemExit("[v10-mix] unsafe replay factor")
if float(packing["repeat_factor_per_epoch"]["semantic_bridge"]) > 1.25:
    raise SystemExit("[v10-mix] bridge replay is too concentrated")
print("[v10-mix] candidate admitted", json.dumps({
    "rows": quality["valid_rows"], "packs": packing["packed_sequences"],
    "replay": packing["repeat_factor_per_epoch"],
}, sort_keys=True))
PY
