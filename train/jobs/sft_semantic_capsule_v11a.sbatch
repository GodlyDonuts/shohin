#!/bin/bash
# Isolated post-bridge semantic-capsule continuation. This tests whether a
# checkpoint with demonstrated semantic competence can form and reuse a
# model-authored compressed state after the source record is removed.
#SBATCH -J shohin-sft-v11a-capsule
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v11a_capsule_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:?set INIT to the V10A checkpoint that passed both semantic gates}
BRIDGE_EVAL=${BRIDGE_EVAL:?set BRIDGE_EVAL to the full held-out V10A bridge result}
COMPOSITION_EVAL=${COMPOSITION_EVAL:?set COMPOSITION_EVAL to the full V10A composition result}
BRIDGE_DATA=${BRIDGE_DATA:-$BASE/artifacts/evals/semantic_bridge_v1_heldout.jsonl}
COMPOSITION_DATA=${COMPOSITION_DATA:-$BASE/artifacts/evals/semantic_composition_transfer_v1.jsonl}
DATA=${DATA:-$BASE/artifacts/sft/semantic_capsule_v1_train.jsonl}
QUALITY=${QUALITY:-$BASE/artifacts/sft/semantic_capsule_v1_train.quality.r1.json}
PACKING=${PACKING:-$BASE/artifacts/sft/semantic_capsule_v1_train.packing.r1.json}
CONTRACTS=${CONTRACTS:-$BASE/artifacts/sft/semantic_capsule_v1_train.response_contracts.r1.json}
PROTOCOL=${PROTOCOL:-$BASE/artifacts/sft/semantic_capsule_v1_train.protocol_audit.r1.json}
OUT=${OUT:?set OUT to a fresh isolated output directory}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
LRMUON=${LRMUON:-0.0008}
LRADAM=${LRADAM:-0.0002}
MIN_BRIDGE_ANSWERS=${MIN_BRIDGE_ANSWERS:-250}
MIN_BRIDGE_TRACE_CONTRACT=${MIN_BRIDGE_TRACE_CONTRACT:-200}
MIN_BRIDGE_TRACE_CONTRACT_PER_FAMILY=${MIN_BRIDGE_TRACE_CONTRACT_PER_FAMILY:-25}
MIN_BRIDGE_PER_FAMILY=${MIN_BRIDGE_PER_FAMILY:-40}
MIN_COMPOSITION_ANSWERS=${MIN_COMPOSITION_ANSWERS:-50}
MIN_COMPOSITION_PER_FAMILY=${MIN_COMPOSITION_PER_FAMILY:-5}

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for required in "$INIT" "$BRIDGE_EVAL" "$COMPOSITION_EVAL" "$BRIDGE_DATA" "$COMPOSITION_DATA" "$DATA" "$QUALITY" "$PACKING" "$CONTRACTS" "$PROTOCOL"; do
  [ -s "$required" ] || { echo "[v11a] missing required artifact: $required" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[v11a] refusing existing output: $OUT" >&2; exit 3; }

"$PY" - "$INIT" "$BRIDGE_EVAL" "$COMPOSITION_EVAL" "$BRIDGE_DATA" "$COMPOSITION_DATA" "$DATA" "$QUALITY" "$PACKING" "$CONTRACTS" "$PROTOCOL" \
  "$MIN_BRIDGE_ANSWERS" "$MIN_BRIDGE_TRACE_CONTRACT" "$MIN_BRIDGE_TRACE_CONTRACT_PER_FAMILY" "$MIN_BRIDGE_PER_FAMILY" "$MIN_COMPOSITION_ANSWERS" "$MIN_COMPOSITION_PER_FAMILY" <<'PY'
import hashlib
import json
import os
import sys

(init, bridge_eval_path, composition_eval_path, bridge_data, composition_data, data, quality_path,
 packing_path, contracts_path, protocol_path, min_bridge_answers, min_bridge_trace_contract,
 min_bridge_trace_contract_per_family, min_bridge_per_family, min_composition_answers,
 min_composition_per_family) = sys.argv[1:]
min_bridge_answers, min_bridge_trace_contract, min_bridge_trace_contract_per_family, min_bridge_per_family, min_composition_answers, min_composition_per_family = map(
    int, (min_bridge_answers, min_bridge_trace_contract, min_bridge_trace_contract_per_family, min_bridge_per_family, min_composition_answers, min_composition_per_family)
)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def exact_checkpoint(result, label):
    observed = os.path.realpath(str(result.get("checkpoint", "")))
    if observed != os.path.realpath(init):
        raise SystemExit("[v11a] {} result does not bind INIT".format(label))

bridge = json.load(open(bridge_eval_path))
composition = json.load(open(composition_eval_path))
quality, packing, contracts, protocol = (json.load(open(path)) for path in (quality_path, packing_path, contracts_path, protocol_path))
exact_checkpoint(bridge, "bridge")
exact_checkpoint(composition, "composition")
if bridge.get("data_sha256") != sha256(bridge_data):
    raise SystemExit("[v11a] bridge result does not bind frozen held-out data")
if composition.get("data_sha256") != sha256(composition_data):
    raise SystemExit("[v11a] composition result does not bind frozen held-out data")
for label, result, minimum, per_family_minimum in (
    ("bridge", bridge, min_bridge_answers, min_bridge_per_family),
    ("composition", composition, min_composition_answers, min_composition_per_family),
):
    summary = result.get("summary", {})
    if summary.get("cases") != 500:
        raise SystemExit("[v11a] {} result is not the full 500-case gate".format(label))
    if int(summary.get("answer_correct", 0)) < minimum:
        raise SystemExit("[v11a] {} answer gate failed".format(label))
    families = summary.get("by_family", {})
    if len(families) != 5 or any(int(stats.get("answer_correct", 0)) < per_family_minimum for stats in families.values()):
        raise SystemExit("[v11a] {} is not family-balanced".format(label))
bridge_summary = bridge.get("summary", {})
if int(bridge_summary.get("trace_contract_correct", 0)) < min_bridge_trace_contract:
    raise SystemExit("[v11a] bridge trace-contract gate failed")
if any(
    int(stats.get("trace_contract_correct", 0)) < min_bridge_trace_contract_per_family
    for stats in bridge_summary.get("by_family", {}).values()
):
    raise SystemExit("[v11a] bridge trace-contract gate is not family-balanced")
data_sha = sha256(data)
invalid_completion_prompts = 0
with open(data) as source:
    for line in source:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("completion_prompt") != row.get("question") or not str(row.get("completion_prompt", "")).endswith(("Answer:", "Result:")):
            invalid_completion_prompts += 1
if invalid_completion_prompts:
    raise SystemExit("[v11a] capsule completion prompts do not match controller prompts")
if quality.get("data_sha256") != data_sha or quality.get("valid_rows") != 360000:
    raise SystemExit("[v11a] capsule quality report does not bind current data")
if quality.get("malformed_rows") or quality.get("duplicate_normalized_questions"):
    raise SystemExit("[v11a] capsule quality gate failed")
if quality.get("eval_overlap", {}).get("exact_prompt_hits") or quality.get("eval_overlap", {}).get("ngram_prompt_hits"):
    raise SystemExit("[v11a] capsule eval-overlap gate failed")
if packing.get("data_sha256", {}).get(data) != data_sha or packing.get("packed_sequences") != 22899:
    raise SystemExit("[v11a] capsule packing gate failed")
if packing.get("group_counts") != {"semantic_capsule": 22899}:
    raise SystemExit("[v11a] capsule group gate failed")
group = contracts.get("groups", {}).get("semantic_capsule", {})
if contracts.get("data_sha256") != data_sha or group.get("rows") != 360000 or group.get("starts_think_rate") != 1.0:
    raise SystemExit("[v11a] capsule response-contract gate failed")
if protocol.get("data_sha256") != data_sha or protocol.get("malformed_train_rows") or protocol.get("invalid_heldout_episodes"):
    raise SystemExit("[v11a] capsule protocol gate failed")
if protocol.get("duplicate_normalized_train_questions") or protocol.get("overlap", {}).get("exact_prompt_hits") or protocol.get("overlap", {}).get("ngram13_hits"):
    raise SystemExit("[v11a] capsule protocol-overlap gate failed")
print("[v11a] admitted", json.dumps({
    "bridge": bridge["summary"], "composition": composition["summary"], "capsule_sha256": data_sha,
}, sort_keys=True))
PY

cd "$BASE/train"
for preflight in 1 2 3; do
  if timeout --kill-after=10s 90s "$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(75)
torch.empty(1, device="cuda", dtype=torch.bfloat16)
PY
  then
    break
  fi
  if [ "$preflight" = 3 ]; then
    echo "[v11a] CUDA unavailable or timed out after allocation" >&2
    exit 75
  fi
  echo "[v11a] CUDA preflight $preflight failed; retrying" >&2
  sleep 30
done

echo "[v11a] init=$INIT data=$DATA out=$OUT epochs=$EPOCHS bs=$BS"
"$PY" sft.py \
  --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --epochs "$EPOCHS" --batch-size "$BS" --lr-muon "$LRMUON" --lr-adam "$LRADAM" \
  --warmup 50 --pack-len 2048 --group-field training_group --prompt-override-field completion_prompt --compile --out "$OUT"
[ -s "$OUT/sft_ep${EPOCHS}.pt" ] || { echo "[v11a] expected final checkpoint missing" >&2; exit 4; }
echo "[v11a] done checkpoint=$OUT/sft_ep${EPOCHS}.pt"
