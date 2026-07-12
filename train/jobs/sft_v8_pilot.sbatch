#!/bin/bash
#SBATCH -J shohin-sft-v8
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v8_%j.out
#SBATCH --open-mode=append
#
# Isolated broad-transfer SFT experiment. It reads a frozen V8 data snapshot
# and a preserved base checkpoint; it never writes to the flagship or promotes
# its output. Public-board and transcript gates decide whether it is useful.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:-$BASE/artifacts/sft/sft_mix_reasoning_v8_candidate_r2.jsonl}
QUALITY=${QUALITY:-$BASE/artifacts/sft/sft_mix_reasoning_v8_candidate_r2.quality.r3.json}
PACKING=${PACKING:-$BASE/artifacts/sft/sft_mix_reasoning_v8_candidate_r2.packing.r3.json}
INIT=${INIT:?set INIT to an explicitly preserved flagship checkpoint}
OUT=${OUT:?set OUT to a fresh isolated SFT output directory}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
LRMUON=${LRMUON:-0.002}
LRADAM=${LRADAM:-0.0005}
WEIGHTS=(math=0.40 procedural=0.45 code=0.10 teacher=0.05)

[ -s "$DATA" ] || { echo "[sft-v8] data missing: $DATA" >&2; exit 2; }
[ -s "$QUALITY" ] || { echo "[sft-v8] quality report missing: $QUALITY" >&2; exit 2; }
[ -s "$PACKING" ] || { echo "[sft-v8] packing report missing: $PACKING" >&2; exit 2; }
[ -f "$INIT" ] || { echo "[sft-v8] init missing: $INIT" >&2; exit 3; }
[ ! -e "$OUT" ] || { echo "[sft-v8] refusing to overwrite $OUT" >&2; exit 4; }

# Bind the training input to both immutable reports. This prevents a convenient
# but different JSONL from being passed under the V8 name after the audit.
"$PY" - "$DATA" "$QUALITY" "$PACKING" <<'PY'
import hashlib
import json
import os
import sys

data, quality_path, packing_path = map(os.path.realpath, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

quality = json.load(open(quality_path))
packing = json.load(open(packing_path))
if os.path.realpath(quality.get("data", "")) != data:
    raise SystemExit("[sft-v8] quality report data mismatch")
if quality.get("data_sha256") != sha256(data):
    raise SystemExit("[sft-v8] quality report does not match current data bytes")
for key in ("malformed_rows", "missing_question_or_response", "duplicate_normalized_questions"):
    if quality.get(key):
        raise SystemExit(f"[sft-v8] quality gate failed: {key}={quality[key]}")
overlap = quality.get("eval_overlap", {})
if overlap.get("exact_prompt_hits") or overlap.get("ngram_prompt_hits"):
    raise SystemExit(f"[sft-v8] eval overlap gate failed: {overlap}")
if int(quality.get("valid_rows", 0)) < 600_000:
    raise SystemExit(f"[sft-v8] insufficient frozen rows: {quality.get('valid_rows')}")

expected = {"math": 0.40, "procedural": 0.45, "code": 0.10, "teacher": 0.05}
if packing.get("pack_len") != 2048 or int(packing.get("packed_sequences", 0)) < 70_000:
    raise SystemExit("[sft-v8] insufficient or incompatible packed capacity")
if {str(key): float(value) for key, value in packing.get("requested_weights", {}).items()} != expected:
    raise SystemExit(f"[sft-v8] unexpected sample weights: {packing.get('requested_weights')}")
if set(packing.get("group_counts", {})) != set(expected):
    raise SystemExit(f"[sft-v8] group mismatch: {packing.get('group_counts')}")
if int(packing["group_counts"].get("code", 0)) < 2_500:
    raise SystemExit(f"[sft-v8] insufficient code capacity: {packing['group_counts']}")
if any(float(value) > 3.0 for value in packing.get("repeat_factor_per_epoch", {}).values()):
    raise SystemExit(f"[sft-v8] a group would be replayed too heavily: {packing['repeat_factor_per_epoch']}")
hashes = packing.get("data_sha256", {})
if hashes.get(data) != sha256(data):
    raise SystemExit("[sft-v8] packing report does not match current data bytes")
print(f"[sft-v8] frozen input gate passed: {quality['valid_rows']} rows, "
      f"{packing['packed_sequences']} packed sequences")
PY

cd "$BASE/train"
if ! "$PY" - <<'PY'
import sys
import torch
try:
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[sft-v8] CUDA allocation unavailable: {exc}", file=sys.stderr)
    sys.exit(75)
PY
then
  exit 75
fi

echo "[sft-v8] init=$INIT data=$DATA out=$OUT epochs=$EPOCHS weights=${WEIGHTS[*]}"
"$PY" sft.py --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --epochs "$EPOCHS" --batch-size "$BS" --lr-muon "$LRMUON" --lr-adam "$LRADAM" \
  --group-field training_group --prompt-override-field completion_prompt \
  --sample-weights "${WEIGHTS[@]}" --compile --out "$OUT"
echo "[sft-v8] completed"
