#!/bin/bash
#SBATCH -J shohin-sft-vrwm
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_vrwm_%j.out
#SBATCH --open-mode=append
#
# Isolated VRWM SFT. The one-GPU request is intentional: sft.py has no audited
# DDP path, so allocating two GPUs would waste one. Use two GPUs for parallel
# isolated ablations or the already-validated DDP pretraining path instead.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:?set DATA to a hash-audited VRWM JSONL}
QUALITY=${QUALITY:?set QUALITY to the matching generic quality report}
FULL_TEXT=${FULL_TEXT:?set FULL_TEXT to the matching full-text overlap report}
PACKING=${PACKING:?set PACKING to the matching packing report}
INIT=${INIT:?set INIT to a preserved raw checkpoint}
OUT=${OUT:?set OUT to a fresh isolated output directory}

for path in "$DATA" "$QUALITY" "$FULL_TEXT" "$PACKING" "$INIT"; do
  [ -s "$path" ] || { echo "[sft-vrwm] required input missing: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[sft-vrwm] refusing existing output: $OUT" >&2; exit 3; }
"$PY" - "$DATA" "$QUALITY" "$FULL_TEXT" "$PACKING" <<'PY'
import hashlib
import json
import os
import sys

data, quality_path, overlap_path, packing_path = map(os.path.realpath, sys.argv[1:])
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

digest = sha256(data)
quality = json.load(open(quality_path))
overlap = json.load(open(overlap_path))
packing = json.load(open(packing_path))
if quality.get("data_sha256") != digest or quality.get("malformed_rows") or quality.get("missing_question_or_response") or quality.get("duplicate_normalized_questions"):
    raise SystemExit("[sft-vrwm] generic quality report does not admit this exact data")
if overlap.get("data_sha256") != digest or overlap.get("overlap", {}).get("exact_rows") or overlap.get("overlap", {}).get("ngram_rows"):
    raise SystemExit("[sft-vrwm] full-text decontamination report does not admit this exact data")
hashes = packing.get("data_sha256", {})
bound_hashes = list(hashes.values())
# Packing reports can be generated locally and consumed on Newton, so an
# absolute path is not a stable identity. This one-file job instead requires
# the report to bind exactly one input whose content hash is this data.
if len(bound_hashes) != 1 or bound_hashes[0] != digest or packing.get("pack_len") != 2048:
    raise SystemExit("[sft-vrwm] packing report does not bind this exact data")
if int(quality.get("valid_rows", 0)) < 100_000 or int(packing.get("packed_sequences", 0)) < 5_000:
    raise SystemExit("[sft-vrwm] insufficient audited VRWM capacity")
print(f"[sft-vrwm] input gate passed rows={quality['valid_rows']} packs={packing['packed_sequences']}")
PY
cd "$BASE/train"
"$PY" - <<'PY'
import sys
import torch
try:
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[sft-vrwm] CUDA unavailable: {exc}", file=sys.stderr)
    sys.exit(75)
PY
"$PY" sft.py --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --epochs "${EPOCHS:-1}" --batch-size "${BS:-16}" --lr-muon "${LRMUON:-0.001}" --lr-adam "${LRADAM:-0.00025}" \
  --group-field training_group --prompt-override-field completion_prompt --sample-weights vrwm=1.0 --compile --out "$OUT"
