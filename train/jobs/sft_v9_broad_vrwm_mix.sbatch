#!/bin/bash
#SBATCH -J shohin-sft-v9
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_v9_%j.out
#SBATCH --open-mode=append
# One-GPU isolated broad-anchor + 5% VRWM SFT. A second GPU would be unused:
# sft.py has no audited DDP implementation. Independent evaluations can use it.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:-$BASE/artifacts/sft/sft_mix_reasoning_v9_broad_vrwm_candidate.jsonl}
MIX_REPORT=${MIX_REPORT:-$BASE/artifacts/sft/sft_mix_reasoning_v9_broad_vrwm_candidate.mix.r1.json}
QUALITY=${QUALITY:-$BASE/artifacts/sft/sft_mix_reasoning_v9_broad_vrwm_candidate.quality.r1.json}
FULL_TEXT=${FULL_TEXT:-$BASE/artifacts/sft/sft_mix_reasoning_v9_broad_vrwm_candidate.training_text_overlap.r1.json}
PACKING=${PACKING:-$BASE/artifacts/sft/sft_mix_reasoning_v9_broad_vrwm_candidate.packing.r1.json}
INIT=${INIT:?set INIT to a preserved raw checkpoint}
OUT=${OUT:?set OUT to a fresh isolated output directory}
WEIGHTS=(math=0.40 procedural=0.42 code=0.08 teacher=0.05 vrwm=0.05)

for path in "$DATA" "$MIX_REPORT" "$QUALITY" "$FULL_TEXT" "$PACKING" "$INIT"; do
  [ -s "$path" ] || { echo "[sft-v9] required input missing: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[sft-v9] refusing existing output: $OUT" >&2; exit 3; }
"$PY" - "$DATA" "$MIX_REPORT" "$QUALITY" "$FULL_TEXT" "$PACKING" <<'PY'
import hashlib
import json
import os
import sys

data, mix_path, quality_path, text_path, packing_path = map(os.path.realpath, sys.argv[1:])
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
digest = sha256(data)
mix, quality, text, packing = (json.load(open(path)) for path in (mix_path, quality_path, text_path, packing_path))
expected = {"math": 0.40, "procedural": 0.42, "code": 0.08, "teacher": 0.05, "vrwm": 0.05}
if mix.get("out") != data or mix.get("out_sha256") != digest:
    raise SystemExit("[sft-v9] mix report does not bind data")
if quality.get("data_sha256") != digest or quality.get("malformed_rows") or quality.get("missing_question_or_response") or quality.get("duplicate_normalized_questions"):
    raise SystemExit("[sft-v9] quality report does not admit data")
if quality.get("eval_overlap", {}).get("exact_prompt_hits") or quality.get("eval_overlap", {}).get("ngram_prompt_hits"):
    raise SystemExit("[sft-v9] prompt overlap gate failed")
if text.get("data_sha256") != digest or text.get("overlap", {}).get("exact_rows") or text.get("overlap", {}).get("ngram_rows"):
    raise SystemExit("[sft-v9] full-text overlap gate failed")
if packing.get("data_sha256", {}).get(data) != digest or packing.get("pack_len") != 2048:
    raise SystemExit("[sft-v9] packing report does not bind data")
if {key: float(value) for key, value in packing.get("requested_weights", {}).items()} != expected:
    raise SystemExit("[sft-v9] unexpected weights")
if set(packing.get("group_counts", {})) != set(expected) or any(float(v) > 3.0 for v in packing.get("repeat_factor_per_epoch", {}).values()):
    raise SystemExit("[sft-v9] group/replay gate failed")
if float(packing["repeat_factor_per_epoch"]["vrwm"]) > 0.30:
    raise SystemExit("[sft-v9] VRWM replay concentration gate failed")
print(f"[sft-v9] frozen input gate passed rows={quality['valid_rows']} packs={packing['packed_sequences']}")
PY
cd "$BASE/train"
"$PY" - <<'PY'
import sys
import torch
try:
    torch.empty(1, device="cuda")
except Exception as exc:
    print(f"[sft-v9] CUDA unavailable: {exc}", file=sys.stderr)
    sys.exit(75)
PY
"$PY" sft.py --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --epochs "${EPOCHS:-1}" --batch-size "${BS:-16}" --lr-muon "${LRMUON:-0.0015}" --lr-adam "${LRADAM:-0.00035}" \
  --group-field training_group --prompt-override-field completion_prompt \
  --sample-weights "${WEIGHTS[@]}" --compile --out "$OUT"
