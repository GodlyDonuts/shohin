#!/bin/bash
#SBATCH -J shohin-microcode-train
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/microcode_train_%j.out
#SBATCH --open-mode=append
# Isolated frozen-base semantic compiler. Never writes a flagship path.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:?set INIT to an immutable base checkpoint}
DATA=${DATA:-$BASE/artifacts/sft/latent_operator_v1_train.jsonl}
AUDIT=${AUDIT:-$BASE/artifacts/sft/latent_operator_v1.audit.r1.json}
QUALITY=${QUALITY:-$BASE/artifacts/sft/latent_operator_v1.quality.r1.json}
FULL_TEXT=${FULL_TEXT:-$BASE/artifacts/sft/latent_operator_v1.training_text_overlap.r1.json}
CONTRACTS=${CONTRACTS:-$BASE/artifacts/sft/latent_operator_v1.response_contracts.r1.json}
CMB_ADMISSION=${CMB_ADMISSION:-$BASE/artifacts/sft/categorical_microcode_v1.admission.json}
OUT=${OUT:?set OUT to a fresh microcode output directory}

for path in "$INIT" "$DATA" "$AUDIT" "$QUALITY" "$FULL_TEXT" "$CONTRACTS" "$CMB_ADMISSION"; do
  [ -s "$path" ] || { echo "[microcode-train] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[microcode-train] refusing existing output: $OUT" >&2; exit 3; }
case "$OUT" in "$BASE"/train/microcode_*) ;; *) echo "[microcode-train] invalid OUT path" >&2; exit 4;; esac

"$PY" - "$DATA" "$AUDIT" "$QUALITY" "$FULL_TEXT" "$CONTRACTS" <<'PY'
import hashlib, json, os, sys
data, audit_path, quality_path, text_path, contracts_path = map(os.path.realpath, sys.argv[1:])
def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()
sha = digest(data)
audit, quality, text, contracts = (json.load(open(path)) for path in
                                   (audit_path, quality_path, text_path, contracts_path))
if audit.get("train_sha256") != sha or any(audit.get(key) for key in
        ("invalid_train_rows", "invalid_heldout_rows", "duplicate_train_questions", "duplicate_heldout_questions")):
    raise SystemExit("[microcode-train] solver audit failed")
if any(audit.get("overlap", {}).get(key) for key in ("exact_prompt_hits", "ngram13_hits")):
    raise SystemExit("[microcode-train] held-out overlap gate failed")
if quality.get("data_sha256") != sha or quality.get("malformed_rows") or quality.get("missing_question_or_response") or quality.get("duplicate_normalized_questions"):
    raise SystemExit("[microcode-train] quality gate failed")
if any(quality.get("eval_overlap", {}).get(key) for key in ("exact_prompt_hits", "ngram_prompt_hits")):
    raise SystemExit("[microcode-train] public-eval overlap gate failed")
if text.get("data_sha256") != sha or any(text.get("overlap", {}).get(key) for key in ("exact_rows", "ngram_rows")):
    raise SystemExit("[microcode-train] full-text overlap gate failed")
group = contracts.get("groups", {}).get("latent_operator", {})
if contracts.get("data_sha256") != sha or set(contracts.get("groups", {})) != {"latent_operator"}:
    raise SystemExit("[microcode-train] response contract mismatch")
if group.get("think_marker_rate") != 0.0 or group.get("state_marker_rate") != 0.0 or group.get("answer_marker_rate") != 1.0:
    raise SystemExit("[microcode-train] response contract is not answer-only")
print("[microcode-train] immutable input gate passed sha256={}".format(sha))
PY

cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys, torch
try:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
except Exception as exc:
    print("[microcode-train] CUDA preflight failed: {}".format(exc), file=sys.stderr)
    sys.exit(75)
PY
then
  exit 75
fi
"$PY" train_categorical_microcode.py \
  --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --admission "$CMB_ADMISSION" \
  --out "$OUT" --layer "${LAYER:-19}" --hidden "${HIDDEN:-256}" \
  --batch-size "${BS:-16}" --max-examples "${MAX_EXAMPLES:-32768}" \
  --epochs "${EPOCHS:-1}" --lr "${LR:-0.001}" --basis-weight "${BASIS_WEIGHT:-1.0}" \
  --seed "${SEED:-20260714}"
