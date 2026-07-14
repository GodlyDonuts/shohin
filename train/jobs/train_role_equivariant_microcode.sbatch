#!/bin/bash
#SBATCH -J shohin-role-equiv
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/role_equiv_train_%j.out
#SBATCH --open-mode=append
# Matched role-factorized compiler arm. Never writes a flagship path.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:?set immutable base checkpoint}
DATA=${DATA:-$BASE/artifacts/sft/role_equivariant_microcode_v3.jsonl}
ADMISSION=${ADMISSION:-$BASE/artifacts/sft/role_equivariant_microcode_v3.admission.json}
QUALITY=${QUALITY:-$BASE/artifacts/sft/role_equivariant_microcode_v3.quality.json}
FULL_TEXT=${FULL_TEXT:-$BASE/artifacts/sft/role_equivariant_microcode_v3.full_text.json}
CONTRACTS=${CONTRACTS:-$BASE/artifacts/sft/role_equivariant_microcode_v3.contracts.json}
OUT=${OUT:?set fresh role-equivariant output directory}

for path in "$INIT" "$DATA" "$ADMISSION" "$QUALITY" "$FULL_TEXT" "$CONTRACTS"; do
  [ -s "$path" ] || { echo "[role-equiv] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[role-equiv] refusing existing output: $OUT" >&2; exit 3; }
case "$OUT" in "$BASE"/train/microcode_role_*) ;; *) echo "[role-equiv] invalid OUT path" >&2; exit 4;; esac

"$PY" - "$DATA" "$ADMISSION" "$QUALITY" "$FULL_TEXT" "$CONTRACTS" <<'PY'
import hashlib, json, sys
data, admission_path, quality_path, text_path, contracts_path = sys.argv[1:]
def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()
sha=digest(data)
admission, quality, text, contracts=(json.load(open(path)) for path in
    (admission_path,quality_path,text_path,contracts_path))
if not admission.get("all_checks_pass") or admission.get("train_sha256") != sha:
    raise SystemExit("[role-equiv] structural admission failed")
if (quality.get("data_sha256") != sha or quality.get("malformed_rows")
        or quality.get("missing_question_or_response") or quality.get("duplicate_normalized_questions")):
    raise SystemExit("[role-equiv] quality admission failed")
if any(quality.get("eval_overlap",{}).get(key) for key in ("exact_prompt_hits","ngram_prompt_hits")):
    raise SystemExit("[role-equiv] public evaluation overlap")
if (text.get("data_sha256") != sha or text.get("malformed_rows") or text.get("malformed_json_rows")
        or any(text.get("overlap",{}).get(key) for key in ("exact_rows","ngram_rows"))):
    raise SystemExit("[role-equiv] full-text overlap")
group=contracts.get("groups",{}).get("role_equivariant_microcode",{})
if (contracts.get("data_sha256") != sha or contracts.get("malformed_rows")
        or contracts.get("missing_response_rows")
        or set(contracts.get("groups",{})) != {"role_equivariant_microcode"}):
    raise SystemExit("[role-equiv] response contract mismatch")
if group.get("think_marker_rate") != 0.0 or group.get("state_marker_rate") != 0.0 or group.get("answer_marker_rate") != 1.0:
    raise SystemExit("[role-equiv] response contract is not answer-only")
print("[role-equiv] immutable input gate passed sha256={}".format(sha))
PY

cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys, torch
try:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.empty(1024,device="cuda",dtype=torch.bfloat16); torch.cuda.synchronize()
except Exception as exc:
    print("[role-equiv] CUDA preflight failed: {}".format(exc),file=sys.stderr); sys.exit(75)
PY
then
  exit 75
fi
"$PY" train_role_equivariant_microcode.py \
  --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --admission "$ADMISSION" --out "$OUT" --layer "${LAYER:-19}" --hidden "${HIDDEN:-256}" \
  --batch-groups "${BATCH_GROUPS:-4}" --max-groups "${MAX_GROUPS:-48000}" \
  --lr "${LR:-0.001}" --basis-weight "${BASIS_WEIGHT:-1.0}" \
  --semantic-weight "${SEMANTIC_WEIGHT:-0.0}" \
  --permutation-weight "${PERMUTATION_WEIGHT:-0.0}" --seed "${SEED:-20260714}"
