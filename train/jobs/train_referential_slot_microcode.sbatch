#!/bin/bash
#SBATCH -J shohin-referential-slot
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/referential_slot_train_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
INIT=${INIT:?set immutable base checkpoint}
DATA=${DATA:-$BASE/artifacts/sft/role_equivariant_microcode_v3.jsonl}
ADMISSION=${ADMISSION:-$BASE/artifacts/sft/role_equivariant_microcode_v3.admission.json}
LABEL_ADMISSION=${LABEL_ADMISSION:-$BASE/artifacts/sft/referential_slot_labels_v4.admission.json}
QUALITY=${QUALITY:-$BASE/artifacts/sft/role_equivariant_microcode_v3.quality.json}
FULL_TEXT=${FULL_TEXT:-$BASE/artifacts/sft/role_equivariant_microcode_v3.full_text.json}
CONTRACTS=${CONTRACTS:-$BASE/artifacts/sft/role_equivariant_microcode_v3.contracts.json}
OUT=${OUT:?set fresh referential output directory}
ROLE_MODE=${ROLE_MODE:?set absolute or pointer}

for path in "$INIT" "$DATA" "$ADMISSION" "$LABEL_ADMISSION" "$QUALITY" "$FULL_TEXT" "$CONTRACTS"; do
  [ -s "$path" ] || { echo "[referential] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[referential] refusing existing output: $OUT" >&2; exit 3; }
case "$OUT" in "$BASE"/train/microcode_referential_*) ;; *) echo "[referential] invalid OUT path" >&2; exit 4;; esac
case "$ROLE_MODE" in absolute|pointer) ;; *) echo "[referential] invalid ROLE_MODE" >&2; exit 5;; esac

"$PY" - "$DATA" "$ADMISSION" "$LABEL_ADMISSION" "$QUALITY" "$FULL_TEXT" "$CONTRACTS" <<'PY'
import hashlib, json, sys
data, admission_path, labels_path, quality_path, text_path, contracts_path = sys.argv[1:]
def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()
sha=digest(data)
admission, labels, quality, text, contracts=(json.load(open(path)) for path in
    (admission_path,labels_path,quality_path,text_path,contracts_path))
if not admission.get("all_checks_pass") or admission.get("train_sha256") != sha:
    raise SystemExit("[referential] structural admission failed")
if not labels.get("all_checks_pass") or labels.get("datasets",{}).get("train",{}).get("sha256") != sha:
    raise SystemExit("[referential] mention-label admission failed")
if quality.get("data_sha256") != sha or quality.get("malformed_rows") or quality.get("duplicate_normalized_questions"):
    raise SystemExit("[referential] quality admission failed")
if any(quality.get("eval_overlap",{}).get(key) for key in ("exact_prompt_hits","ngram_prompt_hits")):
    raise SystemExit("[referential] public evaluation overlap")
if (text.get("data_sha256") != sha or not text.get("all_checks_pass")
        or text.get("exact_rows") or text.get("forbidden_rows")):
    raise SystemExit("[referential] full-text admission failed")
group=contracts.get("groups",{}).get("role_equivariant_microcode",{})
if (contracts.get("data_sha256") != sha or contracts.get("malformed_rows")
        or group.get("think_marker_rate") != 0.0 or group.get("state_marker_rate") != 0.0
        or group.get("answer_marker_rate") != 1.0):
    raise SystemExit("[referential] response contract failed")
print("[referential] immutable gates passed sha256={}".format(sha))
PY

cd "$BASE/train"
timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys, torch
try:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.empty(1024,device="cuda",dtype=torch.bfloat16); torch.cuda.synchronize()
except Exception as exc:
    print("[referential] CUDA preflight failed: {}".format(exc),file=sys.stderr); sys.exit(75)
PY

"$PY" train_referential_slot_microcode.py \
  --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --admission "$ADMISSION" --label-admission "$LABEL_ADMISSION" --out "$OUT" \
  --role-mode "$ROLE_MODE" --layer "${LAYER:-19}" --hidden "${HIDDEN:-256}" \
  --batch-groups "${BATCH_GROUPS:-4}" --max-groups "${MAX_GROUPS:-48000}" \
  --lr "${LR:-0.001}" --basis-weight "${BASIS_WEIGHT:-1.0}" \
  --mention-weight "${MENTION_WEIGHT:-1.0}" --seed "${SEED:-20260714}"
