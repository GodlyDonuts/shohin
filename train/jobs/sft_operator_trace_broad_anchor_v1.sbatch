#!/bin/bash
# One isolated broad-anchor operator-trace SFT.  It cannot modify a flagship
# output, use a live writer file, or submit its own successor.
#SBATCH -J shohin-sft-opanchor
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc36,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_operator_anchor_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=${DATA:?set DATA to an admitted broad-anchor JSONL}
MIX=${MIX:?set MIX to DATA's immutable mix report}
QUALITY=${QUALITY:?set QUALITY to DATA's immutable quality report}
TEXT_OVERLAP=${TEXT_OVERLAP:?set TEXT_OVERLAP to the full-text overlap report}
TRACE_OVERLAP=${TRACE_OVERLAP:?set TRACE_OVERLAP to the fixed trace overlap report}
OOD_OVERLAP=${OOD_OVERLAP:?set OOD_OVERLAP to the state OOD overlap report}
FACTOR_OVERLAP=${FACTOR_OVERLAP:?set FACTOR_OVERLAP to the wording/full factor overlap report}
INIT=${INIT:?set INIT to an immutable retained-SFT checkpoint}
REFERENCE=${REFERENCE:-$INIT}
REPLAY=${REPLAY:?set REPLAY to the frozen prompt-only replay JSONL}
REPLAY_SHA=${REPLAY_SHA:?set REPLAY_SHA to its expected SHA-256}
OUT=${OUT:?set OUT to a fresh train/sft_operator_trace_anchor_* directory}
REQUIRED_EXCLUDED_CONTRACT=${REQUIRED_EXCLUDED_CONTRACT:-}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
LRMUON=${LRMUON:-0.0006}
LRADAM=${LRADAM:-0.00015}
REPLAY_WEIGHT=${REPLAY_WEIGHT:-0.25}
REPLAY_BATCH=${REPLAY_BATCH:-4}
REPLAY_MAX_TOKENS=${REPLAY_MAX_TOKENS:-128}
WEIGHTS=${WEIGHTS:-"math=0.25 procedural=0.30 code=0.05 teacher=0.05 primitives=0.20 operator_trace_contrast=0.15"}

for path in "$DATA" "$MIX" "$QUALITY" "$TEXT_OVERLAP" "$TRACE_OVERLAP" "$OOD_OVERLAP" "$FACTOR_OVERLAP" "$INIT" "$REFERENCE" "$REPLAY"; do
  [ -s "$path" ] || { echo "[opanchor] required input missing: $path" >&2; exit 2; }
done
[ "$(sha256sum "$REPLAY" | awk '{print $1}')" = "$REPLAY_SHA" ] || {
  echo "[opanchor] replay hash mismatch" >&2; exit 3;
}
[ ! -e "$OUT" ] || { echo "[opanchor] refusing existing output: $OUT" >&2; exit 4; }
case "$OUT" in "$BASE"/train/sft_operator_trace_anchor_*) ;; *)
  echo "[opanchor] OUT must be an isolated train/sft_operator_trace_anchor_* directory" >&2; exit 5;;
esac

"$PY" - "$DATA" "$MIX" "$QUALITY" "$TEXT_OVERLAP" "$TRACE_OVERLAP" "$OOD_OVERLAP" "$FACTOR_OVERLAP" "$REQUIRED_EXCLUDED_CONTRACT" <<'PY'
import hashlib
import json
import os
import sys

data, mix_path, quality_path, text_path, trace_path, ood_path, factor_path = [
    os.path.realpath(path) for path in sys.argv[1:8]
]
required_contract = sys.argv[8]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

digest = sha256(data)
mix, quality, text, trace, ood, factor = (json.load(open(path)) for path in (
    mix_path, quality_path, text_path, trace_path, ood_path, factor_path
))
if mix.get("out") != data or mix.get("out_sha256") != digest:
    raise SystemExit("[opanchor] mix report does not bind data")
if any(mix.get(key, 1) for key in (
    "malformed_json_rows", "missing_question_response_or_group", "duplicate_normalized_questions_dropped"
)):
    raise SystemExit("[opanchor] structural mix gate failed")
if not mix.get("eval_filter", {}).get("enabled"):
    raise SystemExit("[opanchor] public-eval filter was not enabled")
if required_contract and not mix.get("excluded_contract_rows", {}).get(required_contract, 0):
    raise SystemExit("[opanchor] expected source-split contract exclusion is absent")
if quality.get("data_sha256") != digest or any(quality.get(key, 1) for key in (
    "malformed_rows", "missing_question_or_response", "duplicate_normalized_questions"
)):
    raise SystemExit("[opanchor] quality report does not bind clean data")
if any(quality.get("eval_overlap", {}).get(key, 1) for key in ("exact_prompt_hits", "ngram_prompt_hits")):
    raise SystemExit("[opanchor] public-eval prompt overlap remains")
if text.get("data_sha256") != digest or text.get("overlap", {}).get("exact_rows") or text.get("overlap", {}).get("ngram_rows"):
    raise SystemExit("[opanchor] public-eval text overlap remains")
for name, report, cases in (("trace", trace, 12), ("ood", ood, 72), ("factor", factor, 600)):
    if report.get("data_sha256") != digest or report.get("cases") != cases:
        raise SystemExit(f"[opanchor] {name} audit does not bind the expected held-out suite")
    if report.get("malformed_rows") or report.get("missing_question") or report.get("exact_prompt_hits") or report.get("ngram_prompt_hits"):
        raise SystemExit(f"[opanchor] {name} held-out overlap remains")
expected_groups = {"math", "procedural", "code", "teacher", "primitives", "operator_trace_contrast"}
if set(mix.get("training_group_rows", {})) != expected_groups:
    raise SystemExit("[opanchor] broad anchor lacks an expected training group")
print(json.dumps({"admitted": True, "data_sha256": digest, "rows": quality["valid_rows"]}, sort_keys=True))
PY

cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    probe = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    del probe
except Exception as exc:
    print(f"[opanchor] CUDA preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
PY
then
  echo "[opanchor] CUDA unavailable after allocation" >&2
  exit 75
fi

echo "[opanchor] init=$INIT reference=$REFERENCE data=$DATA out=$OUT weights=$WEIGHTS"
"$PY" sft.py --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --epochs "$EPOCHS" --batch-size "$BS" --lr-muon "$LRMUON" --lr-adam "$LRADAM" \
  --group-field training_group --prompt-override-field completion_prompt --sample-weights $WEIGHTS \
  --reference "$REFERENCE" --replay-prompts "$REPLAY" --replay-weight "$REPLAY_WEIGHT" \
  --replay-batch-size "$REPLAY_BATCH" --replay-max-tokens "$REPLAY_MAX_TOKENS" \
  --freeze-lexicon --compile --out "$OUT"
[ -s "$OUT/sft_ep${EPOCHS}.pt" ] || { echo "[opanchor] final checkpoint missing" >&2; exit 6; }
echo "[opanchor] done checkpoint=$OUT/sft_ep${EPOCHS}.pt"
