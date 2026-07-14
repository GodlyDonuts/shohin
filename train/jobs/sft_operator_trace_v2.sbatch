#!/bin/bash
# Isolated, behavior-preserving test of factorized operator-to-state binding.
# This never writes a flagship path and is not a promotion mechanism.
#SBATCH -J shohin-sft-optrace
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc36,evc43,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sft_operator_trace_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATA=$BASE/artifacts/sft/operator_trace_contrast_v2_train.jsonl
DATA_REPORT=$BASE/artifacts/sft/operator_trace_contrast_v2_report.json
QUALITY=$BASE/artifacts/sft/operator_trace_contrast_v2_quality.json
TRACE_OVERLAP=$BASE/artifacts/sft/operator_trace_contrast_v2.thinking_trace_overlap.json
OOD_OVERLAP=$BASE/artifacts/sft/operator_trace_contrast_v2.trace_state_ood_v2_overlap.json
FACTOR_OVERLAP=$BASE/artifacts/sft/operator_trace_contrast_v2.factor_wf_overlap.json
INIT=${INIT:?set INIT to the immutable retained-SFT checkpoint}
REFERENCE=${REFERENCE:-$INIT}
REPLAY=${REPLAY:?set REPLAY to the frozen prompt-only replay JSONL}
REPLAY_SHA=${REPLAY_SHA:?set REPLAY_SHA to its expected SHA-256}
OUT=${OUT:?set OUT to a fresh train/sft_operator_trace_* directory}
EPOCHS=${EPOCHS:-1}
BS=${BS:-16}
LRMUON=${LRMUON:-0.0008}
LRADAM=${LRADAM:-0.0002}
REPLAY_WEIGHT=${REPLAY_WEIGHT:-0.25}
REPLAY_BATCH=${REPLAY_BATCH:-4}
REPLAY_MAX_TOKENS=${REPLAY_MAX_TOKENS:-128}

for path in "$DATA" "$DATA_REPORT" "$QUALITY" "$TRACE_OVERLAP" "$OOD_OVERLAP" "$FACTOR_OVERLAP" "$INIT" "$REFERENCE" "$REPLAY"; do
  [ -s "$path" ] || { echo "[optrace] required input missing: $path" >&2; exit 2; }
done
[ "$(sha256sum "$REPLAY" | awk '{print $1}')" = "$REPLAY_SHA" ] || {
  echo "[optrace] replay hash mismatch" >&2; exit 3;
}
[ ! -e "$OUT" ] || { echo "[optrace] refusing existing output: $OUT" >&2; exit 4; }
case "$OUT" in "$BASE"/train/sft_operator_trace_*) ;; *)
  echo "[optrace] OUT must be an isolated train/sft_operator_trace_* directory" >&2; exit 5;;
esac

"$PY" - "$DATA" "$DATA_REPORT" "$QUALITY" "$TRACE_OVERLAP" "$OOD_OVERLAP" "$FACTOR_OVERLAP" <<'PY'
import hashlib
import json
import sys

data, report_path, quality_path, trace_path, ood_path, factor_path = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

data_sha = sha256(data)
report = json.load(open(report_path))
quality = json.load(open(quality_path))
trace = json.load(open(trace_path))
ood = json.load(open(ood_path))
factor = json.load(open(factor_path))
if report.get("train_sha256") != data_sha or report.get("train_rows", 0) < 200_000:
    raise SystemExit("[optrace] data report does not bind a sufficiently large frozen corpus")
if quality.get("data_sha256") != data_sha or quality.get("valid_rows") != report["train_rows"]:
    raise SystemExit("[optrace] quality report does not bind frozen data")
if any(quality.get(key, 1) for key in ("malformed_rows", "missing_question_or_response", "duplicate_normalized_questions")):
    raise SystemExit("[optrace] data quality gate failed")
if any(quality.get("eval_overlap", {}).get(key, 1) for key in ("exact_prompt_hits", "ngram_prompt_hits")):
    raise SystemExit("[optrace] public eval overlap gate failed")
for name, audit in (("trace", trace), ("prior_ood", ood), ("factor_wording_full", factor)):
    if audit.get("data_sha256") != data_sha or audit.get("exact_prompt_hits") or audit.get("ngram_prompt_hits"):
        raise SystemExit("[optrace] {} overlap gate failed".format(name))
if sorted(factor.get("case_regimes", ())) != ["full", "wording"]:
    raise SystemExit("[optrace] factor overlap does not cover wording and full regimes")
print(json.dumps({"data_sha256": data_sha, "rows": report["train_rows"], "admitted": True}, sort_keys=True))
PY

cd "$BASE/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    probe = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    del probe
except Exception as exc:
    print(f"[optrace] CUDA preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
PY
then
  echo "[optrace] CUDA unavailable after allocation" >&2
  exit 75
fi

echo "[optrace] init=$INIT reference=$REFERENCE data=$DATA out=$OUT epochs=$EPOCHS"
"$PY" sft.py --init "$INIT" --data "$DATA" --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --epochs "$EPOCHS" --batch-size "$BS" --lr-muon "$LRMUON" --lr-adam "$LRADAM" \
  --group-field training_group --prompt-override-field completion_prompt \
  --sample-weights operator_trace_contrast=1 \
  --reference "$REFERENCE" --replay-prompts "$REPLAY" --replay-weight "$REPLAY_WEIGHT" \
  --replay-batch-size "$REPLAY_BATCH" --replay-max-tokens "$REPLAY_MAX_TOKENS" \
  --freeze-lexicon --compile --out "$OUT"
[ -s "$OUT/sft_ep${EPOCHS}.pt" ] || { echo "[optrace] final checkpoint missing" >&2; exit 6; }
echo "[optrace] done checkpoint=$OUT/sft_ep${EPOCHS}.pt"
