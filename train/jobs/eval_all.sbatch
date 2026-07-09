#!/bin/bash
#SBATCH -J shohin-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/eval_%j.out
# Full reasoning-benchmark board on one checkpoint. CKPT defaults to the newest SFT epoch (or set
# CKPT=<path> to eval a base/pretrain ckpt). GSM8K uses self-consistency (maj@k) for the headline
# number; the rest are pass@1. sft_ep*.pt / ckpt_*.pt both carry {model,cfg} so eval loads directly.
set -uo pipefail
BASE=/lustre/fs1/home/sa305415/shohin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE/train
if [ -n "${TARGET_STEP:-}" ] && [ -z "${CKPT:-}" ]; then
  CKPT=$(printf "$BASE/train/flagship_out/ckpt_%07d.pt" "$TARGET_STEP")
  if [ ! -f "$CKPT" ]; then
    CKPT="$BASE/train/flagship_out/best_step${TARGET_STEP}.pt"
  fi
fi
CKPT=${CKPT:-$(ls -t $BASE/train/sft_out/sft_ep*.pt 2>/dev/null | head -1)}
[ -z "$CKPT" ] && CKPT=$(ls -t $BASE/train/flagship_out/ckpt_[0-9]*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "[eval-job] checkpoint missing: ${CKPT:-unset}" >&2
  exit 2
fi
TOK=$BASE/artifacts/shohin-tok-32k.json
E=$BASE/artifacts/evals
PY=$BASE/miniforge3/bin/python
HIST=$BASE/artifacts/eval_history
mkdir -p "$HIST"
CKPT_BASE=$(basename "$CKPT" .pt)
RUN_TAG=${RUN_TAG:-$CKPT_BASE}
RUN_LOG=$HIST/${RUN_TAG}_${SLURM_JOB_ID:-local}.log
SUMMARY=$HIST/metrics.jsonl
export CKPT_BASE RUN_TAG SUMMARY RUN_LOG
echo "[eval-job] ckpt=$CKPT run_tag=$RUN_TAG log=$RUN_LOG"
exec > >(tee -a "$RUN_LOG") 2>&1
for i in $(seq 1 30); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1)
  [ "${U:-9999}" -lt 3000 ] 2>/dev/null && break; sleep 10
done
echo "===== SHOHIN BENCHMARK BOARD  ckpt=$(basename $CKPT) ====="
$PY eval_suite.py --ckpt $CKPT --tokenizer $TOK --task gsm8k    --data $E/gsm8k.jsonl    --n ${N:-200} --k ${K:-8} --temp 0.8
$PY eval_suite.py --ckpt $CKPT --tokenizer $TOK --task gsm8k    --data $E/gsm8k.jsonl    --n ${N:-200} --k 1
$PY eval_suite.py --ckpt $CKPT --tokenizer $TOK --task math500  --data $E/math500.jsonl  --n ${N:-200} --k 1
$PY eval_code.py  --ckpt $CKPT --tokenizer $TOK --task humaneval --data $E/humaneval_full.jsonl --n 164 --k 1
$PY eval_code.py  --ckpt $CKPT --tokenizer $TOK --task mbpp      --data $E/mbpp_full.jsonl      --n ${N:-200} --k 1
$PY - "$RUN_LOG" "$SUMMARY" <<'PY'
import datetime as dt
import json
import os
import re
import sys

log_path, summary_path = sys.argv[1], sys.argv[2]
pat = re.compile(
    r"^(gsm8k|math500|humaneval|mbpp)\s+"
    r"(maj@\d+|pass@\d+)\s+"
    r"ckpt=(.*?)\s+step=(.*?)\s+"
    r"(\d+)/(\d+)\s+=\s+([0-9.]+)%"
)
rows = []
with open(log_path, "r", errors="replace") as f:
    for line in f:
        m = pat.search(line.strip())
        if not m:
            continue
        correct, total = int(m.group(5)), int(m.group(6))
        rows.append({
            "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "run_tag": os.environ.get("RUN_TAG"),
            "checkpoint": os.environ.get("CKPT_BASE"),
            "task": m.group(1),
            "metric": m.group(2),
            "ckpt_path": m.group(3),
            "step": m.group(4),
            "correct": correct,
            "total": total,
            "accuracy": correct / max(total, 1),
            "percent": float(m.group(7)),
            "log": log_path,
        })
if rows:
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "a") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[eval-job] appended {len(rows)} metric rows to {summary_path}")
else:
    print("[eval-job] WARNING: no metric rows parsed", file=sys.stderr)
PY
echo "===== board done ====="
