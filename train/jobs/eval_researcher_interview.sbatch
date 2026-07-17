#!/bin/bash
#SBATCH --job-name=researcher-interview
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=evc26,evc31,evc32,evc36,evc44
#SBATCH --no-requeue
#SBATCH --open-mode=truncate
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/researcher_interview_%j_%r.out
#SBATCH -e /lustre/fs1/home/sa305415/shohin/logs/researcher_interview_%j_%r.err

set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:?set SOURCE_ROOT to an isolated clean Git checkout}
DATA_ROOT=${DATA_ROOT:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$DATA_ROOT/miniforge3/bin/python}
CKPT=${CKPT:?set CKPT to an immutable Shohin checkpoint}
CKPT_SHA256=${CKPT_SHA256:?set CKPT_SHA256 to the reviewed checkpoint digest}
TOKENIZER=${TOKENIZER:-$DATA_ROOT/artifacts/shohin-tok-32k.json}
TOKENIZER_SHA256=${TOKENIZER_SHA256:?set TOKENIZER_SHA256 to the reviewed tokenizer digest}
INTERVIEW=${INTERVIEW:-$SOURCE_ROOT/artifacts/evals/researcher_interview_v1.json}
INTERVIEW_SHA256=${INTERVIEW_SHA256:?set INTERVIEW_SHA256 to the frozen interview digest}
SOURCE_COMMIT=${SOURCE_COMMIT:?set SOURCE_COMMIT to the reviewed clean commit}
TAG=${TAG:-$(basename "$CKPT" .pt)}
OUT=${OUT:-$DATA_ROOT/artifacts/eval_history/researcher_interview/${TAG}_${SLURM_JOB_ID}.json}
MAX_NEW=${MAX_NEW:-128}
SEED=${SEED:-20260717}
SOURCE_FILES=(
  artifacts/evals/researcher_interview_v1.json
  train/eval_researcher_interview.py
  train/jobs/eval_researcher_interview.sbatch
  train/eval_suite.py
  train/model.py
)

for path in "$SOURCE_ROOT" "$DATA_ROOT" "$CKPT" "$TOKENIZER" "$INTERVIEW" "$OUT"; do
  if [[ "$path" != /* ]]; then
    echo "researcher-interview paths must be absolute: $path" >&2
    exit 2
  fi
done
case "$OUT" in
  "$DATA_ROOT"/artifacts/eval_history/researcher_interview/*.json) ;;
  *) echo "OUT must be an isolated researcher-interview JSON artifact: $OUT" >&2; exit 2 ;;
esac
if [[ -e "$OUT" ]]; then
  echo "refusing existing output: $OUT" >&2
  exit 2
fi
for pair in "$CKPT:$CKPT_SHA256" "$TOKENIZER:$TOKENIZER_SHA256" "$INTERVIEW:$INTERVIEW_SHA256"; do
  path=${pair%%:*}
  expected=${pair#*:}
  if [[ ! -f "$path" || ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    echo "missing input or malformed SHA256 binding: $path" >&2
    exit 2
  fi
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$observed" != "$expected" ]]; then
    echo "SHA256 mismatch for $path" >&2
    exit 2
  fi
done

RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
if [[ "$RESTART_COUNT" != 0 ]]; then
  echo "refusing restarted researcher-interview job" >&2
  exit 2
fi
JOB_RECORD=$(scontrol show job -o "$SLURM_JOB_ID")
if [[ "$JOB_RECORD" != *" Requeue=0 "* ]]; then
  echo "live Slurm policy does not preserve --no-requeue" >&2
  exit 2
fi
for spec in "StdOut=researcher_interview_${SLURM_JOB_ID}_${RESTART_COUNT}.out" \
            "StdErr=researcher_interview_${SLURM_JOB_ID}_${RESTART_COUNT}.err"; do
  key=${spec%%=*}
  suffix=${spec#*=}
  observed=$(sed -n "s/.* ${key}=\([^ ]*\).*/\1/p" <<<"$JOB_RECORD")
  if [[ "$observed" != *"$suffix" ]]; then
    echo "live Slurm $key path is not restart-distinguishing: $observed" >&2
    exit 2
  fi
done

if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
  echo "SOURCE_ROOT must be a Git checkout" >&2
  exit 2
fi
cd "$SOURCE_ROOT"
if [[ "$(git rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "reviewed source commit is not checked out" >&2
  exit 2
fi
SOURCE_MANIFEST_SHA256=$("$PY" - "$SOURCE_COMMIT" "${SOURCE_FILES[@]}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

commit, *paths = sys.argv[1:]
sources = {}
for name in paths:
    payload = Path(name).read_bytes()
    committed = subprocess.check_output(["git", "show", f"{commit}:{name}"])
    if payload != committed:
        raise SystemExit(f"source differs from reviewed commit: {name}")
    sources[name] = hashlib.sha256(payload).hexdigest()
encoded = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(encoded).hexdigest())
PY
)

GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "$(wc -l <<<"$GPU_NAMES" | tr -d ' ')" != 1 || "$GPU_NAMES" != *H100* ]]; then
  echo "researcher-interview requires exactly one visible H100; observed: $GPU_NAMES" >&2
  exit 75
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$SOURCE_ROOT/train"
if ! timeout --kill-after=10s 90s "$PY" - <<'PY'
import sys
import torch

try:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("allocated device is not an available H100")
    probe = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    probe.add_(1)
    torch.cuda.synchronize()
except Exception as exc:
    print(f"[researcher-interview] H100 preflight failed: {exc}", file=sys.stderr)
    sys.exit(75)
PY
then
  echo "researcher-interview H100 preflight failed after allocation" >&2
  exit 75
fi

"$PY" train/eval_researcher_interview.py \
  --ckpt "$CKPT" \
  --checkpoint-sha256 "$CKPT_SHA256" \
  --tokenizer "$TOKENIZER" \
  --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --interview "$INTERVIEW" \
  --interview-sha256 "$INTERVIEW_SHA256" \
  --out "$OUT" \
  --source-commit "$SOURCE_COMMIT" \
  --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --device cuda \
  --max-new "$MAX_NEW" \
  --seed "$SEED"
echo "[researcher-interview] wrote $OUT"
