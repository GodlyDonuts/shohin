#!/bin/bash
#SBATCH -J shohin-drs-cycle
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/drs_causal_cycle_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CKPT=$BASE/train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt
EPISODES=$BASE/artifacts/evals/digitwise_recurrent_v2_heldout.jsonl
OUT=$BASE/artifacts/evals/drs_causal_cycle_post_drs_r3.json
TOK=$BASE/artifacts/shohin-tok-32k.json
SNAP=$BASE/scratchpad/drs_causal_cycle_snapshot_${SLURM_JOB_ID}

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4

mkdir -p "$BASE/logs" "$(dirname "$OUT")"
[ -f "$CKPT" ] || { echo "missing checkpoint"; exit 2; }
[ -f "$EPISODES" ] || { echo "missing episodes"; exit 3; }
[ -f "$TOK" ] || { echo "missing tokenizer"; exit 4; }
[ ! -e "$OUT" ] || { echo "refusing existing output"; exit 5; }

for attempt in 1 2 3 4 5 6 7 8; do
  if "$PY" - <<'PY'
import torch
assert torch.cuda.is_available()
torch.empty(1, device="cuda", dtype=torch.bfloat16)
print(torch.cuda.get_device_name(0))
PY
  then
    echo "cuda_ok"
    break
  fi
  sleep 15
  [ "$attempt" -eq 8 ] && exit 12
done

umask 077
[ ! -e "$SNAP" ] || { echo "refusing existing snapshot"; exit 13; }
mkdir -p "$SNAP/train" "$SNAP/input"
cleanup() {
  chmod -R u+w "$SNAP" 2>/dev/null || true
  rm -rf "$SNAP"
}
trap cleanup EXIT

for name in probe_drs_causal_cycle.py digitwise_protocol.py eval_suite.py model.py probe_digitwise_workspace.py; do
  source_file=$BASE/train/$name
  [ -f "$source_file" ] && [ ! -L "$source_file" ] || { echo "invalid scientific source $name"; exit 14; }
  cp --reflink=auto -- "$source_file" "$SNAP/train/$name"
done
cp --reflink=auto -- "$CKPT" "$SNAP/input/sft_ep1.pt"
cp --reflink=auto -- "$EPISODES" "$SNAP/input/heldout.jsonl"
cp --reflink=auto -- "$TOK" "$SNAP/input/tokenizer.json"
chmod -R a-w "$SNAP"

(
  cd "$SNAP"
  sha256sum -c - <<'SHA256'
d81ff28db221e706d0b283fc933d718331a05a4a36a505d8b68898340679d82d  train/probe_drs_causal_cycle.py
708489d61c212c402e1533a1483e77bf3fd2d1a057ce924321bb19e4888461f6  train/digitwise_protocol.py
d6f70b8828c967d7f59fae842f3320c6378ae42d5d8fa7b16e0e82ff5620e5e6  train/eval_suite.py
45fc0dc46ceb0f91d08e3f671cbe9ef202ea212e72d5bba8b77356c3fb0983d4  train/model.py
fb545450a93bbc04aac1549efd0a70b863f50e458fd95993cf9935bbe4a53ace  train/probe_digitwise_workspace.py
d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459  input/sft_ep1.pt
89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646  input/heldout.jsonl
87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4  input/tokenizer.json
SHA256
)

export SHOHIN_DRS_CYCLE_SNAPSHOT_ROOT="$SNAP"
export SHOHIN_DRS_CYCLE_CODE_VERIFIED=1
export PYTHONPATH="$SNAP/train"
cd "$SNAP/train"
"$PY" probe_drs_causal_cycle.py \
  --ckpt "$SNAP/input/sft_ep1.pt" \
  --tokenizer "$SNAP/input/tokenizer.json" \
  --episodes "$SNAP/input/heldout.jsonl" \
  --out "$OUT" \
  --transition-index 2 \
  --per-regime 10 \
  --layer 29 \
  --max-new 96 \
  --canonical

echo "done"
