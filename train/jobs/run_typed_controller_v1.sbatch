#!/bin/bash
#SBATCH -J shohin-typed-ctrl-v1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/typed_controller_v1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4

ROOT=$BASE/artifacts/r12/typed_controller_v1
INIT=${INIT:-$BASE/train/flagship_out/best_step200000.pt}
OUT=${OUT:-$BASE/train/sft_typed_controller_v1_200k_r1}
TOK=$BASE/artifacts/shohin-tok-32k.json

mkdir -p "$BASE/logs" "$ROOT"
[ -f "$INIT" ] || { echo "missing init $INIT" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "refusing existing out $OUT" >&2; exit 3; }

cd "$BASE"
echo "[typed-ctrl] generate data"
"$PY" train/generate_typed_controller_v1.py --out-dir "$ROOT" --seed 2026071611 \
  --n-train-cases 12000 --n-heldout-cases 1024

# CUDA preflight
for attempt in 1 2 3 4 5 6 7 8; do
  if "$PY" - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit(11)
torch.empty(1, device="cuda", dtype=torch.bfloat16)
print(torch.cuda.get_device_name(0))
PY
  then echo "[typed-ctrl] cuda ok attempt=$attempt"; break; fi
  sleep 20
  [ "$attempt" -eq 8 ] && exit 12
done

cd "$BASE/train"
echo "[typed-ctrl] SFT init=$INIT out=$OUT"
"$PY" sft.py --init "$INIT" --data "$ROOT/train.jsonl" --tokenizer "$TOK" \
  --epochs 1 --batch-size 16 --lr-muon 0.001 --lr-adam 0.0002 \
  --group-field training_group --prompt-override-field completion_prompt \
  --sample-weights atomic=0.45 rollout=0.40 resume=0.15 \
  --compile --out "$OUT"

CKPT=$OUT/sft_ep1.pt
[ -f "$CKPT" ] || CKPT=$(ls -1 "$OUT"/sft_ep*.pt | head -1)
RAW_EVAL=$ROOT/eval_raw200k.json
SFT_EVAL=$ROOT/eval_sft.json
DECISION=$ROOT/decision.json
[ ! -e "$RAW_EVAL" ] || { echo "refusing $RAW_EVAL" >&2; exit 4; }
[ ! -e "$SFT_EVAL" ] || { echo "refusing $SFT_EVAL" >&2; exit 5; }

echo "[typed-ctrl] eval raw"
"$PY" eval_typed_controller_v1.py --ckpt "$INIT" --tokenizer "$TOK" \
  --heldout "$ROOT/heldout.jsonl" --out "$RAW_EVAL" --device cuda
echo "[typed-ctrl] eval sft"
"$PY" eval_typed_controller_v1.py --ckpt "$CKPT" --tokenizer "$TOK" \
  --heldout "$ROOT/heldout.jsonl" --out "$SFT_EVAL" --device cuda

"$PY" - "$RAW_EVAL" "$SFT_EVAL" "$DECISION" <<'PY'
import json, sys, hashlib
from pathlib import Path
raw_p, sft_p, out_p = map(Path, sys.argv[1:])
raw = json.loads(raw_p.read_text())
sft = json.loads(sft_p.read_text())
typed = sft["typed"]["rollout"]["exact_accuracy"]
direct = sft["direct_control"]["exact_accuracy"]
atomic = sft["typed"]["atomic"]["step_exact_accuracy"]
done = sft["typed"]["rollout"]["done_rate"]
raw_typed = raw["typed"]["rollout"]["exact_accuracy"]
gates = {
    "typed_floor_0_35": typed >= 0.35,
    "typed_minus_direct_0_10": (typed - direct) >= 0.10,
    "atomic_floor_0_70": atomic >= 0.70,
    "done_floor_0_80": done >= 0.80,
    "improves_raw_typed": typed >= raw_typed + 0.10,
}
decision = {
    "protocol": "R12-TYPED-CONTROLLER-v1",
    "advance": all(gates.values()),
    "gates": gates,
    "sft_typed_rollout": typed,
    "sft_direct": direct,
    "sft_atomic": atomic,
    "sft_done_rate": done,
    "raw_typed_rollout": raw_typed,
    "raw_eval": str(raw_p),
    "sft_eval": str(sft_p),
}
out_p.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
blob = out_p.read_bytes()
decision["decision_sha256"] = hashlib.sha256(blob).hexdigest()
out_p.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
print(json.dumps(decision, indent=2, sort_keys=True))
PY
echo "[typed-ctrl] done"
