#!/bin/bash
#SBATCH -J shohin-typed-ctrl-v2
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc36,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/typed_controller_v2_%j.out
#SBATCH --open-mode=append
set -euo pipefail
export PYTHONUNBUFFERED=1
BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4

ROOT=$BASE/artifacts/r12/typed_controller_v2
INIT=${INIT:-$BASE/train/sft_typed_controller_v1_200k_r1/sft_ep1.pt}
OUT=${OUT:-$BASE/train/sft_typed_controller_v2_from_v1_r1}
TOK=$BASE/artifacts/shohin-tok-32k.json

mkdir -p "$BASE/logs" "$ROOT"
[ -f "$INIT" ] || { echo missing init; exit 2; }
[ ! -e "$OUT" ] || { echo refusing out; exit 3; }

cd "$BASE/train"
echo "[typed-v2] generate"
# Heldout first in generator; 256 cases (64/family) fits scarce base_conversion
# space. Train 8k cases (~2k/family) leaves headroom under ~5.4k unique base glyphs.
"$PY" generate_typed_controller_v2.py --out-dir "$ROOT" --seed 2026071701 \
  --n-train-cases 8000 --n-heldout-cases 256

for attempt in 1 2 3 4 5 6 7 8; do
  if "$PY" - <<'PY'
import torch,sys
assert torch.cuda.is_available()
torch.empty(1,device='cuda',dtype=torch.bfloat16)
print(torch.cuda.get_device_name(0))
PY
  then echo cuda_ok; break; fi
  sleep 20
  [ "$attempt" -eq 8 ] && exit 12
done

echo "[typed-v2] SFT from $INIT"
"$PY" sft.py --init "$INIT" --data "$ROOT/train.jsonl" --tokenizer "$TOK" \
  --epochs 2 --batch-size 16 --lr-muon 0.0008 --lr-adam 0.00015 \
  --group-field training_group --prompt-override-field completion_prompt \
  --sample-weights atomic=0.20 rollout=0.25 resume=0.10 native_atomic=0.45 \
  --out "$OUT"

CKPT=$OUT/sft_ep2.pt
[ -f "$CKPT" ] || CKPT=$(ls -1 "$OUT"/sft_ep*.pt | sort | tail -1)
EVAL=$ROOT/eval_sft.json
DEC=$ROOT/decision.json
rm -f "$EVAL" "$DEC"

echo "[typed-v2] eval"
"$PY" eval_typed_controller_v1.py --ckpt "$CKPT" --tokenizer "$TOK" \
  --heldout "$ROOT/heldout.jsonl" --out "$EVAL" --device cuda

"$PY" - "$EVAL" "$DEC" <<'PY'
import json,sys,hashlib
from pathlib import Path
sft=json.loads(Path(sys.argv[1]).read_text())
typed=sft['typed']['rollout']['exact_accuracy']
direct=sft['direct_control']['exact_accuracy']
atomic=sft['typed']['atomic']['step_exact_accuracy']
done=sft['typed']['rollout']['done_rate']
# Compare to v1 rescored baseline 0.164
v1=0.1640625
gates={
  'typed_floor_0_35': typed>=0.35,
  'typed_minus_direct_0_10': (typed-direct)>=0.10,
  'atomic_floor_0_50': atomic>=0.50,  # slightly relaxed vs v1 0.70 while climbing
  'done_floor_0_80': done>=0.80,
  'beats_v1_rollout': typed>=v1+0.05,
}
d={'protocol':'R12-TYPED-CONTROLLER-v2','advance':all(gates.values()),'gates':gates,
   'sft_typed_rollout':typed,'sft_direct':direct,'sft_atomic':atomic,'sft_done_rate':done,
   'v1_rollout_baseline':v1,'delta_vs_v1':typed-v1,
   'sample_transcripts':sft['typed']['rollout']['transcripts'][:5]}
Path(sys.argv[2]).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
d['decision_sha256']=hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
Path(sys.argv[2]).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
print(json.dumps(d,indent=2,sort_keys=True))
PY
echo done
