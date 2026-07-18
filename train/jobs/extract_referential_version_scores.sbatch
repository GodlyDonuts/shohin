#!/bin/bash
#SBATCH -J shohin-r10-scores
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc23,evc26,evc31,evc32,evc34,evc36,evc37,evc38,evc43,evc44,evc45,evc49,evc50
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/r10_scores_%j.out
#SBATCH --open-mode=append
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=4

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
BASE_CKPT=${BASE_CKPT:-$BASE/train/flagship_out/best_step200000.pt}
POINTER=${POINTER:-$BASE/train/microcode_referential_pointer_200k_r4/microcode_adapter_ep1.pt}
TOKENIZER=${TOKENIZER:-$BASE/artifacts/shohin-tok-32k.json}
BOARD_NAME=${BOARD_NAME:?set calibration or confirmation}
ADAPTER=${ADAPTER:?set the frozen R9c no_syndrome adapter}
ADAPTER_SHA256=${ADAPTER_SHA256:?set the exact frozen adapter SHA-256}
DATA=${DATA:?set the exact evaluation JSONL}
DATA_SHA256=${DATA_SHA256:?set the exact board SHA-256}
ADMISSION=${ADMISSION:?set the evaluation structural admission manifest}
ADMISSION_SHA256=${ADMISSION_SHA256:?set the exact structural admission SHA-256}
LABEL_ADMISSION=${LABEL_ADMISSION:?set the evaluation referential-label admission manifest}
LABEL_ADMISSION_SHA256=${LABEL_ADMISSION_SHA256:?set the exact label admission SHA-256}
GATE_MANIFEST=${GATE_MANIFEST:?set the frozen score gate manifest}
GATE_MANIFEST_SHA256=${GATE_MANIFEST_SHA256:?set the exact gate manifest SHA-256}
GATE_ADMISSION=${GATE_ADMISSION:?set the independent gate admission}
GATE_ADMISSION_SHA256=${GATE_ADMISSION_SHA256:?set the exact gate admission SHA-256}
EVALUATOR=${EVALUATOR:?set the canonical evaluator source path}
EVALUATOR_SHA256=${EVALUATOR_SHA256:?set the exact evaluator source SHA-256}
OUT=${OUT:?set a fresh R10 score output}
CODE_REVISION=${CODE_REVISION:?set the exact committed extractor git revision}
EXTRACTOR_SHA256=${EXTRACTOR_SHA256:?set the frozen extractor file SHA-256}

case "$BOARD_NAME" in
  calibration|confirmation) ;;
  *) echo "[r10-scores] invalid BOARD_NAME: $BOARD_NAME" >&2; exit 2 ;;
esac

for path in \
  "$BASE_CKPT" "$POINTER" "$TOKENIZER" "$ADAPTER" "$DATA" \
  "$ADMISSION" "$LABEL_ADMISSION" "$GATE_MANIFEST" "$GATE_ADMISSION" "$EVALUATOR"; do
  [ -s "$path" ] || { echo "[r10-scores] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "[r10-scores] refusing existing output: $OUT" >&2; exit 3; }

cd "$BASE/train"
"$PY" extract_referential_version_scores.py \
  --board-name "$BOARD_NAME" \
  --base "$BASE_CKPT" --pointer-adapter "$POINTER" \
  --adapter "$ADAPTER" --adapter-sha256 "$ADAPTER_SHA256" \
  --data "$DATA" --data-sha256 "$DATA_SHA256" --tokenizer "$TOKENIZER" \
  --admission "$ADMISSION" --admission-sha256 "$ADMISSION_SHA256" \
  --label-admission "$LABEL_ADMISSION" \
  --label-admission-sha256 "$LABEL_ADMISSION_SHA256" \
  --gate-manifest "$GATE_MANIFEST" --gate-manifest-sha256 "$GATE_MANIFEST_SHA256" \
  --gate-admission "$GATE_ADMISSION" --gate-admission-sha256 "$GATE_ADMISSION_SHA256" \
  --evaluator "$EVALUATOR" --evaluator-sha256 "$EVALUATOR_SHA256" \
  --out "$OUT" \
  --batch-size "${BATCH_SIZE:-16}" --seed "${SEED:-20260714}" \
  --code-revision "$CODE_REVISION" --extractor-sha256 "$EXTRACTOR_SHA256"
