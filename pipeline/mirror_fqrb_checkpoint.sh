#!/bin/bash
# Mirror one completed isolated FQRB checkpoint from Newton to this workspace.
# This is a local DR helper: it reads the remote artifact only after it exists,
# verifies its SHA-256, then atomically promotes the local copy.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_BASE=/lustre/fs1/home/sa305415/shohin
REMOTE_CKPT=${REMOTE_CKPT:-$REMOTE_BASE/train/fqrb_200k_l19_r1/cra_ep1.pt}
LOCAL_CKPT=${LOCAL_CKPT:-$ROOT/train/fqrb_200k_l19_r1/cra_ep1.pt}
MANIFEST=${MANIFEST:-$ROOT/train/fqrb_200k_l19_r1/cra_ep1.mirror.json}
MAX_POLLS=${MAX_POLLS:-300}
SLEEP_SECONDS=${SLEEP_SECONDS:-60}

cd "$ROOT"
[ -f .env ] || { echo "[fqrb-mirror] missing local .env" >&2; exit 2; }
set -a
source .env
set +a
export SSHPASS="$NEWTON_PW"
mkdir -p "$(dirname "$LOCAL_CKPT")"

if [ -s "$LOCAL_CKPT" ] && [ -s "$MANIFEST" ]; then
  echo "[fqrb-mirror] verified local checkpoint already present"
  exit 0
fi

for poll in $(seq 1 "$MAX_POLLS"); do
  if sshpass -e ssh -o ConnectTimeout=15 newton "test -s '$REMOTE_CKPT'"; then
    remote_sha=$(sshpass -e ssh -o ConnectTimeout=15 newton "sha256sum '$REMOTE_CKPT'" | awk '{print $1}')
    part="$LOCAL_CKPT.part"
    rm -f "$part"
    sshpass -e scp -q "newton:$REMOTE_CKPT" "$part"
    local_sha=$(sha256sum "$part" | awk '{print $1}')
    if [ "$local_sha" != "$remote_sha" ]; then
      echo "[fqrb-mirror] checksum mismatch after transfer" >&2
      rm -f "$part"
      exit 3
    fi
    mv "$part" "$LOCAL_CKPT"
    REMOTE_CKPT="$REMOTE_CKPT" REMOTE_SHA="$remote_sha" LOCAL_CKPT="$LOCAL_CKPT" python3 - "$MANIFEST" <<'PY'
import json
import os
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.write_text(json.dumps({
    'audit': 'isolated_fqrb_checkpoint_mirror_v1',
    'remote_checkpoint': os.environ['REMOTE_CKPT'],
    'local_checkpoint': os.environ['LOCAL_CKPT'],
    'sha256': os.environ['REMOTE_SHA'],
}, indent=2, sort_keys=True) + '\n')
PY
    echo "[fqrb-mirror] verified $LOCAL_CKPT sha256=$remote_sha"
    exit 0
  fi
  if [ "$poll" = "$MAX_POLLS" ]; then
    echo "[fqrb-mirror] timed out waiting for remote checkpoint" >&2
    exit 4
  fi
  sleep "$SLEEP_SECONDS"
done
