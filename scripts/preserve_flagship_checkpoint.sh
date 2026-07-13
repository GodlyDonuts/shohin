#!/usr/bin/env bash
# Promote one numbered flagship checkpoint and mirror its full optimizer state locally.
# The local copy is downloaded to a resumable .part path and atomically renamed only
# after its MD5 equals the durable Newton copy.
set -euo pipefail

if [ "$#" -ne 1 ] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 STEP" >&2
  exit 2
fi

STEP=$((10#$1))
printf -v STEP_PAD '%07d' "$STEP"

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_HOST=${REMOTE_HOST:-newton}
REMOTE_BASE=${REMOTE_BASE:-/lustre/fs1/home/sa305415/shohin}
LOCAL_OUT=${LOCAL_OUT:-$REPO_DIR/train/flagship_out}
REMOTE_OUT="$REMOTE_BASE/train/flagship_out"
REMOTE_CHECKPOINT="$REMOTE_OUT/ckpt_${STEP_PAD}.pt"
REMOTE_BEST="$REMOTE_OUT/best_step${STEP}.pt"
LOCAL_CHECKPOINT="$LOCAL_OUT/ckpt_${STEP_PAD}.pt"
LOCAL_PART="$LOCAL_CHECKPOINT.part"

mkdir -p "$LOCAL_OUT"

REMOTE_MD5=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE_HOST" \
  "set -euo pipefail
   if test -s '$REMOTE_CHECKPOINT'; then
     source_path='$REMOTE_CHECKPOINT'
   elif test -s '$REMOTE_BEST'; then
     source_path='$REMOTE_BEST'
   else
     echo 'no numbered or durable checkpoint exists for step $STEP' >&2
     exit 5
   fi
   if ! test -e '$REMOTE_BEST'; then
     cp --preserve=mode,timestamps \"\$source_path\" '$REMOTE_BEST'
   fi
   test \"\$(md5sum \"\$source_path\" | awk '{print \$1}')\" = \"\$(md5sum '$REMOTE_BEST' | awk '{print \$1}')\"
   md5sum '$REMOTE_BEST' | awk '{print \$1}'")

if [ -f "$LOCAL_CHECKPOINT" ]; then
  LOCAL_MD5=$(md5 -q "$LOCAL_CHECKPOINT")
  if [ "$LOCAL_MD5" = "$REMOTE_MD5" ]; then
    printf 'checkpoint %s already verified: %s\n' "$STEP" "$LOCAL_CHECKPOINT"
    exit 0
  fi
  echo "existing local checkpoint hash disagrees with Newton: $LOCAL_CHECKPOINT" >&2
  exit 3
fi

# macOS ships openrsync, which lacks the flags needed for a safe append verify.
# OpenSSH sftp's reget resumes the local .part file without restarting it.
sftp -o BatchMode=yes -o ConnectTimeout=20 -b - "$REMOTE_HOST" <<EOF
reget $REMOTE_BEST $LOCAL_PART
EOF

LOCAL_MD5=$(md5 -q "$LOCAL_PART")
if [ "$LOCAL_MD5" != "$REMOTE_MD5" ]; then
  echo "MD5 mismatch after transfer: local=$LOCAL_MD5 remote=$REMOTE_MD5" >&2
  exit 4
fi

mv "$LOCAL_PART" "$LOCAL_CHECKPOINT"
printf 'checkpoint %s promoted and verified: %s\n' "$STEP" "$LOCAL_CHECKPOINT"
