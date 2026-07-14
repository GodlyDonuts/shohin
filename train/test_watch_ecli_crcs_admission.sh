#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
bash -n "$ROOT/pipeline/watch_ecli_crcs_admission.sh"
printf 'CRCS admission watcher syntax: passed\n'
