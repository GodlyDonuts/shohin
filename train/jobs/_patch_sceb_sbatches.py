#!/usr/bin/env python3
from pathlib import Path
import re

EXCLUDE = "#SBATCH --exclude=evc23,evc25,evc26,evc27,evc31,evc36,evc43,evc44,evc45,evc49,evc50"
CUDA = r"""for attempt in 1 2 3 4 5 6 7 8; do
  echo "[job] cuda attempt $attempt on $(hostname)"
  if timeout 45 "$PY" -u - <<'CUDAPY'
import torch, sys
assert torch.cuda.is_available()
torch.empty(1, device="cuda", dtype=torch.bfloat16)
print(torch.cuda.get_device_name(0), flush=True)
CUDAPY
  then echo cuda_ok; break; fi
  sleep 10
  [ "$attempt" -eq 8 ] && exit 12
done"""

for rel in [
    "train/jobs/run_sceb_host_exec.sbatch",
    "train/jobs/run_srr_train.sbatch",
    "train/jobs/run_ssc_halt_first_live.sbatch",
    "train/jobs/run_discrete_controller_heads.sbatch",
]:
    p = Path(rel)
    t = p.read_text()
    t = re.sub(r"#SBATCH --exclude=.*", EXCLUDE, t, count=1)
    if "[job] start" not in t:
        t = t.replace(
            "set -euo pipefail\n",
            'set -euo pipefail\necho "[job] start $(date) host=$(hostname)"\n',
            1,
        )
    t2, n = re.subn(
        r"for attempt in 1 2 3 4 5 6 7 8; do.*?\[ \"\$attempt\" -eq 8 \] && exit 12\ndone",
        CUDA,
        t,
        count=1,
        flags=re.S,
    )
    print(rel, "cuda_replacements", n)
    if n == 1:
        t = t2
    p.write_text(t)
