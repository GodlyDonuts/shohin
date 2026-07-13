#!/usr/bin/env python3
"""Small end-to-end data and audit smoke for append-ledger v1."""
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    data, held, report, audit = (root / "train.jsonl", root / "held.jsonl", root / "report.json", root / "audit.json")
    generator = ROOT / "pipeline" / "generate_append_ledger_v1.py"
    auditor = ROOT / "pipeline" / "audit_append_ledger_v1.py"
    subprocess.run([sys.executable, str(generator), "--train-out", str(data), "--heldout-out", str(held),
                    "--report", str(report), "--train-episodes", "40", "--heldout-per-regime", "4"], check=True)
    subprocess.run([sys.executable, str(auditor), "--data", str(data), "--episodes", str(held), "--out", str(audit)], check=True)
print("append-ledger generator/audit checks: passed")
