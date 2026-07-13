#!/usr/bin/env python3
"""Static isolation contracts for CPU-only token-native ledger jobs."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "pipeline" / "jobs" / "generate_token_native_ledger_v1_stokes.sbatch"
AUDIT = ROOT / "pipeline" / "jobs" / "audit_token_native_ledger_v1_stokes.sbatch"


build, audit = BUILD.read_text(), AUDIT.read_text()
assert "--gres" not in build and "--gres" not in audit
assert "token_native_ledger_v1_train.jsonl" in build
assert "refusing existing artifact" in build
assert "audit_token_native_ledger_v1.py" in audit
assert "refusing existing output" in audit
assert "train_heldout_13gram_hits" in audit
print("token-native ledger job contracts: passed")
