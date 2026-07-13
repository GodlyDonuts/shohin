#!/usr/bin/env python3
"""End-to-end small admission smoke for DCRD data and its independent audit."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    train, heldout, generated, audited = (root / name for name in ("train.jsonl", "heldout.jsonl", "generated.json", "audit.json"))
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "generate_dual_code_reversible_v1.py"),
        "--train-out", str(train), "--heldout-out", str(heldout), "--report", str(generated),
        "--train-episodes", "12", "--heldout-per-regime", "3",
    ], check=True, capture_output=True, text=True)
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "audit_dual_code_reversible_v1.py"),
        "--train", str(train), "--heldout", str(heldout), "--report", str(audited),
    ], check=True, capture_output=True, text=True)
    report = json.loads(audited.read_text())
    assert report["invalid_train_rows"] == report["invalid_heldout_episodes"] == 0
    assert report["train_heldout_exact_prompt_hits"] == report["train_heldout_13gram_hits"] == 0
    assert set(report["train_kinds"]) == {"a_to_b", "b_to_a", "forward_a", "readout", "reverse_b"}
print("dual-code audit checks: passed")
