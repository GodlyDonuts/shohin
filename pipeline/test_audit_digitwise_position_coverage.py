#!/usr/bin/env python3
"""Smoke test for DRS coverage accounting."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    train, heldout, generated, report = (root / name for name in ("train.jsonl", "heldout.jsonl", "generated.json", "coverage.json"))
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "generate_digitwise_recurrent_v1.py"),
        "--train-out", str(train), "--heldout-out", str(heldout), "--report", str(generated),
        "--train-episodes", "64", "--heldout-per-regime", "3", "--seed", "59",
    ], check=True, capture_output=True, text=True)
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "audit_digitwise_position_coverage.py"),
        "--data", str(train), "--episodes", str(heldout), "--out", str(report),
    ], check=True, capture_output=True, text=True)
    result = json.loads(report.read_text())
    assert result["audit"] == "digitwise_position_coverage_v1"
    assert result["train_transition_events"] > 0
    assert set(result["heldout"]) == {"fit_w4", "fit_w6", "value_ood_w4", "value_ood_w6", "width_ood_w8"}
    assert result["heldout"]["width_ood_w8"]["unseen_train_digit_position_events"] > 0
print("digitwise position coverage checks: passed")
