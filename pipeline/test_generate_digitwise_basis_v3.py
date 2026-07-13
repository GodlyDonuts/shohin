#!/usr/bin/env python3
"""End-to-end construction checks for the coverage-complete DRS v3 candidate."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    train, heldout, report = root / "train.jsonl", root / "heldout.jsonl", root / "report.json"
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "generate_digitwise_basis_v3.py"),
        "--train-out", str(train), "--heldout-out", str(heldout), "--report", str(report),
        "--variants", "1", "--heldout-per-regime", "3", "--seed", "71",
    ], check=True, capture_output=True, text=True)
    summary = json.loads(report.read_text())
    assert summary["required_local_contexts"] == summary["covered_local_contexts"] == 3400
    assert summary["missing_local_contexts"] == 0
    assert summary["heldout_by_regime"] == {"recombine_w4": 3, "recombine_w6": 3, "width_ood_w8": 3}
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    assert {row["kind"] for row in rows} == {"transition", "digit", "final"}
    assert all(row["question"] == row["completion_prompt"] and row["response"] for row in rows)
print("digitwise basis v3 generator checks: passed")
