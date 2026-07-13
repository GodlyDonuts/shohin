#!/usr/bin/env python3
"""Admission and coverage-corruption checks for the DRS v3 basis candidate."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from digitwise_basis_protocol import local_context
from digitwise_protocol import parse_state


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    train, heldout, generated, audited = (root / name for name in ("train.jsonl", "heldout.jsonl", "generated.json", "audit.json"))
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "generate_digitwise_basis_v3.py"),
        "--train-out", str(train), "--heldout-out", str(heldout), "--report", str(generated),
        "--variants", "1", "--heldout-per-regime", "3", "--seed", "73",
    ], check=True, capture_output=True, text=True)
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "audit_digitwise_basis_v3.py"),
        "--data", str(train), "--episodes", str(heldout), "--out", str(audited),
    ], check=True, capture_output=True, text=True)
    report = json.loads(audited.read_text())
    assert report["invalid_train_rows"] == report["invalid_heldout_episodes"] == 0
    assert report["missing_local_contexts"] == 0
    assert report["train_heldout_exact_prompt_hits"] == report["train_heldout_13gram_hits"] == 0

    rows = [json.loads(line) for line in train.read_text().splitlines()]
    target = local_context(parse_state(next(row["state"] for row in rows if row["kind"] == "transition")))
    reduced = [
        row for row in rows
        if row["kind"] != "transition" or local_context(parse_state(row["state"])) != target
    ]
    missing_train = root / "missing-context.jsonl"
    missing_train.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in reduced))
    missing_report = root / "missing-context-audit.json"
    process = subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "audit_digitwise_basis_v3.py"),
        "--data", str(missing_train), "--episodes", str(heldout), "--out", str(missing_report),
    ], capture_output=True, text=True)
    assert process.returncode != 0
    assert json.loads(missing_report.read_text())["missing_local_contexts"] >= 1
print("digitwise basis v3 audit checks: passed")
