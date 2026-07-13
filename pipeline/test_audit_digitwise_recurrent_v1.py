#!/usr/bin/env python3
"""Small audit contract for digitwise recurrent data."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    train, heldout, generated = directory / "train.jsonl", directory / "heldout.jsonl", directory / "generated.json"
    audit = directory / "audit.json"
    subprocess.run([
        sys.executable, "pipeline/generate_digitwise_recurrent_v1.py",
        "--train-out", str(train), "--heldout-out", str(heldout), "--report", str(generated),
        "--train-episodes", "30", "--heldout-per-regime", "4", "--seed", "59",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, "pipeline/audit_digitwise_recurrent_v1.py",
        "--data", str(train), "--episodes", str(heldout), "--out", str(audit),
    ], cwd=ROOT, check=True)
    payload = json.loads(audit.read_text())
    assert payload["invalid_rows_or_episodes"] == 0
    assert payload["duplicate_normalized_train_questions"] == 0
    assert payload["counterfactual_pairs"] == 20
    assert payload["overlap"] == {"exact_prompt_hits": 0, "ngram13_hits": 0}
print("digitwise recurrent audit checks: passed")
