#!/usr/bin/env python3
"""Integrity checks for the typed compact-state curriculum generator."""
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "arithmetic", "base_conversion", "state_update", "sort_unique",
    "string_insert", "syllogism", "correction",
}
CONTRACTS = {"write", "repair", "reuse"}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    train, heldout = directory / "train.jsonl", directory / "heldout.jsonl"
    subprocess.run([
        sys.executable, "pipeline/generate_primitives_v3_states.py",
        "--train-out", str(train), "--eval-out", str(heldout),
        "--train-per-family", "4", "--eval-per-family", "2", "--seed", "37",
    ], cwd=ROOT, check=True)
    train_rows, heldout_rows = read_rows(train), read_rows(heldout)
    assert len(train_rows) == 4 * len(FAMILIES) * len(CONTRACTS)
    assert len(heldout_rows) == 2 * len(FAMILIES) * len(CONTRACTS)
    assert set(Counter(row["family"] for row in train_rows)) == FAMILIES
    assert set(Counter(row["contract"] for row in train_rows)) == CONTRACTS
    assert not ({row["source_question"] for row in train_rows} &
                {row["source_question"] for row in heldout_rows})
    for row in train_rows + heldout_rows:
        assert row["training_group"] == "state_protocol"
        assert row["expected_state"].startswith("state=")
        assert row["question"] == row["completion_prompt"]
        if row["contract"] in {"write", "repair"}:
            assert row["response"].startswith(row["expected_state"] + "\n")
        else:
            assert row["response"] == f"The answer is {row['answer']}."
print("primitive v3 typed-state generator checks: passed")
