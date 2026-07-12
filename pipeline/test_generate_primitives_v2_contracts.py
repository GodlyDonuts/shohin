#!/usr/bin/env python3
"""Deterministic integrity checks for the contract-diverse primitive generator."""
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
CONTRACTS = {"qa", "direct", "cot", "review", "scaffold", "compact", "reuse"}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    v1_train, v1_eval = tmp / "v1_train.jsonl", tmp / "v1_eval.jsonl"
    subprocess.run([
        sys.executable, "pipeline/generate_primitives_v1.py",
        "--train-out", str(v1_train), "--eval-out", str(v1_eval),
        "--train-per-family", "8", "--eval-per-family", "4", "--seed", "29",
    ], cwd=ROOT, check=True)
    train, heldout = tmp / "train.jsonl", tmp / "heldout.jsonl"
    subprocess.run([
        sys.executable, "pipeline/generate_primitives_v2_contracts.py",
        "--train-source", str(v1_train), "--eval-source", str(v1_eval),
        "--train-out", str(train), "--eval-out", str(heldout),
        "--train-per-family", "3", "--eval-per-family", "2", "--seed", "31",
    ], cwd=ROOT, check=True)
    train_rows, heldout_rows = rows(train), rows(heldout)
    assert len(train_rows) == 3 * len(FAMILIES) * len(CONTRACTS)
    assert len(heldout_rows) == 2 * len(FAMILIES) * len(CONTRACTS)
    assert set(Counter(row["family"] for row in train_rows)) == FAMILIES
    assert set(Counter(row["contract"] for row in train_rows)) == CONTRACTS
    assert not ({row["source_question"] for row in train_rows} &
                {row["source_question"] for row in heldout_rows})
    assert len({row["completion_prompt"] for row in train_rows}) == len(train_rows)
    for row in train_rows + heldout_rows:
        assert row["question"] == row["completion_prompt"]
        assert row["training_group"] == "contracts"
        assert row["response"].strip()
        if row["contract"] == "compact":
            assert row["response"].startswith("state=")
        if row["contract"] == "reuse":
            assert "previous compact state" in row["completion_prompt"].lower()
print("primitive v2 contract generator checks: passed")
