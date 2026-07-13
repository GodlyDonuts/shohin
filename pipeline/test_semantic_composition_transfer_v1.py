#!/usr/bin/env python3
"""Focused contracts for semantic composition-transfer build and audit."""
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from audit_semantic_composition_transfer_v1 import audit
from generate_semantic_composition_transfer_v1 import FAMILIES, build, write_jsonl


rows = build(4, 17)
assert len(rows) == 4 * len(FAMILIES)
assert {row["family"] for row in rows} == set(FAMILIES)
assert all(row["response"].startswith("<think>") for row in rows)
assert all("The answer is " + row["answer"] + "." in row["response"] for row in rows)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    suite = root / "suite.jsonl"
    train = root / "train.jsonl"
    write_jsonl(suite, rows)
    train.write_text("".join(json.dumps({
        "question": "Unrelated training question {}".format(index),
        "response": "<think>unrelated</think>\nThe answer is 0.",
        "answer": "0",
        "family": "unrelated",
        "source": "test",
    }) + "\n" for index in range(12)))
    result = audit(suite, train)
    assert result["admitted"]
    assert result["rows"] == len(rows)
    assert result["exact_question_hits_against_training"] == 0

print("semantic composition-transfer generation and audit checks: passed")
