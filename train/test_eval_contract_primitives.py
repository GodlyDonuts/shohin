#!/usr/bin/env python3
"""Unit checks for exact-answer extraction and balanced contract selection."""
import json
import tempfile
from pathlib import Path

from eval_contract_primitives import predict, read_rows


assert predict("The answer is 42.", "arithmetic") == "42"
assert predict("state=x\nThe answer is 17.", "state_update") == "17"
assert predict("The answer is [2, 4, 9].", "sort_unique") == "[2,4,9]"
assert predict("The answer is moPQsaic.", "string_insert") == "mopqsaic"
assert predict("Reasoning says no. The answer is no.", "syllogism") == "no"
with tempfile.TemporaryDirectory() as root:
    path = Path(root) / "rows.jsonl"
    rows = [
        {"completion_prompt": "a", "answer": "1", "contract": "direct", "family": "arithmetic"},
        {"completion_prompt": "b", "answer": "2", "contract": "direct", "family": "arithmetic"},
        {"completion_prompt": "c", "answer": "yes", "contract": "review", "family": "syllogism"},
        {"completion_prompt": "d", "answer": "no", "contract": "review", "family": "syllogism"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected = read_rows(path, 0, 1)
    assert len(selected) == 2
    assert {(row["contract"], row["family"]) for row in selected} == {
        ("direct", "arithmetic"), ("review", "syllogism"),
    }
    primitive_path = Path(root) / "primitive.jsonl"
    primitive_path.write_text(json.dumps({"question": "What is 1 plus 2?", "answer": "3", "family": "arithmetic"}) + "\n")
    legacy = read_rows(primitive_path, 0, 0)
    assert legacy[0]["contract"] == "answer"
    assert legacy[0]["completion_prompt"] == "Question: What is 1 plus 2?\nAnswer:"
print("contract primitive evaluator checks: passed")
