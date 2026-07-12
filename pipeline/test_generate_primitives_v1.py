#!/usr/bin/env python3
"""Small deterministic checks for the primitive-reasoning generator."""
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def eval_rg_style_answer(text):
    answer = re.findall(r"(?:the )?answer is\s*([^\n<]+)", text, flags=re.I)
    value = answer[-1] if answer else text.split("\n", 1)[0]
    value = re.sub(r"\s+", " ", value.strip().lower())
    return value.rstrip(". ")


with tempfile.TemporaryDirectory() as root:
    root = Path(root)
    train, heldout = root / "train.jsonl", root / "heldout.jsonl"
    subprocess.run([
        sys.executable, "pipeline/generate_primitives_v1.py",
        "--train-out", str(train), "--eval-out", str(heldout),
        "--train-per-family", "7", "--eval-per-family", "3", "--seed", "17",
    ], check=True)
    train_rows, eval_rows = read(train), read(heldout)
    assert len(train_rows) == 49
    assert len(eval_rows) == 21
    assert not ({row["question"] for row in train_rows} & {row["question"] for row in eval_rows})
    expected = {"arithmetic", "base_conversion", "state_update", "sort_unique", "string_insert", "syllogism", "correction"}
    assert set(Counter(row["family"] for row in train_rows)) == expected
    for row in train_rows + eval_rows:
        assert row["training_group"] == "primitives"
        assert row["response"].startswith("<think>")
        assert "The answer is" in row["response"]
        assert str(row["answer"]).strip()
        assert eval_rg_style_answer(row["response"]) == eval_rg_style_answer(
            f"The answer is {row['answer']}."
        )
print("primitive generator checks: passed")
