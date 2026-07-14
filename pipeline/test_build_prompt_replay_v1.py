#!/usr/bin/env python3
"""CPU contracts for prompt-only SFT replay selection."""
from __future__ import annotations

from build_prompt_replay_v1 import normalized, prompt_for, select_round_robin


assert prompt_for({"question": "What is 2 + 2?"}) == "Question: What is 2 + 2?\nAnswer:"
assert prompt_for({"completion_prompt": "def f(x):\n"}) == "def f(x):\n"
assert normalized(" A\n B ") == "a b"
rows = [
    {"source": "math", "prompt": "a"},
    {"source": "math", "prompt": "b"},
    {"source": "code", "prompt": "c"},
    {"source": "logic", "prompt": "d"},
]
selected = select_round_robin(rows, 4, 7)
assert {row["source"] for row in selected[:3]} == {"math", "code", "logic"}
assert len(select_round_robin(rows, 2, 7)) == 2
print("Prompt replay builder checks: passed")
