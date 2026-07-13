#!/usr/bin/env python3
"""CPU-only contracts for residual-patching workspace-probe construction."""
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from digitwise_protocol import apply_microstep, initial_state, microstep_prompt, canonical_state
from probe_digitwise_workspace import field_prefix, parse_layers, select_pairs, target_digit, transition_examples


def episode(identifier, left, right):
    state = initial_state("add", left, right, 4)
    next_state = apply_microstep(state)
    return {
        "id": identifier,
        "split": "fit_w4",
        "prompt_style": "core",
        "initial_state": canonical_state(state),
        "expected_states": [canonical_state(next_state)] + [canonical_state(apply_microstep(next_state))] * 0,
    }


first = initial_state("add", 19, 8, 4)
next_first = apply_microstep(first)
prompt = microstep_prompt(first)
carry_prefix, carry = field_prefix(prompt, next_first, "carry")
digit_prefix, digit = field_prefix(prompt, next_first, "digit")
assert carry == str(next_first["c"])
assert digit == next_first["r"][0]
assert carry_prefix.endswith(";c=")
assert digit_prefix.endswith(";r=")
assert parse_layers("5, 9, 29, 9", 30) == [5, 9, 29]
try:
    parse_layers("30", 30)
except ValueError:
    pass
else:
    raise AssertionError("out-of-range layer was accepted")

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "episodes.jsonl"
    rows = [episode("a", 19, 8), episode("b", 18, 8), episode("c", 17, 8)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    examples = transition_examples(path, 0)
    pairs = select_pairs(examples, "digit", 1)
    assert len(pairs) == 1
    left, right = pairs[0]["a"], pairs[0]["b"]
    assert target_digit(left, "digit") != target_digit(right, "digit")
    assert left["state"]["p"] == right["state"]["p"] == 0

print("digitwise workspace probe construction checks: passed")
