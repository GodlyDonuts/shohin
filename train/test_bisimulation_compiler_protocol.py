"""Deterministic contracts for CBC state, delta, and source-free prompts."""
from bisimulation_compiler_protocol import (
    apply_operation, canonical_delta, canonical_state, compile_prompt, delta_prompt,
    parse_delta, parse_state, render_event, sum_query_prompt, update_prompt,
)


keys = ("amber", "brass")
initial = {"amber": 17, "brass": 23}
state = canonical_state(initial, keys)
assert state == "cbc:amber=17;brass=23"
assert parse_state("<think>ignored</think>\n" + state, keys) == initial
assert parse_state("cbc:brass=23;amber=17", keys) is None
assert parse_state("cbc:amber=17;brass=23\ncbc:amber=18;brass=23", keys) is None

operations = (
    {"kind": "add", "key": "amber", "value": 5},
    {"kind": "sub", "key": "brass", "value": 3},
    {"kind": "move", "source": "amber", "target": "brass", "value": 4},
    {"kind": "swap", "left": "amber", "right": "brass"},
)
for operation in operations:
    after = apply_operation(initial, operation, keys)
    delta = canonical_delta(operation, keys)
    assert parse_delta(delta, keys) == operation
    assert "cbc-delta" in delta_prompt(state, canonical_state(after, keys), "T-1", 1, "train")
    assert render_event(operation, "tokens", "heldout")

assert "cbc:" in compile_prompt(initial, keys, "workshop", "parts", "T-1", "train", "a")
assert "source discarded" in update_prompt(state, "Add 5 parts to amber.", "T-1", 1, "heldout")
assert "answer=<integer>" in sum_query_prompt(state, keys, "T-1", "train")
print("bisimulation compiler protocol checks: passed")
