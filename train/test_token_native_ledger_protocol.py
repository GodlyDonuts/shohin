#!/usr/bin/env python3
"""Pure contracts for the fixed three-token token-native ledger carrier."""
from token_native_ledger_protocol import (
    CODE_TOKENS, canonical_delta, context_key, expected_answer, expected_delta, final_prompt,
    initial_delta, parse_delta, transition_prompt,
)
from digitwise_factor_protocol import initial_tape


assert len(CODE_TOKENS) == 10
initial = initial_delta()
assert parse_delta(canonical_delta(initial)) == initial
assert parse_delta(canonical_delta({"p": 8, "c": 1, "d": 9})) == {"p": 8, "c": 1, "d": 9}
assert parse_delta(" " + canonical_delta(initial)) is None
assert parse_delta(canonical_delta(initial) + CODE_TOKENS[0]) is None

tape = initial_tape("add", 95, 8, 3)
first = expected_delta(tape, 0, 0)
second = expected_delta(tape, 1, first["c"])
assert first == {"p": 1, "c": 1, "d": 3}
assert second == {"p": 2, "c": 1, "d": 0}
assert expected_answer(tape) == 103
assert canonical_delta(initial) in transition_prompt(tape, initial, "core")
rendered = final_prompt([canonical_delta(first), canonical_delta(second)], context_key(tape), "heldout")
assert "Tape:" not in rendered and canonical_delta(first) in rendered
print("token-native ledger protocol checks: passed")
