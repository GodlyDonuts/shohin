#!/usr/bin/env python3
"""Pure contracts for dual-code reversible scratchpad data construction."""
import random

from dual_code_reversible_protocol import (
    codebook_prompt, encode_state, forward_prompt, invert_microstep, make_codebook,
    parse_state, reverse_prompt, transcode_prompt,
)
from digitwise_protocol import apply_microstep, initial_state


for seed in (3, 17, 101):
    a_book, b_book = make_codebook(seed, "A"), make_codebook(seed, "B")
    assert a_book.aliases != b_book.aliases
    assert "fields" in codebook_prompt(a_book)
    for operation, left, right, width in (("add", 95, 8, 3), ("sub", 1000, 1, 4), ("add", 999, 1, 3)):
        state = initial_state(operation, left, right, width)
        for _ in range(width):
            a_line, b_line = encode_state(state, a_book), encode_state(state, b_book)
            assert parse_state(a_line, a_book) == state
            assert parse_state("<think>trace</think>\n" + b_line, b_book) == state
            assert parse_state(a_line, b_book) is None
            assert parse_state(a_line.replace("dcr:A", "dcr:B", 1), b_book) is None
            assert "dws:" not in forward_prompt(a_book, a_line)
            assert "dcr:B" in transcode_prompt(a_book, b_book, a_line)
            assert "preceding" in reverse_prompt(b_book, b_line)
            next_state = apply_microstep(state)
            assert invert_microstep(next_state) == state
            state = next_state

rng = random.Random(20260713)
for _ in range(120):
    width = rng.randint(1, 8)
    operation = rng.choice(("add", "sub"))
    left, right = rng.randrange(10 ** width), rng.randrange(10 ** width)
    if operation == "sub" and left < right:
        left, right = right, left
    state = initial_state(operation, left, right, width)
    for _ in range(width):
        state = apply_microstep(state)
        assert invert_microstep(state)["p"] == state["p"] - 1

try:
    invert_microstep(initial_state("add", 1, 2, 2))
    raise AssertionError("initial state unexpectedly has a predecessor")
except ValueError:
    pass

print("dual-code reversible protocol checks: passed")
