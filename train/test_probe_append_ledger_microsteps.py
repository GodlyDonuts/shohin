#!/usr/bin/env python3
"""Static contract checks for the ADL microstep likelihood diagnostic."""
from append_ledger_protocol import expected_delta, initial_base
from probe_append_ledger_microsteps import CASES, candidate_lines


lines = candidate_lines()
assert len(lines) == 20 and len(set(lines)) == 20
for _, operation, width, left, right in CASES:
    base = initial_base(operation, left, right, width)
    expected = expected_delta(base, 0, 0)
    assert 0 <= expected["d"] <= 9 and expected["c"] in (0, 1)
print("append-ledger microstep probe checks: passed")
