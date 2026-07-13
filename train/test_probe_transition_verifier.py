#!/usr/bin/env python3
"""Pure contracts for the raw transition-verifier feasibility probe."""
from digitwise_protocol import apply_microstep, initial_state, parse_state
from probe_transition_verifier import make_cases, mutate_candidate, parse_verdict, verdict_prompt


state = initial_state("add", 95, 8, 4)
expected = apply_microstep(state)
for kind in ("digit", "carry", "operand"):
    candidate = mutate_candidate(state, expected, kind)
    assert candidate != expected
    assert parse_state(verdict_prompt(state, candidate, "core").split("Candidate next state: ", 1)[1].split("\n", 1)[0]) == candidate

rows = make_cases(24, 123)
assert len(rows) == 24
assert sum(row["label"] == "valid" for row in rows) == 12
assert {row["style"] for row in rows} == {"core", "heldout"}
assert {row["near_miss"] for row in rows if row["label"] == "invalid"} == {"digit", "carry", "operand"}
assert parse_verdict("verdict=VALID") == "valid"
assert parse_verdict("no decision") is None
assert all("verdict=valid" in row["prompt"] and "verdict=invalid" in row["prompt"] for row in rows)
print("transition verifier probe checks: passed")
