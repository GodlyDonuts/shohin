#!/usr/bin/env python3
"""Pure contracts for the digitwise recurrent scratchpad protocol."""
from digitwise_protocol import (apply_microstep, canonical_state, digit_prompt, final_prompt,
                                initial_state, microstep_prompt, parse_answer, parse_digit,
                                parse_state, state_answer, state_digit)


add = initial_state("add", 95, 8, 3)
assert canonical_state(add) == "dws:op=add;w=3;p=0;c=0;a=590;b=800;r=000;z=0"
add = apply_microstep(add)
assert add["r"] == "300" and add["c"] == 1 and add["p"] == 1
add = apply_microstep(add)
assert add["r"] == "300" and add["c"] == 1 and add["p"] == 2
add = apply_microstep(add)
assert add["z"] == 1 and state_answer(add) == 103
assert state_digit(add, 0) == 3
assert parse_state("<think>x</think>\ndws:op=add;w=3;p=3;c=0;a=590;b=800;r=301;z=1") == add
assert parse_state("dws:op=add;w=3;p=3;c=0;a=590;b=800;r=301;z=1\ndws:op=add;w=3;p=3;c=0;a=590;b=800;r=301;z=1") is None
assert parse_answer("answer=103") == 103
assert parse_digit("digit=7") == 7
assert "State:" in microstep_prompt(initial_state("add", 1, 2, 2))
assert "Machine record:" in microstep_prompt(initial_state("add", 1, 2, 2), style="heldout")
assert "Position:" in digit_prompt(apply_microstep(initial_state("add", 1, 2, 2)), 0)
assert "answer=<integer>" in final_prompt(add)

sub = initial_state("sub", 1000, 1, 4)
for _ in range(4):
    sub = apply_microstep(sub)
assert sub["z"] == 1 and sub["c"] == 0 and state_answer(sub) == 999
print("digitwise protocol checks: passed")
