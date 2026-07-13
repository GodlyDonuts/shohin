#!/usr/bin/env python3
"""Controller checks: transport only, never hidden arithmetic or repair."""
from digitwise_controller import rollout_episode
from digitwise_protocol import apply_microstep, canonical_state, initial_state


state = initial_state("add", 95, 8, 3)
expected = []
for _ in range(3):
    state = apply_microstep(state)
    expected.append(canonical_state(state))
episode = {"initial_state": canonical_state(initial_state("add", 95, 8, 3)), "expected_states": expected}
responses = iter(expected + ["answer=103"])
result = rollout_episode(episode, lambda _: next(responses))
assert result["success"] is True and result["final_correct"] is True
failed = rollout_episode(episode, lambda _: "dws:op=add;w=3;p=1;c=0;a=590;b=800;r=300;z=0")
assert failed["success"] is False and len(failed["rows"]) == 1
print("digitwise controller checks: passed")
