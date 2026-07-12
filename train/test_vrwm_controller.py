#!/usr/bin/env python3
"""Closed-loop controller tests: it transports states but never repairs them."""
from vrwm_controller import rollout_episode


episode = {
    "initial_memory": {"a": 2, "b": 5},
    "operations": [
        {"kind": "add_var", "target": "a", "source": "b"},
        {"kind": "sub_const", "target": "b", "value": 3},
    ],
}
responses = iter(("wm:a=7;b=5", "wm:a=7;b=2"))
success = rollout_episode(episode, lambda _: next(responses))
assert success["success"] is True
assert success["memory"] == {"a": 7, "b": 2}

failure = rollout_episode(episode, lambda _: "wm:a=999;b=5")
assert failure["success"] is False
assert len(failure["rows"]) == 1
print("vrwm controller checks: passed")
