#!/usr/bin/env python3
"""Prove the token-native ledger controller only transports model emissions."""
from digitwise_factor_protocol import canonical_tape, initial_tape
from token_native_ledger_controller import rollout_episode
from token_native_ledger_protocol import canonical_delta, expected_answer, expected_delta, initial_delta


def episode():
    tape = initial_tape("add", 95, 8, 3)
    carry, lines = 0, []
    for position in range(3):
        delta = expected_delta(tape, position, carry)
        lines.append(canonical_delta(delta))
        carry = delta["c"]
    return {
        "tape": canonical_tape(tape),
        "initial_delta": canonical_delta(initial_delta()),
        "expected_deltas": lines,
        "expected_answer": expected_answer(tape),
        "prompt_style": "core",
    }


item = episode()
responses = list(item["expected_deltas"]) + ["answer={}".format(item["expected_answer"])]
result = rollout_episode(item, lambda _prompt, _limit: responses.pop(0))
assert result["success"] and result["strict_closed_loop"] and result["syntactic_closed_loop"]
assert len(result["emitted"]) == 3

wrong = canonical_delta({"p": 1, "c": 0, "d": 3})
responses = [wrong] + item["expected_deltas"][1:] + ["answer={}".format(item["expected_answer"])]
prompts = []
result = rollout_episode(item, lambda prompt, _limit: (prompts.append(prompt), responses.pop(0))[1])
assert not result["success"] and result["syntactic_closed_loop"] and not result["strict_closed_loop"]
assert wrong in prompts[1]
print("token-native ledger controller checks: passed")
