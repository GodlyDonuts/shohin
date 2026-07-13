#!/usr/bin/env python3
"""Pure transport contracts for token-native-ledger evaluation."""
from digitwise_factor_protocol import initial_tape
from eval_token_native_ledger import evaluate_pair, failure_mode
from generate_token_native_ledger_v1 import counterfactual_episode, episode_from_operands
from token_native_ledger_protocol import (
    context_key, final_prompt, initial_delta, parse_delta, transition_prompt,
)


episode = episode_from_operands("test", "recombine_w4", 4, "add", 901, 88, "core")
episode["counterfactual"] = counterfactual_episode(episode)
responses = {}
for item in (episode, episode["counterfactual"]):
    tape = initial_tape(item["operation"], item["left"], item["right"], item["width"])
    prior = initial_delta()
    for line in item["expected_deltas"]:
        responses[transition_prompt(tape, prior, style=item["prompt_style"])] = line
        prior = parse_delta(line)
    responses[final_prompt(item["expected_deltas"], context_key(tape), style=item["prompt_style"])] = (
        "answer={}".format(item["expected_answer"])
    )

calls = []
pair = evaluate_pair(episode, lambda prompt, limit: (calls.append((prompt, limit)), responses[prompt])[1])
assert pair["normal"]["success"] and pair["counterfactual"]["success"]
assert pair["intervention_success"]
assert [limit for _, limit in calls].count(3) == 8
assert [limit for _, limit in calls].count(48) == 2
assert failure_mode(pair["normal"]) == "success"
assert failure_mode({"syntactic_closed_loop": False, "rows": [{"index": 2}], "strict_closed_loop": False, "final_correct": False}) == "malformed_transition_2"
assert failure_mode({"syntactic_closed_loop": True, "rows": [{"index": 0, "correct": False}], "strict_closed_loop": False, "final_correct": False}) == "wrong_transition_0"
print("token-native ledger evaluator checks: passed")
