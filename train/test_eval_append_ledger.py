#!/usr/bin/env python3
"""Pure paired-rollout contract for append-ledger evaluation."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from generate_append_ledger_v1 import counterfactual, episode_from_operands
from eval_append_ledger import evaluate_pair, row_counts


def responses_for(episode):
    responses, block_index = [], 0
    for index, delta in enumerate(episode["expected_deltas"], 1):
        responses.append(delta)
        if index % episode["block_size"] == 0 or index == len(episode["expected_deltas"]):
            responses.append(episode["expected_blocks"][block_index])
            block_index += 1
    responses.append("answer={}".format(episode["expected_answer"]))
    return responses


episode = episode_from_operands("pair", "fit_w8", 8, "add", 95, 8, "heldout", 4)
episode["counterfactual"] = counterfactual(episode)
responses = responses_for(episode) + responses_for(episode["counterfactual"])
prompts = []
result = evaluate_pair(episode, lambda prompt: (prompts.append(prompt), responses.pop(0))[1], prompt_style="core")
assert result["normal"]["final_correct"] is True
assert result["counterfactual"]["final_correct"] is True
assert result["intervention_success"] is True
assert "Append one local decimal delta." in prompts[0]
counts = row_counts(result["normal"]["rows"])
assert counts["first_delta_correct"] == 1
assert counts["delta_correct"] == 8 and counts["block_correct"] == 2
print("append-ledger evaluator checks: passed")
