#!/usr/bin/env python3
"""Pure paired-rollout contract for the digitwise evaluator."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from generate_digitwise_recurrent_v1 import counterfactual_episode, episode_from_operands
from eval_digitwise_recurrent import evaluate_pair


episode = episode_from_operands("pair", "fit_w4", 4, "add", 95, 8, "heldout")
episode["counterfactual"] = counterfactual_episode(episode)
responses = list(episode["expected_states"]) + ["answer={}".format(episode["expected_answer"])]
responses += list(episode["counterfactual"]["expected_states"]) + ["answer={}".format(episode["counterfactual"]["expected_answer"])]
result = evaluate_pair(episode, lambda _: responses.pop(0))
assert result["normal"]["success"] is True
assert result["counterfactual"]["success"] is True
assert result["intervention_success"] is True
print("digitwise evaluator checks: passed")
