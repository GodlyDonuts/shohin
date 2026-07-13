#!/usr/bin/env python3
"""Transport-only checks for append-ledger controller."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from generate_append_ledger_v1 import counterfactual, episode_from_operands
from append_ledger_controller import rollout_episode


episode = episode_from_operands("unit", "fit", 5, "add", 95, 8, "core", 2)
episode["counterfactual"] = counterfactual(episode)
responses = list(episode["expected_deltas"][:2]) + [episode["expected_blocks"][0]]
responses += list(episode["expected_deltas"][2:4]) + [episode["expected_blocks"][1]]
responses += list(episode["expected_deltas"][4:]) + [episode["expected_blocks"][2]]
responses += ["answer={}".format(episode["expected_answer"])]
result = rollout_episode(episode, lambda _: responses.pop(0))
assert result["syntactic_closed_loop"] is True
assert result["exact_chain"] is True
assert result["final_correct"] is True

bad = rollout_episode(episode, lambda _: "adl:step=0;d=0;c=0")
assert bad["syntactic_closed_loop"] is False or bad["exact_chain"] is False
print("append-ledger controller checks: passed")
