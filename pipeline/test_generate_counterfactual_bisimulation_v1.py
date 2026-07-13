#!/usr/bin/env python3
"""Small deterministic contracts for the CBC generator."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "train"))

from generate_counterfactual_bisimulation_v1 import (
    HELDOUT_DOMAINS,
    TRAIN_DOMAINS,
    controller_prompts,
    make_episode,
    rows_for_episode,
)


train = make_episode("train-smoke", "train", TRAIN_DOMAINS[0], 3, __import__("random").Random(7), False)
heldout = make_episode("heldout-smoke", "heldout", HELDOUT_DOMAINS[0], 3, __import__("random").Random(11), True)
assert train["style"] == "train" and heldout["style"] == "heldout"
assert heldout["regime"] == "cbc_len4"
assert heldout["normal"]["query"]["answer"] != heldout["counterfactual"]["query"]["answer"]
assert [step["operation"] for step in heldout["normal"]["steps"]] == [step["operation"] for step in heldout["counterfactual"]["steps"]]
rows = rows_for_episode(train)
assert {row["kind"] for row in rows} == {"compile_a", "compile_b", "update", "inverse_delta", "readout_sum"}
assert all(row["question"] == row["completion_prompt"] and row["response"] for row in rows)
assert all(row["style"] == "train" and row["revision"] >= 0 for row in rows)
assert len(controller_prompts(heldout)) == len(set(controller_prompts(heldout)))
print("counterfactual bisimulation generator checks: passed")
