#!/usr/bin/env python3
"""Focused checks for the counterfactual operator-binding curriculum."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from generate_operator_trace_contrast_v1 import (
    TRAIN_TEMPLATES,
    build_eval,
    build_train,
    normalize,
)


train = build_train(per_family=8, seed=7)
assert len(train) == len(TRAIN_TEMPLATES) * 8 * 2 + 2 * 8
assert {row["contract"] for row in train} == {"direct", "minimal_pair"}
assert all(row["response"].startswith("<think>plan=") for row in train if row["contract"] == "direct")
assert all(len(row["operations"]) == 3 and len(row["states"]) == 4 for row in train)
for row in train:
    if row["contract"] == "minimal_pair":
        differences = sum(a != b for a, b in zip(row["operations"], row["alternate_operations"]))
        assert differences == 1

train_prompts = {normalize(row["completion_prompt"]) for row in train}
cases = build_eval(per_regime_family=4, seed=8, train_prompts=train_prompts)
assert len(cases) == len(TRAIN_TEMPLATES) * 3 * 4
assert {case["regime"] for case in cases} == {"wording", "value", "full"}
assert all(len(case["markers"]) == 3 for case in cases)
assert not train_prompts.intersection(normalize(case["question"]) for case in cases)

print("operator-trace contrast generator checks: passed")
