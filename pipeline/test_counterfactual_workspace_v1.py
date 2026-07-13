#!/usr/bin/env python3
"""Unit contracts for CWI serialization and independent semantic auditing."""

from __future__ import annotations

import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "train"))

from audit_counterfactual_workspace_v1 import (  # noqa: E402
    audit_row,
    heldout_pair_failures,
    required_contexts,
)
from build_counterfactual_workspace_v1 import rows_from_transition  # noqa: E402
from digitwise_factor_protocol import initial_register, initial_tape, local_context  # noqa: E402


base_tape = initial_tape("add", 123, 456, 4)
base_register = initial_register(base_tape)
train_rows = rows_from_transition(
    base_tape,
    base_register,
    split="train",
    episode_id="unit-train",
    transition_index=0,
    prompt_style="core",
)
assert {row["foil_kind"] for row in train_rows} == {"legal", "carry", "result_digit", "program_counter", "tape"}
for row in train_rows:
    record = audit_row(row, expected_partition="train")
    if row["foil_kind"] == "legal":
        assert local_context(record["fixed_tape"], record["previous_register"]) in required_contexts()

corrupted = copy.deepcopy(train_rows[0])
corrupted["response"] = "verdict=legal;field=none;at_p=9;expected=next;observed=next"
try:
    audit_row(corrupted, expected_partition="train")
except ValueError:
    pass
else:
    raise AssertionError("auditor accepted a corrupt semantic response")

counterfactual_tape = initial_tape("add", 124, 456, 4)
heldout_rows = rows_from_transition(
    base_tape,
    base_register,
    split="recombine_w4",
    episode_id="unit-heldout",
    transition_index=0,
    prompt_style="heldout",
    world="base",
) + rows_from_transition(
    counterfactual_tape,
    initial_register(counterfactual_tape),
    split="recombine_w4",
    episode_id="unit-heldout",
    transition_index=0,
    prompt_style="heldout",
    world="counterfactual",
)
audited_heldout = [(row, audit_row(row, expected_partition="heldout")) for row in heldout_rows]
pair_count, failures = heldout_pair_failures(audited_heldout)
assert pair_count == 5
assert failures == 0

print("counterfactual workspace v1 unit checks: passed")
