#!/usr/bin/env python3
"""CPU-only contracts for restricted Jacobian digit-lens data selection."""
import json
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from digitwise_protocol import apply_microstep, canonical_state, initial_state
from probe_restricted_jacobian_digit_lens import (
    local_key,
    pair_directions,
    paired_effect_summary,
    permutation,
    select_readout_records,
    select_eval_pairs,
    split_discovery_and_eval,
)
from probe_digitwise_workspace import transition_examples


def episode(identifier, digit, suffix):
    state = initial_state("add", 10 + digit, 0, 4)
    next_state = apply_microstep(state)
    return {
        "id": "{}-{}".format(identifier, suffix),
        "split": "fit_w4",
        "prompt_style": "core",
        "initial_state": canonical_state(state),
        "expected_states": [canonical_state(next_state)],
    }


with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "episodes.jsonl"
    path.write_text("".join(json.dumps(episode("d{}".format(digit), digit, suffix)) + "\n"
                            for digit in range(10) for suffix in range(3)))
    examples = transition_examples(path, 0)
    records = []
    for example in examples:
        digit = str(example["next_state"]["r"][0])
        records.append({"id": example["id"], "digit": digit, "regime": example["regime"], "example": example})
    discovery, evaluation = split_discovery_and_eval(records, 1)
    assert len(discovery) == 10
    assert {row["id"] for row in discovery}.isdisjoint(row["id"] for row in evaluation)
    assert len(select_readout_records(evaluation, 1)) == 10
    assert len(select_readout_records(evaluation, 0)) == len(evaluation)
    pairs = select_eval_pairs(evaluation, 1)
    assert len(pairs) == 1
    assert local_key(pairs[0]["a"]["example"]) == local_key(pairs[0]["b"]["example"])
    assert pairs[0]["a"]["digit"] != pairs[0]["b"]["digit"]
    directions = pair_directions(pairs[0])
    assert [(side, target["id"], source["id"]) for side, target, source in directions] == [
        ("a", pairs[0]["a"]["id"], pairs[0]["b"]["id"]),
        ("b", pairs[0]["b"]["id"], pairs[0]["a"]["id"]),
    ]

assert permutation("0") == "3"
assert permutation("8") == "1"
assert paired_effect_summary([1.0, 3.0], [0.0, 2.0]) == {
    "mean": 1.0,
    "sample_std": 0.0,
    "sem": 0.0,
    "signal_exceeds_control_count": 2,
}
vectors = torch.eye(2)
assert torch.allclose(torch.linalg.pinv(vectors) @ vectors, torch.eye(2))
print("restricted Jacobian digit-lens selection checks: passed")
