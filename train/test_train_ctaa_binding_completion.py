from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pipeline.build_ctaa_binding_completion_board import orbit_records
from pipeline.ctaa_binding_identification import permutation_parity
from pipeline.ctaa_board_v2 import build_compiler_families
from pipeline.ctaa_name_pool import build_name_pools
from train_ctaa_binding_completion import (
    build_two_slot_chimeras,
    evaluate_readout,
    fixed_schedule,
    tensor_sha256,
    validate_orbit_pair,
)


SEED = 881_729
TOKENIZER = Path(__file__).resolve().parents[1] / "artifacts/tokenizer/tokenizer.json"


def paired_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pools = build_name_pools(TOKENIZER, per_split=16)
    family = build_compiler_families(SEED, per_depth=1)[0]
    orbit = orbit_records(SEED, family, pools, renderer_index=0)
    train = [
        row
        for row in orbit
        if permutation_parity(row["opcode_to_card"]) == 0
    ]
    confirmation = [
        row
        for row in orbit
        if permutation_parity(row["opcode_to_card"]) == 1
    ]
    return train, confirmation


def test_orbit_pair_validation_rejects_semantic_drift() -> None:
    train, confirmation = paired_records()
    audit = validate_orbit_pair(train, confirmation)
    assert audit == {
        "families": 1,
        "rows_per_family_per_partition": 12,
        "combined_bindings_per_family": 24,
        "program_source_overlap": 0,
    }
    changed = [dict(row) for row in confirmation]
    changed[0]["query_position"] = 2
    with pytest.raises(ValueError, match="query_position"):
        validate_orbit_pair(train, changed)


def test_fixed_minibatch_schedule_and_tensor_hash_are_deterministic() -> None:
    first = fixed_schedule(100, 7, 5, 13)
    second = fixed_schedule(100, 7, 5, 13)
    third = fixed_schedule(100, 7, 5, 14)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert tensor_sha256(tensor) == tensor_sha256(tensor.clone())
    assert tensor_sha256(tensor) != tensor_sha256(tensor + 1)


class PerfectStructuredReadout(nn.Module):
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return slots[:, :4, :4] * 20.0


def test_evaluation_reports_raw_projection_and_per_binding_metrics() -> None:
    bindings = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
        ]
    )
    slots = torch.zeros(3, 8, 384)
    for row, binding in enumerate(bindings):
        for opcode, card in enumerate(binding):
            slots[row, opcode, card] = 1.0
    metrics = evaluate_readout(
        PerfectStructuredReadout(),
        slots,
        bindings,
        arm="factorized",
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert metrics["projected_binding_exact"] == 1.0
    assert metrics["raw_binding_exact"] == 1.0
    assert metrics["raw_assignment_valid"] == 1.0
    assert metrics["projection_rescue"] == 0.0
    assert sum(value["rows"] for value in metrics["per_binding"].values()) == 3


def test_two_slot_chimera_uses_only_independent_even_donor_rows() -> None:
    train_raw, _ = paired_records()
    train_bindings = torch.tensor(
        [row["opcode_to_card"] for row in train_raw]
    )
    train_slots = torch.zeros(len(train_raw), 8, 384)
    for row, binding in enumerate(train_bindings):
        for opcode, card in enumerate(binding):
            train_slots[row, opcode, card] = 1.0
    chimeras, targets = build_two_slot_chimeras(
        train_slots,
        train_bindings,
        train_raw,
        limit=12,
    )
    predicted = chimeras[:, :4, :4].argmax(-1)
    assert torch.equal(predicted, targets)
    assert all(permutation_parity(row.tolist()) == 1 for row in targets)
