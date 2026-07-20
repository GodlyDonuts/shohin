from __future__ import annotations

import torch

from assess_sd_cst_complete_physical_fresh import (
    ROWS,
    _hard,
    _output_exact,
    _packet_fields,
    _pointer_exact,
)
from assess_sd_cst_projected_mechanics import packet_arm, semantic_rollout
from sd_cst import STOP_KIND, HardLateQuery, HardProgramTape


def _gold() -> tuple[HardProgramTape, HardLateQuery]:
    kind = torch.zeros((ROWS, 8), dtype=torch.uint8)
    kind[:, 3] = STOP_KIND
    return (
        HardProgramTape(
            torch.zeros(ROWS, dtype=torch.uint8),
            kind,
            torch.zeros((ROWS, 8), dtype=torch.uint8),
            torch.zeros((ROWS, 8), dtype=torch.uint8),
        ),
        HardLateQuery(torch.zeros(ROWS, dtype=torch.uint8)),
    )


def test_independent_packet_and_pointer_recomputation() -> None:
    gold = _gold()
    fields = _packet_fields(gold, gold)
    assert all(bool(value.all()) for value in fields.values())
    predictions = {
        "line": torch.ones((ROWS, 9), dtype=torch.long),
        "binding": torch.ones((ROWS, 3), dtype=torch.long),
        "initial_entity": torch.ones((ROWS, 3), dtype=torch.long),
        "event_entity": torch.ones((ROWS, 8), dtype=torch.long),
    }
    ranges = {
        name: torch.tensor([[[1, 2]] * value.shape[1]] * ROWS, dtype=torch.long)
        for name, value in predictions.items()
    }
    ranges["event_entity"][:, 3] = 0
    exact = _pointer_exact(predictions, ranges, gold[0])
    assert all(bool(value.all()) for value in exact.values())
    predictions["binding"][0, 0] = 2
    assert not bool(_pointer_exact(predictions, ranges, gold[0])["binding"][0])


def test_independent_executor_semantics() -> None:
    tape, query = _gold()
    arm = packet_arm(tape, query)
    parsed_tape, parsed_query = _hard(arm)
    expected = semantic_rollout(parsed_tape, parsed_query)
    output = dict(
        zip(
            ("final_state", "answer", "state_trajectory", "alive_trajectory"),
            expected,
            strict=True,
        )
    )
    assert _output_exact(output, arm)
    output["final_state"] = output["final_state"].clone()
    output["final_state"][0] = 1
    assert not _output_exact(output, arm)
