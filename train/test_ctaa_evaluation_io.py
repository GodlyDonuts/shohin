from __future__ import annotations

import json

import pytest
import torch

from ctaa_evaluation_io import (
    PACKET_INDEX_SCHEMA,
    PROGRAM_PREDICTION_SCHEMA,
    QUERY_PREDICTION_SCHEMA,
    packet_valid_mask,
    read_packet_index,
    read_program_predictions,
    read_query_predictions,
    validate_program_predictions,
    write_json_once,
    write_torch_once,
)


def _program_payload() -> dict[str, object]:
    opcode_schedule = torch.zeros((2, 41), dtype=torch.uint8)
    opcode_schedule[0, 3] = 4
    opcode_schedule[1, 2] = 4
    opcode_schedule[1, 7] = 4
    binding = torch.arange(4, dtype=torch.uint8)[None].expand(2, -1).clone()
    return {
        "schema": PROGRAM_PREDICTION_SCHEMA,
        "family_ids": ["a", "b"],
        "program_source_sha256": "a" * 64,
        "compiler_sha256": "b" * 64,
        "action_cards": torch.zeros((2, 4, 3), dtype=torch.uint8),
        "opcode_to_card": binding,
        "initial_state": torch.zeros((2, 3), dtype=torch.uint8),
        "opcode_schedule": opcode_schedule,
        "schedule": opcode_schedule.clone(),
        "packet_valid": packet_valid_mask(binding, opcode_schedule),
    }


def test_prediction_artifacts_round_trip_and_are_read_only(tmp_path) -> None:
    program_path = tmp_path / "program.pt"
    write_torch_once(program_path, _program_payload())
    loaded = read_program_predictions(program_path)
    assert loaded["family_ids"] == ["a", "b"]
    assert loaded["packet_valid"].tolist() == [True, False]
    assert program_path.stat().st_mode & 0o777 == 0o444

    query_path = tmp_path / "query.pt"
    write_torch_once(
        query_path,
        {
            "schema": QUERY_PREDICTION_SCHEMA,
            "family_ids": ["a"],
            "query_source_sha256": "c" * 64,
            "compiler_sha256": "b" * 64,
            "execution_sha256": "d" * 64,
            "positions": torch.tensor([2], dtype=torch.uint8),
        },
    )
    assert read_query_predictions(query_path)["positions"].tolist() == [2]


def test_validity_mask_cannot_be_forged() -> None:
    payload = _program_payload()
    payload["packet_valid"] = torch.tensor([True, True])
    with pytest.raises(ValueError, match="not derived"):
        validate_program_predictions(payload)


def test_packet_index_supports_mixed_and_zero_valid_rows(tmp_path) -> None:
    path = tmp_path / "index.json"
    write_json_once(
        path,
        {
            "schema": PACKET_INDEX_SCHEMA,
            "program_predictions_sha256": "a" * 64,
            "packet_sha256": "b" * 64,
            "valid_family_ids": ["a"],
            "valid_source_indices": [0],
            "invalid_family_ids": ["b"],
        },
    )
    assert read_packet_index(path)["invalid_family_ids"] == ["b"]
    assert json.loads(path.read_text())["valid_source_indices"] == [0]

    zero = tmp_path / "zero.json"
    write_json_once(
        zero,
        {
            "schema": PACKET_INDEX_SCHEMA,
            "program_predictions_sha256": "a" * 64,
            "packet_sha256": None,
            "valid_family_ids": [],
            "valid_source_indices": [],
            "invalid_family_ids": ["a", "b"],
        },
    )
    assert read_packet_index(zero)["packet_sha256"] is None
