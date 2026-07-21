from __future__ import annotations

import torch

from ctaa_evaluation_io import (
    PROGRAM_PREDICTION_SCHEMA,
    packet_valid_mask,
    read_packet_index,
    write_torch_once,
)
from ctaa_packet_io import read_packet_file
from seal_ctaa_program_packets import seal_predictions


def _write_predictions(path, *, all_invalid: bool) -> None:
    schedule = torch.zeros((2, 41), dtype=torch.uint8)
    if all_invalid:
        schedule[:, 1] = 4
        schedule[:, 2] = 4
    else:
        schedule[0, 3] = 4
        schedule[1, 1] = 4
        schedule[1, 2] = 4
    write_torch_once(
        path,
        {
            "schema": PROGRAM_PREDICTION_SCHEMA,
            "family_ids": ["valid", "invalid"],
            "program_source_sha256": "a" * 64,
            "compiler_sha256": "b" * 64,
            "action_cards": torch.zeros((2, 4, 3), dtype=torch.uint8),
            "initial_state": torch.zeros((2, 3), dtype=torch.uint8),
            "schedule": schedule,
            "packet_valid": packet_valid_mask(schedule),
        },
    )


def test_sealer_keeps_valid_subset_and_preserves_invalid_failure(tmp_path) -> None:
    predictions = tmp_path / "predictions.pt"
    packet = tmp_path / "packet.bin"
    index = tmp_path / "index.json"
    _write_predictions(predictions, all_invalid=False)
    report = seal_predictions(predictions, packet, index)
    assert report["valid_rows"] == 1
    assert read_packet_file(packet).schedule.shape == (1, 41)
    assert read_packet_index(index)["invalid_family_ids"] == ["invalid"]


def test_sealer_records_all_invalid_without_creating_packet(tmp_path) -> None:
    predictions = tmp_path / "predictions.pt"
    packet = tmp_path / "packet.bin"
    index = tmp_path / "index.json"
    _write_predictions(predictions, all_invalid=True)
    report = seal_predictions(predictions, packet, index)
    assert report["valid_rows"] == 0
    assert not packet.exists()
    assert read_packet_index(index)["packet_sha256"] is None

