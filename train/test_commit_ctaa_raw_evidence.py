from __future__ import annotations

import json

import torch

from commit_ctaa_raw_evidence import commit_raw_evidence
from ctaa_evaluation_io import (
    PROGRAM_PREDICTION_SCHEMA,
    QUERY_PREDICTION_SCHEMA,
    packet_valid_mask,
    sha256_file,
    write_json_once,
    write_torch_once,
)
from run_ctaa_packet_executor import EXECUTION_SCHEMA, write_execution_once
from seal_ctaa_program_packets import seal_predictions


def _program_predictions(path, *, all_invalid: bool = False) -> None:
    schedule = torch.zeros((2, 41), dtype=torch.uint8)
    schedule[0, 2] = 4
    schedule[1, 1] = 4
    schedule[1, 3] = 4
    if all_invalid:
        schedule[0, 4] = 4
    write_torch_once(
        path,
        {
            "schema": PROGRAM_PREDICTION_SCHEMA,
            "family_ids": ["f0", "f1"],
            "program_source_sha256": "a" * 64,
            "compiler_sha256": "b" * 64,
            "action_cards": torch.zeros((2, 4, 3), dtype=torch.uint8),
            "initial_state": torch.zeros((2, 3), dtype=torch.uint8),
            "schedule": schedule,
            "packet_valid": packet_valid_mask(schedule),
        },
    )


def test_raw_evidence_preserves_invalid_rows_and_binds_all_stages(tmp_path) -> None:
    program = tmp_path / "program.pt"
    packet = tmp_path / "packet.bin"
    index = tmp_path / "index.json"
    _program_predictions(program)
    seal_predictions(program, packet, index)

    state = torch.zeros((1, 42, 3), dtype=torch.uint8)
    halted = torch.zeros((1, 42), dtype=torch.bool)
    halted[:, 3:] = True
    execution = tmp_path / "execution.pt"
    write_execution_once(
        execution,
        {
            "schema": EXECUTION_SCHEMA,
            "core_kind": "closure_feature",
            "packet_sha256": __import__("json").loads(index.read_text())["packet_sha256"],
            "core_sha256": "c" * 64,
            "state_route": state,
            "halted": halted,
            "composed_cards": state.clone(),
            "composed_states": state.clone(),
        },
    )
    query = tmp_path / "query.pt"
    write_torch_once(
        query,
        {
            "schema": QUERY_PREDICTION_SCHEMA,
            "family_ids": ["f0"],
            "query_source_sha256": "d" * 64,
            "compiler_sha256": "b" * 64,
            "execution_sha256": sha256_file(execution),
            "positions": torch.tensor([1], dtype=torch.uint8),
        },
    )
    answers = tmp_path / "answers.json"
    write_json_once(
        answers,
        {
            "schema": "ctaa_late_query_answer_v1",
            "execution_sha256": sha256_file(execution),
            "query_sha256": "e" * 64,
            "answers": [0],
        },
    )
    output = tmp_path / "evidence"
    receipt = commit_raw_evidence(
        program_predictions_path=program,
        packet_index_path=index,
        execution_path=execution,
        query_predictions_path=query,
        answers_path=answers,
        output_dir=output,
    )
    rows = [json.loads(line) for line in (output / "evidence.jsonl").read_text().splitlines()]
    assert receipt["rows"] == 2
    assert rows[0]["packet_valid"] is True
    assert rows[0]["answer"] == 0
    assert rows[0]["route_agreement"] is True
    assert rows[1]["packet_valid"] is False
    assert rows[1]["state_route"] is None
    assert rows[1]["answer"] is None


def test_all_invalid_predictions_commit_without_executor(tmp_path) -> None:
    program = tmp_path / "program.pt"
    packet = tmp_path / "packet.bin"
    index = tmp_path / "index.json"
    _program_predictions(program, all_invalid=True)
    seal_predictions(program, packet, index)
    output = tmp_path / "evidence"
    receipt = commit_raw_evidence(
        program_predictions_path=program,
        packet_index_path=index,
        output_dir=output,
    )
    assert receipt["valid_packets"] == 0
    assert receipt["answered_rows"] == 0
    assert not packet.exists()

