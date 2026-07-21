from __future__ import annotations

import json

import pytest
import torch

from commit_ctaa_raw_evidence import commit_raw_evidence
from ctaa_neural_core import ClosureFeatureTransitionCore
from ctaa_evaluation_io import (
    PROGRAM_PREDICTION_SCHEMA,
    QUERY_PREDICTION_SCHEMA,
    packet_valid_mask,
    sha256_file,
    write_json_once,
    write_torch_once,
)
from run_ctaa_packet_executor import EXECUTION_SCHEMA, write_execution_once
from ctaa_packet_io import read_packet_file, write_query_file
from ctaa_trunk_compiler import HardCTAAQuery
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


def _core_checkpoint(path) -> ClosureFeatureTransitionCore:
    torch.manual_seed(101)
    core = ClosureFeatureTransitionCore().eval()
    write_torch_once(
        path,
        {
            "schema": "ctaa_recurrent_core_v1",
            "kind": "closure_feature",
            "state": core.state_dict(),
            "training": {
                "schema": "r12_ctaa_v2_core_training_v1",
                "arm": "ctaa_closure",
                "seed": 17,
                "atomic_sha256": "1" * 64,
                "closure_sha256": "2" * 64,
                "updates": 10,
                "batch_size": 8,
                "learning_rate": 1e-3,
            },
        },
    )
    return core


def test_raw_evidence_preserves_invalid_rows_and_binds_all_stages(tmp_path) -> None:
    program = tmp_path / "program.pt"
    packet = tmp_path / "packet.bin"
    index = tmp_path / "index.json"
    _program_predictions(program)
    seal_predictions(program, packet, index)

    execution = tmp_path / "execution.pt"
    core = tmp_path / "core.pt"
    core_module = _core_checkpoint(core)
    trace = read_packet_file(packet).execute_dual(core_module)
    write_execution_once(
        execution,
        {
            "schema": EXECUTION_SCHEMA,
            "core_kind": "closure_feature",
            "packet_sha256": __import__("json").loads(index.read_text())["packet_sha256"],
            "core_sha256": sha256_file(core),
            "state_route": trace.state_route.states.to(torch.uint8),
            "halted": trace.state_route.halted,
            "composed_cards": trace.composed_cards.to(torch.uint8),
            "composed_states": trace.composed_states.to(torch.uint8),
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
    hard_query = tmp_path / "query.bin"
    write_query_file(
        hard_query,
        HardCTAAQuery(position=torch.tensor([1], dtype=torch.uint8)),
    )
    answers = tmp_path / "answers.json"
    expected_answer = int(trace.state_route.states[0, -1, 1])
    write_json_once(
        answers,
        {
            "schema": "ctaa_late_query_answer_v1",
            "execution_sha256": sha256_file(execution),
            "query_sha256": sha256_file(hard_query),
            "answers": [expected_answer],
        },
    )
    output = tmp_path / "evidence"
    receipt = commit_raw_evidence(
        program_predictions_path=program,
        packet_index_path=index,
        execution_path=execution,
        query_predictions_path=query,
        answers_path=answers,
        core_checkpoint_path=core,
        packet_path=packet,
        hard_query_path=hard_query,
        output_dir=output,
    )
    rows = [json.loads(line) for line in (output / "evidence.jsonl").read_text().splitlines()]
    assert receipt["rows"] == 2
    assert isinstance(receipt["query_positions_sha256"], str)
    assert len(receipt["query_positions_sha256"]) == 64
    assert receipt["core_training"]["training_seed"] == 17
    assert receipt["core_training"]["training_arm"] == "ctaa_closure"
    assert rows[0]["packet_valid"] is True
    assert rows[0]["answer"] == expected_answer
    assert rows[0]["route_agreement"] is False
    assert rows[1]["packet_valid"] is False
    assert rows[1]["state_route"] is None
    assert rows[1]["answer"] is None

    execution.chmod(0o644)
    mutated = torch.load(execution, map_location="cpu", weights_only=True)
    mutated["state_route"][0, 0, 0] = (mutated["state_route"][0, 0, 0] + 1) % 3
    torch.save(mutated, execution)
    execution.chmod(0o444)
    with pytest.raises(ValueError, match="deterministic replay"):
        commit_raw_evidence(
            program_predictions_path=program,
            packet_index_path=index,
            execution_path=execution,
            query_predictions_path=query,
            answers_path=answers,
            core_checkpoint_path=core,
            packet_path=packet,
            hard_query_path=hard_query,
            output_dir=tmp_path / "mutated_evidence",
        )


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
    assert receipt["query_positions_sha256"] is None
    assert not packet.exists()
