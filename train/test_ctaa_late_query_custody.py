from __future__ import annotations

import pytest
import torch

from ctaa_evaluation_io import (
    PACKET_INDEX_SCHEMA,
    QUERY_PREDICTION_SCHEMA,
    sha256_file,
    write_json_once,
    write_torch_once,
)
from ctaa_packet_io import read_query_file
from run_ctaa_packet_executor import EXECUTION_SCHEMA, write_execution_once
from run_ctaa_query_compiler import validate_execution_binding
from seal_ctaa_late_queries import seal_queries


def _custody_files(tmp_path):
    execution_path = tmp_path / "execution.pt"
    packet_sha = "b" * 64
    write_execution_once(
        execution_path,
        {
            "schema": EXECUTION_SCHEMA,
            "core_kind": "closure_feature",
            "packet_sha256": packet_sha,
            "core_sha256": "c" * 64,
            "state_route": torch.zeros((1, 42, 3), dtype=torch.uint8),
            "halted": torch.zeros((1, 42), dtype=torch.bool),
            "composed_cards": torch.zeros((1, 42, 3), dtype=torch.uint8),
            "composed_states": torch.zeros((1, 42, 3), dtype=torch.uint8),
        },
    )
    index_path = tmp_path / "index.json"
    write_json_once(
        index_path,
        {
            "schema": PACKET_INDEX_SCHEMA,
            "program_predictions_sha256": "a" * 64,
            "packet_sha256": packet_sha,
            "valid_family_ids": ["f0"],
            "valid_source_indices": [0],
            "invalid_family_ids": ["f1"],
        },
    )
    query_predictions = tmp_path / "query_predictions.pt"
    write_torch_once(
        query_predictions,
        {
            "schema": QUERY_PREDICTION_SCHEMA,
            "family_ids": ["f0"],
            "query_source_sha256": "d" * 64,
            "compiler_sha256": "e" * 64,
            "execution_sha256": sha256_file(execution_path),
            "positions": torch.tensor([2], dtype=torch.uint8),
        },
    )
    return execution_path, index_path, query_predictions


def test_late_query_seals_only_after_immutable_bound_execution(tmp_path) -> None:
    execution, index, predictions = _custody_files(tmp_path)
    output = tmp_path / "query.bin"
    seal_queries(predictions, index, execution, output)
    assert read_query_file(output).position.tolist() == [2]
    assert output.stat().st_mode & 0o777 == 0o444


def test_late_query_rejects_writable_or_wrong_packet_execution(tmp_path) -> None:
    execution, index, _ = _custody_files(tmp_path)
    execution.chmod(0o644)
    with pytest.raises(PermissionError, match="not immutable"):
        validate_execution_binding(execution, __import__("json").loads(index.read_text()))

    execution.chmod(0o444)
    wrong = __import__("json").loads(index.read_text())
    wrong["packet_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="binding"):
        validate_execution_binding(execution, wrong)

