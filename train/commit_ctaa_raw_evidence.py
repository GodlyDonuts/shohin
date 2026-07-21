#!/usr/bin/env python3
"""Commit oracle-blind CTAA predictions and execution traces once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ctaa_evaluation_io import (
    read_packet_index,
    read_program_predictions,
    read_query_predictions,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from run_ctaa_packet_executor import EXECUTION_SCHEMA


RAW_EVIDENCE_SCHEMA = "r12_ctaa_v2_raw_evidence_v1"
RAW_EVIDENCE_RECEIPT_SCHEMA = "r12_ctaa_v2_raw_evidence_receipt_v1"


def _load_execution(path: Path, expected_packet_sha: str, rows: int) -> dict[str, object]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema",
        "core_kind",
        "packet_sha256",
        "core_sha256",
        "state_route",
        "halted",
        "composed_cards",
        "composed_states",
    }
    if not isinstance(value, dict) or set(value) != required or value.get("schema") != EXECUTION_SCHEMA:
        raise ValueError("CTAA raw-evidence execution schema differs")
    state = value["state_route"]
    halted = value["halted"]
    composed_cards = value["composed_cards"]
    composed_states = value["composed_states"]
    if (
        value["packet_sha256"] != expected_packet_sha
        or not isinstance(state, torch.Tensor)
        or state.shape != (rows, 42, 3)
        or state.dtype != torch.uint8
        or not isinstance(halted, torch.Tensor)
        or halted.shape != (rows, 42)
        or halted.dtype != torch.bool
        or not isinstance(composed_cards, torch.Tensor)
        or composed_cards.shape != state.shape
        or composed_cards.dtype != torch.uint8
        or not isinstance(composed_states, torch.Tensor)
        or composed_states.shape != state.shape
        or composed_states.dtype != torch.uint8
    ):
        raise ValueError("CTAA raw-evidence execution geometry differs")
    if int(state.max()) >= 3 or int(composed_states.max()) >= 3 or int(composed_cards.max()) >= 3:
        raise ValueError("CTAA raw-evidence execution leaves categorical domain")
    return value


def _load_answers(path: Path, execution_sha: str, rows: int) -> list[int]:
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "execution_sha256", "query_sha256", "answers"}
        or value.get("schema") != "ctaa_late_query_answer_v1"
        or value.get("execution_sha256") != execution_sha
        or not isinstance(value.get("answers"), list)
        or len(value["answers"]) != rows
        or any(not isinstance(answer, int) or not 0 <= answer < 3 for answer in value["answers"])
    ):
        raise ValueError("CTAA raw-evidence answer artifact differs")
    return value["answers"]


def commit_raw_evidence(
    *,
    program_predictions_path: Path,
    packet_index_path: Path,
    output_dir: Path,
    execution_path: Path | None = None,
    query_predictions_path: Path | None = None,
    answers_path: Path | None = None,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing existing CTAA raw-evidence directory: {output_dir}")
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing CTAA raw-evidence temporary: {temporary}")
    program = read_program_predictions(program_predictions_path)
    index = read_packet_index(packet_index_path)
    if index["program_predictions_sha256"] != sha256_file(program_predictions_path):
        raise ValueError("CTAA raw-evidence packet index is not bound to predictions")
    valid = program["packet_valid"]
    valid_indices = valid.nonzero(as_tuple=False).flatten().tolist()
    invalid_ids = [
        family_id
        for source_index, family_id in enumerate(program["family_ids"])
        if not bool(valid[source_index])
    ]
    if (
        valid_indices != index["valid_source_indices"]
        or [program["family_ids"][source_index] for source_index in valid_indices]
        != index["valid_family_ids"]
        or invalid_ids != index["invalid_family_ids"]
    ):
        raise ValueError("CTAA raw-evidence packet index rows differ")

    execution = None
    execution_sha = None
    query = None
    answers = None
    if execution_path is not None:
        if not valid_indices or index["packet_sha256"] is None:
            raise ValueError("CTAA raw-evidence execution exists without valid packets")
        execution = _load_execution(
            execution_path,
            str(index["packet_sha256"]),
            len(valid_indices),
        )
        execution_sha = sha256_file(execution_path)
    if query_predictions_path is not None:
        if execution is None or execution_sha is None:
            raise ValueError("CTAA raw-evidence query exists before execution")
        query = read_query_predictions(query_predictions_path)
        if (
            query["family_ids"] != index["valid_family_ids"]
            or query["execution_sha256"] != execution_sha
            or query["compiler_sha256"] != program["compiler_sha256"]
        ):
            raise ValueError("CTAA raw-evidence query binding differs")
    if answers_path is not None:
        if query is None or execution_sha is None:
            raise ValueError("CTAA raw-evidence answers exist before late query")
        answers = _load_answers(answers_path, execution_sha, len(valid_indices))

    valid_row = {source_index: row for row, source_index in enumerate(valid_indices)}
    evidence_rows = []
    for source_index, family_id in enumerate(program["family_ids"]):
        packet_valid = bool(valid[source_index])
        execution_row = valid_row.get(source_index)
        state_route = None
        halted = None
        composed_states = None
        route_agreement = False
        query_position = None
        answer = None
        if execution is not None and execution_row is not None:
            state_tensor = execution["state_route"][execution_row]
            composed_tensor = execution["composed_states"][execution_row]
            state_route = state_tensor.tolist()
            halted = execution["halted"][execution_row].tolist()
            composed_states = composed_tensor.tolist()
            route_agreement = bool(torch.equal(state_tensor, composed_tensor))
        if query is not None and execution_row is not None:
            query_position = int(query["positions"][execution_row])
        if answers is not None and execution_row is not None:
            answer = int(answers[execution_row])
        evidence_rows.append(
            {
                "schema": RAW_EVIDENCE_SCHEMA,
                "family_id": family_id,
                "source_index": source_index,
                "packet_valid": packet_valid,
                "predicted_action_cards": program["action_cards"][source_index].tolist(),
                "predicted_initial_state": program["initial_state"][source_index].tolist(),
                "predicted_schedule": program["schedule"][source_index].tolist(),
                "predicted_query_position": query_position,
                "state_route": state_route,
                "halted": halted,
                "composed_states": composed_states,
                "route_agreement": route_agreement,
                "answer": answer,
            }
        )
    temporary.mkdir(parents=True)
    try:
        evidence_path = temporary / "evidence.jsonl"
        row_count, evidence_sha = write_jsonl_once(evidence_path, evidence_rows)
        receipt = {
            "schema": RAW_EVIDENCE_RECEIPT_SCHEMA,
            "rows": row_count,
            "valid_packets": len(valid_indices),
            "executed_rows": len(valid_indices) if execution is not None else 0,
            "queried_rows": len(valid_indices) if query is not None else 0,
            "answered_rows": len(valid_indices) if answers is not None else 0,
            "program_predictions_sha256": sha256_file(program_predictions_path),
            "compiler_sha256": program["compiler_sha256"],
            "packet_index_sha256": sha256_file(packet_index_path),
            "execution_sha256": execution_sha,
            "core_sha256": execution["core_sha256"] if execution is not None else None,
            "core_kind": execution["core_kind"] if execution is not None else None,
            "query_predictions_sha256": (
                sha256_file(query_predictions_path) if query_predictions_path is not None else None
            ),
            "answers_sha256": sha256_file(answers_path) if answers_path is not None else None,
            "evidence_sha256": evidence_sha,
            "oracle_access": 0,
        }
        write_json_once(temporary / "receipt.json", receipt)
        temporary.chmod(0o555)
        temporary.replace(output_dir)
    finally:
        if temporary.exists():
            for path in temporary.iterdir():
                path.chmod(0o600)
                path.unlink()
            temporary.chmod(0o700)
            temporary.rmdir()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-predictions", type=Path, required=True)
    parser.add_argument("--packet-index", type=Path, required=True)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--query-predictions", type=Path)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            commit_raw_evidence(
                program_predictions_path=args.program_predictions,
                packet_index_path=args.packet_index,
                execution_path=args.execution,
                query_predictions_path=args.query_predictions,
                answers_path=args.answers,
                output_dir=args.output_dir,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
