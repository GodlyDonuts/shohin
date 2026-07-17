#!/usr/bin/env python3
"""Focused exhaustive tests for the frozen PLCC CPU mechanics falsifier."""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from pipeline import plcc_cpu_falsifier as plcc


def test_all_400_local_cells_are_exact_and_context_blind() -> None:
    audit = plcc.local_cell_audit()
    assert audit["cell_count"] == 400
    assert audit["exact_cells"] == 400
    assert audit["context_count_per_cell"] == 10
    assert audit["terminal_contexts_per_cell"] == 4
    assert audit["nonterminal_contexts_per_cell"] == 6
    assert audit["context_observation_count"] == 4_000
    assert audit["context_exact"] == 4_000
    assert audit["observed_signature"] == ["op", "a_p", "b_p", "c_p"]
    assert audit["local_table_sha256"] == plcc.FROZEN_LOCAL_TABLE_SHA256
    assert tuple(inspect.signature(plcc.plcc_local_transition).parameters) == (
        "op",
        "a_p",
        "b_p",
        "c_p",
    )


def test_local_cell_source_cannot_read_position_width_or_terminality() -> None:
    tree = ast.parse(inspect.getsource(plcc.plcc_local_transition))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert names.isdisjoint({"position", "location", "width", "terminal"})


def test_exact_known_add_and_sub_cells() -> None:
    assert plcc.plcc_local_transition("ADD", 9, 9, 1) == plcc.LocalOutput(9, 1)
    assert plcc.plcc_local_transition("ADD", 0, 0, 0) == plcc.LocalOutput(0, 0)
    assert plcc.plcc_local_transition("SUB", 0, 9, 1) == plcc.LocalOutput(0, 1)
    assert plcc.plcc_local_transition("SUB", 9, 0, 1) == plcc.LocalOutput(8, 0)


def test_two_column_oracle_prefix_deletion_and_recurrent_control() -> None:
    audit = plcc.two_column_audit()
    assert audit["case_count"] == 40_000
    assert audit["plcc_exact"] == 40_000
    assert audit["explicit_recurrent_exact"] == 40_000
    assert audit["plcc_recurrent_equivalent"] == 40_000
    assert audit["source_prefix_deletion_exact"] == 40_000
    assert audit["intermediate_packet_only"] == 40_000
    assert audit["terminal_endpoint_only"] == 40_000
    assert audit["intermediate_emitted_symbol_count"] == 0
    assert audit["generated_token_count"] == 0
    assert audit["generated_kv_bytes"] == 0
    assert audit["result_tape_slots"] == 0
    assert audit["two_column_board_sha256"] == plcc.FROZEN_TWO_COLUMN_BOARD_SHA256


def test_nonterminal_runtime_returns_only_packet_and_discards_digit() -> None:
    board = plcc.OperandBoard(
        "ADD",
        (plcc.Column(7, 8), plcc.Column(1, 2)),
    )
    first = plcc.advance_packet(board, plcc.Packet(0, 0))
    assert type(first) is plcc.Packet
    assert [field.name for field in fields(first)] == ["location", "polarity"]
    assert first == plcc.Packet(1, 1)
    terminal = plcc.advance_packet(board, first)
    assert terminal == plcc.Endpoint(4, 0)


def test_different_carry_swaps_and_same_carry_shams_are_exhaustive() -> None:
    audit = plcc.carry_swap_audit()
    assert audit["different_carry_swap_checks"] == 400
    assert audit["different_carry_donor_exact"] == 400
    assert audit["different_carry_recipient_divergence"] == 400
    assert audit["same_carry_sham_checks"] == 400
    assert audit["same_carry_sham_exact"] == 400
    assert len(audit["same_carry_prefix_witnesses"]) == 4


def test_cursor_location_swaps_select_source_slot_without_changing_bit() -> None:
    audit = plcc.cursor_swap_audit()
    assert audit["swap_pair_count"] == 40_000
    assert audit["selected_column_observation_count"] == 80_000
    assert audit["selected_column_exact"] == 80_000
    assert audit["polarity_preserved_before_scatter"] == 40_000


def test_width_two_states_are_pairwise_distinguishable() -> None:
    audit = plcc.distinguishability_audit()
    assert audit["state_count"] == 4
    assert audit["state_pair_count"] == 6
    assert audit["distinguishing_witness_count"] == 6
    assert audit["pairwise_distinguishable"]
    assert audit["minimum_total_logical_bits"] == 2
    assert audit["packet_polarity_bits"] == 1
    assert audit["polarity_alone_insufficient"]
    assert len(audit["witnesses"]) == 6


def test_packet_has_exactly_one_support_and_one_payload_bit() -> None:
    audit = plcc.packet_surface_audit()
    assert audit["packet_fields"] == ["location", "polarity"]
    assert audit["packet_slots"] == ["location", "polarity"]
    assert not audit["packet_has_dynamic_dict"]
    assert audit["mutable_payload_bits"] == 1
    assert audit["occupancy_checks"] == 20
    assert audit["exact_one_occupancy"] == 20
    assert audit["packet_recurrent_round_trips"] == 20
    assert audit["learned_address_head_count"] == 0
    assert audit["external_execution_calls"] == 0
    for width in (1, 2, 3, 4):
        for location in range(width):
            for polarity in (0, 1):
                occupancy = plcc.packet_occupancy(
                    plcc.Packet(location, polarity), width
                )
                assert sum(occupancy) == 1
                assert occupancy[location] == 1


def test_hidden_packet_payloads_and_invalid_support_fail_closed() -> None:
    @dataclass(frozen=True, slots=True)
    class HiddenPacket:
        location: int
        polarity: int
        hidden: int

    with pytest.raises(plcc.AuditError, match="exact Packet type"):
        plcc.validate_packet_no_hidden_payload(HiddenPacket(0, 0, 7), 2)  # type: ignore[arg-type]
    with pytest.raises(plcc.AuditError, match="outside the source lattice"):
        plcc.validate_packet_no_hidden_payload(plcc.Packet(2, 0), 2)
    with pytest.raises((TypeError, ValueError)):
        plcc.Packet(0, 2)


def test_hidden_source_board_payload_fails_closed() -> None:
    @dataclass(frozen=True, slots=True)
    class HiddenBoard:
        op: str
        columns: tuple[plcc.Column, ...]
        tape: tuple[int, ...]

    hidden = HiddenBoard("ADD", (plcc.Column(0, 0),), (9,))
    with pytest.raises(plcc.AuditError, match="exact OperandBoard type"):
        plcc.validate_board_closed_world(hidden)  # type: ignore[arg-type]


def test_explicit_recurrent_control_is_an_exact_coordinate_change() -> None:
    audit = plcc.recurrent_equivalence_audit()
    assert audit["coordinate_bijection_width_two"]
    assert audit["endpoint_equivalence_cases"] == 40_000
    assert audit["endpoint_equivalence_total"] == 40_000
    assert audit["resource_vectors_equal"]
    assert audit["computational_class_boundary"] == "finite_state_transducer"
    assert not audit["mechanical_advantage_over_recurrent_control"]


def test_full_report_is_deterministic_hostile_and_closed() -> None:
    first = plcc.run_audit()
    second = plcc.run_audit()
    assert plcc.report_bytes(first) == plcc.report_bytes(second)
    assert first["mechanics_contract_satisfied"]
    assert all(first["gates"].values())
    assert first["mechanical_verdict"] == "equivalent_to_explicit_recurrent_control"
    assert not first["novel_reasoning_primitive_supported"]
    assert not first["neural_pilot_authorized_by_this_report"]
    assert "no novelty" in first["claim_boundary"]
    content = dict(first)
    observed_hash = content.pop("report_content_sha256")
    assert observed_hash == plcc.sha256_bytes(plcc.canonical_json_bytes(content))


def test_report_publication_is_deterministic_read_only_and_no_overwrite(
    tmp_path: Path,
) -> None:
    report = plcc.run_audit()
    destination = tmp_path / "plcc_report.json"
    written_hash = plcc.write_report_once(destination, report)
    payload = destination.read_bytes()
    assert written_hash == plcc.sha256_bytes(payload)
    assert payload == plcc.report_bytes(report)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        plcc.write_report_once(destination, report)


def test_cli_writes_machine_readable_report(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline.plcc_cpu_falsifier",
            "--output",
            os.fspath(destination),
        ],
        cwd=Path(plcc.__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    parsed = json.loads(destination.read_text(encoding="ascii"))
    assert parsed["protocol_id"] == plcc.PROTOCOL_ID
    assert parsed["mechanics_contract_satisfied"]


def test_module_imports_no_accelerator_network_or_subprocess_stack() -> None:
    tree = ast.parse(Path(plcc.__file__).read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    forbidden = {
        "numpy",
        "requests",
        "socket",
        "subprocess",
        "tensorflow",
        "torch",
    }
    assert forbidden.isdisjoint(imported_roots)
