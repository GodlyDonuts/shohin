#!/usr/bin/env python3
"""Deterministic tests for the OBCB-1 finite CPU falsifier."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import stat

import pytest

from pipeline import obcb_falsifier as obcb


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return obcb.build_report()


def test_three_element_carry_monoid_and_45_10_45_counts(
    report: dict[str, object],
) -> None:
    monoid = report["monoid"]
    assert isinstance(monoid, dict)
    assert monoid["event_counts"] == {
        "add": {"K0": 45, "I": 10, "K1": 45},
        "sub": {"K0": 45, "I": 10, "K1": 45},
    }
    assert monoid["closure"] is True
    assert monoid["identity"] is True
    assert monoid["associativity_triples_checked"] == 27
    assert monoid["associative"] is True
    assert monoid["composition_table_outer_after_inner"] == {
        "K0": {"K0": "K0", "I": "K0", "K1": "K0"},
        "I": {"K0": "K0", "I": "I", "K1": "K1"},
        "K1": {"K0": "K1", "I": "K1", "K1": "K1"},
    }


def test_all_400_local_cells_and_all_200_flip_pairs_are_exact(
    report: dict[str, object],
) -> None:
    treatment = report["obcb_treatment"]
    assert isinstance(treatment, dict)
    assert treatment["local_cells_total"] == 400
    assert treatment["local_cells_exact"] == 400
    assert treatment["carry_flip_pairs_total"] == 200
    assert treatment["carry_flip_pairs_exact"] == 200

    seen = 0
    for event, carry in obcb.iter_local_cells():
        result = obcb.oracle_transition(event, carry)
        operator = obcb.classify_event(event)
        assert result.next_carry == operator.apply(carry)
        seen += 1
    assert seen == 400


def test_operator_balanced_allocation_keeps_counterfactual_pairs_together(
    report: dict[str, object],
) -> None:
    allocation = report["balanced_allocation"]
    assert isinstance(allocation, dict)
    expected_pairs = {
        "add": {"K0": 90, "I": 90, "K1": 90},
        "sub": {"K0": 90, "I": 90, "K1": 90},
    }
    expected_rows = {
        operation: {operator: 180 for operator in ("K0", "I", "K1")}
        for operation in obcb.OPERATIONS
    }
    assert allocation["pair_repetitions"] == {"K0": 2, "I": 9, "K1": 2}
    assert allocation["sampled_pairs_by_operation"] == expected_pairs
    assert allocation["sampled_rows_by_operation"] == expected_rows
    assert allocation["total_paired_examples"] == 540
    assert allocation["total_rows"] == 1_080
    assert allocation["counterfactual_pairs_inseparable"] is True
    assert len(allocation["rows_sha256"]) == 64


def test_all_40000_edges_close_under_iteration_and_source_deletion(
    report: dict[str, object],
) -> None:
    treatment = report["obcb_treatment"]
    assert isinstance(treatment, dict)
    assert treatment["two_step_edges_total"] == 40_000
    assert treatment["factual_edges_exact"] == 40_000
    assert treatment["flipped_edges_exact"] == 40_000
    assert treatment["source_poison_edges_invariant"] == 40_000
    assert treatment["first_packets_exact"] == 40_000
    assert treatment["second_packets_exact"] == 40_000
    assert treatment["stateless_edge_commits"] == 40_000


def test_commit_packet_and_resource_vector_are_exactly_one_bit(
    report: dict[str, object],
) -> None:
    assert tuple(field.name for field in fields(obcb.CommitBit)) == ("bit",)
    assert obcb.CommitBit.__slots__ == ("bit",)
    packet = obcb.CommitBit(True)
    assert obcb.exact_packet_contract(packet)
    assert not obcb.exact_packet_contract(obcb.ResultHistoryPacket(True, 7))
    assert not obcb.exact_packet_contract(obcb.HiddenStepPacket(True, 1))
    assert report["resource_vector"] == {
        "added_trainable_parameters": 0,
        "retained_dynamic_bits": 1,
        "source_bytes_after_commit": 0,
        "result_history_symbols": 0,
        "hidden_step_bits": 0,
        "external_memory_bytes": 0,
        "external_execution_calls_at_inference": 0,
        "additional_inference_steps": 0,
    }


@pytest.mark.parametrize(
    ("control", "required_failure"),
    (
        ("commit_ignoring", "all_400_local_cells_exact"),
        ("stale_source_replay", "machine_has_no_postcommit_state"),
        ("shuffled_state", "all_400_local_cells_exact"),
        ("result_history", "local_packets_exactly_one_bit"),
        ("hidden_step", "local_packets_exactly_one_bit"),
    ),
)
def test_named_negative_controls_are_rejected(
    report: dict[str, object],
    control: str,
    required_failure: str,
) -> None:
    controls = report["negative_controls"]
    assert isinstance(controls, dict)
    result = controls[control]
    assert result["rejected"] is True
    assert result["pass"] is False
    assert required_failure in result["failed_checks"]


def test_stale_source_replay_is_favorable_but_poisonable(
    report: dict[str, object],
) -> None:
    controls = report["negative_controls"]
    stale = controls["stale_source_replay"]
    assert stale["local_cells_exact"] == 400
    assert stale["factual_edges_exact"] == 40_000
    assert stale["flipped_edges_exact"] == 40_000
    assert stale["source_poison_edges_invariant"] == 0
    assert stale["poisoned_control_edges"] == 40_000
    assert stale["stateless_edge_commits"] == 0


def test_extra_state_controls_can_be_behaviorally_exact_and_still_rejected(
    report: dict[str, object],
) -> None:
    controls = report["negative_controls"]
    for name in ("result_history", "hidden_step"):
        result = controls[name]
        assert result["local_cells_exact"] == 400
        assert result["factual_edges_exact"] == 40_000
        assert result["flipped_edges_exact"] == 40_000
        assert result["local_packets_exact"] == 0
        assert result["first_packets_exact"] == 0
        assert result["second_packets_exact"] == 0
        assert result["rejected"] is True


def test_report_is_deterministic_and_verifier_fails_closed(
    report: dict[str, object],
) -> None:
    obcb.verify_report(report)
    assert obcb.canonical_report_bytes(report) == obcb.canonical_report_bytes()
    mutated = json.loads(obcb.canonical_report_bytes(report))
    mutated["obcb_treatment"]["factual_edges_exact"] = 39_999
    with pytest.raises(obcb.ContractError):
        obcb.verify_report(mutated)


def test_immutable_writer_refuses_overwrite(
    report: dict[str, object], tmp_path: Path
) -> None:
    output = tmp_path / "obcb.json"
    obcb.write_immutable_report(output, report)
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    loaded = json.loads(output.read_text(encoding="ascii"))
    obcb.verify_report(loaded)
    with pytest.raises(FileExistsError):
        obcb.write_immutable_report(output, report)
