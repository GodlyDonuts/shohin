from __future__ import annotations

import dataclasses
from pathlib import Path

import audit_neural_tcrr_board as audit
import neural_tcrr_board as board
import pytest


def _report(
    value: board.LocalTransitionSlice | None = None,
) -> dict[str, object]:
    return audit.build_audit_report(
        board.build_local_transition_slice() if value is None else value,
        seed=audit.DEFAULT_SEED,
        source_commit="test-commit",
        source_sha256={"test": "receipt"},
    )


def test_audit_admits_only_the_bounded_board_slice() -> None:
    report = _report()
    assert report["decision"] == "admit_source_deleted_local_board_only"
    assert report["packet_count"] == 22
    assert report["transition_count"] == 24
    assert report["split_counts"] == {
        "local_transition_development": 6,
        "local_transition_train": 16,
    }
    assert report["controlled_no_redex_count"] == 4
    assert report["rule_pair_fingerprint_count"] == 51
    assert report["reachable_two_rule_composition_count"] == 8
    assert report["ledger_alignment"] is True
    assert report["model_packets_exclude_offline_ledger"] is True
    assert report["exact_axis_twins_recomputed"] is True
    assert report["max_occurrence_path_depth"] == 4
    assert report["max_capacity"] == 16
    assert report["independent_successor_oracle"] is True
    assert report["neural_runtime_present"] is False
    assert report["independent_oracle"] == {
        "packet_agreement": 22,
        "packet_count": 22,
        "state_count": 47,
        "transition_count": 32,
        "normal_form_count": 21,
        "cyclic_component_count": 1,
    }
    custody = report["export_custody"]
    assert isinstance(custody, dict)
    assert custody["train_packet_count"] == 16
    assert custody["development_packet_count"] == 6
    assert custody["sealed_development_record_count"] == 6


def test_audit_board_digest_is_deterministic() -> None:
    assert _report()["board_sha256"] == _report()["board_sha256"]


def test_audit_rejects_missing_required_twin() -> None:
    value = board.build_local_transition_slice()
    mutant = dataclasses.replace(value, twins=value.twins[:-1])
    with pytest.raises(board.NeuralTcrrBoardError, match="twin ledger"):
        _report(mutant)


def test_audit_rejects_stale_oracle_receipt() -> None:
    value = board.build_local_transition_slice()
    first = value.oracle_agreements[0]
    mutant = dataclasses.replace(
        value,
        oracle_agreements=(
            dataclasses.replace(first, exact_agreement=False),
            *value.oracle_agreements[1:],
        ),
    )
    with pytest.raises(board.NeuralTcrrBoardError, match="oracle ledger"):
        _report(mutant)


def test_source_receipt_refuses_dirty_audit_source(tmp_path: Path) -> None:
    # Exercise the report boundary without depending on the repository state.
    missing = tmp_path / "missing-root"
    missing.mkdir()
    try:
        audit._source_receipt(missing)
    except audit.NeuralTcrrBoardAuditError as error:
        assert "missing audit source" in str(error)
        return
    raise AssertionError("missing-source mutant unexpectedly survived")
