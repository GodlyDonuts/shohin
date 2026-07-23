from __future__ import annotations

import dataclasses
from pathlib import Path

import audit_neural_tcrr_board as audit
import neural_tcrr_board as board


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
    assert report["packet_count"] == 21
    assert report["transition_count"] == 25
    assert report["split_counts"] == {
        "local_transition_development": 6,
        "local_transition_train": 15,
    }
    assert report["identity_namespaces_disjoint"] is True
    assert report["ledger_alignment"] is True
    assert report["model_packets_exclude_offline_ledger"] is True
    assert report["max_occurrence_path_depth"] == 4
    assert report["max_capacity"] == 16
    assert report["independent_successor_oracle"] is False
    assert report["neural_runtime_present"] is False


def test_audit_board_digest_is_deterministic() -> None:
    assert _report()["board_sha256"] == _report()["board_sha256"]


def test_audit_rejects_missing_required_twin() -> None:
    value = board.build_local_transition_slice()
    mutant = dataclasses.replace(value, twins=value.twins[:-1])
    assert _report(mutant)["decision"] == "reject"


def test_audit_rejects_cross_packet_identity_reuse() -> None:
    value = board.build_local_transition_slice()
    first = value.packets[0]
    second = value.packets[1]
    reused = dataclasses.replace(
        second.constructors[0],
        identifier=first.constructors[0].identifier,
    )
    mutant_packet = dataclasses.replace(
        second,
        constructors=(reused, *second.constructors[1:]),
    )
    packets = (first, mutant_packet, *value.packets[2:])
    mutant = dataclasses.replace(value, packets=packets)
    # The packet becomes internally invalid because its rule/graph references
    # still name the old constructor. Fail closed before an admission decision.
    try:
        _report(mutant)
    except board.NeuralTcrrBoardError:
        return
    raise AssertionError("identity-reuse mutant unexpectedly survived")


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

