from __future__ import annotations

import pytest

from ctaa_s4_transport_development import build_report


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return build_report()


def _arm_rows(report: dict[str, object], arm: str) -> list[dict[str, object]]:
    return [row for row in report["results"] if row["arm"] == arm]


def test_s4_transport_development_is_deterministic(
    report: dict[str, object],
) -> None:
    assert report == build_report()
    assert len(report["payload_sha256"]) == 64


def test_s4_transport_development_passes_all_declared_gates(
    report: dict[str, object],
) -> None:
    assert all(report["gates"].values())
    assert (
        report["decision"]
        == "record_retrospective_parameter_tying_signature_only"
    )
    assert report["evidence_status"] == "retrospective_development_not_preregistered"
    assert report["protocol"]["independent_target_oracle_checks"] == 576


def test_s4_tied_transport_generalizes_from_identity_row(
    report: dict[str, object],
) -> None:
    rows = _arm_rows(report, "s4_tied_identity_only")
    assert len(rows) == 5
    assert all(row["supervised_transitions"] == 6 for row in rows)
    assert all(
        result["correct"] == result["total"]
        for row in rows
        for result in row["depths"].values()
    )


def test_dense_control_has_capacity_but_needs_transition_rows(
    report: dict[str, object],
) -> None:
    sparse = _arm_rows(report, "dense24_identity_only")
    complete = _arm_rows(report, "dense24_complete_transition_ceiling")
    assert all(row["supervised_transitions"] == 6 for row in sparse)
    assert all(row["supervised_transitions"] == 144 for row in complete)
    assert all(
        result["correct"] == result["total"]
        for row in complete
        for result in row["depths"].values()
    )
    assert all(
        row["depths"]["4"]["correct"] == 0
        for row in sparse
    )


def test_abelian_control_cannot_recover_nonabelian_composition(
    report: dict[str, object],
) -> None:
    rows = _arm_rows(report, "z24_abelian_identity_only")
    assert all(
        row["depths"]["2"]["correct"] <= 8
        for row in rows
    )
    assert all(
        row["depths"]["4"]["correct"] <= 48
        for row in rows
    )
