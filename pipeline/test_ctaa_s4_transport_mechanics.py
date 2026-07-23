from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctaa_s4_transport_mechanics import (
    SCHEMA,
    build_report,
    canonical_json_bytes,
    write_exclusive,
)


def test_s4_transport_mechanics_do_not_authorize_neural() -> None:
    report = build_report()
    assert report["schema"] == SCHEMA
    assert report["claim_boundary"].endswith("no_reasoning_claim")
    assert report["group"] == {
        "elements": 24,
        "generators": 6,
        "inverse_checks": 24,
        "independent_composition_checks": 576,
        "associativity_checks": 13_824,
        "ordered_generator_pairs": 36,
        "noncommuting_ordered_pairs": 24,
        "abelian_control_order_collapses": 36,
        "coordinate_roundtrip_checks": 13_824,
        "transport_equivariance_checks": 82_944,
        "s4_table_sha256": report["group"]["s4_table_sha256"],
        "z24_table_sha256": report["group"]["z24_table_sha256"],
    }
    assert (
        report["group"]["s4_table_sha256"]
        != report["group"]["z24_table_sha256"]
    )
    assert len(report["order_witnesses"]) == 24
    assert report["executor"] == {
        "one_step_cases": 69_984,
        "one_step_state_and_binding_exact_checks": 139_968,
        "one_step_mass_checks": 69_984,
        "action_maps": 27,
        "post_stop_suffix_invariant": True,
        "mixed_trajectory_mass_conserved": True,
        "particle_gradient_finite_nonzero": True,
        "cue_kernel_gradient_finite_nonzero": True,
    }
    assert all(report["gates"].values())
    assert (
        report["decision"]
        == "record_component_mechanics_only_no_neural_authorization"
    )
    assert len(report["payload_sha256"]) == 64


def test_s4_transport_mechanics_publication_is_exclusive(tmp_path: Path) -> None:
    report = build_report()
    destination = tmp_path / "report.json"
    write_exclusive(destination, canonical_json_bytes(report))
    assert json.loads(destination.read_text(encoding="ascii")) == report
    with pytest.raises(FileExistsError):
        write_exclusive(destination, canonical_json_bytes(report))
