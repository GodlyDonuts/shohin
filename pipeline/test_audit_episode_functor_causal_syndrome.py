from __future__ import annotations

from pipeline.audit_episode_functor_causal_syndrome import (
    audit_frozen_board,
)


def test_frozen_board_single_swap_syndromes_are_identifiable() -> None:
    report = audit_frozen_board()
    assert report["world_count"] == 200
    assert report["faults_per_world"] == [132]
    assert report["total_faults"] == 26_400
    assert report["total_unique_within_world_fingerprints"] == 26_400
    assert report["zero_fingerprint_count"] == 0
    assert report["within_world_collision_count"] == 0
    assert (
        report["decision"]
        == "single_swap_syndromes_identifiable_mechanics_only"
    )
    assert len(report["report_payload_sha256"]) == 64


def test_every_world_has_complete_unique_fault_inventory() -> None:
    report = audit_frozen_board()
    for row in report["world_receipts"]:
        assert row["fault_count"] == 132
        assert row["unique_fingerprint_count"] == 132
        assert row["zero_fingerprint_count"] == 0
        assert row["collision_count"] == 0
        assert row["minimum_changed_coordinates"] > 0
        assert (
            row["maximum_changed_coordinates"]
            >= row["minimum_changed_coordinates"]
        )
        assert len(row["fingerprint_set_sha256"]) == 64
