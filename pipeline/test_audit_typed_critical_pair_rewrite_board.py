from __future__ import annotations

from pathlib import Path

from audit_typed_critical_pair_rewrite_board import build_audit


def test_cpu_mechanics_audit_admits_only_the_frozen_mechanics() -> None:
    root = Path(__file__).resolve().parents[1]
    report = build_audit(
        root=root,
        seed=20260723,
        require_clean=False,
    )
    assert report["decision"] == "admit_cpu_mechanics_only"
    assert report["episode_count"] == 14
    assert report["production_reference_agreement"]
    assert report["storage_reindex_invariant"]
    assert report["all_reachable_states_conserve_capacity"]
    assert report["states"] > report["episode_count"]
    assert report["transitions"] >= report["episode_count"]
    assert report["normal_forms"] >= report["episode_count"]
    assert report["cyclic_components"] == 1
