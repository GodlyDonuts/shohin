from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import relation_complete_transport_falsifier as rct


def test_relation_complete_action_enumeration() -> None:
    candidates = rct.relation_complete_actions()
    assert len(rct.involutions()) == 76
    assert len(candidates) == 120
    assert len(set(candidates)) == 120
    assert set(candidates) == set(rct.regular_action_relabelings())
    assert rct.canonical_table() in candidates
    assert all(not rct.relation_violations(table) for table in candidates)
    assert all(len(rct.orbit(table)) == 6 for table in candidates)


def test_one_erasure_is_uniquely_completed_by_global_relations() -> None:
    result = rct.one_erasure_certificate(rct.canonical_table())
    assert result["unique_exact_completion"]
    assert result["relation_consistent_successors"] == [result["exact_successor"]]
    assert result["identity_only_checks_are_insufficient"]
    assert all(result["wrong_patch_identity_only_relations"].values())
    assert result["wrong_patch_global_violation_count"] > 0


def test_version_space_derives_minimum_four_edge_identification() -> None:
    result = rct.version_space_certificate(
        rct.canonical_table(), rct.relation_complete_actions()
    )
    assert result["edge_subsets_checked"] == 4096
    assert result["minimum_identifying_edges"] == 4
    assert result["minimum_identifying_subset_count"] > 0
    assert result["profile"][0]["minimum_candidates"] == 120
    assert result["profile"][3]["minimum_candidates"] == 2
    assert result["profile"][4]["minimum_candidates"] == 1


def test_report_resource_and_claim_boundaries() -> None:
    report = rct.build_report()
    assert report["all_pass"]
    assert report["globally_relation_complete_transitive_actions"] == 120
    ledger = report["resource_ledger"]
    assert ledger["unconstrained_tables"] == 6**12
    assert ledger["full_atlas_labeled_target_bits"] == 36
    assert ledger["minimum_relation_complete_anchor_bits"] == 12
    assert ledger["presentation_bytes"] > 0
    assert report["equivalence"]["primitive_novelty_authorized"] is False
    assert report["neural_preregistration_drafting_authorized"] is False


def test_cli_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    script = Path(rct.__file__)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    subprocess.run(
        [sys.executable, str(script), "--out", str(first)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, str(script), "--out", str(second)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text())["all_pass"] is True
    rejected = subprocess.run(
        [sys.executable, str(script), "--out", str(first)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "refusing to overwrite" in rejected.stderr
