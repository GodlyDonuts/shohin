from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline.audit_ctaa_board_v2 import V2AuditFailure, audit_long_families, audit_v2
from pipeline.ctaa_board_v2 import build_long_families


def test_revision_two_independent_dry_audit_passes() -> None:
    report = audit_v2(per_class_depth_cell=288)
    assert report["status"] == "pass"
    assert report["atomic_oracle_checks"] == 729
    assert report["closure_execution_checks"] == 19_683
    assert report["development"]["families"] == 13_824
    assert report["confirmation"]["families"] == 13_824
    assert not report["production_seed_generated"]
    assert not report["board_artifact_written"]
    assert not report["jobs_launched"]


def test_revision_two_audit_rejects_class_query_coupling_mutation() -> None:
    families = list(build_long_families(17, "development", per_class_depth_cell=288))
    target = families[0]
    families[0] = replace(target, query_position=(target.query_position + 1) % 3)
    with pytest.raises(V2AuditFailure, match="query/initial balance"):
        audit_long_families(
            families,
            partition="development",
            per_class_depth_cell=288,
        )


def test_revision_two_audit_rejects_repeated_opcode_shortcut_mutation() -> None:
    families = list(build_long_families(19, "development", per_class_depth_cell=288))
    target = families[0]
    active = (target.active[0],) * target.depth
    schedule = (*active, 4, *target.schedule[target.depth + 1 :])
    families[0] = replace(target, schedule=schedule)
    with pytest.raises(V2AuditFailure, match="fewer than three cards"):
        audit_long_families(
            families,
            partition="development",
            per_class_depth_cell=288,
        )
