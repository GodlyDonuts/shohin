from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from ctaa_intervention_protocol import MANDATORY_OPERATIONS, RUNTIME_PANEL_SIZE
from ctaa_run_contract import canonical_json
from ctaa_runtime_execution_projection import (
    EXECUTION_ATTEMPT_SCHEMA,
    EXECUTION_PROJECTION_SCHEMA,
    ExecutionProjectionError,
    make_execution_projection,
    read_execution_projection,
    validate_execution_projection,
    validate_execution_projection_standalone,
    write_execution_projection,
)
from test_ctaa_intervention_protocol import valid_plan


@pytest.fixture(scope="module")
def plan():
    return valid_plan()


@pytest.fixture(scope="module")
def projection(plan):
    return make_execution_projection(plan)


def _recommit(value):
    changed = deepcopy(value)
    changed["attempts_sha256"] = hashlib.sha256(
        canonical_json(changed["attempts"]).encode("ascii")
    ).hexdigest()
    unsigned = {
        key: item for key, item in changed.items() if key != "projection_sha256"
    }
    changed["projection_sha256"] = hashlib.sha256(
        canonical_json(unsigned).encode("ascii")
    ).hexdigest()
    return changed


def _strings(value):
    found = set()
    if isinstance(value, dict):
        for item in value.values():
            found.update(_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_strings(item))
    elif isinstance(value, str):
        found.add(value)
        if value.startswith("{"):
            found.update(_strings(json.loads(value)))
    return found


def test_projection_is_complete_but_query_blind(projection):
    assert projection["schema"] == EXECUTION_PROJECTION_SCHEMA
    assert len(projection["anchors"]) == RUNTIME_PANEL_SIZE
    assert len(projection["attempts"]) == (
        len(MANDATORY_OPERATIONS) - 1
    ) * RUNTIME_PANEL_SIZE
    assert projection["deferred_operation"] == "late_query_swap"
    serialized = json.dumps(projection, sort_keys=True)
    for forbidden in (
        '"answer"',
        '"query_position"',
        '"query_source_sha256"',
        '"resulting_query_source_sha256"',
        '"parent_query_position"',
        '"donor_query_position"',
        '"family_id"',
        '"class_id"',
        '"renderer_index"',
        '"partition"',
        '"padding_mask_sha256"',
        '"midpoint_state_sha256"',
        '"action_card_sha256s"',
    ):
        assert forbidden not in serialized
    assert all(
        set(anchor) == {"anchor_id", "program_source_sha256", "packet_sha256"}
        for anchor in projection["anchors"]
    )
    assert [anchor["anchor_id"] for anchor in projection["anchors"]] == [
        f"oa{index:06d}" for index in range(RUNTIME_PANEL_SIZE)
    ]
    assert all(
        row["schema"] == EXECUTION_ATTEMPT_SCHEMA
        and row["attempt_id"] == f"ot{row['attempt_index']:08d}"
        for row in projection["attempts"]
    )
    assert all(row["operation"] != "late_query_swap" for row in projection["attempts"])
    isolation = [
        row for row in projection["attempts"] if row["operation"] == "query_isolation"
    ]
    assert len(isolation) == RUNTIME_PANEL_SIZE
    assert all(row["mutation_payload_json"] is None for row in isolation)


def test_projection_is_exactly_recomputed(projection, plan):
    assert validate_execution_projection_standalone(projection) == projection
    assert validate_execution_projection(projection, plan) == projection
    changed = deepcopy(projection)
    changed["attempts"][0]["resulting_packet_sha256"] = "f" * 64
    with pytest.raises(ExecutionProjectionError, match="commitment|frozen plan"):
        validate_execution_projection(changed, plan)


def test_original_family_and_anchor_ids_never_cross_projection(projection, plan):
    projected_strings = _strings(projection)
    forbidden = {
        *(anchor.anchor_id for anchor in plan.anchors),
        *(anchor.family_id for anchor in plan.anchors),
        *(attempt.attempt_id for attempt in plan.attempts),
    }
    assert projected_strings.isdisjoint(forbidden)


def test_payload_identifiers_are_opaque_and_authority_derived(projection):
    for row in projection["attempts"]:
        assert row["anchor_id"].startswith("oa")
        if row["donor_anchor_id"] is not None:
            assert row["donor_anchor_id"].startswith("oa")
        payload = row["mutation_payload_json"]
        if payload is None:
            continue
        decoded = json.loads(payload)
        assert decoded["anchor_id"] == row["anchor_id"]
        if "donor_anchor_id" in decoded:
            assert decoded["donor_anchor_id"] == row["donor_anchor_id"]
        if row["operation"] == "renderer_substitution" and "parent_renderer" in decoded:
            assert 0 <= decoded["parent_renderer"] <= 31
            assert 0 <= decoded["target_renderer"] <= 31


def test_projection_rejects_query_field(projection, plan):
    changed = deepcopy(projection)
    changed["anchors"][0]["query_position"] = 0
    with pytest.raises(ExecutionProjectionError, match="leaks query field"):
        validate_execution_projection(changed, plan)


@pytest.mark.parametrize("field", ["family_id", "renderer_index", "partition"])
def test_projection_rejects_removed_query_correlated_fields(projection, field):
    changed = deepcopy(projection)
    changed["anchors"][0][field] = "Dhhh00000000" if field == "family_id" else 0
    with pytest.raises(ExecutionProjectionError, match="leaks query field"):
        validate_execution_projection_standalone(changed)


def test_projection_rejects_literal_query_text_anywhere(projection):
    changed = deepcopy(projection)
    changed["arm_id"] = "READ THE FIRST CELL."
    with pytest.raises(ExecutionProjectionError, match="literal query"):
        validate_execution_projection_standalone(changed)


def test_standalone_projection_rejects_nonopaque_identifier(projection):
    changed = deepcopy(projection)
    original = changed["anchors"][0]["anchor_id"]
    changed["anchors"][0]["anchor_id"] = "family-correlated-id"
    changed["batch_order"] = [
        "family-correlated-id" if item == original else item
        for item in changed["batch_order"]
    ]
    changed["batch_order_sha256"] = hashlib.sha256(
        canonical_json(changed["batch_order"]).encode("ascii")
    ).hexdigest()
    changed = _recommit(changed)
    with pytest.raises(ExecutionProjectionError, match="anchor differs"):
        validate_execution_projection_standalone(changed)


def test_projection_round_trip_is_canonical(tmp_path: Path, plan):
    path = tmp_path / "execution_projection.json"
    digest = write_execution_projection(path, plan)
    assert len(digest) == 64
    assert not path.stat().st_mode & 0o222
    assert read_execution_projection(path, plan) == make_execution_projection(plan)
    assert read_execution_projection(path) == make_execution_projection(plan)


def test_standalone_projection_rejects_reordered_attempts(projection):
    changed = deepcopy(projection)
    changed["attempts"][0], changed["attempts"][1] = (
        changed["attempts"][1],
        changed["attempts"][0],
    )
    with pytest.raises(ExecutionProjectionError, match="attempt order|commitment"):
        validate_execution_projection_standalone(changed)


def test_projection_reader_rejects_writable_file(tmp_path: Path, plan):
    path = tmp_path / "execution_projection.json"
    write_execution_projection(path, plan)
    path.chmod(0o644)
    with pytest.raises(ExecutionProjectionError, match="not immutable"):
        read_execution_projection(path, plan)
