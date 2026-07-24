from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json

import pytest

import pipeline.audit_episode_functor_identifiable_board as audit_module
from pipeline.audit_episode_functor_identifiable_board import (
    _action_overlap_receipt,
    _candidate_projection_receipt,
    _qualification_supervisor_receipt,
    audit_identifiable_pilot,
    write_report,
)
from pipeline.episode_functor_identifiable_board import (
    FAMILIES,
    generate_pilot_rows,
)


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(payload).hexdigest()


def test_atomic_identifiable_pilot_audit_is_deterministic(tmp_path) -> None:
    counts = {
        "confirmation": 3,
        "development": 4,
        "mechanics": 6,
        "train": 8,
    }
    left = audit_identifiable_pilot(
        seed="efc-identifiable-audit-test-v1",
        counts=counts,
    )
    right = audit_identifiable_pilot(
        seed="efc-identifiable-audit-test-v1",
        counts=counts,
    )
    assert left == right
    assert left["latent_world_count"] == 21
    assert left["row_count"] == 8 * 4 + 6 * 8 + 4 * 3 + 3
    assert left["maximum_version_space"] == 1
    assert left["maximum_behavior_classes"] == 1
    assert left["direct_reference_exact"] == left["row_count"]
    assert left["query_renderer_exact"] == 8 * left["row_count"]
    assert left["same_bag_changed_order_rows"] == left["row_count"]
    assert left["unique_canonical_worlds"] == 21
    assert not any(left["split_structural_overlaps"].values())
    assert left["schema"] == "efc-identifiable-cpu-audit-v3"
    assert left["decision"] == "cpu_qualification_candidate_neural_fit_no_go"

    family_audit = left["action_only_family_audit"]
    expected_pairs = {
        f"{left_family}__{right_family}"
        for left_index, left_family in enumerate(FAMILIES)
        for right_family in FAMILIES[left_index + 1 :]
    }
    assert family_audit["family_pair_count"] == len(expected_pairs) == 15
    assert set(family_audit["cross_family_overlap_counts"]) == expected_pairs
    assert not any(family_audit["cross_family_overlap_counts"].values())
    assert set(family_audit["canonical_orbit_count_by_family"]) == set(FAMILIES)
    assert all(family_audit["canonical_orbit_count_by_family"].values())

    candidate = left["candidate_projection"]
    assert candidate["fields"] == ["source"]
    assert candidate["forbidden_attribute_hits"] == 0
    assert candidate["counts_by_split"] == {
        "confirmation": 3,
        "development": 12,
        "mechanics": 48,
        "train": 32,
    }
    supervisor = left["qualification_supervisor"]
    assert supervisor["row_count"] == left["row_count"]
    assert supervisor["candidate_join_key"] == "source_sha256"
    assert supervisor["supervisor_fields_exposed_to_candidate"] == 0
    assert supervisor["answer_label_count"] == 14 * left["row_count"]
    assert supervisor["occurrence_label_count"] == 99 * left["row_count"]
    assert supervisor["record_label_count"] == (
        left["row_count"]
        * 45
        + 2
        * (
            2 * counts["train"]
            + 4 * counts["mechanics"]
            + 2 * counts["development"]
        )
    )
    assert len(supervisor["label_manifest_sha256"]) == 64
    tokenizer = left["trunk_tokenizer"]
    assert tokenizer["exact_byte_coverage_rows"] == left["row_count"]
    assert tokenizer["minimum_tokens"] > 0
    assert tokenizer["maximum_tokens"] >= tokenizer["minimum_tokens"]
    assert tokenizer["parent_context_tokens"] == 2_048
    assert tokenizer["disconnected_window_semantics"]
    assert len(tokenizer["tokenizer_sha256"]) == 64

    structural = left["row_structural_receipt"]
    manifest = structural["manifest"]
    assert len(manifest) == left["row_count"]
    split_receipts: dict[str, list[str]] = defaultdict(list)
    world_receipts: dict[str, list[str]] = defaultdict(list)
    candidate_source_hashes: dict[str, list[str]] = defaultdict(list)
    for row_index, entry in enumerate(manifest):
        assert entry["row_index"] == row_index
        payload = dict(entry)
        row_receipt = payload.pop("row_receipt_sha256")
        assert row_receipt == _canonical_json_sha256(payload)
        split_receipts[entry["split"]].append(row_receipt)
        world_receipts[entry["world_id"]].append(row_receipt)
        candidate_source_hashes[entry["split"]].append(entry["source_sha256"])
    assert structural["manifest_sha256"] == _canonical_json_sha256(manifest)
    assert structural["source_manifest_sha256"] == _canonical_json_sha256(
        [
            {
                "row_index": entry["row_index"],
                "source_sha256": entry["source_sha256"],
            }
            for entry in manifest
        ]
    )
    assert structural["split_manifest_sha256"] == {
        split: _canonical_json_sha256(receipts)
        for split, receipts in sorted(split_receipts.items())
    }
    assert structural["world_manifest_sha256"] == {
        world_id: _canonical_json_sha256(receipts)
        for world_id, receipts in sorted(world_receipts.items())
    }
    assert candidate["payload_manifest_sha256_by_split"] == {
        split: _canonical_json_sha256(source_hashes)
        for split, source_hashes in sorted(candidate_source_hashes.items())
    }

    output = tmp_path / "report.json"
    write_report(output, left)
    assert json.loads(output.read_text()) == left


def test_hostile_reporting_gates_fail_closed(monkeypatch) -> None:
    counts = {
        "confirmation": 1,
        "development": 1,
        "mechanics": 6,
        "train": 3,
    }
    rows = generate_pilot_rows(
        seed="efc-identifiable-hostile-report-test-v1",
        counts=counts,
    )

    monkeypatch.setattr(
        audit_module,
        "canonical_action_bytes",
        lambda transitions: b"forced-cross-family-orbit",
    )
    with pytest.raises(
        ValueError,
        match="action-only canonical orbit crosses families",
    ):
        _action_overlap_receipt(rows)

    monkeypatch.undo()

    @dataclass(frozen=True, slots=True)
    class PoisonedCandidate:
        source: bytes
        machine: object

    def poisoned_projection(rows, *, split):
        return tuple(
            PoisonedCandidate(row.source, row.machine)
            for row in rows
            if row.split == split
        )

    monkeypatch.setattr(
        audit_module,
        "project_candidate_sources",
        poisoned_projection,
    )
    with pytest.raises(
        ValueError,
        match="exposes fields other than source",
    ):
        _candidate_projection_receipt(
            rows,
            splits=("confirmation", "development", "mechanics", "train"),
        )

    monkeypatch.undo()
    duplicate_rows = (rows[0], rows[0])
    with pytest.raises(ValueError, match="duplicated"):
        _qualification_supervisor_receipt(duplicate_rows)
