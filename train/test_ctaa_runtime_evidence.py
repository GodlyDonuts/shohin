from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

import pytest

from ctaa_intervention_protocol import plan_to_dict
from ctaa_runtime_evidence import (
    ATTEMPT_SCHEMA,
    BLOB_SCHEMA,
    EVIDENCE_SCHEMA,
    EXPECTED_ATTEMPT_COUNT,
    RuntimeEvidenceBuilder,
    RuntimeEvidenceError,
    make_custody_receipts,
    make_raw_tensor,
    read_runtime_evidence,
    validate_runtime_evidence,
    write_runtime_evidence,
)
from test_ctaa_intervention_protocol import plan_with_anchors, valid_plan


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def tensor_materials(names: set[str], seed: int) -> dict[str, dict[str, object]]:
    categorical = seed % 3
    materials = {
        "packet": make_raw_tensor("uint8", [56], bytes(range(56))),
        "h19_residual": make_raw_tensor(
            "float32", [1, 576], struct.pack("<576f", *([float(seed)] * 576))
        ),
        "h29_residual": make_raw_tensor(
            "float32", [1, 576], struct.pack("<576f", *([float(seed + 1)] * 576))
        ),
        "state_route": make_raw_tensor("uint8", [42, 3], bytes([categorical] * 126)),
        "composed_route": make_raw_tensor("uint8", [42, 3], bytes([categorical] * 126)),
        "halt_mask": make_raw_tensor("bool", [42], bytes([0] * 42)),
        "terminal_state": make_raw_tensor("uint8", [3], bytes([categorical] * 3)),
        "query_position": make_raw_tensor("uint8", [1], bytes([seed % 3])),
        "answer": make_raw_tensor("uint8", [1], bytes([categorical])),
    }
    return {name: materials[name] for name in names}


@pytest.fixture(scope="module")
def plan():
    base = valid_plan()
    first_id = base.bindings.batch_order[0]
    packet_sha = hashlib.sha256(bytes(range(56))).hexdigest()
    panel = tuple(
        replace(anchor, packet_sha256=packet_sha)
        if anchor.anchor_id == first_id
        else anchor
        for anchor in base.anchors
    )
    return plan_with_anchors(base, panel)


@pytest.fixture(scope="module")
def evidence(plan):
    builder = RuntimeEvidenceBuilder(plan)
    anchor_id = plan.attempts[0].anchor_id
    parent_materials = tensor_materials(
        {
            "packet",
            "h19_residual",
            "h29_residual",
            "state_route",
            "composed_route",
            "halt_mask",
            "terminal_state",
            "query_position",
            "answer",
        },
        1,
    )
    parent_materials["query_position"] = make_raw_tensor("uint8", [1], b"\x00")
    parent = builder.add_snapshot(
        anchor_id=anchor_id,
        role="parent",
        operation=None,
        tensors=parent_materials,
    )
    intervention = builder.add_snapshot(
        anchor_id=anchor_id,
        role="intervention",
        operation="h19_zero",
        tensors=tensor_materials(
            {
                "h19_residual",
                "state_route",
                "composed_route",
                "halt_mask",
                "terminal_state",
                "query_position",
                "answer",
            },
            0,
        ),
    )
    complete = make_custody_receipts(
        packet_receipt_sha256=digest("packet-receipt"),
        source_deletion_receipt_sha256=digest("source-deletion-receipt"),
        query_isolation_receipt_sha256=digest("query-isolation-receipt"),
        execution_receipt_sha256=digest("execution-receipt"),
    )
    builder.add_success(
        parent_snapshot_sha256=parent,
        intervention_snapshot_sha256=intervention,
        donor_snapshot_sha256=None,
        custody_receipts=complete,
    )
    failure_receipts = make_custody_receipts(
        packet_receipt_sha256=None,
        source_deletion_receipt_sha256=None,
        query_isolation_receipt_sha256=None,
        execution_receipt_sha256=digest("execution-receipt"),
    )
    for _ in range(1, EXPECTED_ATTEMPT_COUNT):
        builder.add_failure(
            failure_stage="execution",
            failure_code="execution_error",
            failure_detail_sha256=digest("retained-runtime-failure"),
            custody_receipts=failure_receipts,
        )
    return builder.build()


def clone(evidence: dict[str, object]) -> dict[str, object]:
    changed = dict(evidence)
    changed["blob_catalog"] = dict(evidence["blob_catalog"])
    changed["snapshot_catalog"] = dict(evidence["snapshot_catalog"])
    changed["attempts"] = list(evidence["attempts"])
    return changed


def recommit_attempt(row: dict[str, object]) -> None:
    row["attempt_sha256"] = canonical_sha256(
        {key: value for key, value in row.items() if key != "attempt_sha256"}
    )


def recommit(value: dict[str, object]) -> None:
    value["blob_count"] = len(value["blob_catalog"])
    value["blob_catalog_sha256"] = canonical_sha256(value["blob_catalog"])
    value["snapshot_count"] = len(value["snapshot_catalog"])
    value["snapshot_catalog_sha256"] = canonical_sha256(value["snapshot_catalog"])
    value["attempts_sha256"] = canonical_sha256(value["attempts"])
    value["evidence_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "evidence_sha256"}
    )


def write_raw(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o400)


def test_exact_content_addressed_sidecar_retains_all_attempts(plan, evidence) -> None:
    validated = validate_runtime_evidence(evidence, plan)
    assert validated["schema"] == EVIDENCE_SCHEMA
    assert validated["scored_row_count"] == 40_608
    assert validated["runtime_panel_size"] == 864
    assert validated["operation_count"] == 26
    assert validated["attempt_count"] == EXPECTED_ATTEMPT_COUNT == 22_464
    assert validated["runtime_attempts_affect_scored_denominator"] is False
    assert validated["oracle_access"] == 0
    assert validated["blob_count"] < 18
    assert validated["snapshot_count"] == 2
    attempts = validated["attempts"]
    assert attempts[0]["schema"] == ATTEMPT_SCHEMA
    assert attempts[0]["status"] == "success"
    assert attempts[-1]["status"] == "failure"
    assert not any("pass" in key for key in attempts[0])
    assert [row["attempt_index"] for row in attempts] == list(
        range(EXPECTED_ATTEMPT_COUNT)
    )


@pytest.mark.parametrize("mutation", ["missing", "mismatched"])
def test_every_attempt_requires_one_execution_receipt(
    plan, evidence, mutation: str
) -> None:
    changed = deepcopy(evidence)
    row = changed["attempts"][-1]
    row["custody_receipts"]["execution_receipt_sha256"] = (
        None if mutation == "missing" else digest("substituted-execution-receipt")
    )
    recommit_attempt(row)
    recommit(changed)
    message = "execution_receipt_sha256|one execution receipt"
    with pytest.raises(RuntimeEvidenceError, match=message):
        validate_runtime_evidence(changed, plan)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "reordered", "substituted"]
)
def test_full_attempt_coverage_and_order_fail_closed(
    plan, evidence, mutation: str
) -> None:
    changed = clone(evidence)
    attempts = changed["attempts"]
    if mutation == "missing":
        attempts.pop()
    elif mutation == "duplicate":
        attempts[1] = attempts[0]
    elif mutation == "reordered":
        attempts[0], attempts[1] = attempts[1], attempts[0]
    else:
        replacement = dict(attempts[1])
        replacement["attempt_index"] = 0
        recommit_attempt(replacement)
        attempts[0] = replacement
    recommit(changed)
    with pytest.raises(RuntimeEvidenceError, match="attempt"):
        validate_runtime_evidence(changed, plan)


def test_reference_substitution_is_rejected_even_when_recommitted(
    plan, evidence
) -> None:
    changed = clone(evidence)
    first = dict(changed["attempts"][0])
    first["parent_snapshot_sha256"] = first["intervention_snapshot_sha256"]
    recommit_attempt(first)
    changed["attempts"][0] = first
    recommit(changed)
    with pytest.raises(RuntimeEvidenceError, match="snapshot reference binding"):
        validate_runtime_evidence(changed, plan)


def test_recommitted_noop_residual_intervention_is_rejected(plan, evidence) -> None:
    changed = deepcopy(evidence)
    first = changed["attempts"][0]
    parent = changed["snapshot_catalog"][first["parent_snapshot_sha256"]]
    old_key = first["intervention_snapshot_sha256"]
    intervention = changed["snapshot_catalog"].pop(old_key)
    old_blob = intervention["tensor_refs"]["h19_residual"]["blob_sha256"]
    intervention["tensor_refs"]["h19_residual"] = deepcopy(
        parent["tensor_refs"]["h19_residual"]
    )
    without_hash = {
        key: value for key, value in intervention.items() if key != "snapshot_sha256"
    }
    new_key = canonical_sha256(without_hash)
    intervention["snapshot_sha256"] = new_key
    changed["snapshot_catalog"][new_key] = intervention
    first["intervention_snapshot_sha256"] = new_key
    recommit_attempt(first)
    changed["blob_catalog"].pop(old_blob)
    recommit(changed)
    with pytest.raises(RuntimeEvidenceError, match="residual-zero is a no-op"):
        validate_runtime_evidence(changed, plan)


def test_missing_blob_is_rejected(plan, evidence) -> None:
    changed = clone(evidence)
    changed["blob_catalog"].pop(next(iter(changed["blob_catalog"])))
    recommit(changed)
    with pytest.raises(RuntimeEvidenceError, match="missing blob"):
        validate_runtime_evidence(changed, plan)


def test_unused_blob_is_rejected(plan, evidence) -> None:
    changed = clone(evidence)
    data = b"unused-content-addressed-bytes"
    changed["blob_catalog"][hashlib.sha256(data).hexdigest()] = {
        "schema": BLOB_SCHEMA,
        "encoding": "hex",
        "byte_length": len(data),
        "data_hex": data.hex(),
    }
    recommit(changed)
    with pytest.raises(RuntimeEvidenceError, match="unused entries"):
        validate_runtime_evidence(changed, plan)


def test_duplicate_blob_content_under_substituted_key_is_rejected(
    plan, evidence
) -> None:
    changed = clone(evidence)
    original = next(iter(changed["blob_catalog"].values()))
    changed["blob_catalog"][digest("alternate-key")] = deepcopy(original)
    recommit(changed)
    with pytest.raises(RuntimeEvidenceError, match="content-address"):
        validate_runtime_evidence(changed, plan)


def test_wrong_operation_specific_tensor_set_is_rejected(plan, evidence) -> None:
    changed = clone(evidence)
    first = dict(changed["attempts"][0])
    old_key = first["intervention_snapshot_sha256"]
    snapshot = deepcopy(changed["snapshot_catalog"].pop(old_key))
    parent = changed["snapshot_catalog"][first["parent_snapshot_sha256"]]
    snapshot["tensor_refs"]["packet"] = parent["tensor_refs"]["packet"]
    without_hash = {
        key: item for key, item in snapshot.items() if key != "snapshot_sha256"
    }
    new_key = canonical_sha256(without_hash)
    snapshot["snapshot_sha256"] = new_key
    changed["snapshot_catalog"][new_key] = snapshot
    first["intervention_snapshot_sha256"] = new_key
    recommit_attempt(first)
    changed["attempts"][0] = first
    recommit(changed)
    with pytest.raises(RuntimeEvidenceError, match="operation-specific tensor set"):
        validate_runtime_evidence(changed, plan)


def test_plan_mapping_is_consumed_by_frozen_validator(plan, evidence) -> None:
    assert validate_runtime_evidence(evidence, plan_to_dict(plan)) == evidence


def test_system_tensor_and_attempt_hash_mismatches_are_rejected(plan, evidence) -> None:
    changed = clone(evidence)
    changed["compiler_sha256"] = digest("substituted-compiler")
    recommit(changed)
    with pytest.raises(RuntimeEvidenceError, match="compiler_sha256"):
        validate_runtime_evidence(changed, plan)

    changed = clone(evidence)
    snapshot_key = next(iter(changed["snapshot_catalog"]))
    snapshot = deepcopy(changed["snapshot_catalog"][snapshot_key])
    tensor_name = next(iter(snapshot["tensor_refs"]))
    snapshot["tensor_refs"][tensor_name]["tensor_sha256"] = digest("wrong-tensor")
    changed["snapshot_catalog"][snapshot_key] = snapshot
    recommit(changed)
    with pytest.raises(RuntimeEvidenceError, match="tensor-ref hash"):
        validate_runtime_evidence(changed, plan)

    changed = clone(evidence)
    first = dict(changed["attempts"][0])
    first["attempt_sha256"] = digest("wrong-attempt")
    changed["attempts"][0] = first
    recommit(changed)
    with pytest.raises(RuntimeEvidenceError, match="attempt commitment"):
        validate_runtime_evidence(changed, plan)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("scored_row_count", 40_609),
        ("runtime_attempts_affect_scored_denominator", True),
        ("oracle_access", 1),
    ],
)
def test_denominator_and_oracle_tampering_fail_closed(
    plan, evidence, key: str, value: object
) -> None:
    changed = clone(evidence)
    changed[key] = value
    recommit(changed)
    with pytest.raises(RuntimeEvidenceError, match=key):
        validate_runtime_evidence(changed, plan)


@pytest.mark.parametrize("level", ["top", "attempt", "snapshot", "tensor"])
def test_unknown_fields_are_rejected(plan, evidence, level: str) -> None:
    changed = clone(evidence)
    if level == "top":
        changed["unknown"] = 1
    elif level == "attempt":
        first = dict(changed["attempts"][0])
        first["unknown"] = 1
        changed["attempts"][0] = first
    else:
        first_key = next(iter(changed["snapshot_catalog"]))
        snapshot = deepcopy(changed["snapshot_catalog"][first_key])
        if level == "snapshot":
            snapshot["unknown"] = 1
        else:
            snapshot["tensor_refs"][next(iter(snapshot["tensor_refs"]))]["unknown"] = 1
        changed["snapshot_catalog"][first_key] = snapshot
    with pytest.raises(RuntimeEvidenceError, match="schema"):
        validate_runtime_evidence(changed, plan)


def test_nonfinite_binary_tensor_is_rejected() -> None:
    with pytest.raises(RuntimeEvidenceError, match="non-finite"):
        make_raw_tensor("float32", [1], struct.pack("<f", float("inf")))


def test_canonical_immutable_round_trip(tmp_path: Path, plan, evidence) -> None:
    output = tmp_path / "runtime-evidence.json"
    file_sha = write_runtime_evidence(
        output,
        plan,
        evidence["attempts"],
        evidence["blob_catalog"],
        evidence["snapshot_catalog"],
    )
    metadata = output.lstat()
    assert stat.S_ISREG(metadata.st_mode)
    assert metadata.st_mode & 0o777 == 0o444
    assert metadata.st_nlink == 1
    assert hashlib.sha256(output.read_bytes()).hexdigest() == file_sha
    assert (
        read_runtime_evidence(output, plan, expected_file_sha256=file_sha) == evidence
    )
    with pytest.raises(RuntimeEvidenceError, match="file hash"):
        read_runtime_evidence(output, plan, expected_file_sha256=digest("wrong"))


def test_writable_symlink_and_hardlink_inputs_are_rejected(
    tmp_path: Path, plan, evidence
) -> None:
    payload = (
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    writable = tmp_path / "writable.json"
    writable.write_bytes(payload)
    writable.chmod(0o600)
    with pytest.raises(RuntimeEvidenceError, match="single-link immutable"):
        read_runtime_evidence(writable, plan)
    immutable = tmp_path / "immutable.json"
    write_raw(immutable, payload)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(immutable)
    with pytest.raises(RuntimeEvidenceError, match="single-link immutable"):
        read_runtime_evidence(symlink, plan)
    hardlink = tmp_path / "hardlink.json"
    os.link(immutable, hardlink)
    with pytest.raises(RuntimeEvidenceError, match="single-link immutable"):
        read_runtime_evidence(immutable, plan)
    with pytest.raises(RuntimeEvidenceError, match="single-link immutable"):
        read_runtime_evidence(hardlink, plan)


@pytest.mark.parametrize("existing_kind", ["writable", "symlink", "hardlink"])
def test_writer_refuses_existing_output_forms(
    tmp_path: Path, plan, evidence, existing_kind: str
) -> None:
    output = tmp_path / f"existing-{existing_kind}.json"
    if existing_kind == "writable":
        output.write_text("occupied")
    elif existing_kind == "symlink":
        target = tmp_path / "target.json"
        target.write_text("occupied")
        output.symlink_to(target)
    else:
        target = tmp_path / "target.json"
        target.write_text("occupied")
        os.link(target, output)
    with pytest.raises(FileExistsError, match="existing"):
        write_runtime_evidence(
            output,
            plan,
            evidence["attempts"],
            evidence["blob_catalog"],
            evidence["snapshot_catalog"],
        )


@pytest.mark.parametrize(
    "payload",
    [b'{"schema":"one","schema":"two"}\n', b'{"value":NaN}\n'],
)
def test_duplicate_and_nonfinite_json_inputs_are_rejected(
    tmp_path: Path, plan, payload: bytes
) -> None:
    path = tmp_path / f"bad-{hashlib.sha256(payload).hexdigest()}.json"
    write_raw(path, payload)
    with pytest.raises(RuntimeEvidenceError, match="duplicate|non-finite"):
        read_runtime_evidence(path, plan)


def test_noncanonical_json_is_rejected(tmp_path: Path, plan, evidence) -> None:
    path = tmp_path / "pretty.json"
    write_raw(path, json.dumps(evidence, indent=2).encode("ascii"))
    with pytest.raises(RuntimeEvidenceError, match="not canonical"):
        read_runtime_evidence(path, plan)


def test_incomplete_builder_and_invalid_failure_contract_fail(plan) -> None:
    builder = RuntimeEvidenceBuilder(plan)
    with pytest.raises(RuntimeEvidenceError, match="attempt count"):
        builder.build()
    with pytest.raises(RuntimeEvidenceError, match="stage/code"):
        builder.add_failure(
            failure_stage="execution",
            failure_code="unknown",
            failure_detail_sha256=digest("detail"),
        )
