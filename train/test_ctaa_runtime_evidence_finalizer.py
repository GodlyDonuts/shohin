from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest
import torch

import ctaa_runtime_evidence_finalizer as finalizer
from ctaa_intervention_protocol import GateFamily, InterventionFamily
from ctaa_runtime_execution_artifact import RuntimeExecutionArtifactIndex
from ctaa_runtime_execution_engine import (
    AttemptExecutionRecord,
    ExecutionFailure,
    ExecutionSnapshot,
    ParentExecutionRecord,
    RuntimeExecutionResult,
)
from ctaa_trunk_compiler import HardCTAAPacket
from test_ctaa_intervention_protocol import valid_plan


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _parent(anchor_id: str, *, success: bool = False) -> ParentExecutionRecord:
    snapshot = _snapshot() if success else None
    return ParentExecutionRecord(
        "parent",
        anchor_id,
        "success" if success else "failure",
        _digest(f"source:{anchor_id}"),
        _digest(f"packet:{anchor_id}"),
        snapshot,
        None if success else ExecutionFailure("compile", "program_compile_failed"),
        _digest(f"parent-record:{anchor_id}"),
    )


def _attempt_record(
    *,
    attempt_index: int,
    attempt_id: str,
    operation: str,
    anchor_id: str,
    donor_anchor_id: str | None = None,
    success: bool = False,
) -> AttemptExecutionRecord:
    return AttemptExecutionRecord(
        "attempt",
        attempt_index,
        attempt_id,
        operation,
        anchor_id,
        donor_anchor_id,
        "success" if success else "failure",
        _digest(f"parent-record:{anchor_id}"),
        None,
        None,
        None,
        None,
        _snapshot() if success else None,
        (),
        None
        if success
        else ExecutionFailure("execution", "unexpected_runtime_failure"),
        _digest(f"attempt-record:{attempt_id}"),
    )


def _snapshot() -> ExecutionSnapshot:
    cards = torch.tensor(
        [[(0, 1, 2), (2, 1, 0), (1, 2, 0), (0, 0, 0)]],
        dtype=torch.uint8,
    )
    initial = torch.tensor([[0, 1, 2]], dtype=torch.uint8)
    schedule = torch.tensor([[0, 1, 2, 4] + [0] * 37], dtype=torch.uint8)
    packet = HardCTAAPacket(
        cards,
        initial,
        schedule,
        torch.arange(4, dtype=torch.uint8)[None],
    )
    route = torch.tensor([[0, 1, 2]] * 42, dtype=torch.uint8)
    return ExecutionSnapshot(
        "snapshot",
        packet,
        torch.zeros((1, 576), dtype=torch.float32),
        torch.ones((1, 576), dtype=torch.float32),
        route,
        route.clone(),
        torch.zeros(42, dtype=torch.bool),
        route[-1].clone(),
        (),
        _digest("snapshot"),
    )


def _tiny_mapping(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(finalizer, "RUNTIME_PANEL_SIZE", 2)
    monkeypatch.setattr(finalizer, "EXPECTED_PREQUERY_ATTEMPT_COUNT", 4)
    monkeypatch.setattr(finalizer, "EXPECTED_FINAL_ATTEMPT_COUNT", 6)
    anchors = tuple(
        SimpleNamespace(
            anchor_id=f"full-{index}",
            program_source_sha256=_digest(f"source:oa{index:06d}"),
            packet_sha256=_digest(f"packet:oa{index:06d}"),
        )
        for index in range(2)
    )
    operations = (
        GateFamily.SOURCE_DELETION.value,
        GateFamily.QUERY_ISOLATION.value,
    )
    attempts = []
    projected = []
    records = []
    signed = []
    artifacts = []
    for operation_index, operation in enumerate(operations):
        for row_index in range(2):
            attempt_index = (0 if operation_index == 0 else 4) + row_index
            opaque_anchor = f"oa{row_index:06d}"
            opaque_attempt = f"ot{attempt_index:08d}"
            full = SimpleNamespace(
                attempt_index=attempt_index,
                attempt_id=f"full-attempt-{attempt_index}",
                attempt_plan_sha256=_digest(f"plan:{attempt_index}"),
                operation=operation,
                operation_sha256=_digest(f"operation:{operation}"),
                anchor_id=f"full-{row_index}",
                donor_anchor_id=None,
                resulting_program_source_sha256=None,
                resulting_packet_sha256=None,
            )
            attempts.append(full)
            row = {
                "attempt_index": attempt_index,
                "attempt_id": opaque_attempt,
                "attempt_plan_sha256": full.attempt_plan_sha256,
                "operation": operation,
                "operation_sha256": full.operation_sha256,
                "anchor_id": opaque_anchor,
                "donor_anchor_id": None,
                "resulting_program_source_sha256": None,
                "resulting_packet_sha256": None,
            }
            projected.append(row)
            record = _attempt_record(
                attempt_index=attempt_index,
                attempt_id=opaque_attempt,
                operation=operation,
                anchor_id=opaque_anchor,
            )
            records.append(record)
            artifact = _digest(f"artifact:{attempt_index}")
            artifacts.append(artifact)
            signed.append(
                {
                    "attempt_index": attempt_index,
                    "attempt_id": opaque_attempt,
                    "operation": operation,
                    "status": "failure",
                    "raw_output_artifact_sha256": artifact,
                }
            )
    late = tuple(
        SimpleNamespace(
            attempt_index=2 + index,
            operation=InterventionFamily.LATE_QUERY_SWAP.value,
        )
        for index in range(2)
    )
    full_attempts = tuple(attempts[:2]) + late + tuple(attempts[2:])
    plan = SimpleNamespace(
        anchors=anchors,
        attempts=full_attempts,
        bindings=SimpleNamespace(batch_order=("full-0", "full-1")),
    )
    projection = {
        "anchors": [
            {
                "anchor_id": f"oa{index:06d}",
                "program_source_sha256": anchor.program_source_sha256,
                "packet_sha256": anchor.packet_sha256,
            }
            for index, anchor in enumerate(anchors)
        ],
        "batch_order": ["oa000000", "oa000001"],
        "attempts": projected,
    }
    parents = tuple(_parent(f"oa{index:06d}") for index in range(2))
    execution = SimpleNamespace(parents=parents, attempts=tuple(records))
    index = RuntimeExecutionArtifactIndex(
        _digest("aggregate"),
        tuple(_digest(f"parent-artifact:{index}") for index in range(2)),
        tuple(artifacts),
        tuple(
            {
                "attempt_id": record.attempt_id,
                "status": record.status,
                "raw_output_artifact_sha256": artifact,
            }
            for record, artifact in zip(records, artifacts)
        ),
    )
    receipt = {
        "payload": {
            "attempts": signed,
            "source_deletion_probe_artifact_sha256s": artifacts[:2],
            "query_isolation_probe_artifact_sha256s": artifacts[2:],
        },
        "receipt_sha256": _digest("receipt"),
    }
    return plan, projection, execution, index, receipt


def test_forged_receipt_stops_before_any_query_materialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []

    def forged(*args, **kwargs):
        events.append("receipt")
        raise ValueError("forged Ed25519 signature")

    def plan_touched(*args, **kwargs):
        events.append("plan")
        raise AssertionError("full plan touched before receipt authentication")

    monkeypatch.setattr(finalizer, "_read_authenticated_receipt", forged)
    monkeypatch.setattr(finalizer, "validate_runtime_intervention_plan", plan_touched)
    with pytest.raises(
        finalizer.RuntimeEvidenceFinalizerError,
        match="signed pre-query execution receipt verification failed",
    ):
        finalizer.make_finalized_runtime_evidence(
            plan=valid_plan(),
            execution_projection_path=tmp_path / "projection",
            execution_aggregate_path=tmp_path / "aggregate",
            execution_artifact_directory=tmp_path / "objects",
            execution_aggregate_sha256=_digest("aggregate"),
            execution_receipt_path=tmp_path / "receipt",
            receipt_verification_key=b"x" * 32,
        )
    assert events == ["receipt"]


@pytest.mark.parametrize("mutation", ["missing", "reordered"])
def test_missing_or_reordered_attempt_artifacts_fail_closed(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    plan, projection, execution, index, receipt = _tiny_mapping(monkeypatch)
    records = list(execution.attempts)
    if mutation == "missing":
        records.pop()
    else:
        records[0], records[1] = records[1], records[0]
    execution = SimpleNamespace(parents=execution.parents, attempts=tuple(records))
    with pytest.raises(finalizer.RuntimeEvidenceFinalizerError):
        finalizer._exact_projection_and_artifact_mapping(
            plan=plan,
            projection=projection,
            execution=execution,
            artifact_index=index,
            receipt=receipt,
        )


def test_signed_artifact_substitution_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, projection, execution, index, receipt = _tiny_mapping(monkeypatch)
    receipt["payload"]["attempts"][0]["raw_output_artifact_sha256"] = _digest(
        "substitution"
    )
    with pytest.raises(
        finalizer.RuntimeEvidenceFinalizerError, match="attempt mapping differs"
    ):
        finalizer._exact_projection_and_artifact_mapping(
            plan=plan,
            projection=projection,
            execution=execution,
            artifact_index=index,
            receipt=receipt,
        )


def test_opaque_donor_substitution_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, projection, execution, index, receipt = _tiny_mapping(monkeypatch)
    projection["attempts"][0]["donor_anchor_id"] = "oa000001"
    records = list(execution.attempts)
    records[0] = replace(records[0], donor_anchor_id="oa000001")
    execution = SimpleNamespace(parents=execution.parents, attempts=tuple(records))
    with pytest.raises(
        finalizer.RuntimeEvidenceFinalizerError, match="attempt mapping differs"
    ):
        finalizer._exact_projection_and_artifact_mapping(
            plan=plan,
            projection=projection,
            execution=execution,
            artifact_index=index,
            receipt=receipt,
        )


def test_attempt_output_index_substitution_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, projection, execution, index, receipt = _tiny_mapping(monkeypatch)
    outputs = list(index.attempt_outputs)
    outputs[0] = {
        **outputs[0],
        "raw_output_artifact_sha256": _digest("substituted-index-output"),
    }
    index = replace(index, attempt_outputs=tuple(outputs))
    with pytest.raises(
        finalizer.RuntimeEvidenceFinalizerError, match="attempt mapping differs"
    ):
        finalizer._exact_projection_and_artifact_mapping(
            plan=plan,
            projection=projection,
            execution=execution,
            artifact_index=index,
            receipt=receipt,
        )


def test_parent_batch_order_is_derived_from_frozen_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, projection, execution, index, receipt = _tiny_mapping(monkeypatch)
    plan.bindings.batch_order = ("full-1", "full-0")
    projection["batch_order"] = ["oa000001", "oa000000"]
    execution = SimpleNamespace(
        parents=tuple(reversed(execution.parents)), attempts=execution.attempts
    )
    index = replace(
        index,
        parent_artifact_sha256s=tuple(reversed(index.parent_artifact_sha256s)),
    )

    prepared = finalizer._exact_projection_and_artifact_mapping(
        plan=plan,
        projection=projection,
        execution=execution,
        artifact_index=index,
        receipt=receipt,
    )

    assert tuple(prepared.parent_by_full_anchor) == ("full-1", "full-0")


def test_successful_probe_requires_observation_commitment() -> None:
    record = _attempt_record(
        attempt_index=0,
        attempt_id="ot00000000",
        operation=GateFamily.SOURCE_DELETION.value,
        anchor_id="oa000000",
        success=True,
    )
    with pytest.raises(
        finalizer.RuntimeEvidenceFinalizerError,
        match="lacks an observation commitment",
    ):
        finalizer._validate_extra_artifacts(record)


def test_unrecognized_producer_failure_is_rejected() -> None:
    with pytest.raises(
        finalizer.RuntimeEvidenceFinalizerError,
        match="producer-defined failure",
    ):
        finalizer._validate_failure_origin(
            ExecutionFailure("assessment", "producer_says_pass"),
            operation=GateFamily.ROUTE_AGREEMENT.value,
        )


def test_authenticated_receipt_uses_one_held_fd_snapshot(tmp_path: Path) -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    payload = {
        "schema": finalizer.EXECUTION_RECEIPT_SCHEMA,
        "signing_public_key": public_raw.hex(),
        "oracle_access_count": 0,
    }
    signature = private_key.sign(
        finalizer.canonical_json(payload).encode("ascii")
    ).hex()
    record = {
        "payload": payload,
        "signature": signature,
        "receipt_sha256": hashlib.sha256(
            finalizer.canonical_json(
                {"payload": payload, "signature": signature}
            ).encode("ascii")
        ).hexdigest(),
    }
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="ascii",
    )
    path.chmod(0o400)

    assert finalizer._read_authenticated_receipt(path, public_key) == record


def test_authenticated_receipt_rejects_deferred_query_material(tmp_path: Path) -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    payload = {
        "signing_public_key": public_raw.hex(),
        "query_position": 1,
    }
    signature = private_key.sign(
        finalizer.canonical_json(payload).encode("ascii")
    ).hex()
    record = {
        "payload": payload,
        "signature": signature,
        "receipt_sha256": hashlib.sha256(
            finalizer.canonical_json(
                {"payload": payload, "signature": signature}
            ).encode("ascii")
        ).hexdigest(),
    }
    path = tmp_path / "leaking-receipt.json"
    path.write_text(finalizer.canonical_json(record) + "\n", encoding="ascii")
    path.chmod(0o400)

    with pytest.raises(finalizer.RuntimeEvidenceFinalizerError, match="leaks field"):
        finalizer._read_authenticated_receipt(path, public_key)


def _all_failure_prepared(plan):
    parents = {anchor.anchor_id: _parent(anchor.anchor_id) for anchor in plan.anchors}
    parent_artifacts = {
        anchor.anchor_id: _digest(f"parent-artifact:{anchor.anchor_id}")
        for anchor in plan.anchors
    }
    prequery = {}
    prequery_artifacts = {}
    for attempt in plan.attempts:
        if attempt.operation == InterventionFamily.LATE_QUERY_SWAP.value:
            continue
        prequery[attempt.attempt_index] = _attempt_record(
            attempt_index=attempt.attempt_index,
            attempt_id=attempt.attempt_id,
            operation=attempt.operation,
            anchor_id=attempt.anchor_id,
            donor_anchor_id=attempt.donor_anchor_id,
        )
        prequery_artifacts[attempt.attempt_index] = _digest(
            f"attempt-artifact:{attempt.attempt_index}"
        )
    source = {
        anchor.anchor_id: _digest(f"source-probe:{anchor.anchor_id}")
        for anchor in plan.anchors
    }
    query = {
        anchor.anchor_id: _digest(f"query-probe:{anchor.anchor_id}")
        for anchor in plan.anchors
    }
    return finalizer._PreparedCustody(
        plan,
        {},
        RuntimeExecutionResult("", "", _digest("projection"), 0, False, (), (), ""),
        RuntimeExecutionArtifactIndex("", (), (), ()),
        _digest("receipt"),
        {},
        parents,
        parent_artifacts,
        prequery,
        prequery_artifacts,
        source,
        query,
    )


def test_all_failures_remain_in_exact_25056_denominator() -> None:
    plan = valid_plan()
    evidence = finalizer._populate_builder(_all_failure_prepared(plan)).build()
    attempts = evidence["attempts"]
    assert evidence["attempt_count"] == len(attempts) == 25_056
    assert [row["attempt_index"] for row in attempts] == list(range(25_056))
    assert all(row["status"] == "failure" for row in attempts)
    assert (
        sum(
            row["operation"] == InterventionFamily.LATE_QUERY_SWAP.value
            for row in attempts
        )
        == 864
    )
    first = attempts[0]
    assert first["failure_stage"] == "execution"
    assert first["failure_code"] == "execution_error"
    assert first["failure_detail_sha256"] == _digest("attempt-artifact:0")
    for attempt, row in zip(plan.attempts, attempts):
        expected_detail = (
            _digest(f"parent-artifact:{attempt.anchor_id}")
            if attempt.operation == InterventionFamily.LATE_QUERY_SWAP.value
            else _digest(f"attempt-artifact:{attempt.attempt_index}")
        )
        assert row["failure_detail_sha256"] == expected_detail


class _CaptureBuilder:
    def __init__(self, plan) -> None:
        self.plan = plan
        self.cursor = 0
        self.outcomes = []
        self.snapshots = {}

    def add_snapshot(self, **value):
        digest = _digest(f"snapshot:{len(self.snapshots)}")
        self.snapshots[digest] = value
        return digest

    def add_success(self, **value):
        attempt = self.plan.attempts[self.cursor]
        self.outcomes.append((attempt, "success", value))
        self.cursor += 1

    def add_failure(self, **value):
        attempt = self.plan.attempts[self.cursor]
        self.outcomes.append((attempt, "failure", value))
        self.cursor += 1


def test_all_864_late_queries_use_donor_position_and_parent_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = valid_plan()
    anchors = {item.anchor_id: item for item in base.anchors}
    plan = replace(
        base,
        attempts=tuple(
            replace(
                item,
                resulting_query_source_sha256=anchors[
                    item.donor_anchor_id
                ].query_source_sha256,
            )
            if item.operation == InterventionFamily.LATE_QUERY_SWAP.value
            else item
            for item in base.attempts
        ),
    )
    prepared = _all_failure_prepared(plan)
    success_parents = {
        anchor.anchor_id: _parent(anchor.anchor_id, success=True)
        for anchor in plan.anchors
    }
    prepared = replace(prepared, parent_by_full_anchor=success_parents)
    monkeypatch.setattr(finalizer, "RuntimeEvidenceBuilder", _CaptureBuilder)
    builder = finalizer._populate_builder(prepared)
    late = [
        item
        for item in builder.outcomes
        if item[0].operation == InterventionFamily.LATE_QUERY_SWAP.value
    ]
    assert len(late) == 864
    assert all(status == "success" for _, status, _ in late)
    anchors = {item.anchor_id: item for item in plan.anchors}
    for attempt, _, outcome in late:
        snapshot = builder.snapshots[outcome["intervention_snapshot_sha256"]]
        tensors = snapshot["tensors"]
        donor_position = anchors[attempt.donor_anchor_id].query_position
        assert bytes.fromhex(tensors["query_position"]["data_hex"]) == bytes(
            [donor_position]
        )
        terminal = bytes.fromhex(tensors["terminal_state"]["data_hex"])
        answer = bytes.fromhex(tensors["answer"]["data_hex"])
        assert terminal == b"\x00\x01\x02"
        assert answer == bytes([terminal[donor_position]])


def test_late_query_donor_hash_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = valid_plan()
    anchors = {item.anchor_id: item for item in base.anchors}
    plan = replace(
        base,
        attempts=tuple(
            replace(
                item,
                resulting_query_source_sha256=anchors[
                    item.donor_anchor_id
                ].query_source_sha256,
            )
            if item.operation == InterventionFamily.LATE_QUERY_SWAP.value
            else item
            for item in base.attempts
        ),
    )
    attempts = list(plan.attempts)
    late_index = next(
        index
        for index, item in enumerate(attempts)
        if item.operation == InterventionFamily.LATE_QUERY_SWAP.value
    )
    attempts[late_index] = replace(
        attempts[late_index], resulting_query_source_sha256=_digest("wrong-donor")
    )
    changed_plan = replace(plan, attempts=tuple(attempts))
    prepared = _all_failure_prepared(changed_plan)
    success_parents = {
        anchor.anchor_id: _parent(anchor.anchor_id, success=True)
        for anchor in changed_plan.anchors
    }
    prepared = replace(prepared, parent_by_full_anchor=success_parents)
    monkeypatch.setattr(finalizer, "RuntimeEvidenceBuilder", _CaptureBuilder)
    with pytest.raises(
        finalizer.RuntimeEvidenceFinalizerError,
        match="late-query donor source binding differs",
    ):
        finalizer._populate_builder(prepared)


def test_immutable_reader_rejects_writable_and_hardlinked_inputs(
    tmp_path: Path,
) -> None:
    writable = tmp_path / "writable.json"
    writable.write_bytes(b"{}\n")
    with pytest.raises(finalizer.RuntimeEvidenceFinalizerError, match="immutable"):
        finalizer._read_immutable_bytes(writable, maximum_bytes=100)
    writable.chmod(0o400)
    linked = tmp_path / "linked.json"
    linked.hardlink_to(writable)
    with pytest.raises(finalizer.RuntimeEvidenceFinalizerError, match="single-link"):
        finalizer._read_immutable_bytes(writable, maximum_bytes=100)
