from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import ctaa_runtime_execution_artifact as artifact
import ctaa_runtime_execution_engine as engine
import ctaa_runtime_execution_receipt as execution_receipt
from ctaa_intervention_protocol import (
    InterventionFamily,
    LOCKED_SCORED_ROW_COUNT,
    MANDATORY_OPERATIONS,
)
from ctaa_runtime_execution_engine import (
    RUNTIME_EXECUTION_SCHEMA,
    ExecutionFailure,
    RuntimeExecutionResult,
)
from ctaa_runtime_execution_projection import EXECUTION_PROJECTION_SCHEMA
from ctaa_trunk_compiler import HardCTAAPacket, TrunkResidualBundle
from test_ctaa_intervention_protocol import valid_plan


def _digest(value: bytes | str) -> str:
    raw = value if isinstance(value, bytes) else value.encode("ascii")
    return hashlib.sha256(raw).hexdigest()


class _Core(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eval()

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        target = right.gather(1, left)
        logits = torch.full(
            (*target.shape, 3), -20.0, dtype=torch.float32, device=target.device
        )
        return logits.scatter(-1, target[..., None], 20.0)


def _packet(kind: int) -> HardCTAAPacket:
    cards = (
        ((0, 1, 2), (2, 1, 0), (1, 2, 0), (0, 0, 0))
        if kind == 0
        else ((1, 2, 0), (0, 1, 2), (0, 2, 1), (1, 1, 1))
    )
    schedule = [0, 1, 2, 4] + [0] * 37
    return HardCTAAPacket(
        torch.tensor([cards], dtype=torch.uint8),
        torch.tensor([[kind, (kind + 1) % 3, (kind + 2) % 3]], dtype=torch.uint8),
        torch.tensor([schedule], dtype=torch.uint8),
    )


def _result() -> RuntimeExecutionResult:
    core = _Core()
    parents = []
    for index, anchor_id in enumerate(("anchor-a", "anchor-b")):
        bundle = TrunkResidualBundle(
            torch.arange(6, dtype=torch.float32).reshape(1, 2, 3) + index,
            torch.arange(6, dtype=torch.float32).reshape(1, 2, 3) + index + 10,
            torch.ones(1, 2, dtype=torch.bool),
        )
        snapshot = engine._make_snapshot(
            packet=_packet(index), bundle=bundle, core=core
        )
        parents.append(
            engine._make_parent_record(
                anchor_id=anchor_id,
                program_source_sha256=_digest(f"source:{anchor_id}"),
                expected_packet_sha256=dict(snapshot.artifact_hashes)["packet"],
                snapshot=snapshot,
                failure=None,
            )
        )
    parent_by_id = {record.anchor_id: record for record in parents}
    attempts = []
    full_operation_index = {
        operation: index for index, operation in enumerate(MANDATORY_OPERATIONS)
    }
    operations = [
        operation
        for operation in MANDATORY_OPERATIONS
        if operation != InterventionFamily.LATE_QUERY_SWAP.value
    ]
    for operation in operations:
        for row_index, anchor_id in enumerate(("anchor-a", "anchor-b")):
            index = full_operation_index[operation] * 2 + row_index
            parent = parent_by_id[anchor_id]
            failure = None
            snapshot = parent.snapshot
            extras: tuple[tuple[str, str], ...] = ()
            if operation == InterventionFamily.ENTITY_RECODE.value and row_index == 0:
                failure = ExecutionFailure("compile", "synthetic_failure")
                snapshot = None
            elif operation == "query_isolation" and row_index == 1:
                failure = ExecutionFailure("custody", "denied")
                extras = (("custody_probe", _digest("query-probe")),)
            elif operation == "source_deletion":
                extras = (("custody_probe", _digest(f"source-probe:{anchor_id}")),)
            row = {
                "attempt_index": index,
                "attempt_id": f"{operation}:{anchor_id}",
                "operation": operation,
                "anchor_id": anchor_id,
                "donor_anchor_id": None,
                "resulting_program_source_sha256": None,
                "resulting_packet_sha256": None,
            }
            attempts.append(
                engine._make_attempt_record(
                    row,
                    parent,
                    observed_program_source_sha256=parent.program_source_sha256,
                    snapshot=snapshot,
                    extra_artifact_hashes=extras,
                    failure=failure,
                )
            )
    provisional = RuntimeExecutionResult(
        RUNTIME_EXECUTION_SCHEMA,
        EXECUTION_PROJECTION_SCHEMA,
        _digest("projection"),
        LOCKED_SCORED_ROW_COUNT,
        False,
        tuple(parents),
        tuple(attempts),
        "",
    )
    return RuntimeExecutionResult(
        **{
            **provisional.__dict__,
            "execution_sha256": artifact._execution_sha256(provisional),
        }
    )


@pytest.fixture
def result(monkeypatch: pytest.MonkeyPatch) -> RuntimeExecutionResult:
    monkeypatch.setattr(artifact, "RUNTIME_PANEL_SIZE", 2)
    monkeypatch.setattr(artifact, "EXPECTED_PREQUERY_ATTEMPT_COUNT", 50)
    return _result()


@pytest.fixture
def bundle(
    tmp_path: Path, result: RuntimeExecutionResult
) -> tuple[Path, Path, artifact.RuntimeExecutionArtifactIndex]:
    store = tmp_path / "objects"
    store.mkdir()
    aggregate = tmp_path / "aggregate.json"
    index = artifact.write_runtime_execution_artifact(aggregate, store, result)
    return aggregate, store, index


def _immutable(path: Path, raw: bytes) -> None:
    path.write_bytes(raw)
    path.chmod(0o400)


def _aggregate(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _write_changed_aggregate(
    tmp_path: Path, name: str, value: dict[str, object]
) -> tuple[Path, str]:
    raw = artifact._canonical_json_bytes(value) + b"\n"
    path = tmp_path / name
    _immutable(path, raw)
    return path, _digest(raw)


def _assert_tensor_exact(left: torch.Tensor, right: torch.Tensor) -> None:
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    assert artifact._tensor_raw(left) == artifact._tensor_raw(right)


def _assert_snapshot_exact(left, right) -> None:
    assert left is not None and right is not None
    _assert_tensor_exact(left.packet.action_cards, right.packet.action_cards)
    _assert_tensor_exact(left.packet.initial_state, right.packet.initial_state)
    _assert_tensor_exact(left.packet.schedule, right.packet.schedule)
    for name in (
        "h19_residual",
        "h29_residual",
        "state_route",
        "composed_route",
        "halted",
        "terminal",
    ):
        _assert_tensor_exact(getattr(left, name), getattr(right, name))
    assert left.artifact_hashes == right.artifact_hashes
    assert left.snapshot_sha256 == right.snapshot_sha256


def test_production_contract_is_exactly_21600_attempts() -> None:
    assert artifact.EXPECTED_PREQUERY_ATTEMPT_COUNT == 21_600


def test_full_failure_panel_validates_all_21600_ordered_attempts() -> None:
    parents = []
    for index in range(864):
        anchor_id = f"p-{_digest(f'parent:{index}')[:24]}"
        parents.append(
            engine._make_parent_record(
                anchor_id=anchor_id,
                program_source_sha256=_digest(f"source:{anchor_id}"),
                expected_packet_sha256=_digest(f"packet:{anchor_id}"),
                snapshot=None,
                failure=ExecutionFailure("compile", "parent_compile_failed"),
            )
        )
    attempts = []
    for index in range(21_600):
        parent = parents[index % len(parents)]
        attempts.append(
            engine._make_attempt_record(
                {
                    "attempt_index": 3 + index * 2,
                    "attempt_id": f"a-{_digest(f'attempt:{index}')[:24]}",
                    "operation": f"prequery-op-{index // 864:02d}",
                    "anchor_id": parent.anchor_id,
                    "donor_anchor_id": None,
                    "resulting_program_source_sha256": None,
                    "resulting_packet_sha256": None,
                },
                parent,
                observed_program_source_sha256=parent.program_source_sha256,
                snapshot=None,
                failure=ExecutionFailure("execution", "parent_execution_unavailable"),
            )
        )
    provisional = RuntimeExecutionResult(
        RUNTIME_EXECUTION_SCHEMA,
        EXECUTION_PROJECTION_SCHEMA,
        _digest("full-projection"),
        LOCKED_SCORED_ROW_COUNT,
        False,
        tuple(parents),
        tuple(attempts),
        "",
    )
    complete = RuntimeExecutionResult(
        **{
            **provisional.__dict__,
            "execution_sha256": artifact._execution_sha256(provisional),
        }
    )
    artifact._validate_complete_result(complete)
    assert len(complete.attempts) == 21_600


def test_round_trip_is_lossless_and_receipt_ready(
    bundle, result: RuntimeExecutionResult
) -> None:
    aggregate, store, index = bundle
    replayed, replayed_index = artifact.read_runtime_execution_artifact_bundle(
        aggregate,
        store,
        expected_aggregate_sha256=index.aggregate_sha256,
        expected_projection_sha256=result.projection_sha256,
    )
    assert replayed_index == index
    assert replayed.execution_sha256 == result.execution_sha256
    assert len(replayed.parents) == 2
    assert len(replayed.attempts) == 50
    assert len(index.attempt_outputs) == 50
    assert index.attempt_outputs[0] == {
        "attempt_id": result.attempts[0].attempt_id,
        "status": result.attempts[0].status,
        "raw_output_artifact_sha256": index.attempt_artifact_sha256s[0],
    }
    _assert_snapshot_exact(result.parents[0].snapshot, replayed.parents[0].snapshot)
    successful = next(row for row in replayed.attempts if row.snapshot is not None)
    original = next(
        row for row in result.attempts if row.attempt_id == successful.attempt_id
    )
    _assert_snapshot_exact(original.snapshot, successful.snapshot)


def test_signed_receipt_replays_real_content_addressed_bundle(
    bundle,
    result: RuntimeExecutionResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate, store, index = bundle
    plan = valid_plan()
    bindings = plan.bindings
    projected = {
        "projection_sha256": result.projection_sha256,
        "attempts": [
            {
                "attempt_index": row.attempt_index,
                "attempt_id": row.attempt_id,
                "operation": row.operation,
            }
            for row in result.attempts
        ],
        "plan_sha256": plan.plan_sha256,
        "board_manifest_sha256": bindings.board_manifest_sha256,
        "board_tree_sha256": bindings.board_tree_sha256,
        "run_contract_sha256": bindings.run_contract_sha256,
        "compiler_sha256": bindings.compiler_sha256,
        "core_sha256": bindings.core_sha256,
        "core_kind": bindings.core_kind,
        "tokenizer_sha256": bindings.tokenizer_sha256,
        "base_checkpoint_sha256": bindings.base_checkpoint_sha256,
        "base_raw_evidence_receipt_sha256": (bindings.base_raw_evidence_receipt_sha256),
        "runtime_implementation_sha256": bindings.runtime_implementation_sha256,
        "selection_seed": bindings.selection_seed,
        "selection_seed_receipt_sha256": bindings.selection_seed_receipt_sha256,
        "training_seed": bindings.training_seed,
        "arm_id": bindings.arm_id,
        "anchor_panel_sha256": plan.anchor_panel_sha256,
        "donor_registry_sha256": plan.donor_registry_sha256,
        "batch_order_sha256": bindings.batch_order_sha256,
    }
    monkeypatch.setattr(
        execution_receipt,
        "_load_projection",
        lambda _path, _plan: (projected, _digest("projection-file")),
    )
    signed = execution_receipt.make_runtime_execution_receipt(
        execution_projection_path=aggregate.parent / "projection.json",
        plan=plan,
        execution_aggregate_path=aggregate,
        execution_artifact_directory=store,
        execution_aggregate_sha256=index.aggregate_sha256,
        signing_key=Ed25519PrivateKey.from_private_bytes(b"\x17" * 32),
    )
    payload = signed["payload"]
    assert payload["execution_aggregate_sha256"] == index.aggregate_sha256
    assert payload["execution_sha256"] == result.execution_sha256
    assert [row["raw_output_artifact_sha256"] for row in payload["attempts"]] == list(
        index.attempt_artifact_sha256s
    )


def test_failures_and_all_attempts_are_preserved(bundle) -> None:
    aggregate, store, index = bundle
    replayed = artifact.read_runtime_execution_artifact(
        aggregate, store, expected_aggregate_sha256=index.aggregate_sha256
    )
    assert len(replayed.attempts) == 50
    failures = [row for row in replayed.attempts if row.failure is not None]
    assert [(row.failure.stage, row.failure.code) for row in failures] == [
        ("compile", "synthetic_failure"),
        ("custody", "denied"),
    ]
    assert failures[0].snapshot is None
    assert failures[1].snapshot is not None
    assert failures[1].extra_artifact_hashes == (
        ("custody_probe", _digest("query-probe")),
    )


def test_aggregate_and_content_objects_are_canonical_immutable(bundle) -> None:
    aggregate, store, index = bundle
    assert not aggregate.stat().st_mode & 0o222
    decoded = _aggregate(aggregate)
    assert aggregate.read_bytes() == artifact._canonical_json_bytes(decoded) + b"\n"
    for digest in (
        *index.parent_artifact_sha256s,
        *index.attempt_artifact_sha256s,
    ):
        path = store / f"{digest}{artifact.EXECUTION_ARTIFACT_SUFFIX}"
        assert path.is_file()
        assert path.stat().st_nlink == 1
        assert not path.stat().st_mode & 0o222
        assert _digest(path.read_bytes()) == digest


def test_wrong_aggregate_or_projection_hash_is_rejected(bundle) -> None:
    aggregate, store, index = bundle
    with pytest.raises(artifact.RuntimeExecutionArtifactError, match="substitution"):
        artifact.read_runtime_execution_artifact(
            aggregate, store, expected_aggregate_sha256="f" * 64
        )
    with pytest.raises(artifact.RuntimeExecutionArtifactError, match="projection"):
        artifact.read_runtime_execution_artifact(
            aggregate,
            store,
            expected_aggregate_sha256=index.aggregate_sha256,
            expected_projection_sha256="f" * 64,
        )


def test_duplicate_keys_and_nonfinite_json_are_rejected(tmp_path: Path, bundle) -> None:
    aggregate, store, _ = bundle
    raw = aggregate.read_bytes()
    duplicate = raw.replace(b'"schema":', b'"schema":"duplicate","schema":', 1)
    duplicate_path = tmp_path / "duplicate.json"
    _immutable(duplicate_path, duplicate)
    with pytest.raises(artifact.RuntimeExecutionArtifactError, match="duplicate"):
        artifact.read_runtime_execution_artifact(
            duplicate_path, store, expected_aggregate_sha256=_digest(duplicate)
        )
    nonfinite = raw.replace(b'"attempt_count":50', b'"attempt_count":NaN', 1)
    nonfinite_path = tmp_path / "nonfinite.json"
    _immutable(nonfinite_path, nonfinite)
    with pytest.raises(artifact.RuntimeExecutionArtifactError, match="non-finite"):
        artifact.read_runtime_execution_artifact(
            nonfinite_path, store, expected_aggregate_sha256=_digest(nonfinite)
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered"])
def test_missing_extra_and_reordered_attempts_are_rejected(
    tmp_path: Path, bundle, mutation: str
) -> None:
    aggregate, store, _ = bundle
    changed = _aggregate(aggregate)
    rows = changed["attempt_artifacts"]
    assert isinstance(rows, list)
    if mutation == "missing":
        rows.pop()
        changed["attempt_count"] = len(rows)
    elif mutation == "extra":
        rows.append(deepcopy(rows[-1]))
        changed["attempt_count"] = len(rows)
    else:
        rows[0], rows[1] = rows[1], rows[0]
    changed["attempt_artifacts_sha256"] = artifact._sha256_json(rows)
    path, digest = _write_changed_aggregate(tmp_path, f"{mutation}.json", changed)
    with pytest.raises(artifact.RuntimeExecutionArtifactError):
        artifact.read_runtime_execution_artifact(
            path, store, expected_aggregate_sha256=digest
        )


def test_raw_artifact_hash_substitution_is_rejected(bundle) -> None:
    aggregate, store, index = bundle
    target = store / (
        index.attempt_artifact_sha256s[0] + artifact.EXECUTION_ARTIFACT_SUFFIX
    )
    raw = target.read_bytes() + b"substitution"
    target.unlink()
    _immutable(target, raw)
    with pytest.raises(artifact.RuntimeExecutionArtifactError, match="substitution"):
        artifact.read_runtime_execution_artifact(
            aggregate, store, expected_aggregate_sha256=index.aggregate_sha256
        )


def test_missing_raw_artifact_is_rejected(bundle) -> None:
    aggregate, store, index = bundle
    target = store / (
        index.attempt_artifact_sha256s[-1] + artifact.EXECUTION_ARTIFACT_SUFFIX
    )
    target.unlink()
    with pytest.raises(artifact.RuntimeExecutionArtifactError, match="missing"):
        artifact.read_runtime_execution_artifact(
            aggregate, store, expected_aggregate_sha256=index.aggregate_sha256
        )


def test_tensor_shape_byte_mismatch_is_rejected(tmp_path: Path, bundle) -> None:
    aggregate, store, _ = bundle
    changed_aggregate = _aggregate(aggregate)
    refs = changed_aggregate["attempt_artifacts"]
    assert isinstance(refs, list)
    target_ref = next(
        row for row in refs if isinstance(row, dict) and row["status"] == "success"
    )
    old_digest = target_ref["raw_output_artifact_sha256"]
    old_path = store / f"{old_digest}{artifact.EXECUTION_ARTIFACT_SUFFIX}"
    raw = old_path.read_bytes()
    prefix = (
        len(artifact.EXECUTION_ARTIFACT_MAGIC) + artifact.EXECUTION_ARTIFACT_HEADER.size
    )
    header_length = artifact.EXECUTION_ARTIFACT_HEADER.unpack(
        raw[len(artifact.EXECUTION_ARTIFACT_MAGIC) : prefix]
    )[0]
    header_end = prefix + header_length
    header = json.loads(raw[prefix:header_end])
    descriptor = next(row for row in header["blobs"] if row["name"] == "state_route")
    descriptor["shape"] = [999]
    encoded = artifact._canonical_json_bytes(header)
    changed_raw = (
        artifact.EXECUTION_ARTIFACT_MAGIC
        + artifact.EXECUTION_ARTIFACT_HEADER.pack(len(encoded))
        + encoded
        + raw[header_end:]
    )
    changed_digest = _digest(changed_raw)
    _immutable(
        store / f"{changed_digest}{artifact.EXECUTION_ARTIFACT_SUFFIX}", changed_raw
    )
    target_ref["raw_output_artifact_sha256"] = changed_digest
    changed_aggregate["attempt_artifacts_sha256"] = artifact._sha256_json(refs)
    changed_path, aggregate_digest = _write_changed_aggregate(
        tmp_path, "shape-mismatch.json", changed_aggregate
    )
    with pytest.raises(
        artifact.RuntimeExecutionArtifactError, match="shape/byte length"
    ):
        artifact.read_runtime_execution_artifact(
            changed_path, store, expected_aggregate_sha256=aggregate_digest
        )


def test_query_answer_oracle_and_late_query_leakage_is_rejected(
    tmp_path: Path, bundle
) -> None:
    aggregate, store, _ = bundle
    for index, (key, value) in enumerate(
        (
            ("answer", 1),
            ("oracle_access", 1),
            ("query_source_sha256", "a" * 64),
            ("late-query-swap", True),
        )
    ):
        changed = _aggregate(aggregate)
        changed[key] = value
        path, digest = _write_changed_aggregate(tmp_path, f"leak-{index}.json", changed)
        with pytest.raises(artifact.RuntimeExecutionArtifactError, match="leaks"):
            artifact.read_runtime_execution_artifact(
                path, store, expected_aggregate_sha256=digest
            )


def test_symlink_hardlink_and_writable_files_are_rejected(
    tmp_path: Path, bundle
) -> None:
    aggregate, store, index = bundle
    hardlink = tmp_path / "aggregate-hardlink.json"
    os.link(aggregate, hardlink)
    with pytest.raises(artifact.RuntimeExecutionArtifactError, match="single-link"):
        artifact.read_runtime_execution_artifact(
            hardlink, store, expected_aggregate_sha256=index.aggregate_sha256
        )
    hardlink.unlink()
    target = store / (
        index.attempt_artifact_sha256s[0] + artifact.EXECUTION_ARTIFACT_SUFFIX
    )
    target.chmod(0o600)
    with pytest.raises(artifact.RuntimeExecutionArtifactError, match="immutable"):
        artifact.read_runtime_execution_artifact(
            aggregate, store, expected_aggregate_sha256=index.aggregate_sha256
        )


def test_symlinked_raw_artifact_and_parent_directory_are_rejected(
    tmp_path: Path, bundle, result: RuntimeExecutionResult
) -> None:
    aggregate, store, index = bundle
    target = store / (
        index.attempt_artifact_sha256s[0] + artifact.EXECUTION_ARTIFACT_SUFFIX
    )
    backup = tmp_path / "backup.ctaaexec"
    _immutable(backup, target.read_bytes())
    target.unlink()
    target.symlink_to(backup)
    with pytest.raises(artifact.RuntimeExecutionArtifactError, match="symlinked"):
        artifact.read_runtime_execution_artifact(
            aggregate, store, expected_aggregate_sha256=index.aggregate_sha256
        )
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(artifact.RuntimeExecutionArtifactError, match="directory"):
        artifact.write_runtime_execution_artifact(
            tmp_path / "other.json", alias, result
        )


def test_writer_refuses_existing_aggregate(
    bundle, result: RuntimeExecutionResult
) -> None:
    aggregate, store, _ = bundle
    with pytest.raises(FileExistsError):
        artifact.write_runtime_execution_artifact(aggregate, store, result)


def test_incomplete_result_is_rejected_before_publication(
    tmp_path: Path, result: RuntimeExecutionResult
) -> None:
    incomplete = RuntimeExecutionResult(
        **{**result.__dict__, "attempts": result.attempts[:-1]}
    )
    store = tmp_path / "objects"
    store.mkdir()
    aggregate = tmp_path / "aggregate.json"
    with pytest.raises(artifact.RuntimeExecutionArtifactError, match="coverage"):
        artifact.write_runtime_execution_artifact(aggregate, store, incomplete)
    assert not aggregate.exists()
    assert not list(store.iterdir())
