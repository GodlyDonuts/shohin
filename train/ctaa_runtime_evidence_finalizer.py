"""Fail-closed post-receipt finalization of CTAA runtime evidence.

The query-blind runner publishes lossless execution artifacts and a signed
receipt before this module is invoked.  This module replays those artifacts,
proves their opaque identities are the exact projection of the frozen full
plan, and only then materializes query positions and answers.  It never
accepts caller-supplied tensors, query positions, answers, or statuses:
pre-query tensors and statuses must replay from the signed, content-addressed
execution bundle and pass the independent structural checks below.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import torch

from ctaa_intervention_protocol import (
    GateFamily,
    InterventionFamily,
    MANDATORY_OPERATIONS,
    RUNTIME_PANEL_SIZE,
    RuntimeInterventionPlan,
    validate_runtime_intervention_plan,
)
from ctaa_packet_io import packet_body
from ctaa_run_contract import canonical_json
from ctaa_runtime_evidence import (
    RuntimeEvidenceBuilder,
    make_custody_receipts,
    make_raw_tensor,
)
from ctaa_runtime_execution_artifact import (
    RuntimeExecutionArtifactIndex,
    read_runtime_execution_artifact_bundle,
)
from ctaa_runtime_execution_engine import (
    AttemptExecutionRecord,
    ExecutionFailure,
    ExecutionSnapshot,
    ParentExecutionRecord,
    RuntimeExecutionResult,
)
from ctaa_runtime_execution_projection import (
    EXECUTION_PROJECTION_SCHEMA,
    validate_execution_projection,
)
from ctaa_runtime_execution_receipt import (
    EXECUTION_RECEIPT_SCHEMA,
    validate_runtime_execution_receipt,
)


EXPECTED_FINAL_ATTEMPT_COUNT = RUNTIME_PANEL_SIZE * len(MANDATORY_OPERATIONS)
EXPECTED_PREQUERY_ATTEMPT_COUNT = EXPECTED_FINAL_ATTEMPT_COUNT - RUNTIME_PANEL_SIZE
_MAX_PROJECTION_BYTES = 256 * 1024 * 1024
_MAX_RECEIPT_BYTES = 128 * 1024 * 1024
_RECEIPT_RECORD_KEYS = frozenset({"payload", "signature", "receipt_sha256"})
_RECEIPT_FORBIDDEN_FRAGMENTS = (
    "answer",
    "query_position",
    "query_source",
    "late_query_swap",
)

_H19_OPERATIONS = frozenset(
    {
        "h19_zero",
        "h19_batch_rotate",
        "h19_donor_transplant",
    }
)
_H29_OPERATIONS = frozenset(
    {
        "h29_zero",
        "h29_batch_rotate",
        "h29_donor_transplant",
    }
)
_MIDPOINT_OPERATIONS = frozenset({"midpoint_donor_state", "midpoint_donor_action"})
_GATE_OPERATIONS = frozenset(item.value for item in GateFamily)
_PARENT_TENSORS = frozenset(
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
    }
)
_OUTPUT_TENSORS = frozenset(
    {
        "state_route",
        "composed_route",
        "halt_mask",
        "terminal_state",
        "query_position",
        "answer",
    }
)
_ENGINE_FAILURE_CODES_BY_STAGE = {
    "source": frozenset({"source_transform_noop"}),
    "compile": frozenset(
        {
            "donor_residual_unavailable",
            "materializer_unavailable",
            "packet_batch_geometry",
            "packet_materialization_failed",
            "packet_type",
            "parent_packet_replay_mismatch",
            "parent_residual_unavailable",
            "program_compile_failed",
            "residual_bundle_geometry",
            "residual_bundle_type",
            "residual_compile_failed",
            "residual_donor_geometry",
            "residual_donor_noop",
            "residual_zero_noop",
        }
    ),
    "packet": frozenset(
        {
            "committed_packet_replay_mismatch",
            "donor_packet_unavailable",
            "packet_mutation_failed",
            "packet_operation_unknown",
            "parent_packet_unavailable",
        }
    ),
    "custody": frozenset(
        {
            "probe_execution_failed",
            "probe_observation_type",
            "probe_unavailable",
            "sealed_parent_unavailable",
            "source_poison_commitment",
            "source_poison_encoding",
        }
    ),
    "execution": frozenset(
        {
            "midpoint_donor_unavailable",
            "midpoint_execution_failed",
            "parent_execution_unavailable",
            "trace_execution_failed",
            "unexpected_runtime_failure",
        }
    ),
    "projection": frozenset(
        {
            "deferred_input_payload_disclosed",
            "mutation_payload_invalid",
            "operation_not_executable",
        }
    ),
}


class RuntimeEvidenceFinalizerError(ValueError):
    """The post-receipt custody chain or its deterministic replay differs."""


@dataclass(frozen=True)
class _PreparedCustody:
    plan: RuntimeInterventionPlan
    projection: dict[str, object]
    execution: RuntimeExecutionResult
    artifact_index: RuntimeExecutionArtifactIndex
    receipt_sha256: str
    opaque_to_full_anchor: Mapping[str, str]
    parent_by_full_anchor: Mapping[str, ParentExecutionRecord]
    parent_artifact_by_full_anchor: Mapping[str, str]
    prequery_records: Mapping[int, AttemptExecutionRecord]
    prequery_artifacts: Mapping[int, str]
    source_probe_artifacts: Mapping[str, str]
    query_probe_artifacts: Mapping[str, str]


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _metadata_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_immutable_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    """Read one immutable, single-link regular file through a held descriptor."""

    path = Path(path).absolute()
    try:
        parent = path.parent
        parent_metadata = parent.lstat()
        if parent.resolve(strict=True) != parent or not stat.S_ISDIR(
            parent_metadata.st_mode
        ):
            raise RuntimeEvidenceFinalizerError(
                "finalizer input parent is not a direct directory"
            )
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeEvidenceFinalizerError("finalizer input is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o222
        or metadata.st_size > maximum_bytes
    ):
        raise RuntimeEvidenceFinalizerError(
            "finalizer input is not a bounded single-link immutable file"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeEvidenceFinalizerError(
            "finalizer input cannot be opened safely"
        ) from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise RuntimeEvidenceFinalizerError("finalizer input exceeds its bound")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        _metadata_identity(metadata) != _metadata_identity(before)
        or _metadata_identity(before) != _metadata_identity(after)
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or after.st_mode & 0o222
    ):
        raise RuntimeEvidenceFinalizerError("finalizer input changed while being read")
    return b"".join(chunks)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeEvidenceFinalizerError("projection contains a duplicate key")
        result[key] = value
    return result


def _decode_canonical_object(raw: bytes, *, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise RuntimeEvidenceFinalizerError(
            f"{label} contains non-finite constant {value}"
        )

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeEvidenceFinalizerError(f"{label} JSON differs") from error
    if not isinstance(value, dict):
        raise RuntimeEvidenceFinalizerError(f"{label} root differs")
    if raw != (canonical_json(value) + "\n").encode("ascii"):
        raise RuntimeEvidenceFinalizerError(f"{label} is not canonical")
    return value


def _decode_projection(raw: bytes) -> dict[str, object]:
    return _decode_canonical_object(raw, label="projection")


def _reject_receipt_query_material(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).casefold()
            if any(fragment in lowered for fragment in _RECEIPT_FORBIDDEN_FRAGMENTS):
                raise RuntimeEvidenceFinalizerError(
                    f"pre-query receipt leaks field: {key}"
                )
            _reject_receipt_query_material(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_receipt_query_material(item)
    elif isinstance(value, str):
        lowered = value.casefold()
        if any(fragment in lowered for fragment in _RECEIPT_FORBIDDEN_FRAGMENTS):
            raise RuntimeEvidenceFinalizerError(
                "pre-query receipt leaks deferred query data"
            )


def _verification_key(
    value: bytes | Ed25519PublicKey,
) -> tuple[Ed25519PublicKey, bytes]:
    if isinstance(value, Ed25519PublicKey):
        raw = value.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return value, raw
    if not isinstance(value, bytes) or len(value) != 32:
        raise RuntimeEvidenceFinalizerError("invalid receipt verification key")
    try:
        return Ed25519PublicKey.from_public_bytes(value), value
    except ValueError as error:
        raise RuntimeEvidenceFinalizerError(
            "invalid receipt verification key"
        ) from error


def _read_authenticated_receipt(
    path: Path,
    verification_key: bytes | Ed25519PublicKey,
) -> dict[str, object]:
    """Authenticate exact held-FD receipt bytes before touching the full plan."""

    raw = _read_immutable_bytes(path, maximum_bytes=_MAX_RECEIPT_BYTES)
    record = _decode_canonical_object(raw, label="execution receipt")
    if set(record) != _RECEIPT_RECORD_KEYS:
        raise RuntimeEvidenceFinalizerError("execution receipt record schema differs")
    _reject_receipt_query_material(record)
    payload = record.get("payload")
    signature = record.get("signature")
    receipt_sha = record.get("receipt_sha256")
    if (
        not isinstance(payload, dict)
        or not isinstance(signature, str)
        or len(signature) != 128
        or any(character not in "0123456789abcdef" for character in signature)
        or not isinstance(receipt_sha, str)
        or len(receipt_sha) != 64
        or any(character not in "0123456789abcdef" for character in receipt_sha)
    ):
        raise RuntimeEvidenceFinalizerError("execution receipt authentication differs")
    public_key, raw_key = _verification_key(verification_key)
    if payload.get("signing_public_key") != raw_key.hex():
        raise RuntimeEvidenceFinalizerError(
            "execution receipt uses the wrong signing key"
        )
    try:
        public_key.verify(
            bytes.fromhex(signature), canonical_json(payload).encode("ascii")
        )
    except (InvalidSignature, ValueError) as error:
        raise RuntimeEvidenceFinalizerError(
            "Ed25519 receipt signature verification failed"
        ) from error
    expected_receipt_sha = hashlib.sha256(
        canonical_json({"payload": payload, "signature": signature}).encode("ascii")
    ).hexdigest()
    if receipt_sha != expected_receipt_sha:
        raise RuntimeEvidenceFinalizerError("execution receipt hash differs")
    return record


def _opaque_anchor_id(index: int) -> str:
    return f"oa{index:06d}"


def _opaque_attempt_id(index: int) -> str:
    return f"ot{index:08d}"


def _validate_failure_origin(
    failure: ExecutionFailure,
    *,
    operation: str | None,
    extra_artifact_hashes: tuple[tuple[str, str], ...] = (),
) -> None:
    allowed = _ENGINE_FAILURE_CODES_BY_STAGE.get(failure.stage)
    if allowed is not None and failure.code in allowed:
        return
    # A negative physical custody observation carries a runner-defined code,
    # but it is admissible only when the same artifact commits the observation.
    if (
        failure.stage == "custody"
        and operation
        in {GateFamily.SOURCE_DELETION.value, GateFamily.QUERY_ISOLATION.value}
        and len(extra_artifact_hashes) == 1
        and extra_artifact_hashes[0][0] == "custody_probe"
    ):
        return
    raise RuntimeEvidenceFinalizerError(
        "execution artifact contains a producer-defined failure"
    )


def _validate_extra_artifacts(record: AttemptExecutionRecord) -> None:
    extras = record.extra_artifact_hashes
    probe_operations = {
        GateFamily.SOURCE_DELETION.value,
        GateFamily.QUERY_ISOLATION.value,
    }
    if record.operation in probe_operations:
        if extras and (len(extras) != 1 or extras[0][0] != "custody_probe"):
            raise RuntimeEvidenceFinalizerError("custody probe artifact differs")
        if record.status == "success" and not extras:
            raise RuntimeEvidenceFinalizerError(
                "successful custody probe lacks an observation commitment"
            )
        return
    if record.operation in _MIDPOINT_OPERATIONS:
        expected_name = (
            "injected_state"
            if record.operation == "midpoint_donor_state"
            else "injected_action"
        )
        if record.status == "success":
            if len(extras) != 1 or extras[0][0] != expected_name:
                raise RuntimeEvidenceFinalizerError(
                    "midpoint injection artifact differs"
                )
        elif extras:
            raise RuntimeEvidenceFinalizerError(
                "failed midpoint contains producer-authored artifacts"
            )
        return
    if extras:
        raise RuntimeEvidenceFinalizerError(
            "execution artifact contains unconsumed producer data"
        )


def _exact_projection_and_artifact_mapping(
    *,
    plan: RuntimeInterventionPlan,
    projection: Mapping[str, object],
    execution: RuntimeExecutionResult,
    artifact_index: RuntimeExecutionArtifactIndex,
    receipt: Mapping[str, object],
) -> _PreparedCustody:
    anchors = projection.get("anchors")
    attempts = projection.get("attempts")
    if not isinstance(anchors, list) or not isinstance(attempts, list):
        raise RuntimeEvidenceFinalizerError("projection coverage differs")
    if (
        len(anchors) != RUNTIME_PANEL_SIZE
        or len(attempts) != EXPECTED_PREQUERY_ATTEMPT_COUNT
        or len(plan.attempts) != EXPECTED_FINAL_ATTEMPT_COUNT
        or len(execution.parents) != RUNTIME_PANEL_SIZE
        or len(execution.attempts) != EXPECTED_PREQUERY_ATTEMPT_COUNT
        or len(artifact_index.parent_artifact_sha256s) != RUNTIME_PANEL_SIZE
        or len(artifact_index.attempt_artifact_sha256s)
        != EXPECTED_PREQUERY_ATTEMPT_COUNT
        or len(artifact_index.attempt_outputs) != EXPECTED_PREQUERY_ATTEMPT_COUNT
    ):
        raise RuntimeEvidenceFinalizerError("finalizer exact coverage differs")

    opaque_to_full: dict[str, str] = {}
    full_to_opaque: dict[str, str] = {}
    projected_anchor_by_opaque: dict[str, Mapping[str, object]] = {}
    anchor_by_full = {anchor.anchor_id: anchor for anchor in plan.anchors}
    for index, (anchor, projected) in enumerate(zip(plan.anchors, anchors)):
        if not isinstance(projected, Mapping):
            raise RuntimeEvidenceFinalizerError("projected anchor schema differs")
        opaque = _opaque_anchor_id(index)
        if (
            projected.get("anchor_id") != opaque
            or projected.get("program_source_sha256") != anchor.program_source_sha256
            or projected.get("packet_sha256") != anchor.packet_sha256
        ):
            raise RuntimeEvidenceFinalizerError("opaque anchor mapping differs")
        opaque_to_full[opaque] = anchor.anchor_id
        full_to_opaque[anchor.anchor_id] = opaque
        projected_anchor_by_opaque[opaque] = projected

    batch_order = projection.get("batch_order")
    expected_batch_order = [
        full_to_opaque[anchor_id] for anchor_id in plan.bindings.batch_order
    ]
    if (
        not isinstance(batch_order, list)
        or batch_order != expected_batch_order
        or [record.anchor_id for record in execution.parents] != batch_order
    ):
        raise RuntimeEvidenceFinalizerError("parent artifact batch order differs")
    parent_by_full: dict[str, ParentExecutionRecord] = {}
    parent_artifacts: dict[str, str] = {}
    for record, artifact_sha in zip(
        execution.parents, artifact_index.parent_artifact_sha256s
    ):
        full_anchor_id = opaque_to_full.get(record.anchor_id)
        projected = projected_anchor_by_opaque.get(record.anchor_id)
        if full_anchor_id is None or projected is None:
            raise RuntimeEvidenceFinalizerError("opaque parent identity differs")
        anchor = anchor_by_full[full_anchor_id]
        if (
            record.program_source_sha256 != anchor.program_source_sha256
            or record.expected_packet_sha256 != anchor.packet_sha256
        ):
            raise RuntimeEvidenceFinalizerError("opaque parent mapping differs")
        if record.failure is not None:
            if record.failure.stage not in {"compile", "execution"}:
                raise RuntimeEvidenceFinalizerError(
                    "parent artifact contains a producer-defined failure"
                )
            _validate_failure_origin(record.failure, operation=None)
        if record.snapshot is not None:
            packet_sha = hashlib.sha256(packet_body(record.snapshot.packet)).hexdigest()
            if packet_sha != anchor.packet_sha256:
                raise RuntimeEvidenceFinalizerError("parent packet artifact differs")
        parent_by_full[anchor.anchor_id] = record
        parent_artifacts[anchor.anchor_id] = artifact_sha

    full_prequery = tuple(
        item
        for item in plan.attempts
        if item.operation != InterventionFamily.LATE_QUERY_SWAP.value
    )
    if len(full_prequery) != EXPECTED_PREQUERY_ATTEMPT_COUNT:
        raise RuntimeEvidenceFinalizerError("full-plan pre-query coverage differs")
    receipt_payload = receipt.get("payload")
    if not isinstance(receipt_payload, Mapping):
        raise RuntimeEvidenceFinalizerError("signed receipt payload differs")
    receipt_attempts = receipt_payload.get("attempts")
    if not isinstance(receipt_attempts, list) or len(receipt_attempts) != len(attempts):
        raise RuntimeEvidenceFinalizerError("signed receipt attempt coverage differs")

    records: dict[int, AttemptExecutionRecord] = {}
    artifact_hashes: dict[int, str] = {}
    source_probes: dict[str, str] = {}
    query_probes: dict[str, str] = {}
    for full, projected, record, artifact_sha, output, signed_row in zip(
        full_prequery,
        attempts,
        execution.attempts,
        artifact_index.attempt_artifact_sha256s,
        artifact_index.attempt_outputs,
        receipt_attempts,
    ):
        if (
            not isinstance(projected, Mapping)
            or not isinstance(output, Mapping)
            or not isinstance(signed_row, Mapping)
        ):
            raise RuntimeEvidenceFinalizerError("attempt mapping schema differs")
        opaque_anchor = projected.get("anchor_id")
        opaque_donor = projected.get("donor_anchor_id")
        expected_donor = (
            None
            if full.donor_anchor_id is None
            else full_to_opaque.get(full.donor_anchor_id)
        )
        if (
            projected.get("attempt_index") != full.attempt_index
            or projected.get("attempt_id") != _opaque_attempt_id(full.attempt_index)
            or projected.get("attempt_plan_sha256") != full.attempt_plan_sha256
            or projected.get("operation") != full.operation
            or projected.get("operation_sha256") != full.operation_sha256
            or projected.get("resulting_program_source_sha256")
            != full.resulting_program_source_sha256
            or projected.get("resulting_packet_sha256") != full.resulting_packet_sha256
            or opaque_to_full.get(str(opaque_anchor)) != full.anchor_id
            or opaque_donor != expected_donor
            or record.attempt_index != full.attempt_index
            or record.attempt_id != projected.get("attempt_id")
            or record.operation != full.operation
            or record.anchor_id != opaque_anchor
            or record.donor_anchor_id != opaque_donor
            or record.parent_record_sha256
            != parent_by_full[full.anchor_id].record_sha256
            or record.committed_program_source_sha256
            != full.resulting_program_source_sha256
            or record.committed_packet_sha256 != full.resulting_packet_sha256
            or output.get("attempt_id") != record.attempt_id
            or output.get("status") != record.status
            or output.get("raw_output_artifact_sha256") != artifact_sha
            or signed_row.get("attempt_index") != full.attempt_index
            or signed_row.get("attempt_id") != record.attempt_id
            or signed_row.get("operation") != full.operation
            or signed_row.get("status") != record.status
            or signed_row.get("raw_output_artifact_sha256") != artifact_sha
        ):
            raise RuntimeEvidenceFinalizerError(
                "opaque/full-plan attempt mapping differs"
            )
        _validate_extra_artifacts(record)
        if record.failure is not None:
            _validate_failure_origin(
                record.failure,
                operation=record.operation,
                extra_artifact_hashes=record.extra_artifact_hashes,
            )
        if record.status == "success":
            expected_program = (
                full.resulting_program_source_sha256
                if full.resulting_program_source_sha256 is not None
                else anchor_by_full[full.anchor_id].program_source_sha256
            )
            if record.observed_program_source_sha256 != expected_program:
                raise RuntimeEvidenceFinalizerError(
                    "successful program observation differs from its commitment"
                )
            if (
                full.resulting_packet_sha256 is not None
                and record.observed_packet_sha256 != full.resulting_packet_sha256
            ):
                raise RuntimeEvidenceFinalizerError(
                    "successful packet observation differs from its commitment"
                )
        records[full.attempt_index] = record
        artifact_hashes[full.attempt_index] = artifact_sha
        destination = (
            source_probes
            if full.operation == GateFamily.SOURCE_DELETION.value
            else query_probes
            if full.operation == GateFamily.QUERY_ISOLATION.value
            else None
        )
        if destination is not None:
            if full.anchor_id in destination:
                raise RuntimeEvidenceFinalizerError("duplicate custody probe mapping")
            destination[full.anchor_id] = artifact_sha
    all_anchor_ids = {item.anchor_id for item in plan.anchors}
    if set(source_probes) != all_anchor_ids or set(query_probes) != all_anchor_ids:
        raise RuntimeEvidenceFinalizerError("custody probe coverage differs")
    if receipt_payload.get("source_deletion_probe_artifact_sha256s") != list(
        source_probes.values()
    ) or receipt_payload.get("query_isolation_probe_artifact_sha256s") != list(
        query_probes.values()
    ):
        raise RuntimeEvidenceFinalizerError("signed custody probe binding differs")

    receipt_sha = receipt.get("receipt_sha256")
    if not isinstance(receipt_sha, str) or len(receipt_sha) != 64:
        raise RuntimeEvidenceFinalizerError("signed receipt identity differs")
    return _PreparedCustody(
        plan=plan,
        projection=dict(projection),
        execution=execution,
        artifact_index=artifact_index,
        receipt_sha256=receipt_sha,
        opaque_to_full_anchor=opaque_to_full,
        parent_by_full_anchor=parent_by_full,
        parent_artifact_by_full_anchor=parent_artifacts,
        prequery_records=records,
        prequery_artifacts=artifact_hashes,
        source_probe_artifacts=source_probes,
        query_probe_artifacts=query_probes,
    )


def _prepare_custody(
    *,
    plan: RuntimeInterventionPlan | Mapping[str, object],
    execution_projection_path: Path,
    execution_aggregate_path: Path,
    execution_artifact_directory: Path,
    execution_aggregate_sha256: str,
    execution_receipt_path: Path,
    receipt_verification_key: bytes | Ed25519PublicKey,
) -> _PreparedCustody:
    """Finish all receipt, artifact, and identity checks before query tensors exist."""

    try:
        authenticated_receipt = _read_authenticated_receipt(
            execution_receipt_path, receipt_verification_key
        )
    except Exception as error:
        raise RuntimeEvidenceFinalizerError(
            "signed pre-query execution receipt verification failed"
        ) from error

    # The full plan carries deferred query material.  It must not be inspected
    # until the exact receipt bytes have authenticated under the authority key.
    frozen = validate_runtime_intervention_plan(plan)
    try:
        receipt = validate_runtime_execution_receipt(
            authenticated_receipt,
            execution_projection_path=execution_projection_path,
            plan=frozen,
            execution_aggregate_path=execution_aggregate_path,
            execution_artifact_directory=execution_artifact_directory,
            execution_aggregate_sha256=execution_aggregate_sha256,
            verification_key=receipt_verification_key,
        )
    except Exception as error:
        raise RuntimeEvidenceFinalizerError(
            "signed pre-query execution receipt verification failed"
        ) from error
    payload = receipt.get("payload")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != EXECUTION_RECEIPT_SCHEMA
    ):
        raise RuntimeEvidenceFinalizerError("execution receipt version differs")

    raw_projection = _read_immutable_bytes(
        execution_projection_path, maximum_bytes=_MAX_PROJECTION_BYTES
    )
    if hashlib.sha256(raw_projection).hexdigest() != payload.get(
        "execution_projection_file_sha256"
    ):
        raise RuntimeEvidenceFinalizerError("signed projection file binding differs")
    decoded_projection = _decode_projection(raw_projection)
    try:
        projection = validate_execution_projection(decoded_projection, frozen)
    except ValueError as error:
        raise RuntimeEvidenceFinalizerError(
            "execution projection replay failed"
        ) from error
    if projection.get("schema") != EXECUTION_PROJECTION_SCHEMA:
        raise RuntimeEvidenceFinalizerError("execution projection version differs")
    try:
        execution, artifact_index = read_runtime_execution_artifact_bundle(
            execution_aggregate_path,
            execution_artifact_directory,
            expected_aggregate_sha256=execution_aggregate_sha256,
            expected_projection_sha256=str(projection["projection_sha256"]),
        )
    except Exception as error:
        raise RuntimeEvidenceFinalizerError(
            "execution artifact replay failed"
        ) from error
    if (
        payload.get("execution_aggregate_sha256") != artifact_index.aggregate_sha256
        or payload.get("execution_sha256") != execution.execution_sha256
        or payload.get("execution_projection_sha256")
        != projection.get("projection_sha256")
    ):
        raise RuntimeEvidenceFinalizerError("signed execution binding differs")
    return _exact_projection_and_artifact_mapping(
        plan=frozen,
        projection=projection,
        execution=execution,
        artifact_index=artifact_index,
        receipt=receipt,
    )


def _tensor_bytes(value: torch.Tensor) -> tuple[str, list[int], bytes]:
    if sys.byteorder != "little":
        raise RuntimeEvidenceFinalizerError("finalizer requires little-endian tensors")
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.layout != torch.strided
        or value.requires_grad
        or not value.is_contiguous()
    ):
        raise RuntimeEvidenceFinalizerError(
            "execution tensor is not immutable CPU material"
        )
    dtype = {
        torch.bool: "bool",
        torch.uint8: "uint8",
        torch.int64: "int64",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }.get(value.dtype)
    if dtype is None:
        raise RuntimeEvidenceFinalizerError("execution tensor dtype differs")
    raw = value.view(torch.uint8).numpy().tobytes()
    return dtype, list(value.shape), raw


def _raw_tensor(value: torch.Tensor) -> dict[str, object]:
    dtype, shape, raw = _tensor_bytes(value)
    return make_raw_tensor(dtype, shape, raw)


def _snapshot_materials(
    snapshot: ExecutionSnapshot,
    *,
    query_position: int,
    names: frozenset[str],
) -> dict[str, dict[str, object]]:
    if type(query_position) is not int or not 0 <= query_position < 3:
        raise RuntimeEvidenceFinalizerError("late query position differs")
    if snapshot.composed_route is None:
        raise RuntimeEvidenceFinalizerError("execution snapshot lacks composed route")
    terminal = snapshot.terminal
    if terminal.dtype != torch.uint8 or terminal.shape != (3,):
        raise RuntimeEvidenceFinalizerError("execution terminal differs")
    answer = int(terminal[query_position].item())
    available: dict[str, dict[str, object]] = {
        "packet": make_raw_tensor(
            "uint8",
            [snapshot.packet.bytes_per_row],
            packet_body(snapshot.packet),
        ),
        "state_route": _raw_tensor(snapshot.state_route),
        "composed_route": _raw_tensor(snapshot.composed_route),
        "halt_mask": _raw_tensor(snapshot.halted),
        "terminal_state": _raw_tensor(terminal),
        "query_position": make_raw_tensor("uint8", [1], bytes([query_position])),
        "answer": make_raw_tensor("uint8", [1], bytes([answer])),
    }
    if snapshot.h19_residual is not None:
        available["h19_residual"] = _raw_tensor(snapshot.h19_residual)
    if snapshot.h29_residual is not None:
        available["h29_residual"] = _raw_tensor(snapshot.h29_residual)
    if not names.issubset(available):
        raise RuntimeEvidenceFinalizerError(
            "execution snapshot tensor coverage differs"
        )
    return {name: available[name] for name in names}


def _operation_tensors(operation: str) -> frozenset[str]:
    if operation in _H19_OPERATIONS:
        return _OUTPUT_TENSORS | {"h19_residual"}
    if operation in _H29_OPERATIONS:
        return _OUTPUT_TENSORS | {"h29_residual"}
    if operation in _MIDPOINT_OPERATIONS:
        return _OUTPUT_TENSORS
    if operation == InterventionFamily.LATE_QUERY_SWAP.value:
        return frozenset({"terminal_state", "query_position", "answer"})
    if operation in _GATE_OPERATIONS:
        return frozenset()
    return _OUTPUT_TENSORS | {"packet"}


def _generic_failure(
    failure: ExecutionFailure,
    *,
    operation: str,
) -> tuple[str, str]:
    stage = failure.stage.casefold()
    if stage == "source":
        return "source", "source_transform_error"
    if stage == "compile":
        return "compile", "compiler_error"
    if stage == "packet":
        return "packet", "packet_validation_error"
    if stage == "query":
        return "query", "query_custody_error"
    if stage == "custody":
        if operation == GateFamily.SOURCE_DELETION.value:
            return "custody", "source_deletion_probe_error"
        if operation == GateFamily.QUERY_ISOLATION.value:
            return "custody", "query_isolation_probe_error"
        return "custody", "artifact_custody_error"
    if stage == "execution":
        return "execution", "execution_error"
    if stage == "projection":
        return "validation", "schema_validation_error"
    raise RuntimeEvidenceFinalizerError("execution failure stage is not admissible")


def _custody_receipts(prepared: _PreparedCustody, anchor_id: str) -> dict[str, object]:
    return make_custody_receipts(
        packet_receipt_sha256=prepared.parent_artifact_by_full_anchor[anchor_id],
        source_deletion_receipt_sha256=prepared.source_probe_artifacts[anchor_id],
        query_isolation_receipt_sha256=prepared.query_probe_artifacts[anchor_id],
        execution_receipt_sha256=prepared.receipt_sha256,
    )


def _populate_builder(prepared: _PreparedCustody) -> RuntimeEvidenceBuilder:
    """Materialize query data only after ``_prepare_custody`` has returned."""

    plan = prepared.plan
    builder = RuntimeEvidenceBuilder(plan)
    anchors = {item.anchor_id: item for item in plan.anchors}
    parent_snapshot_cache: dict[str, str] = {}

    def parent_snapshot(anchor_id: str) -> str:
        cached = parent_snapshot_cache.get(anchor_id)
        if cached is not None:
            return cached
        record = prepared.parent_by_full_anchor[anchor_id]
        if record.status != "success" or record.snapshot is None:
            raise RuntimeEvidenceFinalizerError("requested parent execution failed")
        anchor = anchors[anchor_id]
        snapshot_sha = builder.add_snapshot(
            anchor_id=anchor_id,
            role="parent",
            operation=None,
            tensors=_snapshot_materials(
                record.snapshot,
                query_position=anchor.query_position,
                names=_PARENT_TENSORS,
            ),
        )
        parent_snapshot_cache[anchor_id] = snapshot_sha
        return snapshot_sha

    for attempt in plan.attempts:
        receipts = _custody_receipts(prepared, attempt.anchor_id)
        if attempt.operation == InterventionFamily.LATE_QUERY_SWAP.value:
            donor_id = attempt.donor_anchor_id
            if donor_id is None or donor_id not in anchors:
                raise RuntimeEvidenceFinalizerError("late-query donor mapping differs")
            parent_record = prepared.parent_by_full_anchor[attempt.anchor_id]
            donor_record = prepared.parent_by_full_anchor[donor_id]
            if (
                anchors[attempt.anchor_id].query_position
                == anchors[donor_id].query_position
            ):
                raise RuntimeEvidenceFinalizerError("late-query donor is a no-op")
            if (
                parent_record.status != "success"
                or parent_record.snapshot is None
                or donor_record.status != "success"
                or donor_record.snapshot is None
            ):
                failed_record = (
                    parent_record if parent_record.status != "success" else donor_record
                )
                detail = prepared.parent_artifact_by_full_anchor[
                    attempt.anchor_id if failed_record is parent_record else donor_id
                ]
                builder.add_failure(
                    failure_stage="query",
                    failure_code="query_custody_error",
                    failure_detail_sha256=detail,
                    custody_receipts=receipts,
                    parent_snapshot_sha256=(
                        None
                        if parent_record.status != "success"
                        else parent_snapshot(attempt.anchor_id)
                    ),
                    donor_snapshot_sha256=(
                        None
                        if donor_record.status != "success"
                        else parent_snapshot(donor_id)
                    ),
                )
                continue
            if (
                attempt.resulting_query_source_sha256
                != anchors[donor_id].query_source_sha256
            ):
                raise RuntimeEvidenceFinalizerError(
                    "late-query donor source binding differs"
                )
            parent_ref = parent_snapshot(attempt.anchor_id)
            donor_ref = parent_snapshot(donor_id)
            intervention_ref = builder.add_snapshot(
                anchor_id=attempt.anchor_id,
                role="intervention",
                operation=attempt.operation,
                tensors=_snapshot_materials(
                    parent_record.snapshot,
                    query_position=anchors[donor_id].query_position,
                    names=_operation_tensors(attempt.operation),
                ),
            )
            builder.add_success(
                parent_snapshot_sha256=parent_ref,
                intervention_snapshot_sha256=intervention_ref,
                donor_snapshot_sha256=donor_ref,
                custody_receipts=receipts,
            )
            continue

        record = prepared.prequery_records.get(attempt.attempt_index)
        artifact_sha = prepared.prequery_artifacts.get(attempt.attempt_index)
        if record is None or artifact_sha is None:
            raise RuntimeEvidenceFinalizerError(
                "pre-query attempt is missing after replay"
            )
        parent_record = prepared.parent_by_full_anchor[attempt.anchor_id]
        parent_ref = (
            parent_snapshot(attempt.anchor_id)
            if parent_record.status == "success" and parent_record.snapshot is not None
            else None
        )
        donor_ref = None
        if attempt.donor_anchor_id is not None:
            donor_record = prepared.parent_by_full_anchor[attempt.donor_anchor_id]
            if donor_record.status == "success" and donor_record.snapshot is not None:
                donor_ref = parent_snapshot(attempt.donor_anchor_id)
        intervention_ref = None
        tensor_names = _operation_tensors(attempt.operation)
        if record.snapshot is not None and tensor_names:
            intervention_ref = builder.add_snapshot(
                anchor_id=attempt.anchor_id,
                role="intervention",
                operation=attempt.operation,
                tensors=_snapshot_materials(
                    record.snapshot,
                    query_position=anchors[attempt.anchor_id].query_position,
                    names=tensor_names,
                ),
            )
        if record.status == "success":
            if parent_ref is None or (tensor_names and intervention_ref is None):
                raise RuntimeEvidenceFinalizerError(
                    "successful artifact lacks evidence tensors"
                )
            if attempt.donor_anchor_id is not None and donor_ref is None:
                raise RuntimeEvidenceFinalizerError(
                    "successful donor artifact is unavailable"
                )
            builder.add_success(
                parent_snapshot_sha256=parent_ref,
                intervention_snapshot_sha256=intervention_ref,
                donor_snapshot_sha256=donor_ref,
                custody_receipts=receipts,
            )
        else:
            if record.failure is None:
                raise RuntimeEvidenceFinalizerError(
                    "failed artifact lacks a typed failure"
                )
            failure_stage, failure_code = _generic_failure(
                record.failure, operation=attempt.operation
            )
            builder.add_failure(
                failure_stage=failure_stage,
                failure_code=failure_code,
                failure_detail_sha256=artifact_sha,
                custody_receipts=receipts,
                parent_snapshot_sha256=parent_ref,
                intervention_snapshot_sha256=intervention_ref,
                donor_snapshot_sha256=donor_ref,
            )
    return builder


def make_finalized_runtime_evidence(
    *,
    plan: RuntimeInterventionPlan | Mapping[str, object],
    execution_projection_path: Path,
    execution_aggregate_path: Path,
    execution_artifact_directory: Path,
    execution_aggregate_sha256: str,
    execution_receipt_path: Path,
    receipt_verification_key: bytes | Ed25519PublicKey,
) -> dict[str, object]:
    """Verify custody, then build all 22,464 evidence attempts in plan order."""

    prepared = _prepare_custody(
        plan=plan,
        execution_projection_path=execution_projection_path,
        execution_aggregate_path=execution_aggregate_path,
        execution_artifact_directory=execution_artifact_directory,
        execution_aggregate_sha256=execution_aggregate_sha256,
        execution_receipt_path=execution_receipt_path,
        receipt_verification_key=receipt_verification_key,
    )
    evidence = _populate_builder(prepared).build()
    attempts = evidence.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != EXPECTED_FINAL_ATTEMPT_COUNT:
        raise RuntimeEvidenceFinalizerError("final evidence coverage differs")
    return evidence


def finalize_runtime_evidence(
    output_path: Path,
    *,
    plan: RuntimeInterventionPlan | Mapping[str, object],
    execution_projection_path: Path,
    execution_aggregate_path: Path,
    execution_artifact_directory: Path,
    execution_aggregate_sha256: str,
    execution_receipt_path: Path,
    receipt_verification_key: bytes | Ed25519PublicKey,
) -> str:
    """Verify custody and atomically publish one immutable evidence sidecar."""

    prepared = _prepare_custody(
        plan=plan,
        execution_projection_path=execution_projection_path,
        execution_aggregate_path=execution_aggregate_path,
        execution_artifact_directory=execution_artifact_directory,
        execution_aggregate_sha256=execution_aggregate_sha256,
        execution_receipt_path=execution_receipt_path,
        receipt_verification_key=receipt_verification_key,
    )
    builder = _populate_builder(prepared)
    return builder.write(Path(output_path))


write_finalized_runtime_evidence = finalize_runtime_evidence
