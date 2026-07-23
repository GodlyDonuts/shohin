"""Content-addressed immutable evidence for CTAA runtime interventions.

The sidecar records execution completion or failure, never a producer-authored
scientific pass bit. Raw bytes are deduplicated in a SHA-256 blob catalog;
attempts reference typed snapshots so an assessor can independently replay and
recompute every retained outcome without repeating trunk residuals 26 times.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import struct
from typing import Mapping, Sequence

from ctaa_intervention_protocol import (
    LOCKED_SCORED_ROW_COUNT,
    MANDATORY_INTERVENTIONS,
    MANDATORY_OPERATIONS,
    RUNTIME_PANEL_SIZE,
    RuntimeInterventionPlan,
    validate_runtime_intervention_plan,
)
from ctaa_neural_core import CTAA_MAX_STEPS, CTAA_WIDTH


EVIDENCE_SCHEMA = "r12_ctaa_runtime_evidence_v2"
ATTEMPT_SCHEMA = "r12_ctaa_runtime_attempt_evidence_v2"
SNAPSHOT_SCHEMA = "r12_ctaa_runtime_snapshot_ref_v1"
TENSOR_REF_SCHEMA = "r12_ctaa_runtime_tensor_ref_v1"
BLOB_SCHEMA = "r12_ctaa_runtime_blob_v1"
RAW_TENSOR_SCHEMA = "r12_ctaa_runtime_raw_tensor_material_v1"
CUSTODY_SCHEMA = "r12_ctaa_runtime_custody_receipts_v1"
EXPECTED_ATTEMPT_COUNT = RUNTIME_PANEL_SIZE * len(MANDATORY_OPERATIONS)

FAILURE_CODES_BY_STAGE = {
    "source": frozenset({"source_transform_error"}),
    "compile": frozenset({"compiler_error"}),
    "packet": frozenset({"packet_validation_error"}),
    "execution": frozenset(
        {"execution_error", "intervention_error", "resource_error", "timeout"}
    ),
    "query": frozenset({"query_custody_error"}),
    "custody": frozenset(
        {
            "artifact_custody_error",
            "query_isolation_probe_error",
            "source_deletion_probe_error",
        }
    ),
    "assessment": frozenset({"route_agreement_error"}),
    "validation": frozenset({"internal_error", "schema_validation_error"}),
}

_DTYPE_SIZES = {
    "bool": 1,
    "uint8": 1,
    "int64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}
_TENSOR_NAMES = frozenset(
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
_PARENT_TENSORS = _TENSOR_NAMES
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
_H19_OPERATIONS = frozenset({"h19_zero", "h19_batch_rotate", "h19_donor_transplant"})
_H29_OPERATIONS = frozenset({"h29_zero", "h29_batch_rotate", "h29_donor_transplant"})
_MIDPOINT_OPERATIONS = frozenset({"midpoint_donor_state", "midpoint_donor_action"})
_GATE_OPERATIONS = frozenset(
    {"source_deletion", "query_isolation", "state_route_composed_route_agreement"}
)
_INTERVENTION_OPERATIONS = frozenset(item.value for item in MANDATORY_INTERVENTIONS)

_RAW_TENSOR_KEYS = frozenset({"schema", "dtype", "shape", "data_hex"})
_BLOB_KEYS = frozenset({"schema", "encoding", "byte_length", "data_hex"})
_TENSOR_REF_KEYS = frozenset(
    {"schema", "dtype", "shape", "blob_sha256", "tensor_sha256"}
)
_SNAPSHOT_KEYS = frozenset(
    {
        "schema",
        "anchor_id",
        "role",
        "operation",
        "tensor_refs",
        "snapshot_sha256",
    }
)
_CUSTODY_KEYS = frozenset(
    {
        "schema",
        "packet_receipt_sha256",
        "source_deletion_receipt_sha256",
        "query_isolation_receipt_sha256",
        "execution_receipt_sha256",
    }
)
_ATTEMPT_KEYS = frozenset(
    {
        "schema",
        "attempt_index",
        "attempt_id",
        "attempt_plan_sha256",
        "anchor_id",
        "family_id",
        "operation",
        "operation_sha256",
        "kind",
        "stage",
        "timing",
        "donor_anchor_id",
        "donor_derangement_sha256",
        "mutation_payload_sha256",
        "resulting_program_source_sha256",
        "resulting_query_source_sha256",
        "resulting_packet_sha256",
        "status",
        "failure_stage",
        "failure_code",
        "failure_detail_sha256",
        "parent_snapshot_sha256",
        "intervention_snapshot_sha256",
        "donor_snapshot_sha256",
        "custody_receipts",
        "attempt_sha256",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "runtime_plan_schema",
        "plan_sha256",
        "board_manifest_sha256",
        "board_tree_sha256",
        "run_contract_sha256",
        "base_checkpoint_sha256",
        "base_raw_evidence_receipt_sha256",
        "selection_seed",
        "selection_seed_receipt_sha256",
        "training_seed",
        "arm_id",
        "partition",
        "compiler_sha256",
        "core_sha256",
        "core_kind",
        "tokenizer_sha256",
        "anchor_panel_sha256",
        "donor_registry_sha256",
        "batch_order_sha256",
        "runtime_implementation_sha256",
        "scored_row_count",
        "runtime_panel_size",
        "operation_count",
        "attempt_count",
        "runtime_attempts_affect_scored_denominator",
        "oracle_access",
        "blob_count",
        "blob_catalog_sha256",
        "blob_catalog",
        "snapshot_count",
        "snapshot_catalog_sha256",
        "snapshot_catalog",
        "attempts_sha256",
        "attempts",
        "evidence_sha256",
    }
)


class RuntimeEvidenceError(ValueError):
    """Raised when runtime evidence violates its immutable contract."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeEvidenceError(
            "CTAA runtime evidence is not canonical JSON"
        ) from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not _is_sha256(value):
        raise RuntimeEvidenceError(f"CTAA runtime {label} is not a canonical SHA-256")


def _exact_mapping(
    value: object, keys: frozenset[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeEvidenceError(f"CTAA runtime {label} schema differs")
    return value


def _require_plain_json(value: object, label: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeEvidenceError(f"CTAA runtime {label} contains non-finite data")
        raise RuntimeEvidenceError(f"CTAA runtime {label} contains an untyped float")
    if isinstance(value, list):
        for item in value:
            _require_plain_json(item, label)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeEvidenceError(f"CTAA runtime {label} has a non-string key")
            _require_plain_json(item, label)
        return
    raise RuntimeEvidenceError(f"CTAA runtime {label} contains a non-JSON value")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeEvidenceError(f"CTAA runtime duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(data: bytes) -> dict[str, object]:
    def reject_nonfinite(value: str) -> None:
        raise RuntimeEvidenceError(f"CTAA runtime non-finite JSON constant: {value}")

    try:
        decoded = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeEvidenceError("CTAA runtime evidence JSON differs") from error
    if not isinstance(decoded, dict):
        raise RuntimeEvidenceError("CTAA runtime evidence root differs")
    _require_plain_json(decoded, "evidence")
    return decoded


def _product(shape: Sequence[int]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def _validate_finite_float_bytes(dtype: str, data: bytes) -> None:
    if dtype == "bfloat16":
        values = struct.unpack(f"<{len(data) // 2}H", data)
        if any(value & 0x7F80 == 0x7F80 for value in values):
            raise RuntimeEvidenceError("CTAA runtime tensor contains non-finite data")
        return
    formats = {"float16": "e", "float32": "f"}
    if dtype in formats:
        width = _DTYPE_SIZES[dtype]
        values = struct.unpack(f"<{len(data) // width}{formats[dtype]}", data)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeEvidenceError("CTAA runtime tensor contains non-finite data")


def make_raw_tensor(dtype: str, shape: Sequence[int], data: bytes) -> dict[str, object]:
    """Create transient tensor material; published snapshots contain only refs."""

    value = {
        "schema": RAW_TENSOR_SCHEMA,
        "dtype": dtype,
        "shape": list(shape),
        "data_hex": bytes(data).hex(),
    }
    _validate_raw_tensor(value, "new")
    return value


def _validate_raw_tensor(value: object, label: str) -> tuple[str, list[int], bytes]:
    row = _exact_mapping(value, _RAW_TENSOR_KEYS, f"{label} raw tensor")
    dtype = row["dtype"]
    shape = row["shape"]
    data_hex = row["data_hex"]
    if row["schema"] != RAW_TENSOR_SCHEMA or dtype not in _DTYPE_SIZES:
        raise RuntimeEvidenceError(f"CTAA runtime {label} tensor metadata differs")
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(item) is not int or item <= 0 for item in shape)
    ):
        raise RuntimeEvidenceError(f"CTAA runtime {label} tensor shape differs")
    if (
        not isinstance(data_hex, str)
        or len(data_hex) % 2
        or any(character not in "0123456789abcdef" for character in data_hex)
    ):
        raise RuntimeEvidenceError(f"CTAA runtime {label} tensor bytes differ")
    data = bytes.fromhex(data_hex)
    if len(data) != _product(shape) * _DTYPE_SIZES[str(dtype)]:
        raise RuntimeEvidenceError(f"CTAA runtime {label} tensor byte count differs")
    if dtype == "bool" and any(item not in (0, 1) for item in data):
        raise RuntimeEvidenceError(f"CTAA runtime {label} bool tensor differs")
    _validate_finite_float_bytes(str(dtype), data)
    return str(dtype), list(shape), data


def _validate_tensor_contract(name: str, dtype: str, shape: list[int]) -> None:
    fixed = {
        "packet": ("uint8", [56]),
        "state_route": ("uint8", [CTAA_MAX_STEPS + 1, CTAA_WIDTH]),
        "composed_route": ("uint8", [CTAA_MAX_STEPS + 1, CTAA_WIDTH]),
        "halt_mask": ("bool", [CTAA_MAX_STEPS + 1]),
        "terminal_state": ("uint8", [CTAA_WIDTH]),
        "query_position": ("uint8", [1]),
        "answer": ("uint8", [1]),
    }
    if name in fixed and (dtype, shape) != fixed[name]:
        raise RuntimeEvidenceError(f"CTAA runtime {name} tensor contract differs")
    if name in {"h19_residual", "h29_residual"} and (
        dtype not in {"float16", "bfloat16", "float32"} or shape[-1] != 576
    ):
        raise RuntimeEvidenceError(f"CTAA runtime {name} tensor contract differs")


def _operation_tensor_set(operation: str) -> frozenset[str]:
    if operation in _H19_OPERATIONS:
        return _OUTPUT_TENSORS | {"h19_residual"}
    if operation in _H29_OPERATIONS:
        return _OUTPUT_TENSORS | {"h29_residual"}
    if operation in _MIDPOINT_OPERATIONS:
        return _OUTPUT_TENSORS
    if operation == "late_query_swap":
        return frozenset({"terminal_state", "query_position", "answer"})
    if operation in _GATE_OPERATIONS:
        return frozenset()
    if operation in _INTERVENTION_OPERATIONS:
        return _OUTPUT_TENSORS | {"packet"}
    raise RuntimeEvidenceError("CTAA runtime snapshot operation is unknown")


def _expected_snapshot_tensors(role: str, operation: str | None) -> frozenset[str]:
    if role == "parent" and operation is None:
        return _PARENT_TENSORS
    if role == "intervention" and isinstance(operation, str):
        required = _operation_tensor_set(operation)
        if required:
            return required
    raise RuntimeEvidenceError("CTAA runtime snapshot role/operation differs")


def _blob_entry(data: bytes) -> dict[str, object]:
    return {
        "schema": BLOB_SCHEMA,
        "encoding": "hex",
        "byte_length": len(data),
        "data_hex": data.hex(),
    }


def _validate_blob(key: str, value: object) -> bytes:
    row = _exact_mapping(value, _BLOB_KEYS, "blob")
    if (
        row["schema"] != BLOB_SCHEMA
        or row["encoding"] != "hex"
        or type(row["byte_length"]) is not int
        or row["byte_length"] < 0
        or not isinstance(row["data_hex"], str)
        or len(row["data_hex"]) % 2
        or any(character not in "0123456789abcdef" for character in row["data_hex"])
    ):
        raise RuntimeEvidenceError("CTAA runtime blob schema differs")
    data = bytes.fromhex(row["data_hex"])
    if len(data) != row["byte_length"] or hashlib.sha256(data).hexdigest() != key:
        raise RuntimeEvidenceError("CTAA runtime blob content-address differs")
    return data


def _tensor_ref(dtype: str, shape: list[int], blob_sha256: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": TENSOR_REF_SCHEMA,
        "dtype": dtype,
        "shape": shape,
        "blob_sha256": blob_sha256,
    }
    value["tensor_sha256"] = _canonical_sha256(value)
    return value


def _validate_tensor_ref(
    name: str, value: object, blobs: Mapping[str, bytes]
) -> dict[str, object]:
    row = dict(_exact_mapping(value, _TENSOR_REF_KEYS, f"{name} tensor ref"))
    if row["schema"] != TENSOR_REF_SCHEMA or row["dtype"] not in _DTYPE_SIZES:
        raise RuntimeEvidenceError(f"CTAA runtime {name} tensor-ref metadata differs")
    shape = row["shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(item) is not int or item <= 0 for item in shape)
    ):
        raise RuntimeEvidenceError(f"CTAA runtime {name} tensor-ref shape differs")
    _require_sha256(row["blob_sha256"], f"{name} blob reference")
    if row["blob_sha256"] not in blobs:
        raise RuntimeEvidenceError(f"CTAA runtime {name} references a missing blob")
    dtype = str(row["dtype"])
    data = blobs[str(row["blob_sha256"])]
    if len(data) != _product(shape) * _DTYPE_SIZES[dtype]:
        raise RuntimeEvidenceError(f"CTAA runtime {name} referenced byte count differs")
    if dtype == "bool" and any(item not in (0, 1) for item in data):
        raise RuntimeEvidenceError(f"CTAA runtime {name} bool tensor differs")
    _validate_finite_float_bytes(dtype, data)
    _validate_tensor_contract(name, dtype, shape)
    expected_sha = _canonical_sha256(
        {key: item for key, item in row.items() if key != "tensor_sha256"}
    )
    if row["tensor_sha256"] != expected_sha:
        raise RuntimeEvidenceError(f"CTAA runtime {name} tensor-ref hash differs")
    return row


def _validate_snapshot(
    key: str,
    value: object,
    blobs: Mapping[str, bytes],
    anchor_ids: frozenset[str],
) -> tuple[dict[str, object], frozenset[str]]:
    row = dict(_exact_mapping(value, _SNAPSHOT_KEYS, "snapshot"))
    if (
        row["schema"] != SNAPSHOT_SCHEMA
        or row["anchor_id"] not in anchor_ids
        or row["role"] not in {"parent", "intervention"}
        or not isinstance(row["tensor_refs"], Mapping)
    ):
        raise RuntimeEvidenceError("CTAA runtime snapshot metadata differs")
    expected_names = _expected_snapshot_tensors(
        str(row["role"]),
        row["operation"],  # type: ignore[arg-type]
    )
    if set(row["tensor_refs"]) != expected_names:
        raise RuntimeEvidenceError("CTAA runtime operation-specific tensor set differs")
    refs = {
        name: _validate_tensor_ref(name, tensor, blobs)
        for name, tensor in row["tensor_refs"].items()
    }
    row["tensor_refs"] = refs
    expected_sha = _canonical_sha256(
        {name: item for name, item in row.items() if name != "snapshot_sha256"}
    )
    if key != expected_sha or row["snapshot_sha256"] != expected_sha:
        raise RuntimeEvidenceError("CTAA runtime snapshot content-address differs")
    return row, frozenset(str(item["blob_sha256"]) for item in refs.values())


def make_custody_receipts(
    *,
    packet_receipt_sha256: str | None,
    source_deletion_receipt_sha256: str | None,
    query_isolation_receipt_sha256: str | None,
    execution_receipt_sha256: str | None,
) -> dict[str, object]:
    value = {
        "schema": CUSTODY_SCHEMA,
        "packet_receipt_sha256": packet_receipt_sha256,
        "source_deletion_receipt_sha256": source_deletion_receipt_sha256,
        "query_isolation_receipt_sha256": query_isolation_receipt_sha256,
        "execution_receipt_sha256": execution_receipt_sha256,
    }
    return _validate_custody_receipts(value, required=frozenset())


def _required_receipts(operation: str) -> frozenset[str]:
    if operation == "source_deletion":
        return frozenset({"source_deletion_receipt_sha256"})
    if operation == "query_isolation":
        return frozenset({"query_isolation_receipt_sha256"})
    if operation == "state_route_composed_route_agreement":
        return frozenset({"execution_receipt_sha256"})
    return frozenset(
        {
            "packet_receipt_sha256",
            "source_deletion_receipt_sha256",
            "query_isolation_receipt_sha256",
            "execution_receipt_sha256",
        }
    )


def _validate_custody_receipts(
    value: object, *, required: frozenset[str]
) -> dict[str, object]:
    row = dict(_exact_mapping(value, _CUSTODY_KEYS, "custody receipts"))
    if row["schema"] != CUSTODY_SCHEMA:
        raise RuntimeEvidenceError("CTAA runtime custody receipt schema differs")
    for key, item in row.items():
        if key != "schema":
            _require_sha256(item, key, nullable=key not in required)
    return row


def _plan_indexes(plan: RuntimeInterventionPlan) -> tuple[dict[str, object], ...]:
    anchors = {item.anchor_id: item for item in plan.anchors}
    operations = {item.operation: item for item in plan.operations}
    donors = {
        item.operation: item.derangement_sha256 for item in plan.donor_derangements
    }
    return tuple(
        {
            "attempt": attempt,
            "operation": operations[attempt.operation],
            "family_id": anchors[attempt.anchor_id].family_id,
            "donor_derangement_sha256": donors.get(attempt.operation),
        }
        for attempt in plan.attempts
    )


def _attempt_prefix(index: Mapping[str, object]) -> dict[str, object]:
    attempt = index["attempt"]
    operation = index["operation"]
    return {
        "schema": ATTEMPT_SCHEMA,
        "attempt_index": attempt.attempt_index,  # type: ignore[attr-defined]
        "attempt_id": attempt.attempt_id,  # type: ignore[attr-defined]
        "attempt_plan_sha256": attempt.attempt_plan_sha256,  # type: ignore[attr-defined]
        "anchor_id": attempt.anchor_id,  # type: ignore[attr-defined]
        "family_id": index["family_id"],
        "operation": attempt.operation,  # type: ignore[attr-defined]
        "operation_sha256": attempt.operation_sha256,  # type: ignore[attr-defined]
        "kind": operation.kind.value,  # type: ignore[attr-defined]
        "stage": operation.stage.value,  # type: ignore[attr-defined]
        "timing": operation.timing,  # type: ignore[attr-defined]
        "donor_anchor_id": attempt.donor_anchor_id,  # type: ignore[attr-defined]
        "donor_derangement_sha256": index["donor_derangement_sha256"],
        "mutation_payload_sha256": attempt.mutation_payload_sha256,  # type: ignore[attr-defined]
        "resulting_program_source_sha256": (
            attempt.resulting_program_source_sha256  # type: ignore[attr-defined]
        ),
        "resulting_query_source_sha256": (
            attempt.resulting_query_source_sha256  # type: ignore[attr-defined]
        ),
        "resulting_packet_sha256": attempt.resulting_packet_sha256,  # type: ignore[attr-defined]
    }


def _commit_attempt(value: dict[str, object]) -> dict[str, object]:
    value["attempt_sha256"] = _canonical_sha256(value)
    return value


class RuntimeEvidenceBuilder:
    """Intern raw bytes and append all attempt outcomes in frozen plan order."""

    def __init__(self, plan: RuntimeInterventionPlan | Mapping[str, object]) -> None:
        self.plan = validate_runtime_intervention_plan(plan)
        self._indexes = _plan_indexes(self.plan)
        self._anchor_ids = frozenset(item.anchor_id for item in self.plan.anchors)
        self._blobs: dict[str, dict[str, object]] = {}
        self._snapshots: dict[str, dict[str, object]] = {}
        self._attempts: list[dict[str, object]] = []

    def add_snapshot(
        self,
        *,
        anchor_id: str,
        role: str,
        operation: str | None,
        tensors: Mapping[str, Mapping[str, object]],
    ) -> str:
        if anchor_id not in self._anchor_ids:
            raise RuntimeEvidenceError("CTAA runtime snapshot anchor differs")
        expected_names = _expected_snapshot_tensors(role, operation)
        if set(tensors) != expected_names:
            raise RuntimeEvidenceError(
                "CTAA runtime operation-specific tensor set differs"
            )
        refs: dict[str, dict[str, object]] = {}
        for name, material in tensors.items():
            dtype, shape, data = _validate_raw_tensor(material, name)
            _validate_tensor_contract(name, dtype, shape)
            blob_sha = hashlib.sha256(data).hexdigest()
            entry = _blob_entry(data)
            existing = self._blobs.setdefault(blob_sha, entry)
            if existing != entry:  # pragma: no cover - SHA-256 collision guard
                raise RuntimeEvidenceError("CTAA runtime blob collision differs")
            refs[name] = _tensor_ref(dtype, shape, blob_sha)
        snapshot: dict[str, object] = {
            "schema": SNAPSHOT_SCHEMA,
            "anchor_id": anchor_id,
            "role": role,
            "operation": operation,
            "tensor_refs": refs,
        }
        snapshot_sha = _canonical_sha256(snapshot)
        snapshot["snapshot_sha256"] = snapshot_sha
        existing_snapshot = self._snapshots.setdefault(snapshot_sha, snapshot)
        if existing_snapshot != snapshot:  # pragma: no cover - collision guard
            raise RuntimeEvidenceError("CTAA runtime snapshot collision differs")
        return snapshot_sha

    def _next(self) -> Mapping[str, object]:
        if len(self._attempts) >= EXPECTED_ATTEMPT_COUNT:
            raise RuntimeEvidenceError("CTAA runtime evidence has excess attempts")
        return self._indexes[len(self._attempts)]

    def add_success(
        self,
        *,
        parent_snapshot_sha256: str,
        intervention_snapshot_sha256: str | None,
        donor_snapshot_sha256: str | None,
        custody_receipts: Mapping[str, object],
    ) -> None:
        index = self._next()
        operation = index["attempt"].operation  # type: ignore[attr-defined]
        value = {
            **_attempt_prefix(index),
            "status": "success",
            "failure_stage": None,
            "failure_code": None,
            "failure_detail_sha256": None,
            "parent_snapshot_sha256": parent_snapshot_sha256,
            "intervention_snapshot_sha256": intervention_snapshot_sha256,
            "donor_snapshot_sha256": donor_snapshot_sha256,
            "custody_receipts": _validate_custody_receipts(
                custody_receipts, required=_required_receipts(operation)
            ),
        }
        self._attempts.append(_commit_attempt(value))

    def add_failure(
        self,
        *,
        failure_stage: str,
        failure_code: str,
        failure_detail_sha256: str,
        custody_receipts: Mapping[str, object] | None = None,
        parent_snapshot_sha256: str | None = None,
        intervention_snapshot_sha256: str | None = None,
        donor_snapshot_sha256: str | None = None,
    ) -> None:
        if failure_code not in FAILURE_CODES_BY_STAGE.get(failure_stage, frozenset()):
            raise RuntimeEvidenceError("CTAA runtime failure stage/code differs")
        _require_sha256(failure_detail_sha256, "failure detail")
        index = self._next()
        receipts = custody_receipts or make_custody_receipts(
            packet_receipt_sha256=None,
            source_deletion_receipt_sha256=None,
            query_isolation_receipt_sha256=None,
            execution_receipt_sha256=None,
        )
        value = {
            **_attempt_prefix(index),
            "status": "failure",
            "failure_stage": failure_stage,
            "failure_code": failure_code,
            "failure_detail_sha256": failure_detail_sha256,
            "parent_snapshot_sha256": parent_snapshot_sha256,
            "intervention_snapshot_sha256": intervention_snapshot_sha256,
            "donor_snapshot_sha256": donor_snapshot_sha256,
            "custody_receipts": _validate_custody_receipts(
                receipts, required=frozenset()
            ),
        }
        self._attempts.append(_commit_attempt(value))

    def build(self) -> dict[str, object]:
        return make_runtime_evidence(
            self.plan, self._attempts, self._blobs, self._snapshots
        )

    def write(self, path: Path) -> str:
        return write_runtime_evidence(
            path, self.plan, self._attempts, self._blobs, self._snapshots
        )


def _expected_top_level(plan: RuntimeInterventionPlan) -> dict[str, object]:
    bindings = plan.bindings
    return {
        "schema": EVIDENCE_SCHEMA,
        "runtime_plan_schema": plan.schema,
        "plan_sha256": plan.plan_sha256,
        "board_manifest_sha256": bindings.board_manifest_sha256,
        "board_tree_sha256": bindings.board_tree_sha256,
        "run_contract_sha256": bindings.run_contract_sha256,
        "base_checkpoint_sha256": bindings.base_checkpoint_sha256,
        "base_raw_evidence_receipt_sha256": (bindings.base_raw_evidence_receipt_sha256),
        "selection_seed": bindings.selection_seed,
        "selection_seed_receipt_sha256": bindings.selection_seed_receipt_sha256,
        "training_seed": bindings.training_seed,
        "arm_id": bindings.arm_id,
        "partition": bindings.partition.value,
        "compiler_sha256": bindings.compiler_sha256,
        "core_sha256": bindings.core_sha256,
        "core_kind": bindings.core_kind,
        "tokenizer_sha256": bindings.tokenizer_sha256,
        "anchor_panel_sha256": plan.anchor_panel_sha256,
        "donor_registry_sha256": plan.donor_registry_sha256,
        "batch_order_sha256": bindings.batch_order_sha256,
        "runtime_implementation_sha256": bindings.runtime_implementation_sha256,
        "scored_row_count": LOCKED_SCORED_ROW_COUNT,
        "runtime_panel_size": RUNTIME_PANEL_SIZE,
        "operation_count": len(MANDATORY_OPERATIONS),
        "attempt_count": EXPECTED_ATTEMPT_COUNT,
        "runtime_attempts_affect_scored_denominator": False,
        "oracle_access": 0,
    }


def _validate_snapshot_reference(
    value: object,
    snapshots: Mapping[str, Mapping[str, object]],
    *,
    nullable: bool,
    anchor_id: str,
    role: str,
    operation: str | None,
) -> str | None:
    if nullable and value is None:
        return None
    _require_sha256(value, "snapshot reference")
    if value not in snapshots:
        raise RuntimeEvidenceError("CTAA runtime attempt references a missing snapshot")
    snapshot = snapshots[str(value)]
    if (
        snapshot["anchor_id"] != anchor_id
        or snapshot["role"] != role
        or snapshot["operation"] != operation
    ):
        raise RuntimeEvidenceError("CTAA runtime snapshot reference binding differs")
    return str(value)


def _snapshot_bytes(
    snapshot: Mapping[str, object],
    name: str,
    blobs: Mapping[str, bytes],
) -> bytes:
    refs = snapshot["tensor_refs"]
    if not isinstance(refs, Mapping) or name not in refs:
        raise RuntimeEvidenceError(f"CTAA runtime snapshot lacks {name}")
    reference = refs[name]
    if not isinstance(reference, Mapping):
        raise RuntimeEvidenceError(f"CTAA runtime {name} reference differs")
    digest = reference.get("blob_sha256")
    if not isinstance(digest, str) or digest not in blobs:
        raise RuntimeEvidenceError(f"CTAA runtime {name} blob differs")
    return blobs[digest]


def _validate_answer_consistency(
    snapshot: Mapping[str, object], blobs: Mapping[str, bytes]
) -> None:
    terminal = _snapshot_bytes(snapshot, "terminal_state", blobs)
    position = _snapshot_bytes(snapshot, "query_position", blobs)
    answer = _snapshot_bytes(snapshot, "answer", blobs)
    route = _snapshot_bytes(snapshot, "state_route", blobs)
    if (
        len(terminal) != CTAA_WIDTH
        or len(position) != 1
        or len(answer) != 1
        or position[0] >= CTAA_WIDTH
        or route[-CTAA_WIDTH:] != terminal
        or answer[0] != terminal[position[0]]
    ):
        raise RuntimeEvidenceError("CTAA runtime answer/terminal route differs")


def _validate_success_mechanics(
    *,
    attempt: object,
    operation: str,
    parent: Mapping[str, object],
    intervention: Mapping[str, object] | None,
    donor: Mapping[str, object] | None,
    blobs: Mapping[str, bytes],
    anchors: Mapping[str, object],
) -> None:
    parent_packet = _snapshot_bytes(parent, "packet", blobs)
    anchor = anchors[attempt.anchor_id]  # type: ignore[attr-defined]
    if hashlib.sha256(parent_packet).hexdigest() != anchor.packet_sha256:  # type: ignore[attr-defined]
        raise RuntimeEvidenceError("CTAA runtime parent packet/anchor differs")
    _validate_answer_consistency(parent, blobs)
    parent_query = _snapshot_bytes(parent, "query_position", blobs)
    if parent_query != bytes([anchor.query_position]):  # type: ignore[attr-defined]
        raise RuntimeEvidenceError("CTAA runtime parent query/anchor differs")
    if intervention is None:
        return

    if operation != "late_query_swap":
        _validate_answer_consistency(intervention, blobs)
        if _snapshot_bytes(intervention, "query_position", blobs) != parent_query:
            raise RuntimeEvidenceError("CTAA runtime intervention query changed early")

    resulting_packet = attempt.resulting_packet_sha256  # type: ignore[attr-defined]
    refs = intervention["tensor_refs"]
    if not isinstance(refs, Mapping):
        raise RuntimeEvidenceError("CTAA runtime intervention refs differ")
    if resulting_packet is not None:
        packet = _snapshot_bytes(intervention, "packet", blobs)
        if (
            hashlib.sha256(packet).hexdigest() != resulting_packet
            or packet == parent_packet
        ):
            raise RuntimeEvidenceError(
                "CTAA runtime committed packet intervention differs"
            )

    residual_name = (
        "h19_residual"
        if operation in _H19_OPERATIONS
        else "h29_residual"
        if operation in _H29_OPERATIONS
        else None
    )
    if residual_name is not None:
        mutated = _snapshot_bytes(intervention, residual_name, blobs)
        original = _snapshot_bytes(parent, residual_name, blobs)
        if operation.endswith("_zero"):
            if any(mutated) or mutated == original:
                raise RuntimeEvidenceError("CTAA runtime residual-zero is a no-op")
        else:
            if donor is None:
                raise RuntimeEvidenceError("CTAA runtime residual donor is absent")
            expected = _snapshot_bytes(donor, residual_name, blobs)
            if mutated != expected or mutated == original:
                raise RuntimeEvidenceError("CTAA runtime residual donor differs")

    if operation == "packet_transplant":
        if donor is None:
            raise RuntimeEvidenceError("CTAA runtime packet donor is absent")
        packet = _snapshot_bytes(intervention, "packet", blobs)
        expected = _snapshot_bytes(donor, "packet", blobs)
        if packet != expected or packet == parent_packet:
            raise RuntimeEvidenceError("CTAA runtime packet transplant differs")

    if operation == "late_query_swap":
        if donor is None:
            raise RuntimeEvidenceError("CTAA runtime late-query donor is absent")
        terminal = _snapshot_bytes(intervention, "terminal_state", blobs)
        position = _snapshot_bytes(intervention, "query_position", blobs)
        answer = _snapshot_bytes(intervention, "answer", blobs)
        donor_position = _snapshot_bytes(donor, "query_position", blobs)
        if (
            terminal != _snapshot_bytes(parent, "terminal_state", blobs)
            or position != donor_position
            or position == parent_query
            or answer != bytes([terminal[position[0]]])
        ):
            raise RuntimeEvidenceError("CTAA runtime late-query swap differs")


def _validate_attempt(
    value: object,
    expected: Mapping[str, object],
    snapshots: Mapping[str, Mapping[str, object]],
    blobs: Mapping[str, bytes],
    anchors: Mapping[str, object],
) -> tuple[dict[str, object], frozenset[str]]:
    row = dict(_exact_mapping(value, _ATTEMPT_KEYS, "attempt evidence"))
    for key, item in _attempt_prefix(expected).items():
        if row[key] != item:
            raise RuntimeEvidenceError(f"CTAA runtime attempt {key} binding differs")
    attempt = expected["attempt"]
    operation = attempt.operation  # type: ignore[attr-defined]
    used: set[str] = set()
    status = row["status"]
    if status == "success":
        if any(
            row[key] is not None
            for key in ("failure_stage", "failure_code", "failure_detail_sha256")
        ):
            raise RuntimeEvidenceError("CTAA runtime success carries failure metadata")
        parent = _validate_snapshot_reference(
            row["parent_snapshot_sha256"],
            snapshots,
            nullable=False,
            anchor_id=attempt.anchor_id,  # type: ignore[attr-defined]
            role="parent",
            operation=None,
        )
        intervention_required = operation not in _GATE_OPERATIONS
        intervention = _validate_snapshot_reference(
            row["intervention_snapshot_sha256"],
            snapshots,
            nullable=not intervention_required,
            anchor_id=attempt.anchor_id,  # type: ignore[attr-defined]
            role="intervention",
            operation=operation,
        )
        if not intervention_required and intervention is not None:
            raise RuntimeEvidenceError("CTAA runtime gate has an intervention snapshot")
        donor_required = attempt.donor_anchor_id is not None  # type: ignore[attr-defined]
        donor = _validate_snapshot_reference(
            row["donor_snapshot_sha256"],
            snapshots,
            nullable=not donor_required,
            anchor_id=attempt.donor_anchor_id or "",  # type: ignore[attr-defined]
            role="parent",
            operation=None,
        )
        if not donor_required and donor is not None:
            raise RuntimeEvidenceError(
                "CTAA runtime non-donor attempt has a donor snapshot"
            )
        used.update(item for item in (parent, intervention, donor) if item is not None)
        _validate_success_mechanics(
            attempt=attempt,
            operation=operation,
            parent=snapshots[str(parent)],
            intervention=(
                snapshots[str(intervention)] if intervention is not None else None
            ),
            donor=snapshots[str(donor)] if donor is not None else None,
            blobs=blobs,
            anchors=anchors,
        )
        row["custody_receipts"] = _validate_custody_receipts(
            row["custody_receipts"], required=_required_receipts(operation)
        )
    elif status == "failure":
        stage = row["failure_stage"]
        code = row["failure_code"]
        if (
            not isinstance(stage, str)
            or not isinstance(code, str)
            or code not in FAILURE_CODES_BY_STAGE.get(stage, frozenset())
        ):
            raise RuntimeEvidenceError("CTAA runtime failure stage/code differs")
        _require_sha256(row["failure_detail_sha256"], "failure detail")
        optional_refs = (
            (
                "parent_snapshot_sha256",
                attempt.anchor_id,  # type: ignore[attr-defined]
                "parent",
                None,
            ),
            (
                "intervention_snapshot_sha256",
                attempt.anchor_id,  # type: ignore[attr-defined]
                "intervention",
                operation,
            ),
            (
                "donor_snapshot_sha256",
                attempt.donor_anchor_id or "",  # type: ignore[attr-defined]
                "parent",
                None,
            ),
        )
        for key, anchor_id, role, snapshot_operation in optional_refs:
            if row[key] is not None:
                if role == "intervention" and operation in _GATE_OPERATIONS:
                    raise RuntimeEvidenceError(
                        "CTAA runtime gate has an intervention snapshot"
                    )
                reference = _validate_snapshot_reference(
                    row[key],
                    snapshots,
                    nullable=False,
                    anchor_id=anchor_id,
                    role=role,
                    operation=snapshot_operation,
                )
                used.add(str(reference))
        row["custody_receipts"] = _validate_custody_receipts(
            row["custody_receipts"], required=frozenset({"execution_receipt_sha256"})
        )
    else:
        raise RuntimeEvidenceError("CTAA runtime attempt status differs")
    expected_sha = _canonical_sha256(
        {key: item for key, item in row.items() if key != "attempt_sha256"}
    )
    if row["attempt_sha256"] != expected_sha:
        raise RuntimeEvidenceError("CTAA runtime attempt commitment differs")
    return row, frozenset(used)


def make_runtime_evidence(
    plan: RuntimeInterventionPlan | Mapping[str, object],
    attempts: Sequence[Mapping[str, object]],
    blob_catalog: Mapping[str, Mapping[str, object]],
    snapshot_catalog: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    frozen_plan = validate_runtime_intervention_plan(plan)
    evidence: dict[str, object] = {
        **_expected_top_level(frozen_plan),
        "blob_count": len(blob_catalog),
        "blob_catalog_sha256": _canonical_sha256(blob_catalog),
        "blob_catalog": dict(blob_catalog),
        "snapshot_count": len(snapshot_catalog),
        "snapshot_catalog_sha256": _canonical_sha256(snapshot_catalog),
        "snapshot_catalog": dict(snapshot_catalog),
        "attempts_sha256": _canonical_sha256(attempts),
        "attempts": list(attempts),
    }
    evidence["evidence_sha256"] = _canonical_sha256(evidence)
    return validate_runtime_evidence(evidence, frozen_plan)


def validate_runtime_evidence(
    value: Mapping[str, object],
    plan: RuntimeInterventionPlan | Mapping[str, object],
) -> dict[str, object]:
    frozen_plan = validate_runtime_intervention_plan(plan)
    _require_plain_json(value, "evidence")
    evidence = dict(_exact_mapping(value, _EVIDENCE_KEYS, "evidence"))
    for key, expected in _expected_top_level(frozen_plan).items():
        if evidence[key] != expected:
            raise RuntimeEvidenceError(f"CTAA runtime evidence {key} binding differs")
    raw_blobs = evidence["blob_catalog"]
    if not isinstance(raw_blobs, Mapping):
        raise RuntimeEvidenceError("CTAA runtime blob catalog differs")
    blobs: dict[str, bytes] = {}
    for key, item in raw_blobs.items():
        _require_sha256(key, "blob key")
        blobs[str(key)] = _validate_blob(str(key), item)
    if evidence["blob_count"] != len(blobs) or evidence[
        "blob_catalog_sha256"
    ] != _canonical_sha256(raw_blobs):
        raise RuntimeEvidenceError("CTAA runtime blob catalog commitment differs")
    raw_snapshots = evidence["snapshot_catalog"]
    if not isinstance(raw_snapshots, Mapping):
        raise RuntimeEvidenceError("CTAA runtime snapshot catalog differs")
    anchor_ids = frozenset(item.anchor_id for item in frozen_plan.anchors)
    snapshots: dict[str, dict[str, object]] = {}
    snapshot_blobs: dict[str, frozenset[str]] = {}
    for key, item in raw_snapshots.items():
        _require_sha256(key, "snapshot key")
        snapshot, used_blobs = _validate_snapshot(str(key), item, blobs, anchor_ids)
        snapshots[str(key)] = snapshot
        snapshot_blobs[str(key)] = used_blobs
    if evidence["snapshot_count"] != len(snapshots) or evidence[
        "snapshot_catalog_sha256"
    ] != _canonical_sha256(raw_snapshots):
        raise RuntimeEvidenceError("CTAA runtime snapshot catalog commitment differs")
    raw_attempts = evidence["attempts"]
    if (
        not isinstance(raw_attempts, list)
        or len(raw_attempts) != EXPECTED_ATTEMPT_COUNT
    ):
        raise RuntimeEvidenceError("CTAA runtime evidence attempt count differs")
    indexes = _plan_indexes(frozen_plan)
    anchors = {item.anchor_id: item for item in frozen_plan.anchors}
    attempts = []
    used_snapshots: set[str] = set()
    for index, item in enumerate(raw_attempts):
        attempt, references = _validate_attempt(
            item, indexes[index], snapshots, blobs, anchors
        )
        attempts.append(attempt)
        used_snapshots.update(references)
    execution_receipts = {
        item["custody_receipts"]["execution_receipt_sha256"] for item in attempts
    }
    if len(execution_receipts) != 1 or None in execution_receipts:
        raise RuntimeEvidenceError(
            "CTAA runtime evidence does not have one execution receipt"
        )
    if len({item["attempt_id"] for item in attempts}) != EXPECTED_ATTEMPT_COUNT:
        raise RuntimeEvidenceError("CTAA runtime attempt identity is duplicated")
    if set(snapshots) != used_snapshots:
        raise RuntimeEvidenceError(
            "CTAA runtime snapshot catalog contains unused entries"
        )
    used_blobs = (
        set().union(*(snapshot_blobs[key] for key in used_snapshots))
        if used_snapshots
        else set()
    )
    if set(blobs) != used_blobs:
        raise RuntimeEvidenceError("CTAA runtime blob catalog contains unused entries")
    expected_attempts_sha = _canonical_sha256(attempts)
    if evidence["attempts_sha256"] != expected_attempts_sha:
        raise RuntimeEvidenceError("CTAA runtime attempt registry hash differs")
    evidence["blob_catalog"] = dict(raw_blobs)
    evidence["snapshot_catalog"] = snapshots
    evidence["attempts"] = attempts
    expected_evidence_sha = _canonical_sha256(
        {key: item for key, item in evidence.items() if key != "evidence_sha256"}
    )
    if evidence["evidence_sha256"] != expected_evidence_sha:
        raise RuntimeEvidenceError("CTAA runtime evidence commitment differs")
    return evidence


def _read_immutable_bytes(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeEvidenceError(
            "CTAA runtime evidence input is unavailable"
        ) from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_mode & 0o222
        or metadata.st_nlink != 1
    ):
        raise RuntimeEvidenceError(
            "CTAA runtime evidence input is not a single-link immutable file"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeEvidenceError(
            "CTAA runtime evidence cannot be opened safely"
        ) from error
    try:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if (
        identity(metadata) != identity(before)
        or identity(before) != identity(after)
        or after.st_mode & 0o222
        or after.st_nlink != 1
    ):
        raise RuntimeEvidenceError("CTAA runtime evidence changed while being read")
    return b"".join(chunks)


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return (_canonical_json(dict(value)) + "\n").encode("ascii")


def read_runtime_evidence(
    path: Path,
    plan: RuntimeInterventionPlan | Mapping[str, object],
    *,
    expected_file_sha256: str | None = None,
) -> dict[str, object]:
    payload = _read_immutable_bytes(Path(path))
    observed_sha = hashlib.sha256(payload).hexdigest()
    if expected_file_sha256 is not None and observed_sha != expected_file_sha256:
        raise RuntimeEvidenceError("CTAA runtime evidence file hash differs")
    evidence = _decode_json(payload)
    if payload != _canonical_bytes(evidence):
        raise RuntimeEvidenceError("CTAA runtime evidence file is not canonical JSON")
    return validate_runtime_evidence(evidence, plan)


def _safe_output_parent(path: Path) -> Path:
    absolute = path.absolute()
    parent = absolute.parent
    try:
        resolved = parent.resolve(strict=True)
        metadata = parent.lstat()
    except OSError as error:
        raise RuntimeEvidenceError(
            "CTAA runtime evidence output parent differs"
        ) from error
    if resolved != parent or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeEvidenceError("CTAA runtime evidence output parent is not direct")
    return absolute


def _publish_immutable_once(path: Path, payload: bytes) -> None:
    target = _safe_output_parent(path)
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing CTAA runtime evidence: {target}")
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RuntimeEvidenceError(
                    "CTAA runtime evidence write made no progress"
                )
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, target, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.lexists(temporary):
            os.chmod(temporary, 0o600, follow_symlinks=False)
            temporary.unlink()


def write_runtime_evidence(
    path: Path,
    plan: RuntimeInterventionPlan | Mapping[str, object],
    attempts: Sequence[Mapping[str, object]],
    blob_catalog: Mapping[str, Mapping[str, object]],
    snapshot_catalog: Mapping[str, Mapping[str, object]],
) -> str:
    """Validate and atomically publish one canonical immutable sidecar."""

    target = _safe_output_parent(Path(path))
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing CTAA runtime evidence: {target}")
    frozen_plan = validate_runtime_intervention_plan(plan)
    evidence = make_runtime_evidence(
        frozen_plan, attempts, blob_catalog, snapshot_catalog
    )
    payload = _canonical_bytes(evidence)
    file_sha256 = hashlib.sha256(payload).hexdigest()
    _publish_immutable_once(target, payload)
    read_runtime_evidence(target, frozen_plan, expected_file_sha256=file_sha256)
    return file_sha256
