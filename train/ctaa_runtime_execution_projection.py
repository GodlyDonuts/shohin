"""Query-blind execution projection for a frozen CTAA runtime plan.

The complete runtime plan commits late-query positions and donor queries.  It
must therefore stay outside the source/compile/packet/execution sandbox.  This
module derives the only plan view that may cross that boundary.  The view is
fully recomputable from the signed plan while omitting every query-bearing
field and the late-query intervention itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from ctaa_intervention_protocol import (
    GateFamily,
    InterventionFamily,
    MANDATORY_OPERATIONS,
    RUNTIME_PANEL_SIZE,
    RuntimeInterventionPlan,
    validate_runtime_intervention_plan,
)
from ctaa_run_contract import canonical_json


EXECUTION_PROJECTION_SCHEMA = "r12_ctaa_runtime_execution_projection_v2"
EXECUTION_ATTEMPT_SCHEMA = "r12_ctaa_runtime_execution_attempt_v2"

_ANCHOR_KEYS = (
    "anchor_id",
    "program_source_sha256",
    "packet_sha256",
)
_FORBIDDEN_KEYS = frozenset(
    {
        "action_card_sha256s",
        "answer",
        "answer_sha256",
        "class_id",
        "depth",
        "donor_query_position",
        "family_id",
        "midpoint_action_sha256",
        "midpoint_state_sha256",
        "midpoint_suffix_sha256",
        "parent_query_position",
        "partition",
        "query_position",
        "query_source",
        "query_source_sha256",
        "renderer_index",
        "resulting_query_source_sha256",
        "shift_cell",
    }
)
_PROJECTION_KEYS = frozenset(
    {
        "schema",
        "plan_schema",
        "plan_sha256",
        "board_manifest_sha256",
        "board_tree_sha256",
        "run_contract_sha256",
        "compiler_sha256",
        "core_sha256",
        "core_kind",
        "tokenizer_sha256",
        "base_checkpoint_sha256",
        "base_raw_evidence_receipt_sha256",
        "selection_seed",
        "selection_seed_receipt_sha256",
        "training_seed",
        "arm_id",
        "runtime_implementation_sha256",
        "anchor_panel_sha256",
        "donor_registry_sha256",
        "batch_order",
        "batch_order_sha256",
        "runtime_panel_size",
        "scored_row_count",
        "runtime_attempts_affect_scored_denominator",
        "anchors",
        "attempts",
        "attempts_sha256",
        "deferred_operation",
        "projection_sha256",
    }
)
_ATTEMPT_KEYS = frozenset(
    {
        "schema",
        "attempt_index",
        "attempt_id",
        "attempt_plan_sha256",
        "operation",
        "operation_sha256",
        "anchor_id",
        "donor_anchor_id",
        "mutation_payload_json",
        "mutation_payload_sha256",
        "resulting_program_source_sha256",
        "resulting_packet_sha256",
    }
)
_EXECUTION_OPERATIONS = tuple(
    operation
    for operation in MANDATORY_OPERATIONS
    if operation != InterventionFamily.LATE_QUERY_SWAP.value
)
_HASH_FIELDS = (
    "plan_sha256",
    "board_manifest_sha256",
    "board_tree_sha256",
    "run_contract_sha256",
    "compiler_sha256",
    "core_sha256",
    "tokenizer_sha256",
    "base_checkpoint_sha256",
    "base_raw_evidence_receipt_sha256",
    "selection_seed_receipt_sha256",
    "runtime_implementation_sha256",
    "anchor_panel_sha256",
    "donor_registry_sha256",
    "batch_order_sha256",
    "attempts_sha256",
)
_OPAQUE_ANCHOR_RE = re.compile(r"oa[0-9]{6}\Z")
_OPAQUE_ATTEMPT_RE = re.compile(r"ot[0-9]{8}\Z")
_LITERAL_QUERY_FRAGMENTS = (
    "READ THE ",
    "REPORT VALUE AT ",
    "query_source",
    "answer_sha256",
)


class ExecutionProjectionError(ValueError):
    """The execution-only plan view differs from its frozen derivation."""


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExecutionProjectionError(f"duplicate execution-projection key: {key}")
        result[key] = value
    return result


def _decode(raw: bytes) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ExecutionProjectionError(
            f"non-finite execution-projection constant: {value}"
        )

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExecutionProjectionError("execution-projection JSON differs") from error
    if not isinstance(value, dict):
        raise ExecutionProjectionError("execution-projection root differs")
    return value


def _assert_query_blind(value: object) -> None:
    if isinstance(value, Mapping):
        leaked = _FORBIDDEN_KEYS.intersection(value)
        if leaked:
            raise ExecutionProjectionError(
                f"execution projection leaks query field: {sorted(leaked)[0]}"
            )
        for item in value.values():
            _assert_query_blind(item)
    elif isinstance(value, list):
        for item in value:
            _assert_query_blind(item)
    elif isinstance(value, str) and any(
        fragment.casefold() in value.casefold() for fragment in _LITERAL_QUERY_FRAGMENTS
    ):
        raise ExecutionProjectionError(
            "execution projection leaks literal query material"
        )


def _opaque_anchor_id(index: int) -> str:
    return f"oa{index:06d}"


def _opaque_attempt_id(index: int) -> str:
    return f"ot{index:08d}"


def _project_payload(
    raw: str,
    *,
    anchor_ids: Mapping[str, str],
) -> tuple[str, str]:
    decoded = _decode(raw.encode("ascii"))
    _assert_query_blind(decoded)
    for key in ("anchor_id", "donor_anchor_id"):
        if key not in decoded:
            continue
        original = decoded[key]
        if original is None and key == "donor_anchor_id":
            continue
        if not isinstance(original, str) or original not in anchor_ids:
            raise ExecutionProjectionError(
                "execution-projection payload identity differs"
            )
        decoded[key] = anchor_ids[original]
    if decoded.get("operation") == InterventionFamily.RENDERER_SUBSTITUTION.value:
        renderer_keys = ("parent_renderer", "target_renderer")
        if any(key in decoded for key in renderer_keys):
            if not all(key in decoded for key in renderer_keys):
                raise ExecutionProjectionError(
                    "execution-projection renderer payload differs"
                )
            for key in renderer_keys:
                value = decoded[key]
                if type(value) is not int or not 0 <= value <= 63:
                    raise ExecutionProjectionError(
                        "execution-projection renderer payload differs"
                    )
                decoded[key] = value & 31
    projected = canonical_json(decoded)
    return projected, hashlib.sha256(projected.encode("ascii")).hexdigest()


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_execution_projection_standalone(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the redacted view without exposing the full query-bearing plan."""

    if set(value) != _PROJECTION_KEYS:
        raise ExecutionProjectionError("execution-projection schema differs")
    projection = dict(value)
    _assert_query_blind(projection)
    if (
        projection["schema"] != EXECUTION_PROJECTION_SCHEMA
        or projection["deferred_operation"] != InterventionFamily.LATE_QUERY_SWAP.value
        or projection["runtime_panel_size"] != RUNTIME_PANEL_SIZE
        or projection["runtime_attempts_affect_scored_denominator"] is not False
    ):
        raise ExecutionProjectionError("execution-projection identity differs")
    if any(not _is_hash(projection[key]) for key in _HASH_FIELDS):
        raise ExecutionProjectionError("execution-projection hash differs")
    expected_projection_sha = _sha256(
        {key: item for key, item in projection.items() if key != "projection_sha256"}
    )
    if projection["projection_sha256"] != expected_projection_sha:
        raise ExecutionProjectionError("execution-projection commitment differs")
    anchors = projection["anchors"]
    batch_order = projection["batch_order"]
    attempts = projection["attempts"]
    if (
        not isinstance(anchors, list)
        or len(anchors) != RUNTIME_PANEL_SIZE
        or not isinstance(batch_order, list)
        or len(batch_order) != RUNTIME_PANEL_SIZE
        or not isinstance(attempts, list)
        or len(attempts) != len(_EXECUTION_OPERATIONS) * RUNTIME_PANEL_SIZE
    ):
        raise ExecutionProjectionError("execution-projection coverage differs")
    anchor_ids = []
    for anchor_index, anchor in enumerate(anchors):
        if not isinstance(anchor, Mapping) or set(anchor) != set(_ANCHOR_KEYS):
            raise ExecutionProjectionError("execution-projection anchor schema differs")
        anchor_id = anchor.get("anchor_id")
        if (
            not isinstance(anchor_id, str)
            or _OPAQUE_ANCHOR_RE.fullmatch(anchor_id) is None
            or anchor_id != _opaque_anchor_id(anchor_index)
            or not _is_hash(anchor.get("program_source_sha256"))
            or not _is_hash(anchor.get("packet_sha256"))
        ):
            raise ExecutionProjectionError("execution-projection anchor differs")
        anchor_ids.append(anchor_id)
    if len(set(anchor_ids)) != RUNTIME_PANEL_SIZE or set(batch_order) != set(
        anchor_ids
    ):
        raise ExecutionProjectionError("execution-projection batch order differs")
    if any(
        not isinstance(anchor_id, str) or _OPAQUE_ANCHOR_RE.fullmatch(anchor_id) is None
        for anchor_id in batch_order
    ) or projection["batch_order_sha256"] != _sha256(batch_order):
        raise ExecutionProjectionError("execution-projection batch commitment differs")
    if projection["attempts_sha256"] != _sha256(attempts):
        raise ExecutionProjectionError(
            "execution-projection attempt commitment differs"
        )
    cursor = 0
    full_operation_index = {
        operation: index for index, operation in enumerate(MANDATORY_OPERATIONS)
    }
    for operation in _EXECUTION_OPERATIONS:
        for row_index, anchor_id in enumerate(batch_order):
            row = attempts[cursor]
            if not isinstance(row, Mapping) or set(row) != _ATTEMPT_KEYS:
                raise ExecutionProjectionError(
                    "execution-projection attempt schema differs"
                )
            expected_index = (
                full_operation_index[operation] * RUNTIME_PANEL_SIZE + row_index
            )
            if (
                row.get("schema") != EXECUTION_ATTEMPT_SCHEMA
                or row.get("attempt_index") != expected_index
                or row.get("operation") != operation
                or row.get("anchor_id") != anchor_id
                or row.get("attempt_id") != _opaque_attempt_id(expected_index)
            ):
                raise ExecutionProjectionError(
                    "execution-projection attempt order differs"
                )
            donor_id = row.get("donor_anchor_id")
            if donor_id is not None and (
                not isinstance(donor_id, str)
                or _OPAQUE_ANCHOR_RE.fullmatch(donor_id) is None
                or donor_id not in anchor_ids
                or donor_id == anchor_id
            ):
                raise ExecutionProjectionError(
                    "execution-projection donor identity differs"
                )
            for key in (
                "attempt_plan_sha256",
                "operation_sha256",
                "mutation_payload_sha256",
            ):
                if not _is_hash(row.get(key)):
                    raise ExecutionProjectionError(
                        "execution-projection attempt hash differs"
                    )
            for key in (
                "resulting_program_source_sha256",
                "resulting_packet_sha256",
            ):
                if row.get(key) is not None and not _is_hash(row.get(key)):
                    raise ExecutionProjectionError(
                        "execution-projection result hash differs"
                    )
            payload = row.get("mutation_payload_json")
            if operation == GateFamily.QUERY_ISOLATION.value:
                if payload is not None:
                    raise ExecutionProjectionError(
                        "execution-projection query gate payload is disclosed"
                    )
            else:
                if not isinstance(payload, str):
                    raise ExecutionProjectionError(
                        "execution-projection mutation payload differs"
                    )
                decoded = _decode(payload.encode("ascii"))
                _assert_query_blind(decoded)
                if (
                    canonical_json(decoded) != payload
                    or hashlib.sha256(payload.encode("ascii")).hexdigest()
                    != row["mutation_payload_sha256"]
                ):
                    raise ExecutionProjectionError(
                        "execution-projection mutation commitment differs"
                    )
                if decoded.get("anchor_id") != anchor_id:
                    raise ExecutionProjectionError(
                        "execution-projection payload anchor differs"
                    )
                if (
                    "donor_anchor_id" in decoded
                    and decoded["donor_anchor_id"] != donor_id
                ):
                    raise ExecutionProjectionError(
                        "execution-projection payload donor differs"
                    )
                if operation == InterventionFamily.RENDERER_SUBSTITUTION.value:
                    renderer_keys = ("parent_renderer", "target_renderer")
                    if any(key in decoded for key in renderer_keys):
                        if not all(key in decoded for key in renderer_keys):
                            raise ExecutionProjectionError(
                                "execution-projection renderer payload differs"
                            )
                        for key in renderer_keys:
                            value = decoded[key]
                            if type(value) is not int or not 0 <= value <= 31:
                                raise ExecutionProjectionError(
                                    "execution-projection renderer payload differs"
                                )
            cursor += 1
    return projection


def make_execution_projection(
    plan: RuntimeInterventionPlan | Mapping[str, object],
) -> dict[str, object]:
    frozen = validate_runtime_intervention_plan(plan)
    bindings = frozen.bindings
    deferred = InterventionFamily.LATE_QUERY_SWAP.value
    anchor_ids = {
        anchor.anchor_id: _opaque_anchor_id(index)
        for index, anchor in enumerate(frozen.anchors)
    }
    anchors = [
        {
            "anchor_id": anchor_ids[anchor.anchor_id],
            "program_source_sha256": anchor.program_source_sha256,
            "packet_sha256": anchor.packet_sha256,
        }
        for anchor in frozen.anchors
    ]
    attempts = []
    for attempt in frozen.attempts:
        if attempt.operation == deferred:
            continue
        mutation_payload: str | None = attempt.mutation_payload_json
        mutation_payload_sha256 = attempt.mutation_payload_sha256
        if attempt.operation == GateFamily.QUERY_ISOLATION.value:
            mutation_payload = None
        else:
            mutation_payload, mutation_payload_sha256 = _project_payload(
                attempt.mutation_payload_json,
                anchor_ids=anchor_ids,
            )
        row = {
            "schema": EXECUTION_ATTEMPT_SCHEMA,
            "attempt_index": attempt.attempt_index,
            "attempt_id": _opaque_attempt_id(attempt.attempt_index),
            "attempt_plan_sha256": attempt.attempt_plan_sha256,
            "operation": attempt.operation,
            "operation_sha256": attempt.operation_sha256,
            "anchor_id": anchor_ids[attempt.anchor_id],
            "donor_anchor_id": (
                None
                if attempt.donor_anchor_id is None
                else anchor_ids[attempt.donor_anchor_id]
            ),
            "mutation_payload_json": mutation_payload,
            "mutation_payload_sha256": mutation_payload_sha256,
            "resulting_program_source_sha256": (
                attempt.resulting_program_source_sha256
            ),
            "resulting_packet_sha256": attempt.resulting_packet_sha256,
        }
        if set(row) != _ATTEMPT_KEYS:
            raise AssertionError("execution-attempt projection schema differs")
        attempts.append(row)
    value: dict[str, object] = {
        "schema": EXECUTION_PROJECTION_SCHEMA,
        "plan_schema": frozen.schema,
        "plan_sha256": frozen.plan_sha256,
        "board_manifest_sha256": bindings.board_manifest_sha256,
        "board_tree_sha256": bindings.board_tree_sha256,
        "run_contract_sha256": bindings.run_contract_sha256,
        "compiler_sha256": bindings.compiler_sha256,
        "core_sha256": bindings.core_sha256,
        "core_kind": bindings.core_kind,
        "tokenizer_sha256": bindings.tokenizer_sha256,
        "base_checkpoint_sha256": bindings.base_checkpoint_sha256,
        "base_raw_evidence_receipt_sha256": (bindings.base_raw_evidence_receipt_sha256),
        "selection_seed": bindings.selection_seed,
        "selection_seed_receipt_sha256": bindings.selection_seed_receipt_sha256,
        "training_seed": bindings.training_seed,
        "arm_id": bindings.arm_id,
        "runtime_implementation_sha256": bindings.runtime_implementation_sha256,
        "anchor_panel_sha256": frozen.anchor_panel_sha256,
        "donor_registry_sha256": frozen.donor_registry_sha256,
        "batch_order": [anchor_ids[anchor_id] for anchor_id in bindings.batch_order],
        "batch_order_sha256": _sha256(
            [anchor_ids[anchor_id] for anchor_id in bindings.batch_order]
        ),
        "runtime_panel_size": bindings.runtime_panel_size,
        "scored_row_count": bindings.scored_row_count,
        "runtime_attempts_affect_scored_denominator": (
            bindings.runtime_attempts_affect_scored_denominator
        ),
        "anchors": anchors,
        "attempts": attempts,
        "attempts_sha256": _sha256(attempts),
        "deferred_operation": deferred,
    }
    _assert_query_blind(value)
    value["projection_sha256"] = _sha256(value)
    return value


def validate_execution_projection(
    value: Mapping[str, object],
    plan: RuntimeInterventionPlan | Mapping[str, object],
) -> dict[str, object]:
    validate_execution_projection_standalone(value)
    expected = make_execution_projection(plan)
    if dict(value) != expected:
        raise ExecutionProjectionError("execution projection differs from frozen plan")
    return expected


def write_execution_projection(
    path: Path,
    plan: RuntimeInterventionPlan | Mapping[str, object],
) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing execution projection: {path}")
    value = make_execution_projection(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode("ascii")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing existing execution temporary: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
        temporary.chmod(0o444)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.chmod(0o600)
            temporary.unlink()
    path.chmod(0o444)
    return hashlib.sha256(payload).hexdigest()


def read_execution_projection(
    path: Path,
    plan: RuntimeInterventionPlan | Mapping[str, object] | None = None,
) -> dict[str, object]:
    metadata = path.lstat()
    if not path.is_file() or path.is_symlink() or metadata.st_mode & 0o222:
        raise ExecutionProjectionError("execution projection is not immutable")
    raw = path.read_bytes()
    value = _decode(raw)
    if raw != (canonical_json(value) + "\n").encode("ascii"):
        raise ExecutionProjectionError("execution projection is not canonical")
    if plan is None:
        return validate_execution_projection_standalone(value)
    return validate_execution_projection(value, plan)
