"""Pure, query-blind execution engine for frozen CTAA runtime projections.

The engine deliberately accepts neither a complete runtime plan nor any late
query material.  Its only contract is a standalone-validated execution
projection plus hash-addressed, pre-tokenized program bytes.  It performs no
filesystem access, signing, or scientific assessment.  Every projected
attempt produces exactly one ordered raw record, including runtime failures.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping, Protocol, Sequence

import torch
import torch.nn as nn

from ctaa_intervention_protocol import (
    ATTEMPT_PLAN_SCHEMA,
    GateFamily,
    InterventionFamily,
    LOCKED_SCORED_ROW_COUNT,
)
from ctaa_neural_core import CTAA_ACTION_COUNT, CTAA_WIDTH, HardExecutionTrace
from ctaa_packet_io import packet_body
from ctaa_run_contract import canonical_json
from ctaa_runtime_execution_projection import (
    EXECUTION_PROJECTION_SCHEMA,
    validate_execution_projection_standalone,
)
from ctaa_runtime_interventions import (
    binding_only_counterfactual,
    card_only_counterfactual,
    card_storage_reindex,
    compensated_opcode_relabel,
    execute_with_midpoint_intervention,
    future_schedule_counterfactual,
    packet_transplant,
    post_stop_poison,
)
from ctaa_runtime_plan_replay import _payload as _validate_replay_payload
from ctaa_trunk_compiler import HardCTAAPacket, TrunkResidualBundle


RUNTIME_EXECUTION_SCHEMA = "r12_ctaa_query_blind_runtime_execution_v1"
PROGRAM_ARTIFACT_SCHEMA = "r12_ctaa_query_blind_program_artifact_v1"
SNAPSHOT_SCHEMA = "r12_ctaa_prequery_execution_snapshot_v1"
PARENT_RECORD_SCHEMA = "r12_ctaa_prequery_parent_record_v1"
ATTEMPT_RECORD_SCHEMA = "r12_ctaa_prequery_attempt_record_v1"

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "answer",
        "answer_sha256",
        "oracle",
        "oracle_access",
        "oracle_access_count",
        "parent_query_position",
        "donor_query_position",
        "query_position",
        "query_source",
        "query_source_sha256",
        "resulting_query_source_sha256",
    }
)
_RESIDUAL_OPERATIONS = frozenset(
    {
        InterventionFamily.H19_ZERO.value,
        InterventionFamily.H19_BATCH_ROTATE.value,
        InterventionFamily.H19_DONOR_TRANSPLANT.value,
        InterventionFamily.H29_ZERO.value,
        InterventionFamily.H29_BATCH_ROTATE.value,
        InterventionFamily.H29_DONOR_TRANSPLANT.value,
    }
)
_SOURCE_OPERATIONS = frozenset(
    {
        InterventionFamily.ENTITY_RECODE.value,
        InterventionFamily.WITNESS_RECODE.value,
        InterventionFamily.OPCODE_RECODE.value,
        InterventionFamily.RENDERER_SUBSTITUTION.value,
        InterventionFamily.RULE_LINE_SHUFFLE.value,
        InterventionFamily.WITNESS_CORRUPTION.value,
        InterventionFamily.PAIRED_SHUFFLED_LAW.value,
        InterventionFamily.SCHEDULE_ORDER_TWIN.value,
        InterventionFamily.STOP_RELOCATION.value,
    }
)
_PACKET_OPERATIONS = frozenset(
    {
        InterventionFamily.CARD_ONLY_COUNTERFACTUAL.value,
        InterventionFamily.BINDING_ONLY_COUNTERFACTUAL.value,
        InterventionFamily.COMPENSATED_OPCODE_RELABEL.value,
        InterventionFamily.CARD_STORAGE_REINDEX.value,
        InterventionFamily.FUTURE_MASK.value,
        InterventionFamily.POST_STOP_POISON.value,
        InterventionFamily.PACKET_TRANSPLANT.value,
    }
)
_MIDPOINT_OPERATIONS = frozenset(
    {
        InterventionFamily.MIDPOINT_DONOR_STATE.value,
        InterventionFamily.MIDPOINT_DONOR_ACTION.value,
    }
)
_PROBE_OPERATIONS = frozenset(
    {GateFamily.SOURCE_DELETION.value, GateFamily.QUERY_ISOLATION.value}
)


class RuntimeExecutionError(ValueError):
    """The query-blind execution contract is invalid before execution."""


class _StageFault(RuntimeError):
    def __init__(self, stage: str, code: str) -> None:
        super().__init__(f"{stage}:{code}")
        self.stage = stage
        self.code = code


@dataclass(frozen=True)
class ProgramArtifact:
    """Immutable in-memory source bytes and their precommitted token IDs."""

    schema: str
    source: bytes
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.schema != PROGRAM_ARTIFACT_SCHEMA
            or not isinstance(self.source, bytes)
            or not self.source
            or not isinstance(self.token_ids, tuple)
            or not self.token_ids
            or any(type(token) is not int or token < 0 for token in self.token_ids)
        ):
            raise RuntimeExecutionError("CTAA program artifact differs")

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source).hexdigest()


@dataclass(frozen=True)
class ProbeObservation:
    """Opaque raw custody-probe observation supplied by the sandbox runner."""

    succeeded: bool
    code: str
    raw: bytes

    def __post_init__(self) -> None:
        if (
            type(self.succeeded) is not bool
            or not isinstance(self.code, str)
            or not self.code
            or not self.code.isascii()
            or not isinstance(self.raw, bytes)
            or not self.raw
        ):
            raise RuntimeExecutionError("CTAA custody observation differs")


class CustodyProbe(Protocol):
    """Runner-owned physical probe; no source or deferred input crosses it."""

    def observe(
        self,
        *,
        operation: str,
        anchor_id: str,
        parent_record_sha256: str,
    ) -> ProbeObservation: ...


@dataclass(frozen=True)
class ExecutionFailure:
    stage: str
    code: str


@dataclass(frozen=True)
class ExecutionSnapshot:
    schema: str
    packet: HardCTAAPacket
    h19_residual: torch.Tensor | None
    h29_residual: torch.Tensor | None
    state_route: torch.Tensor
    composed_route: torch.Tensor | None
    halted: torch.Tensor
    terminal: torch.Tensor
    artifact_hashes: tuple[tuple[str, str], ...]
    snapshot_sha256: str


@dataclass(frozen=True)
class ParentExecutionRecord:
    schema: str
    anchor_id: str
    status: str
    program_source_sha256: str
    expected_packet_sha256: str
    snapshot: ExecutionSnapshot | None
    failure: ExecutionFailure | None
    record_sha256: str


@dataclass(frozen=True)
class AttemptExecutionRecord:
    schema: str
    attempt_index: int
    attempt_id: str
    operation: str
    anchor_id: str
    donor_anchor_id: str | None
    status: str
    parent_record_sha256: str
    committed_program_source_sha256: str | None
    committed_packet_sha256: str | None
    observed_program_source_sha256: str | None
    observed_packet_sha256: str | None
    snapshot: ExecutionSnapshot | None
    extra_artifact_hashes: tuple[tuple[str, str], ...]
    failure: ExecutionFailure | None
    record_sha256: str


@dataclass(frozen=True)
class RuntimeExecutionResult:
    schema: str
    execution_projection_schema: str
    projection_sha256: str
    scored_row_count: int
    runtime_attempts_affect_scored_denominator: bool
    parents: tuple[ParentExecutionRecord, ...]
    attempts: tuple[AttemptExecutionRecord, ...]
    execution_sha256: str


@dataclass(frozen=True)
class _ParentRuntime:
    record: ParentExecutionRecord
    bundle: TrunkResidualBundle | None


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _reject_forbidden_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in _FORBIDDEN_INPUT_KEYS or any(
                fragment in normalized
                for fragment in ("answer", "oracle", "query_position", "query_source")
            ):
                raise RuntimeExecutionError(
                    f"CTAA runtime input contains forbidden field: {key}"
                )
            if key == "mutation_payload_json" and isinstance(item, str):
                try:
                    decoded = json.loads(item)
                except json.JSONDecodeError as error:
                    raise RuntimeExecutionError(
                        "CTAA runtime mutation payload is malformed"
                    ) from error
                _reject_forbidden_fields(decoded)
                continue
            _reject_forbidden_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden_fields(item)


def _require_frozen(module: object, label: str) -> None:
    if getattr(module, "training", None) is not False:
        raise RuntimeExecutionError(f"CTAA {label} is not frozen in eval mode")
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        raise RuntimeExecutionError(f"CTAA {label} is not module-compatible")
    if any(parameter.requires_grad for parameter in parameters()):
        raise RuntimeExecutionError(f"CTAA {label} has trainable parameters")


def _module_device(module: object) -> torch.device:
    parameters = getattr(module, "parameters")
    first = next(iter(parameters()), None)
    return first.device if first is not None else torch.device("cpu")


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().contiguous().cpu()
    return tensor.view(torch.uint8).numpy().tobytes()


def _tensor_hash(name: str, value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    header = canonical_json(
        {
            "schema": "r12_ctaa_tensor_artifact_v1",
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + _tensor_bytes(tensor)).hexdigest()


def _clone_packet(packet: HardCTAAPacket) -> HardCTAAPacket:
    return HardCTAAPacket(
        packet.action_cards.detach().cpu().clone(),
        packet.initial_state.detach().cpu().clone(),
        packet.opcode_schedule.detach().cpu().clone(),
        packet.opcode_to_card.detach().cpu().clone(),
    )


def _slice_packet(packet: HardCTAAPacket, index: int = 0) -> HardCTAAPacket:
    return HardCTAAPacket(
        packet.action_cards[index : index + 1].clone(),
        packet.initial_state[index : index + 1].clone(),
        packet.opcode_schedule[index : index + 1].clone(),
        packet.opcode_to_card[index : index + 1].clone(),
    )


def _validate_bundle(bundle: object, batch: int) -> TrunkResidualBundle:
    if not isinstance(bundle, TrunkResidualBundle):
        raise _StageFault("compile", "residual_bundle_type")
    if (
        bundle.early.ndim != 3
        or bundle.late.shape != bundle.early.shape
        or bundle.early.shape[0] != batch
        or bundle.valid.dtype != torch.bool
        or bundle.valid.shape != bundle.early.shape[:2]
    ):
        raise _StageFault("compile", "residual_bundle_geometry")
    return bundle


def _concat_bundles(
    parent: TrunkResidualBundle, donor: TrunkResidualBundle
) -> TrunkResidualBundle:
    if (
        parent.early.shape != donor.early.shape
        or parent.late.shape != donor.late.shape
        or parent.valid.shape != donor.valid.shape
        or not torch.equal(parent.valid, donor.valid.to(parent.valid.device))
    ):
        raise _StageFault("compile", "residual_donor_geometry")
    return TrunkResidualBundle(
        early=torch.cat((parent.early, donor.early.to(parent.early.device))),
        late=torch.cat((parent.late, donor.late.to(parent.late.device))),
        valid=torch.cat((parent.valid, donor.valid.to(parent.valid.device))),
    )


def _materialize_packet(compiler: object, output: object) -> HardCTAAPacket:
    materialize = getattr(compiler, "materialize_program", None)
    if not callable(materialize):
        raise _StageFault("compile", "materializer_unavailable")
    try:
        packet = materialize(output)
    except Exception as error:  # noqa: BLE001 - retained as typed failure
        raise _StageFault("compile", "packet_materialization_failed") from error
    if not isinstance(packet, HardCTAAPacket):
        raise _StageFault("compile", "packet_type")
    return packet


def _compile_artifact(
    compiler: object,
    artifact: ProgramArtifact,
) -> tuple[TrunkResidualBundle, HardCTAAPacket]:
    device = _module_device(compiler)
    ids = torch.tensor([artifact.token_ids], dtype=torch.long, device=device)
    try:
        bundle = _validate_bundle(compiler.encode_source(ids), 1)
        output = compiler.compile_program_from_residuals(bundle)
        packet = _materialize_packet(compiler, output)
    except _StageFault:
        raise
    except Exception as error:  # noqa: BLE001 - retained as typed failure
        raise _StageFault("compile", "program_compile_failed") from error
    if packet.opcode_schedule.shape[0] != 1:
        raise _StageFault("compile", "packet_batch_geometry")
    return bundle, packet


def _snapshot_payload(snapshot: ExecutionSnapshot) -> dict[str, object]:
    return {
        "schema": snapshot.schema,
        "artifact_hashes": [list(item) for item in snapshot.artifact_hashes],
    }


def _make_snapshot(
    *,
    packet: HardCTAAPacket,
    bundle: TrunkResidualBundle | None,
    core: nn.Module,
    trace: HardExecutionTrace | None = None,
    composed_route: torch.Tensor | None = None,
) -> ExecutionSnapshot:
    try:
        if trace is None:
            dual = packet.execute_dual(core)
            trace = dual.state_route
            composed_route = dual.composed_states
        if (
            trace.states.shape[0] != 1
            or trace.halted.shape != trace.states.shape[:2]
            or trace.states.shape[1] != packet.opcode_schedule.shape[1] + 1
        ):
            raise ValueError("trace geometry")
        state_route = trace.states[0].to(torch.uint8).detach().cpu().clone()
        halted = trace.halted[0].detach().cpu().clone()
        terminal = state_route[-1].clone()
        composed = (
            None
            if composed_route is None
            else composed_route[0].to(torch.uint8).detach().cpu().clone()
        )
        h19 = None if bundle is None else bundle.early[0].detach().cpu().clone()
        h29 = None if bundle is None else bundle.late[0].detach().cpu().clone()
        frozen_packet = _clone_packet(_slice_packet(packet))
    except Exception as error:  # noqa: BLE001 - retained as typed failure
        raise _StageFault("execution", "trace_execution_failed") from error
    artifacts: list[tuple[str, str]] = [
        ("packet", hashlib.sha256(packet_body(frozen_packet)).hexdigest()),
        ("state_route", _tensor_hash("state_route", state_route)),
        ("halted", _tensor_hash("halted", halted)),
        ("terminal", _tensor_hash("terminal", terminal)),
    ]
    if composed is not None:
        artifacts.append(("composed_route", _tensor_hash("composed_route", composed)))
    if h19 is not None:
        artifacts.append(("h19_residual", _tensor_hash("h19_residual", h19)))
    if h29 is not None:
        artifacts.append(("h29_residual", _tensor_hash("h29_residual", h29)))
    frozen_artifacts = tuple(sorted(artifacts))
    provisional = ExecutionSnapshot(
        SNAPSHOT_SCHEMA,
        frozen_packet,
        h19,
        h29,
        state_route,
        composed,
        halted,
        terminal,
        frozen_artifacts,
        "",
    )
    return ExecutionSnapshot(
        **{
            **provisional.__dict__,
            "snapshot_sha256": _sha256_json(_snapshot_payload(provisional)),
        }
    )


def _parent_payload(record: ParentExecutionRecord) -> dict[str, object]:
    return {
        "schema": record.schema,
        "anchor_id": record.anchor_id,
        "status": record.status,
        "program_source_sha256": record.program_source_sha256,
        "expected_packet_sha256": record.expected_packet_sha256,
        "snapshot_sha256": (
            None if record.snapshot is None else record.snapshot.snapshot_sha256
        ),
        "failure": (
            None
            if record.failure is None
            else {"stage": record.failure.stage, "code": record.failure.code}
        ),
    }


def _make_parent_record(
    *,
    anchor_id: str,
    program_source_sha256: str,
    expected_packet_sha256: str,
    snapshot: ExecutionSnapshot | None,
    failure: ExecutionFailure | None,
) -> ParentExecutionRecord:
    status = "success" if failure is None else "failure"
    provisional = ParentExecutionRecord(
        PARENT_RECORD_SCHEMA,
        anchor_id,
        status,
        program_source_sha256,
        expected_packet_sha256,
        snapshot,
        failure,
        "",
    )
    return ParentExecutionRecord(
        **{
            **provisional.__dict__,
            "record_sha256": _sha256_json(_parent_payload(provisional)),
        }
    )


def _attempt_payload(record: AttemptExecutionRecord) -> dict[str, object]:
    return {
        "schema": record.schema,
        "attempt_index": record.attempt_index,
        "attempt_id": record.attempt_id,
        "operation": record.operation,
        "anchor_id": record.anchor_id,
        "donor_anchor_id": record.donor_anchor_id,
        "status": record.status,
        "parent_record_sha256": record.parent_record_sha256,
        "committed_program_source_sha256": record.committed_program_source_sha256,
        "committed_packet_sha256": record.committed_packet_sha256,
        "observed_program_source_sha256": record.observed_program_source_sha256,
        "observed_packet_sha256": record.observed_packet_sha256,
        "snapshot_sha256": (
            None if record.snapshot is None else record.snapshot.snapshot_sha256
        ),
        "extra_artifact_hashes": [list(item) for item in record.extra_artifact_hashes],
        "failure": (
            None
            if record.failure is None
            else {"stage": record.failure.stage, "code": record.failure.code}
        ),
    }


def _make_attempt_record(
    row: Mapping[str, object],
    parent: ParentExecutionRecord,
    *,
    observed_program_source_sha256: str | None,
    snapshot: ExecutionSnapshot | None,
    extra_artifact_hashes: Sequence[tuple[str, str]] = (),
    failure: ExecutionFailure | None = None,
) -> AttemptExecutionRecord:
    status = "success" if failure is None else "failure"
    observed_packet = None
    if snapshot is not None:
        observed_packet = dict(snapshot.artifact_hashes)["packet"]
    provisional = AttemptExecutionRecord(
        ATTEMPT_RECORD_SCHEMA,
        int(row["attempt_index"]),
        str(row["attempt_id"]),
        str(row["operation"]),
        str(row["anchor_id"]),
        None if row["donor_anchor_id"] is None else str(row["donor_anchor_id"]),
        status,
        parent.record_sha256,
        (
            None
            if row["resulting_program_source_sha256"] is None
            else str(row["resulting_program_source_sha256"])
        ),
        (
            None
            if row["resulting_packet_sha256"] is None
            else str(row["resulting_packet_sha256"])
        ),
        observed_program_source_sha256,
        observed_packet,
        snapshot,
        tuple(sorted(extra_artifact_hashes)),
        failure,
        "",
    )
    return AttemptExecutionRecord(
        **{
            **provisional.__dict__,
            "record_sha256": _sha256_json(_attempt_payload(provisional)),
        }
    )


def _failed_attempt(
    row: Mapping[str, object],
    parent: ParentExecutionRecord,
    fault: _StageFault,
    *,
    observed_program_source_sha256: str | None = None,
    snapshot: ExecutionSnapshot | None = None,
    extra_artifact_hashes: Sequence[tuple[str, str]] = (),
) -> AttemptExecutionRecord:
    return _make_attempt_record(
        row,
        parent,
        observed_program_source_sha256=observed_program_source_sha256,
        snapshot=snapshot,
        extra_artifact_hashes=extra_artifact_hashes,
        failure=ExecutionFailure(fault.stage, fault.code),
    )


def _program_registry(
    projection: Mapping[str, object],
    artifacts: Mapping[str, ProgramArtifact],
) -> dict[str, ProgramArtifact]:
    if not isinstance(artifacts, Mapping):
        raise RuntimeExecutionError("CTAA program artifact registry differs")
    anchors = projection["anchors"]
    attempts = projection["attempts"]
    assert isinstance(anchors, list) and isinstance(attempts, list)
    required = {str(anchor["program_source_sha256"]) for anchor in anchors}
    for row in attempts:
        assert isinstance(row, Mapping)
        if row["operation"] in _SOURCE_OPERATIONS:
            digest = row["resulting_program_source_sha256"]
            if not _is_hash(digest):
                raise RuntimeExecutionError(
                    "CTAA source intervention lacks a program commitment"
                )
            required.add(str(digest))
    if set(artifacts) != required:
        raise RuntimeExecutionError("CTAA program artifact registry coverage differs")
    result: dict[str, ProgramArtifact] = {}
    for digest, artifact in artifacts.items():
        if not _is_hash(digest) or not isinstance(artifact, ProgramArtifact):
            raise RuntimeExecutionError("CTAA program artifact registry entry differs")
        if artifact.source_sha256 != digest:
            raise RuntimeExecutionError("CTAA program artifact hash differs")
        result[digest] = artifact
    return result


def _payload(row: Mapping[str, object]) -> dict[str, object]:
    operation = str(row["operation"])
    if operation == GateFamily.QUERY_ISOLATION.value:
        if row["mutation_payload_json"] is not None:
            raise _StageFault("projection", "deferred_input_payload_disclosed")
        return {}
    from ctaa_intervention_protocol import AnchorOperationCommitment

    commitment = AnchorOperationCommitment(
        schema=ATTEMPT_PLAN_SCHEMA,
        attempt_index=int(row["attempt_index"]),
        attempt_id=str(row["attempt_id"]),
        operation=operation,
        operation_sha256=str(row["operation_sha256"]),
        anchor_id=str(row["anchor_id"]),
        donor_anchor_id=(
            None if row["donor_anchor_id"] is None else str(row["donor_anchor_id"])
        ),
        mutation_payload_json=str(row["mutation_payload_json"]),
        mutation_payload_sha256=str(row["mutation_payload_sha256"]),
        resulting_program_source_sha256=(
            None
            if row["resulting_program_source_sha256"] is None
            else str(row["resulting_program_source_sha256"])
        ),
        resulting_query_source_sha256=None,
        resulting_packet_sha256=(
            None
            if row["resulting_packet_sha256"] is None
            else str(row["resulting_packet_sha256"])
        ),
        attempt_plan_sha256=str(row["attempt_plan_sha256"]),
    )
    try:
        return _validate_replay_payload(commitment)
    except Exception as error:  # noqa: BLE001 - retained as typed failure
        raise _StageFault("projection", "mutation_payload_invalid") from error


def _parent_runtime(
    anchor: Mapping[str, object],
    artifacts: Mapping[str, ProgramArtifact],
    compiler: object,
    core: nn.Module,
) -> _ParentRuntime:
    anchor_id = str(anchor["anchor_id"])
    program_hash = str(anchor["program_source_sha256"])
    expected_packet_hash = str(anchor["packet_sha256"])
    try:
        artifact = artifacts[program_hash]
        bundle, packet = _compile_artifact(compiler, artifact)
        snapshot = _make_snapshot(packet=packet, bundle=bundle, core=core)
        observed_packet_hash = dict(snapshot.artifact_hashes)["packet"]
        if observed_packet_hash != expected_packet_hash:
            raise _StageFault("compile", "parent_packet_replay_mismatch")
        record = _make_parent_record(
            anchor_id=anchor_id,
            program_source_sha256=program_hash,
            expected_packet_sha256=expected_packet_hash,
            snapshot=snapshot,
            failure=None,
        )
        return _ParentRuntime(record, bundle)
    except _StageFault as fault:
        record = _make_parent_record(
            anchor_id=anchor_id,
            program_source_sha256=program_hash,
            expected_packet_sha256=expected_packet_hash,
            snapshot=None,
            failure=ExecutionFailure(fault.stage, fault.code),
        )
        return _ParentRuntime(record, None)


def _source_attempt(
    row: Mapping[str, object],
    parent: _ParentRuntime,
    artifacts: Mapping[str, ProgramArtifact],
    compiler: object,
    core: nn.Module,
) -> AttemptExecutionRecord:
    digest = str(row["resulting_program_source_sha256"])
    if digest == parent.record.program_source_sha256:
        raise _StageFault("source", "source_transform_noop")
    artifact = artifacts[digest]
    bundle, packet = _compile_artifact(compiler, artifact)
    snapshot = _make_snapshot(packet=packet, bundle=bundle, core=core)
    return _make_attempt_record(
        row,
        parent.record,
        observed_program_source_sha256=digest,
        snapshot=snapshot,
    )


def _residual_attempt(
    row: Mapping[str, object],
    parent: _ParentRuntime,
    donor: _ParentRuntime | None,
    compiler: object,
    core: nn.Module,
) -> AttemptExecutionRecord:
    if parent.bundle is None:
        raise _StageFault("compile", "parent_residual_unavailable")
    operation = str(row["operation"])
    bundle = parent.bundle
    donor_bundle = None if donor is None else donor.bundle
    selected_parent = bundle.early if operation.startswith("h19_") else bundle.late
    if operation.endswith("_zero"):
        selected_child = torch.zeros_like(selected_parent)
        if torch.equal(selected_child, selected_parent):
            raise _StageFault("compile", "residual_zero_noop")
        try:
            output = compiler.compile_program_from_residuals(
                bundle, intervention=operation
            )
        except Exception as error:  # noqa: BLE001
            raise _StageFault("compile", "residual_compile_failed") from error
        child_packet = _materialize_packet(compiler, output)
    else:
        if donor_bundle is None:
            raise _StageFault("compile", "donor_residual_unavailable")
        selected_donor = (
            donor_bundle.early if operation.startswith("h19_") else donor_bundle.late
        )
        selected_child = selected_donor.to(selected_parent.device)
        if torch.equal(selected_child, selected_parent):
            raise _StageFault("compile", "residual_donor_noop")
        try:
            if operation.endswith("_batch_rotate"):
                paired = _concat_bundles(bundle, donor_bundle)
                rotation = torch.tensor(
                    [1, 0], dtype=torch.long, device=paired.valid.device
                )
                output = compiler.compile_program_from_residuals(
                    paired,
                    intervention=operation,
                    batch_rotation=rotation,
                )
                child_packet = _slice_packet(_materialize_packet(compiler, output))
            else:
                output = compiler.compile_program_from_residuals(
                    bundle,
                    intervention=operation,
                    donor=donor_bundle,
                )
                child_packet = _materialize_packet(compiler, output)
        except _StageFault:
            raise
        except Exception as error:  # noqa: BLE001
            raise _StageFault("compile", "residual_compile_failed") from error
    child_bundle = TrunkResidualBundle(
        early=(selected_child if operation.startswith("h19_") else bundle.early),
        late=(selected_child if operation.startswith("h29_") else bundle.late),
        valid=bundle.valid,
    )
    snapshot = _make_snapshot(packet=child_packet, bundle=child_bundle, core=core)
    return _make_attempt_record(
        row,
        parent.record,
        observed_program_source_sha256=parent.record.program_source_sha256,
        snapshot=snapshot,
    )


def _packet_attempt(
    row: Mapping[str, object],
    payload: Mapping[str, object],
    parent: _ParentRuntime,
    donor: _ParentRuntime | None,
    core: nn.Module,
) -> AttemptExecutionRecord:
    if parent.record.snapshot is None:
        raise _StageFault("packet", "parent_packet_unavailable")
    packet = parent.record.snapshot.packet
    operation = str(row["operation"])
    try:
        if operation == InterventionFamily.CARD_ONLY_COUNTERFACTUAL.value:
            card_address = torch.tensor(
                [int(payload["card_address"])], dtype=torch.long
            )
            coordinate = torch.tensor(
                [int(payload["coordinate"])], dtype=torch.long
            )
            child = card_only_counterfactual(
                packet, card_address, coordinate
            ).packet
        elif operation == InterventionFamily.BINDING_ONLY_COUNTERFACTUAL.value:
            order = torch.tensor(
                [payload["new_to_old_opcode"]], dtype=torch.long
            )
            child = binding_only_counterfactual(packet, order).packet
        elif operation == InterventionFamily.COMPENSATED_OPCODE_RELABEL.value:
            old_to_new = torch.tensor(
                [payload["old_to_new_opcode"]], dtype=torch.long
            )
            child = compensated_opcode_relabel(packet, old_to_new).packet
        elif operation == InterventionFamily.CARD_STORAGE_REINDEX.value:
            order = torch.tensor([payload["storage_order"]], dtype=torch.long)
            child = card_storage_reindex(packet, order).packet
        elif operation == InterventionFamily.FUTURE_MASK.value:
            boundary = torch.tensor(
                [int(payload["first_exposure_step"])], dtype=torch.long
            )
            child = future_schedule_counterfactual(packet, boundary).packet
        elif operation == InterventionFamily.POST_STOP_POISON.value:
            child = post_stop_poison(packet).packet
        elif operation == InterventionFamily.PACKET_TRANSPLANT.value:
            if donor is None or donor.record.snapshot is None:
                raise _StageFault("packet", "donor_packet_unavailable")
            child = packet_transplant(packet, donor.record.snapshot.packet).packet
        else:  # pragma: no cover - operation partition invariant
            raise _StageFault("packet", "packet_operation_unknown")
    except _StageFault:
        raise
    except Exception as error:  # noqa: BLE001
        raise _StageFault("packet", "packet_mutation_failed") from error
    snapshot = _make_snapshot(packet=child, bundle=parent.bundle, core=core)
    observed = dict(snapshot.artifact_hashes)["packet"]
    expected = row["resulting_packet_sha256"]
    parent_exact = (
        dict(parent.record.snapshot.artifact_hashes)["packet"]
        == parent.record.expected_packet_sha256
    )
    donor_exact = True
    if operation == InterventionFamily.PACKET_TRANSPLANT.value:
        assert donor is not None and donor.record.snapshot is not None
        donor_exact = (
            dict(donor.record.snapshot.artifact_hashes)["packet"]
            == donor.record.expected_packet_sha256
        )
    if parent_exact and donor_exact and expected is not None and observed != expected:
        raise _StageFault("packet", "committed_packet_replay_mismatch")
    return _make_attempt_record(
        row,
        parent.record,
        observed_program_source_sha256=parent.record.program_source_sha256,
        snapshot=snapshot,
    )


def _midpoint_attempt(
    row: Mapping[str, object],
    payload: Mapping[str, object],
    parent: _ParentRuntime,
    donor: _ParentRuntime | None,
    core: nn.Module,
) -> AttemptExecutionRecord:
    if parent.record.snapshot is None or donor is None or donor.record.snapshot is None:
        raise _StageFault("execution", "midpoint_donor_unavailable")
    operation = str(row["operation"])
    midpoint = int(payload["midpoint_step"])
    midpoint_tensor = torch.tensor([midpoint], dtype=torch.long)
    extra: list[tuple[str, str]] = []
    donor_state: torch.Tensor | None = None
    donor_action: torch.Tensor | None = None
    try:
        if operation == InterventionFamily.MIDPOINT_DONOR_STATE.value:
            donor_state = donor.record.snapshot.state_route[midpoint][None].long()
            trace = execute_with_midpoint_intervention(
                core,
                parent.record.snapshot.packet,
                operation=operation,
                midpoint_step=midpoint_tensor,
                donor_state=donor_state,
            )
            extra.append(
                ("injected_state", _tensor_hash("injected_state", donor_state))
            )
        else:
            slot = int(payload["donor_card_slot"])
            donor_action = donor.record.snapshot.packet.action_cards[:, slot].long()
            trace = execute_with_midpoint_intervention(
                core,
                parent.record.snapshot.packet,
                operation=operation,
                midpoint_step=midpoint_tensor,
                donor_action=donor_action,
            )
            extra.append(
                ("injected_action", _tensor_hash("injected_action", donor_action))
            )
        composed_route = _compose_midpoint_route(
            core,
            parent.record.snapshot.packet,
            operation=operation,
            midpoint=midpoint,
            donor_state=donor_state,
            donor_action=donor_action,
        )
    except Exception as error:  # noqa: BLE001
        raise _StageFault("execution", "midpoint_execution_failed") from error
    snapshot = _make_snapshot(
        packet=parent.record.snapshot.packet,
        bundle=parent.bundle,
        core=core,
        trace=trace,
        composed_route=composed_route,
    )
    return _make_attempt_record(
        row,
        parent.record,
        observed_program_source_sha256=parent.record.program_source_sha256,
        snapshot=snapshot,
        extra_artifact_hashes=extra,
    )


def _compose_midpoint_route(
    core: nn.Module,
    packet: HardCTAAPacket,
    *,
    operation: str,
    midpoint: int,
    donor_state: torch.Tensor | None,
    donor_action: torch.Tensor | None,
) -> torch.Tensor:
    """Independently compose the execution after one midpoint injection."""

    cards = packet.action_cards.long()
    schedule = packet.resolved_schedule.long()
    initial = packet.initial_state.long()
    batch = schedule.shape[0]
    identity = torch.arange(CTAA_WIDTH, device=schedule.device)[None].expand(batch, -1)
    composed = identity
    segment_initial = initial
    composed_state = initial
    states = [composed_state]
    halted = torch.zeros(batch, dtype=torch.bool, device=schedule.device)
    batch_index = torch.arange(batch, device=schedule.device)
    for step, event in enumerate(schedule.unbind(1)):
        is_stop = event.eq(CTAA_ACTION_COUNT)
        active = ~(halted | is_stop)
        if (
            operation == InterventionFamily.MIDPOINT_DONOR_STATE.value
            and step == midpoint
        ):
            if donor_state is None:
                raise ValueError("midpoint donor state is absent")
            segment_initial = donor_state.to(initial.device)
            composed = identity
            composed_state = segment_initial
        selected = cards[batch_index, event.clamp_max(CTAA_ACTION_COUNT - 1)]
        if (
            operation == InterventionFamily.MIDPOINT_DONOR_ACTION.value
            and step == midpoint
        ):
            if donor_action is None:
                raise ValueError("midpoint donor action is absent")
            selected = donor_action.to(selected.device)
        active_index = batch_index[active]
        if active_index.numel():
            candidate = core(selected[active], composed[active]).argmax(-1)
            composed = composed.clone()
            composed[active_index] = candidate
            from_composed = core(composed[active], segment_initial[active]).argmax(-1)
            composed_state = composed_state.clone()
            composed_state[active_index] = from_composed
        halted = halted | is_stop
        states.append(composed_state)
    return torch.stack(states, dim=1)


def _source_poison_attempt(
    row: Mapping[str, object],
    payload: Mapping[str, object],
    parent: _ParentRuntime,
) -> AttemptExecutionRecord:
    if parent.record.snapshot is None:
        raise _StageFault("custody", "sealed_parent_unavailable")
    try:
        poison = bytes.fromhex(str(payload["poison_bytes_hex"]))
    except ValueError as error:
        raise _StageFault("custody", "source_poison_encoding") from error
    digest = hashlib.sha256(poison).hexdigest()
    if (
        digest != payload["poison_bytes_sha256"]
        or digest != row["resulting_program_source_sha256"]
        or digest == parent.record.program_source_sha256
    ):
        raise _StageFault("custody", "source_poison_commitment")
    return _make_attempt_record(
        row,
        parent.record,
        observed_program_source_sha256=digest,
        snapshot=parent.record.snapshot,
    )


def _probe_attempt(
    row: Mapping[str, object],
    parent: _ParentRuntime,
    probe: CustodyProbe | None,
) -> AttemptExecutionRecord:
    if probe is None:
        raise _StageFault("custody", "probe_unavailable")
    try:
        observation = probe.observe(
            operation=str(row["operation"]),
            anchor_id=str(row["anchor_id"]),
            parent_record_sha256=parent.record.record_sha256,
        )
    except Exception as error:  # noqa: BLE001
        raise _StageFault("custody", "probe_execution_failed") from error
    if not isinstance(observation, ProbeObservation):
        raise _StageFault("custody", "probe_observation_type")
    artifact = hashlib.sha256(observation.raw).hexdigest()
    if not observation.succeeded:
        return _failed_attempt(
            row,
            parent.record,
            _StageFault("custody", observation.code),
            observed_program_source_sha256=parent.record.program_source_sha256,
            snapshot=parent.record.snapshot,
            extra_artifact_hashes=(("custody_probe", artifact),),
        )
    return _make_attempt_record(
        row,
        parent.record,
        observed_program_source_sha256=parent.record.program_source_sha256,
        snapshot=parent.record.snapshot,
        extra_artifact_hashes=(("custody_probe", artifact),),
    )


def _execute_attempt(
    row: Mapping[str, object],
    parent: _ParentRuntime,
    donor: _ParentRuntime | None,
    artifacts: Mapping[str, ProgramArtifact],
    compiler: object,
    core: nn.Module,
    probe: CustodyProbe | None,
) -> AttemptExecutionRecord:
    observed_program = parent.record.program_source_sha256
    try:
        payload = _payload(row)
        if parent.record.snapshot is None:
            raise _StageFault("execution", "parent_execution_unavailable")
        operation = str(row["operation"])
        if operation in _SOURCE_OPERATIONS:
            return _source_attempt(row, parent, artifacts, compiler, core)
        if operation in _RESIDUAL_OPERATIONS:
            return _residual_attempt(row, parent, donor, compiler, core)
        if operation in _PACKET_OPERATIONS:
            return _packet_attempt(row, payload, parent, donor, core)
        if operation in _MIDPOINT_OPERATIONS:
            return _midpoint_attempt(row, payload, parent, donor, core)
        if operation == InterventionFamily.SOURCE_POISON.value:
            return _source_poison_attempt(row, payload, parent)
        if operation in _PROBE_OPERATIONS:
            return _probe_attempt(row, parent, probe)
        if operation == GateFamily.ROUTE_AGREEMENT.value:
            return _make_attempt_record(
                row,
                parent.record,
                observed_program_source_sha256=parent.record.program_source_sha256,
                snapshot=parent.record.snapshot,
            )
        raise _StageFault("projection", "operation_not_executable")
    except _StageFault as fault:
        return _failed_attempt(
            row,
            parent.record,
            fault,
            observed_program_source_sha256=observed_program,
        )
    except Exception:  # noqa: BLE001 - no projected attempt may be omitted
        return _failed_attempt(
            row,
            parent.record,
            _StageFault("execution", "unexpected_runtime_failure"),
            observed_program_source_sha256=observed_program,
        )


def execute_runtime_projection(
    *,
    projection: Mapping[str, object],
    program_artifacts: Mapping[str, ProgramArtifact],
    compiler: object,
    core: nn.Module,
    custody_probe: CustodyProbe | None = None,
) -> RuntimeExecutionResult:
    """Execute one complete pre-query projection without opening deferred data."""

    _reject_forbidden_fields(projection)
    try:
        frozen_projection = validate_execution_projection_standalone(projection)
    except ValueError as error:
        raise RuntimeExecutionError(
            "CTAA execution projection validation failed"
        ) from error
    if (
        frozen_projection["scored_row_count"] != LOCKED_SCORED_ROW_COUNT
        or frozen_projection["runtime_attempts_affect_scored_denominator"] is not False
    ):
        raise RuntimeExecutionError("CTAA scored-denominator metadata differs")
    if any(
        isinstance(row, Mapping)
        and row.get("operation") == InterventionFamily.LATE_QUERY_SWAP.value
        for row in frozen_projection.get("attempts", ())
    ):
        raise RuntimeExecutionError("CTAA deferred operation entered runtime execution")
    _require_frozen(compiler, "compiler")
    _require_frozen(core, "core")
    artifacts = _program_registry(frozen_projection, program_artifacts)
    anchors = frozen_projection["anchors"]
    attempts = frozen_projection["attempts"]
    assert isinstance(anchors, list) and isinstance(attempts, list)
    anchor_by_id = {str(anchor["anchor_id"]): anchor for anchor in anchors}
    order = frozen_projection["batch_order"]
    assert isinstance(order, list)

    parents: dict[str, _ParentRuntime] = {}
    with torch.inference_mode():
        for anchor_id in order:
            anchor = anchor_by_id[str(anchor_id)]
            parents[str(anchor_id)] = _parent_runtime(anchor, artifacts, compiler, core)
        records: list[AttemptExecutionRecord] = []
        for row in attempts:
            assert isinstance(row, Mapping)
            parent = parents[str(row["anchor_id"])]
            donor_id = row["donor_anchor_id"]
            donor = None if donor_id is None else parents.get(str(donor_id))
            records.append(
                _execute_attempt(
                    row,
                    parent,
                    donor,
                    artifacts,
                    compiler,
                    core,
                    custody_probe,
                )
            )
    if len(records) != len(attempts):  # pragma: no cover - construction invariant
        raise AssertionError("CTAA runtime attempt retention differs")
    parent_records = tuple(parents[str(anchor_id)].record for anchor_id in order)
    attempt_records = tuple(records)
    execution_payload = {
        "schema": RUNTIME_EXECUTION_SCHEMA,
        "execution_projection_schema": EXECUTION_PROJECTION_SCHEMA,
        "projection_sha256": frozen_projection["projection_sha256"],
        "scored_row_count": LOCKED_SCORED_ROW_COUNT,
        "runtime_attempts_affect_scored_denominator": False,
        "parents": [record.record_sha256 for record in parent_records],
        "attempts": [record.record_sha256 for record in attempt_records],
    }
    return RuntimeExecutionResult(
        RUNTIME_EXECUTION_SCHEMA,
        EXECUTION_PROJECTION_SCHEMA,
        str(frozen_projection["projection_sha256"]),
        LOCKED_SCORED_ROW_COUNT,
        False,
        parent_records,
        attempt_records,
        _sha256_json(execution_payload),
    )
