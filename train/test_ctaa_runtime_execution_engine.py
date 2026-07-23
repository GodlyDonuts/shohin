from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json

import pytest
import torch
import torch.nn as nn

import ctaa_runtime_execution_engine as engine
from ctaa_intervention_protocol import (
    GateFamily,
    InterventionFamily,
    LOCKED_SCORED_ROW_COUNT,
    MANDATORY_OPERATIONS,
    OPERATION_SPECS,
)
from ctaa_packet_io import packet_body
from ctaa_runtime_execution_engine import (
    PROGRAM_ARTIFACT_SCHEMA,
    ProgramArtifact,
    ProbeObservation,
    RuntimeExecutionError,
    execute_runtime_projection,
)
from ctaa_runtime_plan_replay import PAYLOAD_SCHEMA
from ctaa_trunk_compiler import HardCTAAPacket, TrunkResidualBundle


def _digest(value: bytes | str) -> str:
    raw = value if isinstance(value, bytes) else value.encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _packet(kind: int, batch: int = 1) -> HardCTAAPacket:
    cards = (
        ((0, 1, 2), (2, 1, 0), (1, 2, 0), (0, 0, 0))
        if kind == 0
        else ((1, 2, 0), (0, 1, 2), (0, 2, 1), (1, 1, 1))
    )
    initial = (0, 1, 2) if kind == 0 else (2, 0, 1)
    schedule = [1, 2, 0, 4] + [0] * 37
    return HardCTAAPacket(
        torch.tensor([cards] * batch, dtype=torch.uint8),
        torch.tensor([initial] * batch, dtype=torch.uint8),
        torch.tensor([schedule] * batch, dtype=torch.uint8),
        torch.arange(4, dtype=torch.uint8)[None].expand(batch, -1).clone(),
    )


class FakeCompiler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eval()

    def encode_source(self, ids: torch.Tensor) -> TrunkResidualBundle:
        if ids.shape[1] > 1 and bool(ids[:, 1].eq(99).any()):
            raise RuntimeError("synthetic compile fault")
        key = ids[:, 0].float().remainder(2)
        early = key[:, None, None].expand(-1, 2, 3).clone() + 1.0
        late = key[:, None, None].expand(-1, 2, 3).clone() + 4.0
        valid = torch.ones(ids.shape[0], 2, dtype=torch.bool)
        return TrunkResidualBundle(early, late, valid)

    def compile_program_from_residuals(
        self,
        bundle: TrunkResidualBundle,
        *,
        intervention: str = "native",
        donor: TrunkResidualBundle | None = None,
        batch_rotation: torch.Tensor | None = None,
    ) -> HardCTAAPacket:
        early, late = bundle.early, bundle.late
        if intervention == "h19_zero":
            early = torch.zeros_like(early)
        elif intervention == "h29_zero":
            late = torch.zeros_like(late)
        elif intervention in {"h19_batch_rotate", "h29_batch_rotate"}:
            assert batch_rotation is not None
            if intervention.startswith("h19"):
                early = early.index_select(0, batch_rotation)
            else:
                late = late.index_select(0, batch_rotation)
        elif intervention in {"h19_donor_transplant", "h29_donor_transplant"}:
            assert donor is not None
            if intervention.startswith("h19"):
                early = donor.early
            else:
                late = donor.late
        key = (early[:, 0, 0] + 2 * late[:, 0, 0]).round().long().remainder(2)
        packets = [_packet(int(item)) for item in key.tolist()]
        return HardCTAAPacket(
            torch.cat([item.action_cards for item in packets]),
            torch.cat([item.initial_state for item in packets]),
            torch.cat([item.opcode_schedule for item in packets]),
            torch.cat([item.opcode_to_card for item in packets]),
        )

    @staticmethod
    def materialize_program(output: HardCTAAPacket) -> HardCTAAPacket:
        return output


class FakeCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eval()

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        target = right.gather(1, left)
        logits = torch.full(
            (*target.shape, 3), -20.0, dtype=torch.float32, device=target.device
        )
        return logits.scatter(-1, target[..., None], 20.0)


class PassingProbe:
    def observe(
        self, *, operation: str, anchor_id: str, parent_record_sha256: str
    ) -> ProbeObservation:
        return ProbeObservation(
            True,
            "denied",
            f"{operation}:{anchor_id}:{parent_record_sha256}".encode("ascii"),
        )


def _extras(operation: str, anchor: str, donor: str | None) -> dict[str, object]:
    if operation.startswith("h19_") or operation.startswith("h29_"):
        return {
            "residual_layer": 19 if operation.startswith("h19_") else 29,
            "token_start": 0,
            "token_stop": 2,
            "channel_start": 0,
            "channel_stop": "model_width",
            "padding_mask_sha256": _digest(f"mask:{anchor}"),
            "donor_anchor_id": donor,
        }
    if operation in {
        InterventionFamily.ENTITY_RECODE.value,
        InterventionFamily.WITNESS_RECODE.value,
        InterventionFamily.OPCODE_RECODE.value,
    }:
        return {"old_to_new": {"old": "new"}}
    if operation == InterventionFamily.RENDERER_SUBSTITUTION.value:
        return {"parent_renderer": 0, "target_renderer": 1}
    if operation == InterventionFamily.RULE_LINE_SHUFFLE.value:
        return {"rule_order": [1, 0, 2, 3]}
    if operation == InterventionFamily.CARD_ONLY_COUNTERFACTUAL.value:
        return {
            "card_address": 0,
            "coordinate": 0,
            "before": 0,
            "after": 1,
        }
    if operation == InterventionFamily.BINDING_ONLY_COUNTERFACTUAL.value:
        return {
            "old_to_new_opcode": [1, 2, 0, 3],
            "new_to_old_opcode": [2, 0, 1, 3],
        }
    if operation == InterventionFamily.COMPENSATED_OPCODE_RELABEL.value:
        return {"old_to_new_opcode": [1, 2, 0, 3]}
    if operation == InterventionFamily.CARD_STORAGE_REINDEX.value:
        return {"storage_order": [1, 0, 3, 2], "inverse": [1, 0, 3, 2]}
    if operation == InterventionFamily.WITNESS_CORRUPTION.value:
        return {"slot": 0, "position": 0, "before": 0, "after": 1}
    if operation == InterventionFamily.PAIRED_SHUFFLED_LAW.value:
        return {"law_order": [1, 0, 3, 2]}
    if operation == InterventionFamily.SCHEDULE_ORDER_TWIN.value:
        return {"swapped_active_slots": [0, 1]}
    if operation == InterventionFamily.SOURCE_POISON.value:
        poison = f"poison:{anchor}".encode("ascii")
        return {
            "replacement_offset": 0,
            "replacement_length": 7,
            "poison_bytes_hex": poison.hex(),
            "poison_bytes_sha256": _digest(poison),
        }
    if operation == InterventionFamily.FUTURE_MASK.value:
        return {"first_exposure_step": 1, "changed_slots": [1, 2]}
    if operation == InterventionFamily.STOP_RELOCATION.value:
        return {"old_stop_index": 3, "new_stop_index": 1, "displaced_event": 2}
    if operation == InterventionFamily.POST_STOP_POISON.value:
        return {
            "stop_index": 3,
            "changed_slots": list(range(4, 41)),
            "replacement_rule": "(event+1)%4",
        }
    if operation == InterventionFamily.MIDPOINT_DONOR_STATE.value:
        return {"midpoint_step": 1, "donor_state_sha256": _digest("donor-state")}
    if operation == InterventionFamily.MIDPOINT_DONOR_ACTION.value:
        return {
            "midpoint_step": 1,
            "donor_card_slot": 3,
            "donor_action_sha256": _digest("donor-action"),
        }
    if operation == InterventionFamily.PACKET_TRANSPLANT.value:
        return {"literal_donor_packet_sha256": _digest("donor-packet")}
    if operation == GateFamily.SOURCE_DELETION.value:
        return {
            "probe_stage": "source_blind_packet_executor",
            "probe_targets": ["program_source", "board_root"],
            "required_result": "all_open_attempts_denied",
            "allowed_errno": ["EACCES", "ENOENT"],
        }
    if operation == GateFamily.ROUTE_AGREEMENT.value:
        return {
            "positions": list(range(42)),
            "comparison": "exact_uint8_state_route_equals_composed_route",
            "required_tensor_shape": [42, 3],
        }
    raise AssertionError(operation)


def _payload(operation: str, anchor: str, donor: str | None) -> str | None:
    if operation == GateFamily.QUERY_ISOLATION.value:
        return None
    value = {
        "schema": PAYLOAD_SCHEMA,
        "operation": operation,
        "anchor_id": anchor,
        "timing": OPERATION_SPECS[operation].timing,
        **_extras(operation, anchor, donor),
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _fixture_contract() -> tuple[dict[str, object], dict[str, ProgramArtifact]]:
    compiler = FakeCompiler()
    parent_sources = {"a": b"parent-a", "b": b"parent-b"}
    parent_tokens = {"a": (0,), "b": (1,)}
    artifacts: dict[str, ProgramArtifact] = {}
    anchors = []
    for anchor in ("a", "b"):
        artifact = ProgramArtifact(
            PROGRAM_ARTIFACT_SCHEMA, parent_sources[anchor], parent_tokens[anchor]
        )
        digest = artifact.source_sha256
        artifacts[digest] = artifact
        bundle = compiler.encode_source(torch.tensor([parent_tokens[anchor]]))
        packet = compiler.materialize_program(
            compiler.compile_program_from_residuals(bundle)
        )
        anchors.append(
            {
                "anchor_id": anchor,
                "program_source_sha256": digest,
                "packet_sha256": _digest(packet_body(packet)),
            }
        )
    anchor_map = {row["anchor_id"]: row for row in anchors}
    donor_operations = {
        InterventionFamily.H19_BATCH_ROTATE.value,
        InterventionFamily.H19_DONOR_TRANSPLANT.value,
        InterventionFamily.H29_BATCH_ROTATE.value,
        InterventionFamily.H29_DONOR_TRANSPLANT.value,
        InterventionFamily.MIDPOINT_DONOR_STATE.value,
        InterventionFamily.MIDPOINT_DONOR_ACTION.value,
        InterventionFamily.PACKET_TRANSPLANT.value,
    }
    operations = [
        operation
        for operation in MANDATORY_OPERATIONS
        if operation != InterventionFamily.LATE_QUERY_SWAP.value
    ]
    attempts = []
    cursor = 0
    for operation in operations:
        for anchor in ("a", "b"):
            donor = (
                ("b" if anchor == "a" else "a")
                if operation in donor_operations
                else None
            )
            payload = _payload(operation, anchor, donor)
            program_hash = None
            packet_hash = None
            if operation in engine._SOURCE_OPERATIONS:
                source = f"variant:{operation}:{anchor}".encode("ascii")
                artifact = ProgramArtifact(
                    PROGRAM_ARTIFACT_SCHEMA, source, parent_tokens[anchor]
                )
                artifacts[artifact.source_sha256] = artifact
                program_hash = artifact.source_sha256
            elif operation == InterventionFamily.SOURCE_POISON.value:
                assert payload is not None
                program_hash = json.loads(payload)["poison_bytes_sha256"]
            parent_packet = _packet(1 if anchor == "a" else 0)
            if operation == InterventionFamily.CARD_ONLY_COUNTERFACTUAL.value:
                packet_hash = _digest(
                    packet_body(
                        engine.card_only_counterfactual(
                            parent_packet,
                            torch.tensor([0], dtype=torch.long),
                            torch.tensor([0], dtype=torch.long),
                        ).packet
                    )
                )
            elif operation == InterventionFamily.BINDING_ONLY_COUNTERFACTUAL.value:
                packet_hash = _digest(
                    packet_body(
                        engine.binding_only_counterfactual(
                            parent_packet,
                            torch.tensor([[2, 0, 1, 3]], dtype=torch.long),
                        ).packet
                    )
                )
            elif operation == InterventionFamily.COMPENSATED_OPCODE_RELABEL.value:
                packet_hash = _digest(
                    packet_body(
                        engine.compensated_opcode_relabel(
                            parent_packet,
                            torch.tensor([[1, 2, 0, 3]], dtype=torch.long),
                        ).packet
                    )
                )
            elif operation == InterventionFamily.CARD_STORAGE_REINDEX.value:
                packet_hash = _digest(
                    packet_body(
                        engine.card_storage_reindex(
                            parent_packet,
                            torch.tensor([[1, 0, 3, 2]], dtype=torch.long),
                        ).packet
                    )
                )
            elif operation == InterventionFamily.FUTURE_MASK.value:
                packet_hash = _digest(
                    packet_body(
                        engine.future_schedule_counterfactual(
                            parent_packet, torch.tensor([1], dtype=torch.long)
                        ).packet
                    )
                )
            elif operation == InterventionFamily.POST_STOP_POISON.value:
                packet_hash = _digest(
                    packet_body(engine.post_stop_poison(parent_packet).packet)
                )
            elif operation == InterventionFamily.PACKET_TRANSPLANT.value:
                packet_hash = anchor_map[donor]["packet_sha256"]
            payload_hash = _digest(payload or "redacted")
            attempts.append(
                {
                    "schema": "r12_ctaa_runtime_execution_attempt_v1",
                    "attempt_index": cursor,
                    "attempt_id": f"{operation}:{anchor}",
                    "attempt_plan_sha256": _digest(f"plan:{cursor}"),
                    "operation": operation,
                    "operation_sha256": _digest(f"operation:{operation}"),
                    "anchor_id": anchor,
                    "donor_anchor_id": donor,
                    "mutation_payload_json": payload,
                    "mutation_payload_sha256": payload_hash,
                    "resulting_program_source_sha256": program_hash,
                    "resulting_packet_sha256": packet_hash,
                }
            )
            cursor += 1
    projection = {
        "schema": "r12_ctaa_runtime_execution_projection_v1",
        "projection_sha256": _digest("projection"),
        "scored_row_count": LOCKED_SCORED_ROW_COUNT,
        "runtime_attempts_affect_scored_denominator": False,
        "runtime_panel_size": 2,
        "batch_order": ["a", "b"],
        "anchors": anchors,
        "attempts": attempts,
    }
    return projection, artifacts


@pytest.fixture
def contract(monkeypatch: pytest.MonkeyPatch):
    projection, artifacts = _fixture_contract()
    monkeypatch.setattr(
        engine, "validate_execution_projection_standalone", lambda value: dict(value)
    )
    return projection, artifacts


def _run(contract, *, probe=PassingProbe(), compiler=None, core=None):
    projection, artifacts = contract
    return execute_runtime_projection(
        projection=projection,
        program_artifacts=artifacts,
        compiler=compiler or FakeCompiler(),
        core=core or FakeCore(),
        custody_probe=probe,
    )


def test_executes_every_prequery_operation_in_projection_order(contract) -> None:
    result = _run(contract)
    expected_operations = [
        operation
        for operation in MANDATORY_OPERATIONS
        if operation != InterventionFamily.LATE_QUERY_SWAP.value
    ]
    assert len(result.parents) == 2
    assert len(result.attempts) == 2 * len(expected_operations)
    assert [row.operation for row in result.attempts[::2]] == expected_operations
    assert Counter(row.operation for row in result.attempts) == {
        operation: 2 for operation in expected_operations
    }
    assert all(
        row.attempt_id == f"{row.operation}:{row.anchor_id}" for row in result.attempts
    )
    assert result.scored_row_count == 40_608
    assert result.runtime_attempts_affect_scored_denominator is False
    assert all(row.status == "success" for row in result.attempts)


def test_raw_records_capture_packets_residuals_routes_halt_and_terminal(
    contract,
) -> None:
    result = _run(contract)
    assert all(parent.status == "success" for parent in result.parents)
    snapshot = result.parents[0].snapshot
    assert snapshot is not None
    assert snapshot.packet.bytes_per_row == 60
    assert snapshot.h19_residual is not None
    assert snapshot.h29_residual is not None
    assert snapshot.state_route.shape == (42, 3)
    assert snapshot.composed_route is not None
    assert snapshot.composed_route.shape == (42, 3)
    assert snapshot.halted.shape == (42,)
    assert torch.equal(snapshot.terminal, snapshot.state_route[-1])
    assert set(dict(snapshot.artifact_hashes)) == {
        "packet",
        "h19_residual",
        "h29_residual",
        "state_route",
        "composed_route",
        "halted",
        "terminal",
    }


def test_residual_zero_and_donor_bytes_are_observable(contract) -> None:
    result = _run(contract)
    parents = {row.anchor_id: row for row in result.parents}
    attempts = {row.attempt_id: row for row in result.attempts}
    zero = attempts["h19_zero:a"]
    donor = attempts["h29_donor_transplant:a"]
    assert zero.status == "success" and zero.snapshot is not None
    assert donor.status == "success" and donor.snapshot is not None
    assert torch.count_nonzero(zero.snapshot.h19_residual) == 0
    assert torch.equal(
        donor.snapshot.h29_residual,
        parents["b"].snapshot.h29_residual,
    )


def test_midpoint_interventions_retain_independent_composed_routes(contract) -> None:
    result = _run(contract)
    midpoint = [
        row for row in result.attempts if row.operation in engine._MIDPOINT_OPERATIONS
    ]
    assert len(midpoint) == 4
    assert all(row.status == "success" for row in midpoint)
    assert all(row.snapshot is not None for row in midpoint)
    assert all(row.snapshot.composed_route.shape == (42, 3) for row in midpoint)


def test_custody_probe_absence_retains_failures_without_omission(contract) -> None:
    result = _run(contract, probe=None)
    failures = [row for row in result.attempts if row.failure is not None]
    assert len(result.attempts) == 56
    assert {(row.operation, row.failure.code) for row in failures} >= {
        ("source_deletion", "probe_unavailable"),
        ("query_isolation", "probe_unavailable"),
    }
    assert sum(row.operation in engine._PROBE_OPERATIONS for row in failures) == 4


def test_compile_fault_is_typed_and_attempt_order_is_retained(contract) -> None:
    projection, artifacts = contract
    changed = dict(artifacts)
    target = next(
        row
        for row in projection["attempts"]
        if row["operation"] == "entity_recode" and row["anchor_id"] == "a"
    )
    digest = target["resulting_program_source_sha256"]
    original = changed[digest]
    changed[digest] = ProgramArtifact(PROGRAM_ARTIFACT_SCHEMA, original.source, (0, 99))
    result = execute_runtime_projection(
        projection=projection,
        program_artifacts=changed,
        compiler=FakeCompiler(),
        core=FakeCore(),
        custody_probe=PassingProbe(),
    )
    record = next(
        row for row in result.attempts if row.attempt_id == target["attempt_id"]
    )
    assert len(result.attempts) == 56
    assert record.status == "failure"
    assert record.failure == engine.ExecutionFailure(
        "compile", "program_compile_failed"
    )


def test_parent_packet_commitment_mismatch_fails_parent_and_all_child_attempts(
    contract,
) -> None:
    projection, artifacts = contract
    changed = deepcopy(projection)
    changed["anchors"][0]["packet_sha256"] = "f" * 64
    result = execute_runtime_projection(
        projection=changed,
        program_artifacts=artifacts,
        compiler=FakeCompiler(),
        core=FakeCore(),
        custody_probe=PassingProbe(),
    )
    parent = next(row for row in result.parents if row.anchor_id == "a")
    assert parent.status == "failure"
    assert parent.failure == engine.ExecutionFailure(
        "compile", "parent_packet_replay_mismatch"
    )
    children = [row for row in result.attempts if row.anchor_id == "a"]
    assert len(children) == 28
    assert all(row.status == "failure" for row in children)
    assert all(row.failure.code == "parent_execution_unavailable" for row in children)


def test_direct_packet_noop_is_retained_as_typed_failure(contract) -> None:
    projection, artifacts = contract
    changed = deepcopy(projection)
    target = next(
        row
        for row in changed["attempts"]
        if row["attempt_id"] == "card_storage_reindex:a"
    )
    payload = json.loads(target["mutation_payload_json"])
    payload["storage_order"] = [0, 1, 2, 3]
    payload["inverse"] = [0, 1, 2, 3]
    target["mutation_payload_json"] = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    target["mutation_payload_sha256"] = _digest(target["mutation_payload_json"])
    result = execute_runtime_projection(
        projection=changed,
        program_artifacts=artifacts,
        compiler=FakeCompiler(),
        core=FakeCore(),
        custody_probe=PassingProbe(),
    )
    record = next(
        row for row in result.attempts if row.attempt_id == target["attempt_id"]
    )
    assert record.status == "failure"
    assert record.failure == engine.ExecutionFailure("packet", "packet_mutation_failed")


@pytest.mark.parametrize(
    "field,value",
    [
        ("query_position", 1),
        ("query_source", "hidden"),
        ("answer", 2),
        ("oracle_access_count", 0),
        ("oracle_payload", "hidden"),
    ],
)
def test_forbidden_fields_are_rejected_before_execution(contract, field, value) -> None:
    projection, artifacts = contract
    changed = deepcopy(projection)
    changed[field] = value
    with pytest.raises(RuntimeExecutionError, match="forbidden field"):
        execute_runtime_projection(
            projection=changed,
            program_artifacts=artifacts,
            compiler=FakeCompiler(),
            core=FakeCore(),
            custody_probe=PassingProbe(),
        )


def test_late_query_operation_cannot_enter_engine(contract) -> None:
    projection, artifacts = contract
    changed = deepcopy(projection)
    changed["attempts"][0]["operation"] = "late_query_swap"
    with pytest.raises(RuntimeExecutionError, match="deferred operation"):
        execute_runtime_projection(
            projection=changed,
            program_artifacts=artifacts,
            compiler=FakeCompiler(),
            core=FakeCore(),
            custody_probe=PassingProbe(),
        )


def test_forbidden_field_cannot_hide_inside_payload(contract) -> None:
    projection, artifacts = contract
    changed = deepcopy(projection)
    payload = json.loads(changed["attempts"][0]["mutation_payload_json"])
    payload["answer"] = 2
    changed["attempts"][0]["mutation_payload_json"] = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    with pytest.raises(RuntimeExecutionError, match="forbidden field"):
        execute_runtime_projection(
            projection=changed,
            program_artifacts=artifacts,
            compiler=FakeCompiler(),
            core=FakeCore(),
            custody_probe=PassingProbe(),
        )


def test_program_registry_rejects_uncommitted_extra_material(contract) -> None:
    projection, artifacts = contract
    changed = dict(artifacts)
    extra = ProgramArtifact(PROGRAM_ARTIFACT_SCHEMA, b"extra", (0,))
    changed[extra.source_sha256] = extra
    with pytest.raises(RuntimeExecutionError, match="registry coverage"):
        execute_runtime_projection(
            projection=projection,
            program_artifacts=changed,
            compiler=FakeCompiler(),
            core=FakeCore(),
            custody_probe=PassingProbe(),
        )


def test_compiler_and_core_must_be_frozen(contract) -> None:
    projection, artifacts = contract
    compiler = FakeCompiler()
    compiler.train()
    with pytest.raises(RuntimeExecutionError, match="compiler is not frozen"):
        execute_runtime_projection(
            projection=projection,
            program_artifacts=artifacts,
            compiler=compiler,
            core=FakeCore(),
            custody_probe=PassingProbe(),
        )
