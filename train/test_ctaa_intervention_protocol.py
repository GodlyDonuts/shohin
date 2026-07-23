from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib

import pytest

from ctaa_intervention_protocol import (
    ANCHORS_PER_CLASS_DEPTH,
    LOCKED_SCORED_ROW_COUNT,
    MANDATORY_GATES,
    MANDATORY_INTERVENTIONS,
    MANDATORY_OPERATIONS,
    OPERATION_SPECS,
    RUNTIME_PANEL_SIZE,
    AnchorBinding,
    DonorConstraint,
    DonorDerangement,
    DonorPair,
    GateFamily,
    HashRelation,
    InterventionFamily,
    OperationKind,
    OutcomeExpectation,
    Partition,
    PlanBindings,
    ProtocolValidationError,
    RuntimeInterventionPlan,
    _canonical_sha256,
    _validate_donor_pair,
    anchor_to_dict,
    make_donor_derangement,
    make_anchor_operation_commitment,
    make_runtime_intervention_plan,
    operation_semantics_sha256,
    plan_to_dict,
    validate_runtime_intervention_plan,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def bindings() -> PlanBindings:
    batch_order = tuple(
        f"anchor-c{class_index}-d{depth}-i{index:03d}"
        for class_index in range(3)
        for depth in (16, 32)
        for index in range(ANCHORS_PER_CLASS_DEPTH)
    )
    return PlanBindings(
        board_manifest_sha256=digest("board-manifest"),
        board_tree_sha256=digest("board-tree"),
        compiler_sha256=digest("compiler"),
        tokenizer_sha256=digest("tokenizer"),
        base_checkpoint_sha256=digest("base-checkpoint"),
        run_contract_sha256=digest("run-contract"),
        selection_seed=934_702,
        selection_seed_receipt_sha256=digest("seed-receipt"),
        arm_id="treatment",
        training_seed=1234,
        core_sha256=digest("core"),
        core_kind="closure_feature",
        base_raw_evidence_receipt_sha256=digest("base-receipt"),
        runtime_implementation_sha256=digest("runtime-implementation"),
        partition=Partition.DEVELOPMENT,
        batch_order=batch_order,
        batch_order_sha256=_canonical_sha256(
            {
                "schema": "r12_ctaa_v2_runtime_batch_order_v1",
                "value": list(batch_order),
            }
        ),
        scored_row_count=LOCKED_SCORED_ROW_COUNT,
        runtime_panel_size=RUNTIME_PANEL_SIZE,
        runtime_attempts_affect_scored_denominator=False,
    )


def anchors() -> tuple[AnchorBinding, ...]:
    rows = []
    for class_index in range(3):
        for depth in (16, 32):
            for index in range(ANCHORS_PER_CLASS_DEPTH):
                cell = index % 18
                label = f"c{class_index}-d{depth}-i{index:03d}"
                rows.append(
                    AnchorBinding(
                        anchor_id=f"anchor-{label}",
                        family_id=f"family-{label}",
                        class_id=f"class-{class_index}",
                        depth=depth,
                        shift_cell="hhh",
                        renderer_index=index % 16,
                        query_state_cell_id=f"cell-{cell:02d}",
                        query_position=cell % 3,
                        partition=Partition.DEVELOPMENT,
                        program_source_sha256=digest(f"program-{label}"),
                        query_source_sha256=digest(f"query-cell-{cell:02d}"),
                        packet_sha256=digest(f"packet-{label}"),
                        padding_mask_sha256=digest("common-padding"),
                        midpoint_suffix_sha256=digest(
                            f"suffix-c{class_index}-d{depth}"
                        ),
                        midpoint_state_sha256=digest(f"state-{label}"),
                        midpoint_action_sha256=digest(f"action-{label}"),
                        action_card_sha256s=tuple(
                            digest(f"card-{slot}-{label}") for slot in range(4)
                        ),
                    )
                )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.class_id,
                row.depth,
                row.renderer_index,
                row.query_state_cell_id,
                row.anchor_id,
            ),
        )
    )


def donor_derangements(
    panel: tuple[AnchorBinding, ...],
) -> tuple[DonorDerangement, ...]:
    by_stratum: dict[tuple[str, int], list[AnchorBinding]] = {}
    for anchor in panel:
        by_stratum.setdefault((anchor.class_id, anchor.depth), []).append(anchor)
    donor_by_id = {}
    for stratum in by_stratum.values():
        ordered = sorted(stratum, key=lambda item: item.anchor_id)
        for index, anchor in enumerate(ordered):
            donor_by_id[anchor.anchor_id] = ordered[
                (index + 1) % len(ordered)
            ].anchor_id
    pairs = tuple(
        DonorPair(anchor.anchor_id, donor_by_id[anchor.anchor_id])
        for anchor in sorted(panel, key=lambda item: item.anchor_id)
    )
    return tuple(
        make_donor_derangement(operation=spec.operation, pairs=pairs)
        for spec in OPERATION_SPECS.values()
        if spec.expected.donor_constraint is not DonorConstraint.NONE
    )


def attempt_plans(
    panel: tuple[AnchorBinding, ...],
    donors: tuple[DonorDerangement, ...],
    plan_bindings: PlanBindings,
):
    panel_sha = _canonical_sha256([anchor_to_dict(item) for item in panel])
    donor_hashes = {item.operation: item.derangement_sha256 for item in donors}
    attempts = []
    index = 0
    for operation, spec in OPERATION_SPECS.items():
        donor_map = next(
            (
                {pair.anchor_id: pair.donor_anchor_id for pair in item.pairs}
                for item in donors
                if item.operation == operation
            ),
            {},
        )
        operation_sha = operation_semantics_sha256(
            spec, panel_sha, donor_hashes.get(operation)
        )
        for anchor_id in plan_bindings.batch_order:
            attempts.append(
                make_anchor_operation_commitment(
                    attempt_index=index,
                    operation=operation,
                    operation_sha256=operation_sha,
                    anchor_id=anchor_id,
                    donor_anchor_id=donor_map.get(anchor_id),
                    mutation_payload={
                        "schema": "test_mutation_v1",
                        "operation": operation,
                        "anchor_id": anchor_id,
                    },
                )
            )
            index += 1
    return tuple(attempts)


@lru_cache(maxsize=1)
def valid_plan() -> RuntimeInterventionPlan:
    panel = anchors()
    plan_bindings = bindings()
    donors = donor_derangements(panel)
    return make_runtime_intervention_plan(
        bindings=plan_bindings,
        anchors=panel,
        donor_derangements=donors,
        attempts=attempt_plans(panel, donors, plan_bindings),
    )


def plan_with_anchors(
    plan: RuntimeInterventionPlan, panel: tuple[AnchorBinding, ...]
) -> RuntimeInterventionPlan:
    ordered = tuple(
        sorted(
            panel,
            key=lambda row: (
                row.class_id,
                row.depth,
                row.renderer_index,
                row.query_state_cell_id,
                row.anchor_id,
            ),
        )
    )
    return make_runtime_intervention_plan(
        bindings=plan.bindings,
        anchors=ordered,
        donor_derangements=plan.donor_derangements,
        attempts=attempt_plans(ordered, plan.donor_derangements, plan.bindings),
    )


def test_balanced_board_level_plan_validates_and_round_trips() -> None:
    plan = valid_plan()
    assert len(plan.anchors) == 864
    assert len(MANDATORY_INTERVENTIONS) == 26
    assert len(MANDATORY_GATES) == 3
    assert len(plan.operations) == 29
    assert validate_runtime_intervention_plan(plan) == plan
    assert validate_runtime_intervention_plan(plan_to_dict(plan)) == plan


def test_runtime_panel_cannot_change_locked_scored_denominator() -> None:
    plan = valid_plan()
    assert plan.bindings.scored_row_count == 40_608
    assert plan.bindings.runtime_attempts_affect_scored_denominator is False
    for changed in (
        replace(plan.bindings, scored_row_count=40_609),
        replace(plan.bindings, runtime_panel_size=865),
        replace(plan.bindings, runtime_attempts_affect_scored_denominator=True),
    ):
        with pytest.raises(ProtocolValidationError, match="denominator|panel size"):
            validate_runtime_intervention_plan(replace(plan, bindings=changed))


def test_interventions_and_gates_are_distinct_and_alpha_duplicate_is_absent() -> None:
    plan = valid_plan()
    kinds = {item.operation: item.kind for item in plan.operations}
    assert "alpha_recode" not in kinds
    assert all(
        kinds[item.value] is OperationKind.INTERVENTION
        for item in MANDATORY_INTERVENTIONS
    )
    assert all(kinds[item.value] is OperationKind.GATE for item in MANDATORY_GATES)
    assert tuple(item.operation for item in plan.operations) == MANDATORY_OPERATIONS


def test_invalid_invariants_are_resolved_explicitly() -> None:
    specs = OPERATION_SPECS
    post_stop = specs[InterventionFamily.POST_STOP_POISON.value].expected
    assert post_stop.packet is HashRelation.DIFFERENT_FROM_PARENT
    assert post_stop.outcome is OutcomeExpectation.ACTIVE_PREFIX_INVARIANCE
    future = specs[InterventionFamily.FUTURE_MASK.value].expected
    assert future.outcome is OutcomeExpectation.PREFIX_BEFORE_EXPOSURE_INVARIANCE
    assert "before" in future.comparison_scope
    for family in (
        InterventionFamily.ENTITY_RECODE,
        InterventionFamily.WITNESS_RECODE,
        InterventionFamily.OPCODE_RECODE,
    ):
        assert specs[family.value].expected.query_source is HashRelation.SAME_AS_PARENT
    deletion = specs[GateFamily.SOURCE_DELETION.value]
    assert deletion.kind is OperationKind.GATE
    assert deletion.expected.program_source is HashRelation.UNAVAILABLE_DURING_STAGE


@pytest.mark.parametrize(
    "field",
    [
        "board_manifest_sha256",
        "board_tree_sha256",
        "compiler_sha256",
        "tokenizer_sha256",
        "base_checkpoint_sha256",
        "run_contract_sha256",
        "selection_seed_receipt_sha256",
        "batch_order_sha256",
    ],
)
def test_all_provenance_hashes_are_bound_and_canonical(field: str) -> None:
    plan = valid_plan()
    bad = replace(plan.bindings, **{field: "A" * 64})
    with pytest.raises(ProtocolValidationError, match="canonical SHA-256"):
        validate_runtime_intervention_plan(replace(plan, bindings=bad))


def test_seed_arm_and_partition_are_bound() -> None:
    plan = valid_plan()
    with pytest.raises(ProtocolValidationError, match="selection seed"):
        validate_runtime_intervention_plan(
            replace(plan, bindings=replace(plan.bindings, selection_seed=-1))
        )
    with pytest.raises(ProtocolValidationError, match="arm id"):
        validate_runtime_intervention_plan(
            replace(plan, bindings=replace(plan.bindings, arm_id="bad arm"))
        )
    first = replace(plan.anchors[0], partition=Partition.CONFIRMATION)
    with pytest.raises(ProtocolValidationError, match="partition"):
        validate_runtime_intervention_plan(
            replace(plan, anchors=(first,) + plan.anchors[1:])
        )


def test_anchor_panel_requires_hhh_canonical_balanced_selection() -> None:
    plan = valid_plan()
    first = replace(plan.anchors[0], shift_cell="hhl")
    with pytest.raises(ProtocolValidationError, match="hhh"):
        validate_runtime_intervention_plan(
            replace(plan, anchors=(first,) + plan.anchors[1:])
        )
    first = replace(plan.anchors[0], renderer_index=1)
    with pytest.raises(ProtocolValidationError, match="renderer balance"):
        validate_runtime_intervention_plan(
            plan_with_anchors(plan, (first,) + plan.anchors[1:])
        )
    first = replace(plan.anchors[0], query_state_cell_id="cell-01")
    with pytest.raises(ProtocolValidationError, match="query-state balance"):
        validate_runtime_intervention_plan(
            plan_with_anchors(plan, (first,) + plan.anchors[1:])
        )
    with pytest.raises(ProtocolValidationError, match="canonical order"):
        validate_runtime_intervention_plan(
            replace(plan, anchors=tuple(reversed(plan.anchors)))
        )


def replace_derangement(
    plan: RuntimeInterventionPlan, operation: str, **changes: object
) -> RuntimeInterventionPlan:
    values = list(plan.donor_derangements)
    index = next(i for i, item in enumerate(values) if item.operation == operation)
    values[index] = replace(values[index], **changes)
    return replace(plan, donor_derangements=tuple(values))


def test_donor_maps_are_complete_one_to_one_derangements() -> None:
    plan = valid_plan()
    operation = InterventionFamily.H19_DONOR_TRANSPLANT.value
    item = next(
        value for value in plan.donor_derangements if value.operation == operation
    )
    pairs = list(item.pairs)
    pairs[0] = replace(pairs[0], donor_anchor_id=pairs[0].anchor_id)
    with pytest.raises(ProtocolValidationError, match="fixed point"):
        validate_runtime_intervention_plan(
            replace_derangement(plan, operation, pairs=tuple(pairs))
        )
    pairs = list(item.pairs)
    pairs[0] = replace(pairs[0], donor_anchor_id=pairs[1].donor_anchor_id)
    with pytest.raises(ProtocolValidationError, match="not one-to-one"):
        validate_runtime_intervention_plan(
            replace_derangement(plan, operation, pairs=tuple(pairs))
        )
    with pytest.raises(ProtocolValidationError, match="parent coverage"):
        validate_runtime_intervention_plan(
            replace_derangement(plan, operation, pairs=item.pairs[1:])
        )


def test_residual_donor_requires_exact_padding_and_different_packet() -> None:
    plan = valid_plan()
    operation = InterventionFamily.H19_BATCH_ROTATE.value
    item = next(
        value for value in plan.donor_derangements if value.operation == operation
    )
    first_pair = item.pairs[0]
    donor_index = next(
        i
        for i, anchor in enumerate(plan.anchors)
        if anchor.anchor_id == first_pair.donor_anchor_id
    )
    donor = replace(
        plan.anchors[donor_index], padding_mask_sha256=digest("wrong-padding")
    )
    changed = plan.anchors[:donor_index] + (donor,) + plan.anchors[donor_index + 1 :]
    with pytest.raises(ProtocolValidationError, match="padding mask"):
        validate_runtime_intervention_plan(plan_with_anchors(plan, changed))


def test_midpoint_and_query_donor_constraints_are_enforced() -> None:
    plan = valid_plan()
    midpoint_op = InterventionFamily.MIDPOINT_DONOR_STATE.value
    midpoint = next(
        item for item in plan.donor_derangements if item.operation == midpoint_op
    )
    pair = midpoint.pairs[0]
    donor_index = next(
        i
        for i, item in enumerate(plan.anchors)
        if item.anchor_id == pair.donor_anchor_id
    )
    donor = replace(
        plan.anchors[donor_index], midpoint_suffix_sha256=digest("wrong-suffix")
    )
    changed = plan.anchors[:donor_index] + (donor,) + plan.anchors[donor_index + 1 :]
    with pytest.raises(ProtocolValidationError, match="midpoint donor suffix"):
        validate_runtime_intervention_plan(plan_with_anchors(plan, changed))

    query_op = InterventionFamily.LATE_QUERY_SWAP.value
    query_map = next(
        item for item in plan.donor_derangements if item.operation == query_op
    )
    pair = query_map.pairs[0]
    parent = next(item for item in plan.anchors if item.anchor_id == pair.anchor_id)
    donor_index = next(
        i
        for i, item in enumerate(plan.anchors)
        if item.anchor_id == pair.donor_anchor_id
    )
    donor = replace(
        plan.anchors[donor_index], query_source_sha256=parent.query_source_sha256
    )
    changed = plan.anchors[:donor_index] + (donor,) + plan.anchors[donor_index + 1 :]
    with pytest.raises(ProtocolValidationError, match="does not change query"):
        validate_runtime_intervention_plan(plan_with_anchors(plan, changed))


def test_operation_specific_donors_cannot_be_semantic_noops() -> None:
    plan = valid_plan()
    parent = plan.anchors[0]
    candidate = next(
        item
        for item in plan.anchors[1:]
        if item.depth == parent.depth and item.class_id == parent.class_id
    )
    same_query_position = replace(
        candidate,
        query_position=parent.query_position,
        query_source_sha256=digest("different-wording-same-position"),
    )
    with pytest.raises(ProtocolValidationError, match="does not change query"):
        _validate_donor_pair(
            DonorConstraint.LATE_QUERY_DIFFERENT, parent, same_query_position
        )

    same_state = replace(
        candidate,
        midpoint_suffix_sha256=parent.midpoint_suffix_sha256,
        midpoint_state_sha256=parent.midpoint_state_sha256,
    )
    with pytest.raises(ProtocolValidationError, match="injected register"):
        _validate_donor_pair(
            DonorConstraint.MIDPOINT_STATE_DIFFERENT_MATCHED_SUFFIX,
            parent,
            same_state,
        )

    same_actions = replace(
        candidate,
        midpoint_suffix_sha256=parent.midpoint_suffix_sha256,
        action_card_sha256s=(parent.midpoint_action_sha256,) * 4,
    )
    with pytest.raises(ProtocolValidationError, match="injected register"):
        _validate_donor_pair(
            DonorConstraint.MIDPOINT_ACTION_DIFFERENT_MATCHED_SUFFIX,
            parent,
            same_actions,
        )


def test_stage_timing_parameters_and_kind_cannot_be_rewritten() -> None:
    plan = valid_plan()
    operations = list(plan.operations)
    operations[0] = replace(operations[0], timing="after_final_norm")
    with pytest.raises(ProtocolValidationError, match="semantic contract"):
        validate_runtime_intervention_plan(replace(plan, operations=tuple(operations)))
    operations = list(plan.operations)
    operations[0] = replace(operations[0], parameters=(("mutation", "noop"),))
    with pytest.raises(ProtocolValidationError, match="semantic contract"):
        validate_runtime_intervention_plan(replace(plan, operations=tuple(operations)))
    operations = list(plan.operations)
    gate_index = MANDATORY_OPERATIONS.index(GateFamily.QUERY_ISOLATION.value)
    operations[gate_index] = replace(
        operations[gate_index], kind=OperationKind.INTERVENTION
    )
    with pytest.raises(ProtocolValidationError, match="semantic contract"):
        validate_runtime_intervention_plan(replace(plan, operations=tuple(operations)))


def test_operation_set_order_and_donor_binding_are_fail_closed() -> None:
    plan = valid_plan()
    with pytest.raises(ProtocolValidationError, match="operation order/set"):
        validate_runtime_intervention_plan(
            replace(plan, operations=plan.operations[:-1])
        )
    operations = list(plan.operations)
    operations[0], operations[1] = operations[1], operations[0]
    with pytest.raises(ProtocolValidationError, match="operation order/set"):
        validate_runtime_intervention_plan(replace(plan, operations=tuple(operations)))
    donor_index = MANDATORY_OPERATIONS.index(InterventionFamily.H19_BATCH_ROTATE.value)
    operations = list(plan.operations)
    operations[donor_index] = replace(
        operations[donor_index], donor_derangement_sha256=digest("forged-donor")
    )
    with pytest.raises(ProtocolValidationError, match="donor binding"):
        validate_runtime_intervention_plan(replace(plan, operations=tuple(operations)))


def test_residual_capture_points_bind_zero_based_block_indices() -> None:
    plan = valid_plan()
    for operation in plan.operations[:6]:
        parameters = dict(operation.parameters)
        assert parameters["block_index_base"] == "zero"
        expected = "19" if operation.operation.startswith("h19_") else "29"
        assert parameters["block_index"] == expected


def test_commitments_and_unknown_nested_fields_detect_mutation() -> None:
    plan = valid_plan()
    with pytest.raises(ProtocolValidationError, match="plan commitment"):
        validate_runtime_intervention_plan(
            replace(plan, plan_sha256=digest("forged-plan"))
        )
    payload = plan_to_dict(plan)
    payload["operations"][0]["expected"]["retry_allowed"] = False  # type: ignore[index]
    with pytest.raises(ProtocolValidationError, match="schema"):
        validate_runtime_intervention_plan(payload)


def test_attempt_registry_rejects_missing_and_substituted_records() -> None:
    plan = valid_plan()
    with pytest.raises(ProtocolValidationError, match="attempt count"):
        validate_runtime_intervention_plan(replace(plan, attempts=plan.attempts[:-1]))
    changed = list(plan.attempts)
    changed[0] = replace(changed[0], mutation_payload_json='{"forged":true}')
    with pytest.raises(ProtocolValidationError, match="payload commitment"):
        validate_runtime_intervention_plan(replace(plan, attempts=tuple(changed)))


@pytest.mark.parametrize(
    "payload",
    ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'),
)
def test_hostile_attempt_payload_json_is_rejected(payload: str) -> None:
    plan = valid_plan()
    changed = list(plan.attempts)
    changed[0] = replace(changed[0], mutation_payload_json=payload)
    with pytest.raises(ProtocolValidationError):
        validate_runtime_intervention_plan(replace(plan, attempts=tuple(changed)))


def test_unknown_attempt_operation_fails_closed() -> None:
    plan = valid_plan()
    changed = list(plan.attempts)
    changed[0] = replace(changed[0], operation="unknown_operation")
    with pytest.raises(
        ProtocolValidationError, match="(operation|coverage/order) differs"
    ):
        validate_runtime_intervention_plan(replace(plan, attempts=tuple(changed)))
