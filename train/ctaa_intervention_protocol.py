"""Frozen board-level protocol for CTAA runtime interventions and gates.

The runtime panel is a repeated-measure sidecar over 864 balanced ``hhh``
anchors.  It never creates scored child rows and therefore cannot alter the
locked 40,608-row CTAA denominator.  This module commits the experiment plan;
it does not inspect model outputs or execute an intervention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Mapping, Sequence


PLAN_SCHEMA = "r12_ctaa_runtime_intervention_plan_v3"
DERANGEMENT_SCHEMA = "r12_ctaa_donor_derangement_v1"
OPERATION_SCHEMA = "r12_ctaa_runtime_operation_v1"
ATTEMPT_PLAN_SCHEMA = "r12_ctaa_anchor_operation_commitment_v1"
LOCKED_SCORED_ROW_COUNT = 40_608
RUNTIME_PANEL_SIZE = 864
ANCHORS_PER_CLASS_DEPTH = 144
RENDERER_COUNT = 16
ANCHORS_PER_RENDERER = 9
QUERY_STATE_CELL_COUNT = 18
ANCHORS_PER_QUERY_STATE_CELL = 8
ANCHORS_PER_QUERY_POSITION = 48
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")


class ProtocolValidationError(ValueError):
    """Raised when a runtime intervention plan is incomplete or inconsistent."""


class InterventionStage(str, Enum):
    SOURCE = "source"
    COMPILE = "compile"
    PACKET = "packet"
    EXECUTION = "execution"
    QUERY = "query"
    CUSTODY = "custody"
    ASSESSMENT = "assessment"


class OperationKind(str, Enum):
    INTERVENTION = "intervention"
    GATE = "gate"


class Partition(str, Enum):
    DEVELOPMENT = "development"
    CONFIRMATION = "confirmation"


class InterventionFamily(str, Enum):
    H19_ZERO = "h19_zero"
    H19_BATCH_ROTATE = "h19_batch_rotate"
    H19_DONOR_TRANSPLANT = "h19_donor_transplant"
    H29_ZERO = "h29_zero"
    H29_BATCH_ROTATE = "h29_batch_rotate"
    H29_DONOR_TRANSPLANT = "h29_donor_transplant"
    ENTITY_RECODE = "entity_recode"
    WITNESS_RECODE = "witness_recode"
    OPCODE_RECODE = "opcode_recode"
    RENDERER_SUBSTITUTION = "renderer_substitution"
    RULE_LINE_SHUFFLE = "rule_line_shuffle"
    CARD_STORAGE_REINDEX = "card_storage_reindex"
    WITNESS_CORRUPTION = "witness_corruption"
    PAIRED_SHUFFLED_LAW = "paired_shuffled_law"
    SCHEDULE_ORDER_TWIN = "schedule_order_twin"
    SOURCE_POISON = "source_poison"
    FUTURE_MASK = "future_mask"
    STOP_RELOCATION = "stop_relocation"
    LATE_QUERY_SWAP = "late_query_swap"
    POST_STOP_POISON = "post_stop_poison"
    MIDPOINT_DONOR_STATE = "midpoint_donor_state"
    MIDPOINT_DONOR_ACTION = "midpoint_donor_action"
    PACKET_TRANSPLANT = "packet_transplant"


class GateFamily(str, Enum):
    SOURCE_DELETION = "source_deletion"
    QUERY_ISOLATION = "query_isolation"
    ROUTE_AGREEMENT = "state_route_composed_route_agreement"


MANDATORY_INTERVENTIONS = tuple(InterventionFamily)
MANDATORY_GATES = tuple(GateFamily)
MANDATORY_OPERATIONS = tuple(item.value for item in MANDATORY_INTERVENTIONS) + tuple(
    item.value for item in MANDATORY_GATES
)


class HashRelation(str, Enum):
    SAME_AS_PARENT = "same_as_parent"
    DIFFERENT_FROM_PARENT = "different_from_parent"
    SAME_AS_DONOR = "same_as_donor"
    UNAVAILABLE_DURING_STAGE = "unavailable_during_stage"
    NOT_APPLICABLE = "not_applicable"


class OutcomeExpectation(str, Enum):
    CAUSAL_DISRUPTION = "causal_disruption"
    TERMINAL_INVARIANCE = "terminal_invariance"
    DONOR_FOLLOWING = "donor_following"
    ACTIVE_PREFIX_INVARIANCE = "active_prefix_invariance"
    PREFIX_BEFORE_EXPOSURE_INVARIANCE = "prefix_before_exposure_invariance"
    QUERY_RECOMPUTED_FROM_PARENT_TERMINAL = "query_recomputed_from_parent_terminal"
    CUSTODY_RECEIPT = "custody_receipt"
    ROUTE_AGREEMENT = "route_agreement"


class DonorConstraint(str, Enum):
    NONE = "none"
    RESIDUAL_EXACT_MASK = "residual_exact_mask"
    LATE_QUERY_DIFFERENT = "late_query_different"
    MIDPOINT_STATE_DIFFERENT_MATCHED_SUFFIX = "midpoint_state_different_matched_suffix"
    MIDPOINT_ACTION_DIFFERENT_MATCHED_SUFFIX = (
        "midpoint_action_different_matched_suffix"
    )
    PACKET_DIFFERENT = "packet_different"


@dataclass(frozen=True)
class PlanBindings:
    board_manifest_sha256: str
    board_tree_sha256: str
    compiler_sha256: str
    tokenizer_sha256: str
    base_checkpoint_sha256: str
    run_contract_sha256: str
    selection_seed: int
    selection_seed_receipt_sha256: str
    arm_id: str
    training_seed: int
    core_sha256: str
    core_kind: str
    base_raw_evidence_receipt_sha256: str
    runtime_implementation_sha256: str
    partition: Partition
    batch_order: tuple[str, ...]
    batch_order_sha256: str
    scored_row_count: int
    runtime_panel_size: int
    runtime_attempts_affect_scored_denominator: bool


@dataclass(frozen=True)
class AnchorBinding:
    anchor_id: str
    family_id: str
    class_id: str
    depth: int
    shift_cell: str
    renderer_index: int
    query_state_cell_id: str
    query_position: int
    partition: Partition
    program_source_sha256: str
    query_source_sha256: str
    packet_sha256: str
    padding_mask_sha256: str
    midpoint_suffix_sha256: str
    midpoint_state_sha256: str
    midpoint_action_sha256: str
    action_card_sha256s: tuple[str, ...]


@dataclass(frozen=True)
class DonorPair:
    anchor_id: str
    donor_anchor_id: str


@dataclass(frozen=True)
class DonorDerangement:
    schema: str
    operation: str
    constraint: DonorConstraint
    pairs: tuple[DonorPair, ...]
    derangement_sha256: str


@dataclass(frozen=True)
class ExpectedSemantics:
    program_source: HashRelation
    query_source: HashRelation
    packet: HashRelation
    outcome: OutcomeExpectation
    comparison_scope: str
    identical_right_padding_masks: bool
    donor_constraint: DonorConstraint


@dataclass(frozen=True)
class OperationSpec:
    operation: str
    kind: OperationKind
    stage: InterventionStage
    relation: str
    timing: str
    expected: ExpectedSemantics
    parameters: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class OperationCommitment:
    schema: str
    operation: str
    kind: OperationKind
    stage: InterventionStage
    relation: str
    timing: str
    expected: ExpectedSemantics
    parameters: tuple[tuple[str, str], ...]
    anchor_panel_sha256: str
    donor_derangement_sha256: str | None
    attempt_count: int
    attempts_sha256: str
    operation_sha256: str


@dataclass(frozen=True)
class AnchorOperationCommitment:
    schema: str
    attempt_index: int
    attempt_id: str
    operation: str
    operation_sha256: str
    anchor_id: str
    donor_anchor_id: str | None
    mutation_payload_json: str
    mutation_payload_sha256: str
    resulting_program_source_sha256: str | None
    resulting_query_source_sha256: str | None
    resulting_packet_sha256: str | None
    attempt_plan_sha256: str


@dataclass(frozen=True)
class RuntimeInterventionPlan:
    schema: str
    bindings: PlanBindings
    anchors: tuple[AnchorBinding, ...]
    anchor_panel_sha256: str
    donor_derangements: tuple[DonorDerangement, ...]
    donor_registry_sha256: str
    operations: tuple[OperationCommitment, ...]
    attempts: tuple[AnchorOperationCommitment, ...]
    attempts_sha256: str
    plan_sha256: str


def _params(**values: str) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(values.items()))


def _expected(
    *,
    program: HashRelation = HashRelation.SAME_AS_PARENT,
    query: HashRelation = HashRelation.SAME_AS_PARENT,
    packet: HashRelation = HashRelation.SAME_AS_PARENT,
    outcome: OutcomeExpectation,
    scope: str,
    donor: DonorConstraint = DonorConstraint.NONE,
    exact_padding: bool = False,
) -> ExpectedSemantics:
    return ExpectedSemantics(
        program, query, packet, outcome, scope, exact_padding, donor
    )


def _spec(
    operation: InterventionFamily | GateFamily,
    kind: OperationKind,
    stage: InterventionStage,
    relation: str,
    timing: str,
    expected: ExpectedSemantics,
    **parameters: str,
) -> OperationSpec:
    return OperationSpec(
        operation.value,
        kind,
        stage,
        relation,
        timing,
        expected,
        _params(**parameters),
    )


_I = OperationKind.INTERVENTION
_G = OperationKind.GATE
_SAME = HashRelation.SAME_AS_PARENT
_DIFF = HashRelation.DIFFERENT_FROM_PARENT
_DONOR = HashRelation.SAME_AS_DONOR


_SPECS = [
    _spec(
        InterventionFamily.H19_ZERO,
        _I,
        InterventionStage.COMPILE,
        "residual_zero",
        "post_block_19_pre_block_20",
        _expected(
            packet=_DIFF,
            outcome=OutcomeExpectation.CAUSAL_DISRUPTION,
            scope="field_and_terminal_effect",
        ),
        block_index="19",
        block_index_base="zero",
        capture_point="raw_residual_post_block_19_pre_block_20",
        mutation="zero_selected_residual",
    ),
    _spec(
        InterventionFamily.H19_BATCH_ROTATE,
        _I,
        InterventionStage.COMPILE,
        "residual_batch_rotate",
        "post_block_19_pre_block_20",
        _expected(
            packet=_DIFF,
            outcome=OutcomeExpectation.DONOR_FOLLOWING,
            scope="field_level_donor_transport",
            donor=DonorConstraint.RESIDUAL_EXACT_MASK,
            exact_padding=True,
        ),
        block_index="19",
        block_index_base="zero",
        capture_point="raw_residual_post_block_19_pre_block_20",
        mutation="frozen_batch_derangement",
    ),
    _spec(
        InterventionFamily.H19_DONOR_TRANSPLANT,
        _I,
        InterventionStage.COMPILE,
        "residual_donor_transplant",
        "post_block_19_pre_block_20",
        _expected(
            packet=_DIFF,
            outcome=OutcomeExpectation.DONOR_FOLLOWING,
            scope="field_level_donor_transport",
            donor=DonorConstraint.RESIDUAL_EXACT_MASK,
            exact_padding=True,
        ),
        block_index="19",
        block_index_base="zero",
        capture_point="raw_residual_post_block_19_pre_block_20",
        mutation="frozen_donor_residual",
    ),
    _spec(
        InterventionFamily.H29_ZERO,
        _I,
        InterventionStage.COMPILE,
        "residual_zero",
        "post_block_29_pre_final_norm",
        _expected(
            packet=_DIFF,
            outcome=OutcomeExpectation.CAUSAL_DISRUPTION,
            scope="field_and_terminal_effect",
        ),
        block_index="29",
        block_index_base="zero",
        capture_point="raw_residual_post_block_29_pre_final_norm",
        mutation="zero_selected_residual",
    ),
    _spec(
        InterventionFamily.H29_BATCH_ROTATE,
        _I,
        InterventionStage.COMPILE,
        "residual_batch_rotate",
        "post_block_29_pre_final_norm",
        _expected(
            packet=_DIFF,
            outcome=OutcomeExpectation.DONOR_FOLLOWING,
            scope="field_level_donor_transport",
            donor=DonorConstraint.RESIDUAL_EXACT_MASK,
            exact_padding=True,
        ),
        block_index="29",
        block_index_base="zero",
        capture_point="raw_residual_post_block_29_pre_final_norm",
        mutation="frozen_batch_derangement",
    ),
    _spec(
        InterventionFamily.H29_DONOR_TRANSPLANT,
        _I,
        InterventionStage.COMPILE,
        "residual_donor_transplant",
        "post_block_29_pre_final_norm",
        _expected(
            packet=_DIFF,
            outcome=OutcomeExpectation.DONOR_FOLLOWING,
            scope="field_level_donor_transport",
            donor=DonorConstraint.RESIDUAL_EXACT_MASK,
            exact_padding=True,
        ),
        block_index="29",
        block_index_base="zero",
        capture_point="raw_residual_post_block_29_pre_final_norm",
        mutation="frozen_donor_residual",
    ),
    _spec(
        InterventionFamily.ENTITY_RECODE,
        _I,
        InterventionStage.SOURCE,
        "alpha_invariance",
        "pre_compile_source_transform",
        _expected(
            program=_DIFF,
            outcome=OutcomeExpectation.TERMINAL_INVARIANCE,
            scope="complete_execution",
        ),
        namespace="entity_symbols_only",
        fixed_position_query="unchanged",
    ),
    _spec(
        InterventionFamily.WITNESS_RECODE,
        _I,
        InterventionStage.SOURCE,
        "alpha_invariance",
        "pre_compile_source_transform",
        _expected(
            program=_DIFF,
            outcome=OutcomeExpectation.TERMINAL_INVARIANCE,
            scope="complete_execution",
        ),
        namespace="witness_symbols_only",
        fixed_position_query="unchanged",
    ),
    _spec(
        InterventionFamily.OPCODE_RECODE,
        _I,
        InterventionStage.SOURCE,
        "alpha_invariance",
        "pre_compile_source_transform",
        _expected(
            program=_DIFF,
            outcome=OutcomeExpectation.TERMINAL_INVARIANCE,
            scope="complete_execution",
        ),
        namespace="opcode_symbols_only",
        fixed_position_query="unchanged",
    ),
    _spec(
        InterventionFamily.RENDERER_SUBSTITUTION,
        _I,
        InterventionStage.SOURCE,
        "renderer_invariance",
        "pre_compile_source_transform",
        _expected(
            program=_DIFF,
            outcome=OutcomeExpectation.TERMINAL_INVARIANCE,
            scope="complete_execution",
        ),
        renderer="frozen_alternate_renderer_same_partition_coset",
        fixed_position_query="unchanged",
    ),
    _spec(
        InterventionFamily.RULE_LINE_SHUFFLE,
        _I,
        InterventionStage.SOURCE,
        "renderer_invariance",
        "pre_compile_source_transform",
        _expected(
            program=_DIFF,
            outcome=OutcomeExpectation.TERMINAL_INVARIANCE,
            scope="complete_execution",
        ),
        shuffle="seed_frozen_physical_rule_line_permutation",
    ),
    _spec(
        InterventionFamily.CARD_STORAGE_REINDEX,
        _I,
        InterventionStage.PACKET,
        "card_storage_reindex_invariance",
        "post_seal_pre_execution",
        _expected(
            packet=_DIFF,
            outcome=OutcomeExpectation.TERMINAL_INVARIANCE,
            scope="complete_execution",
        ),
        transform="packet_card_storage_and_schedule_rebinding",
    ),
    _spec(
        InterventionFamily.WITNESS_CORRUPTION,
        _I,
        InterventionStage.SOURCE,
        "witness_corruption_sensitivity",
        "pre_compile_source_transform",
        _expected(
            program=_DIFF,
            packet=_DIFF,
            outcome=OutcomeExpectation.CAUSAL_DISRUPTION,
            scope="coherent_alternate_witness_law",
        ),
        corruption="coherent_alternate_witness_mapping",
    ),
    _spec(
        InterventionFamily.PAIRED_SHUFFLED_LAW,
        _I,
        InterventionStage.SOURCE,
        "shuffled_law_sensitivity",
        "pre_compile_source_transform",
        _expected(
            program=_DIFF,
            packet=_DIFF,
            outcome=OutcomeExpectation.CAUSAL_DISRUPTION,
            scope="coherent_alternate_action_law",
        ),
        shuffle="paired_seed_frozen_action_law",
    ),
    _spec(
        InterventionFamily.SCHEDULE_ORDER_TWIN,
        _I,
        InterventionStage.SOURCE,
        "schedule_order_sensitivity",
        "pre_compile_source_transform",
        _expected(
            program=_DIFF,
            packet=_DIFF,
            outcome=OutcomeExpectation.CAUSAL_DISRUPTION,
            scope="complete_execution",
        ),
        transform="active_schedule_order_twin",
    ),
    _spec(
        InterventionFamily.SOURCE_POISON,
        _I,
        InterventionStage.CUSTODY,
        "sealed_packet_source_invariance",
        "post_seal_pre_execution",
        _expected(
            program=_DIFF,
            outcome=OutcomeExpectation.TERMINAL_INVARIANCE,
            scope="identical_sealed_packet_replay",
        ),
        target="isolated_compiler_workspace_source_copy",
    ),
    _spec(
        InterventionFamily.FUTURE_MASK,
        _I,
        InterventionStage.EXECUTION,
        "future_prefix_invariance",
        "before_first_mutated_slot_exposure",
        _expected(
            packet=_DIFF,
            outcome=OutcomeExpectation.PREFIX_BEFORE_EXPOSURE_INVARIANCE,
            scope="states_strictly_before_first_mutated_slot_is_current",
        ),
        mutation="unseen_future_slots_only",
    ),
    _spec(
        InterventionFamily.STOP_RELOCATION,
        _I,
        InterventionStage.SOURCE,
        "stop_relocation_sensitivity",
        "pre_compile_source_transform",
        _expected(
            program=_DIFF,
            packet=_DIFF,
            outcome=OutcomeExpectation.CAUSAL_DISRUPTION,
            scope="complete_execution",
        ),
        transform="relocate_first_active_stop",
    ),
    _spec(
        InterventionFamily.LATE_QUERY_SWAP,
        _I,
        InterventionStage.QUERY,
        "late_query_recomputation",
        "after_immutable_execution_receipt",
        _expected(
            query=_DONOR,
            outcome=OutcomeExpectation.QUERY_RECOMPUTED_FROM_PARENT_TERMINAL,
            scope="same_parent_terminal_alternate_query",
            donor=DonorConstraint.LATE_QUERY_DIFFERENT,
        ),
        execution_policy="do_not_rerun",
    ),
    _spec(
        InterventionFamily.POST_STOP_POISON,
        _I,
        InterventionStage.PACKET,
        "hard_halt_active_prefix_invariance",
        "post_seal_pre_execution",
        _expected(
            packet=_DIFF,
            outcome=OutcomeExpectation.ACTIVE_PREFIX_INVARIANCE,
            scope="states_and_terminal_from_active_prefix_through_first_stop",
        ),
        mutation="packet_suffix_strictly_after_first_stop",
    ),
    _spec(
        InterventionFamily.MIDPOINT_DONOR_STATE,
        _I,
        InterventionStage.EXECUTION,
        "midpoint_donor_state_following",
        "after_midpoint_before_next_transition",
        _expected(
            outcome=OutcomeExpectation.DONOR_FOLLOWING,
            scope="hybrid_transition_and_terminal_oracle",
            donor=DonorConstraint.MIDPOINT_STATE_DIFFERENT_MATCHED_SUFFIX,
        ),
        injection="host_owned_state_register",
    ),
    _spec(
        InterventionFamily.MIDPOINT_DONOR_ACTION,
        _I,
        InterventionStage.EXECUTION,
        "midpoint_donor_action_following",
        "at_midpoint_current_action_before_transition",
        _expected(
            outcome=OutcomeExpectation.DONOR_FOLLOWING,
            scope="immediate_transition_and_hybrid_terminal_oracle",
            donor=DonorConstraint.MIDPOINT_ACTION_DIFFERENT_MATCHED_SUFFIX,
        ),
        injection="host_owned_current_action",
    ),
    _spec(
        InterventionFamily.PACKET_TRANSPLANT,
        _I,
        InterventionStage.PACKET,
        "packet_donor_transplant",
        "post_seal_pre_execution",
        _expected(
            packet=_DONOR,
            outcome=OutcomeExpectation.DONOR_FOLLOWING,
            scope="donor_packet_execution",
            donor=DonorConstraint.PACKET_DIFFERENT,
        ),
        transplant="literal_sealed_packet_bytes",
    ),
    _spec(
        GateFamily.SOURCE_DELETION,
        _G,
        InterventionStage.CUSTODY,
        "source_inaccessibility",
        "entire_execution_window",
        _expected(
            program=HashRelation.UNAVAILABLE_DURING_STAGE,
            outcome=OutcomeExpectation.CUSTODY_RECEIPT,
            scope="source_root_inaccessible_during_execution",
        ),
        receipt="sandbox_source_inaccessibility_receipt",
    ),
    _spec(
        GateFamily.QUERY_ISOLATION,
        _G,
        InterventionStage.QUERY,
        "query_noninterference",
        "execution_until_immutable_receipt",
        _expected(
            query=HashRelation.UNAVAILABLE_DURING_STAGE,
            outcome=OutcomeExpectation.CUSTODY_RECEIPT,
            scope="query_disclosed_only_after_execution_receipt",
        ),
        receipt="query_disclosure_order_receipt",
    ),
    _spec(
        GateFamily.ROUTE_AGREEMENT,
        _G,
        InterventionStage.ASSESSMENT,
        "state_route_composed_route_agreement",
        "post_execution_assessment",
        _expected(
            outcome=OutcomeExpectation.ROUTE_AGREEMENT,
            scope="state_route_equals_composed_route_per_position",
        ),
        assessment="independent_exact_route_comparison",
    ),
]


OPERATION_SPECS: Mapping[str, OperationSpec] = MappingProxyType(
    {spec.operation: spec for spec in _SPECS}
)
if (
    tuple(OPERATION_SPECS) != MANDATORY_OPERATIONS
):  # pragma: no cover - import invariant
    raise RuntimeError("CTAA operation specification order differs")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolValidationError(
                "CTAA runtime attempt payload has duplicate keys"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ProtocolValidationError(
        f"CTAA runtime attempt payload has non-finite constant: {value}"
    )


def bindings_to_dict(value: PlanBindings) -> dict[str, object]:
    return {
        "board_manifest_sha256": value.board_manifest_sha256,
        "board_tree_sha256": value.board_tree_sha256,
        "compiler_sha256": value.compiler_sha256,
        "tokenizer_sha256": value.tokenizer_sha256,
        "base_checkpoint_sha256": value.base_checkpoint_sha256,
        "run_contract_sha256": value.run_contract_sha256,
        "selection_seed": value.selection_seed,
        "selection_seed_receipt_sha256": value.selection_seed_receipt_sha256,
        "arm_id": value.arm_id,
        "training_seed": value.training_seed,
        "core_sha256": value.core_sha256,
        "core_kind": value.core_kind,
        "base_raw_evidence_receipt_sha256": value.base_raw_evidence_receipt_sha256,
        "runtime_implementation_sha256": value.runtime_implementation_sha256,
        "partition": value.partition.value,
        "batch_order": list(value.batch_order),
        "batch_order_sha256": value.batch_order_sha256,
        "scored_row_count": value.scored_row_count,
        "runtime_panel_size": value.runtime_panel_size,
        "runtime_attempts_affect_scored_denominator": (
            value.runtime_attempts_affect_scored_denominator
        ),
    }


def anchor_to_dict(value: AnchorBinding) -> dict[str, object]:
    return {
        "anchor_id": value.anchor_id,
        "family_id": value.family_id,
        "class_id": value.class_id,
        "depth": value.depth,
        "shift_cell": value.shift_cell,
        "renderer_index": value.renderer_index,
        "query_state_cell_id": value.query_state_cell_id,
        "query_position": value.query_position,
        "partition": value.partition.value,
        "program_source_sha256": value.program_source_sha256,
        "query_source_sha256": value.query_source_sha256,
        "packet_sha256": value.packet_sha256,
        "padding_mask_sha256": value.padding_mask_sha256,
        "midpoint_suffix_sha256": value.midpoint_suffix_sha256,
        "midpoint_state_sha256": value.midpoint_state_sha256,
        "midpoint_action_sha256": value.midpoint_action_sha256,
        "action_card_sha256s": list(value.action_card_sha256s),
    }


def pair_to_dict(value: DonorPair) -> dict[str, str]:
    return {"anchor_id": value.anchor_id, "donor_anchor_id": value.donor_anchor_id}


def derangement_to_dict(
    value: DonorDerangement, *, include_hash: bool = True
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": value.schema,
        "operation": value.operation,
        "constraint": value.constraint.value,
        "pairs": [pair_to_dict(pair) for pair in value.pairs],
    }
    if include_hash:
        result["derangement_sha256"] = value.derangement_sha256
    return result


def expected_to_dict(value: ExpectedSemantics) -> dict[str, object]:
    return {
        "program_source": value.program_source.value,
        "query_source": value.query_source.value,
        "packet": value.packet.value,
        "outcome": value.outcome.value,
        "comparison_scope": value.comparison_scope,
        "identical_right_padding_masks": value.identical_right_padding_masks,
        "donor_constraint": value.donor_constraint.value,
    }


def operation_to_dict(
    value: OperationCommitment,
    *,
    include_hash: bool = True,
    include_attempts: bool = True,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": value.schema,
        "operation": value.operation,
        "kind": value.kind.value,
        "stage": value.stage.value,
        "relation": value.relation,
        "timing": value.timing,
        "expected": expected_to_dict(value.expected),
        "parameters": [list(pair) for pair in value.parameters],
        "anchor_panel_sha256": value.anchor_panel_sha256,
        "donor_derangement_sha256": value.donor_derangement_sha256,
    }
    if include_attempts:
        result["attempt_count"] = value.attempt_count
        result["attempts_sha256"] = value.attempts_sha256
    if include_hash:
        result["operation_sha256"] = value.operation_sha256
    return result


def attempt_to_dict(
    value: AnchorOperationCommitment, *, include_hash: bool = True
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": value.schema,
        "attempt_index": value.attempt_index,
        "attempt_id": value.attempt_id,
        "operation": value.operation,
        "operation_sha256": value.operation_sha256,
        "anchor_id": value.anchor_id,
        "donor_anchor_id": value.donor_anchor_id,
        "mutation_payload_json": value.mutation_payload_json,
        "mutation_payload_sha256": value.mutation_payload_sha256,
        "resulting_program_source_sha256": value.resulting_program_source_sha256,
        "resulting_query_source_sha256": value.resulting_query_source_sha256,
        "resulting_packet_sha256": value.resulting_packet_sha256,
    }
    if include_hash:
        result["attempt_plan_sha256"] = value.attempt_plan_sha256
    return result


def plan_to_dict(
    value: RuntimeInterventionPlan, *, include_hash: bool = True
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": value.schema,
        "bindings": bindings_to_dict(value.bindings),
        "anchors": [anchor_to_dict(anchor) for anchor in value.anchors],
        "anchor_panel_sha256": value.anchor_panel_sha256,
        "donor_derangements": [
            derangement_to_dict(item) for item in value.donor_derangements
        ],
        "donor_registry_sha256": value.donor_registry_sha256,
        "operations": [operation_to_dict(item) for item in value.operations],
        "attempts": [attempt_to_dict(item) for item in value.attempts],
        "attempts_sha256": value.attempts_sha256,
    }
    if include_hash:
        result["plan_sha256"] = value.plan_sha256
    return result


def make_donor_derangement(
    *, operation: str, pairs: Sequence[DonorPair]
) -> DonorDerangement:
    spec = OPERATION_SPECS.get(operation)
    if spec is None or spec.expected.donor_constraint is DonorConstraint.NONE:
        raise ProtocolValidationError("CTAA donor derangement operation differs")
    value = DonorDerangement(
        DERANGEMENT_SCHEMA,
        operation,
        spec.expected.donor_constraint,
        tuple(pairs),
        "",
    )
    return replace(
        value,
        derangement_sha256=_canonical_sha256(
            derangement_to_dict(value, include_hash=False)
        ),
    )


def _make_operation(
    spec: OperationSpec,
    anchor_panel_sha256: str,
    donor_sha256: str | None,
    attempt_count: int,
    attempts_sha256: str,
) -> OperationCommitment:
    value = OperationCommitment(
        OPERATION_SCHEMA,
        spec.operation,
        spec.kind,
        spec.stage,
        spec.relation,
        spec.timing,
        spec.expected,
        spec.parameters,
        anchor_panel_sha256,
        donor_sha256,
        attempt_count,
        attempts_sha256,
        "",
    )
    return replace(
        value,
        operation_sha256=_canonical_sha256(
            operation_to_dict(value, include_hash=False, include_attempts=False)
        ),
    )


def operation_semantics_sha256(
    spec: OperationSpec,
    anchor_panel_sha256: str,
    donor_sha256: str | None,
) -> str:
    return _make_operation(
        spec, anchor_panel_sha256, donor_sha256, 0, "0" * 64
    ).operation_sha256


def make_anchor_operation_commitment(
    *,
    attempt_index: int,
    operation: str,
    operation_sha256: str,
    anchor_id: str,
    donor_anchor_id: str | None,
    mutation_payload: Mapping[str, object],
    resulting_program_source_sha256: str | None = None,
    resulting_query_source_sha256: str | None = None,
    resulting_packet_sha256: str | None = None,
) -> AnchorOperationCommitment:
    payload_json = json.dumps(
        mutation_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    payload_sha = hashlib.sha256(payload_json.encode("ascii")).hexdigest()
    value = AnchorOperationCommitment(
        ATTEMPT_PLAN_SCHEMA,
        attempt_index,
        f"{operation}:{anchor_id}",
        operation,
        operation_sha256,
        anchor_id,
        donor_anchor_id,
        payload_json,
        payload_sha,
        resulting_program_source_sha256,
        resulting_query_source_sha256,
        resulting_packet_sha256,
        "",
    )
    return replace(
        value,
        attempt_plan_sha256=_canonical_sha256(
            attempt_to_dict(value, include_hash=False)
        ),
    )


def make_runtime_intervention_plan(
    *,
    bindings: PlanBindings,
    anchors: Sequence[AnchorBinding],
    donor_derangements: Sequence[DonorDerangement],
    attempts: Sequence[AnchorOperationCommitment],
) -> RuntimeInterventionPlan:
    frozen_anchors = tuple(anchors)
    panel_sha = _canonical_sha256([anchor_to_dict(item) for item in frozen_anchors])
    frozen_derangements = tuple(donor_derangements)
    donor_registry_sha = _canonical_sha256(
        [derangement_to_dict(item) for item in frozen_derangements]
    )
    donor_hashes = {
        item.operation: item.derangement_sha256 for item in frozen_derangements
    }
    frozen_attempts = tuple(attempts)
    by_operation = {
        operation: tuple(
            item for item in frozen_attempts if item.operation == operation
        )
        for operation in MANDATORY_OPERATIONS
    }
    operations = tuple(
        _make_operation(
            spec,
            panel_sha,
            donor_hashes.get(spec.operation),
            len(by_operation[spec.operation]),
            _canonical_sha256(
                [attempt_to_dict(item) for item in by_operation[spec.operation]]
            ),
        )
        for spec in _SPECS
    )
    attempts_sha = _canonical_sha256(
        [attempt_to_dict(item) for item in frozen_attempts]
    )
    value = RuntimeInterventionPlan(
        PLAN_SCHEMA,
        bindings,
        frozen_anchors,
        panel_sha,
        frozen_derangements,
        donor_registry_sha,
        operations,
        frozen_attempts,
        attempts_sha,
        "",
    )
    return replace(
        value,
        plan_sha256=_canonical_sha256(plan_to_dict(value, include_hash=False)),
    )


_BINDING_KEYS = frozenset(
    bindings_to_dict(
        PlanBindings(
            *(["0" * 64] * 6),
            0,
            "0" * 64,
            "a",
            0,
            "0" * 64,
            "closure_feature",
            "0" * 64,
            "0" * 64,
            Partition.DEVELOPMENT,
            ("a",) * RUNTIME_PANEL_SIZE,
            "0" * 64,
            40_608,
            864,
            False,
        )
    )
)
_ANCHOR_KEYS = frozenset(
    anchor_to_dict(
        AnchorBinding(
            anchor_id="a",
            family_id="a",
            class_id="a",
            depth=1,
            shift_cell="hhh",
            renderer_index=0,
            query_state_cell_id="a",
            query_position=0,
            partition=Partition.DEVELOPMENT,
            program_source_sha256="0" * 64,
            query_source_sha256="0" * 64,
            packet_sha256="0" * 64,
            padding_mask_sha256="0" * 64,
            midpoint_suffix_sha256="0" * 64,
            midpoint_state_sha256="0" * 64,
            midpoint_action_sha256="0" * 64,
            action_card_sha256s=("0" * 64,) * 4,
        )
    )
)
_PAIR_KEYS = frozenset({"anchor_id", "donor_anchor_id"})
_DERANGEMENT_KEYS = frozenset(
    {"schema", "operation", "constraint", "pairs", "derangement_sha256"}
)
_EXPECTED_KEYS = frozenset(
    expected_to_dict(next(iter(OPERATION_SPECS.values())).expected)
)
_OPERATION_KEYS = frozenset(
    {
        "schema",
        "operation",
        "kind",
        "stage",
        "relation",
        "timing",
        "expected",
        "parameters",
        "anchor_panel_sha256",
        "donor_derangement_sha256",
        "attempt_count",
        "attempts_sha256",
        "operation_sha256",
    }
)
_ATTEMPT_KEYS = frozenset(
    {
        "schema",
        "attempt_index",
        "attempt_id",
        "operation",
        "operation_sha256",
        "anchor_id",
        "donor_anchor_id",
        "mutation_payload_json",
        "mutation_payload_sha256",
        "resulting_program_source_sha256",
        "resulting_query_source_sha256",
        "resulting_packet_sha256",
        "attempt_plan_sha256",
    }
)
_PLAN_KEYS = frozenset(
    {
        "schema",
        "bindings",
        "anchors",
        "anchor_panel_sha256",
        "donor_derangements",
        "donor_registry_sha256",
        "operations",
        "attempts",
        "attempts_sha256",
        "plan_sha256",
    }
)


def _exact_mapping(
    value: object, keys: frozenset[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ProtocolValidationError(f"CTAA {label} schema differs")
    return value


def _enum(enum_type: type[Enum], value: object, label: str) -> Enum:
    if not isinstance(value, str):
        raise ProtocolValidationError(f"CTAA {label} differs")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ProtocolValidationError(f"CTAA {label} is unknown") from error


def _parse_bindings(value: object) -> PlanBindings:
    row = _exact_mapping(value, _BINDING_KEYS, "plan bindings")
    return PlanBindings(
        board_manifest_sha256=row["board_manifest_sha256"],  # type: ignore[arg-type]
        board_tree_sha256=row["board_tree_sha256"],  # type: ignore[arg-type]
        compiler_sha256=row["compiler_sha256"],  # type: ignore[arg-type]
        tokenizer_sha256=row["tokenizer_sha256"],  # type: ignore[arg-type]
        base_checkpoint_sha256=row["base_checkpoint_sha256"],  # type: ignore[arg-type]
        run_contract_sha256=row["run_contract_sha256"],  # type: ignore[arg-type]
        selection_seed=row["selection_seed"],  # type: ignore[arg-type]
        selection_seed_receipt_sha256=row["selection_seed_receipt_sha256"],  # type: ignore[arg-type]
        arm_id=row["arm_id"],  # type: ignore[arg-type]
        training_seed=row["training_seed"],  # type: ignore[arg-type]
        core_sha256=row["core_sha256"],  # type: ignore[arg-type]
        core_kind=row["core_kind"],  # type: ignore[arg-type]
        base_raw_evidence_receipt_sha256=row["base_raw_evidence_receipt_sha256"],  # type: ignore[arg-type]
        runtime_implementation_sha256=row["runtime_implementation_sha256"],  # type: ignore[arg-type]
        partition=_enum(Partition, row["partition"], "partition"),  # type: ignore[arg-type]
        batch_order=tuple(row["batch_order"]),  # type: ignore[arg-type]
        batch_order_sha256=row["batch_order_sha256"],  # type: ignore[arg-type]
        scored_row_count=row["scored_row_count"],  # type: ignore[arg-type]
        runtime_panel_size=row["runtime_panel_size"],  # type: ignore[arg-type]
        runtime_attempts_affect_scored_denominator=row[
            "runtime_attempts_affect_scored_denominator"
        ],  # type: ignore[arg-type]
    )


def _parse_anchor(value: object) -> AnchorBinding:
    row = _exact_mapping(value, _ANCHOR_KEYS, "anchor")
    action_cards = row["action_card_sha256s"]
    if not isinstance(action_cards, list) or len(action_cards) != 4:
        raise ProtocolValidationError("CTAA anchor action-card catalog differs")
    return AnchorBinding(
        anchor_id=row["anchor_id"],  # type: ignore[arg-type]
        family_id=row["family_id"],  # type: ignore[arg-type]
        class_id=row["class_id"],  # type: ignore[arg-type]
        depth=row["depth"],  # type: ignore[arg-type]
        shift_cell=row["shift_cell"],  # type: ignore[arg-type]
        renderer_index=row["renderer_index"],  # type: ignore[arg-type]
        query_state_cell_id=row["query_state_cell_id"],  # type: ignore[arg-type]
        query_position=row["query_position"],  # type: ignore[arg-type]
        partition=_enum(Partition, row["partition"], "anchor partition"),  # type: ignore[arg-type]
        program_source_sha256=row["program_source_sha256"],  # type: ignore[arg-type]
        query_source_sha256=row["query_source_sha256"],  # type: ignore[arg-type]
        packet_sha256=row["packet_sha256"],  # type: ignore[arg-type]
        padding_mask_sha256=row["padding_mask_sha256"],  # type: ignore[arg-type]
        midpoint_suffix_sha256=row["midpoint_suffix_sha256"],  # type: ignore[arg-type]
        midpoint_state_sha256=row["midpoint_state_sha256"],  # type: ignore[arg-type]
        midpoint_action_sha256=row["midpoint_action_sha256"],  # type: ignore[arg-type]
        action_card_sha256s=tuple(action_cards),  # type: ignore[arg-type]
    )


def _parse_derangement(value: object) -> DonorDerangement:
    row = _exact_mapping(value, _DERANGEMENT_KEYS, "donor derangement")
    pairs = row["pairs"]
    if not isinstance(pairs, list):
        raise ProtocolValidationError("CTAA donor pair list differs")
    parsed_pairs = []
    for item in pairs:
        pair = _exact_mapping(item, _PAIR_KEYS, "donor pair")
        parsed_pairs.append(DonorPair(pair["anchor_id"], pair["donor_anchor_id"]))  # type: ignore[arg-type]
    return DonorDerangement(
        schema=row["schema"],  # type: ignore[arg-type]
        operation=row["operation"],  # type: ignore[arg-type]
        constraint=_enum(DonorConstraint, row["constraint"], "donor constraint"),  # type: ignore[arg-type]
        pairs=tuple(parsed_pairs),
        derangement_sha256=row["derangement_sha256"],  # type: ignore[arg-type]
    )


def _parse_expected(value: object) -> ExpectedSemantics:
    row = _exact_mapping(value, _EXPECTED_KEYS, "expected semantics")
    if type(row["identical_right_padding_masks"]) is not bool:
        raise ProtocolValidationError("CTAA padding-mask requirement differs")
    if not isinstance(row["comparison_scope"], str):
        raise ProtocolValidationError("CTAA comparison scope differs")
    return ExpectedSemantics(
        program_source=_enum(HashRelation, row["program_source"], "program relation"),  # type: ignore[arg-type]
        query_source=_enum(HashRelation, row["query_source"], "query relation"),  # type: ignore[arg-type]
        packet=_enum(HashRelation, row["packet"], "packet relation"),  # type: ignore[arg-type]
        outcome=_enum(OutcomeExpectation, row["outcome"], "outcome expectation"),  # type: ignore[arg-type]
        comparison_scope=row["comparison_scope"],
        identical_right_padding_masks=row["identical_right_padding_masks"],
        donor_constraint=_enum(
            DonorConstraint, row["donor_constraint"], "donor constraint"
        ),  # type: ignore[arg-type]
    )


def _parse_operation(value: object) -> OperationCommitment:
    row = _exact_mapping(value, _OPERATION_KEYS, "operation commitment")
    raw_parameters = row["parameters"]
    if not isinstance(raw_parameters, list) or any(
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(isinstance(item, str) for item in pair)
        for pair in raw_parameters
    ):
        raise ProtocolValidationError("CTAA operation parameters differ")
    donor_sha = row["donor_derangement_sha256"]
    if donor_sha is not None and not isinstance(donor_sha, str):
        raise ProtocolValidationError("CTAA operation donor commitment differs")
    return OperationCommitment(
        schema=row["schema"],  # type: ignore[arg-type]
        operation=row["operation"],  # type: ignore[arg-type]
        kind=_enum(OperationKind, row["kind"], "operation kind"),  # type: ignore[arg-type]
        stage=_enum(InterventionStage, row["stage"], "intervention stage"),  # type: ignore[arg-type]
        relation=row["relation"],  # type: ignore[arg-type]
        timing=row["timing"],  # type: ignore[arg-type]
        expected=_parse_expected(row["expected"]),
        parameters=tuple((pair[0], pair[1]) for pair in raw_parameters),
        anchor_panel_sha256=row["anchor_panel_sha256"],  # type: ignore[arg-type]
        donor_derangement_sha256=donor_sha,
        attempt_count=row["attempt_count"],  # type: ignore[arg-type]
        attempts_sha256=row["attempts_sha256"],  # type: ignore[arg-type]
        operation_sha256=row["operation_sha256"],  # type: ignore[arg-type]
    )


def _parse_attempt(value: object) -> AnchorOperationCommitment:
    row = _exact_mapping(value, _ATTEMPT_KEYS, "anchor-operation commitment")
    nullable_hashes = (
        "resulting_program_source_sha256",
        "resulting_query_source_sha256",
        "resulting_packet_sha256",
    )
    for key in nullable_hashes:
        if row[key] is not None and not isinstance(row[key], str):
            raise ProtocolValidationError("CTAA attempt result commitment differs")
    donor = row["donor_anchor_id"]
    if donor is not None and not isinstance(donor, str):
        raise ProtocolValidationError("CTAA attempt donor differs")
    return AnchorOperationCommitment(
        schema=row["schema"],  # type: ignore[arg-type]
        attempt_index=row["attempt_index"],  # type: ignore[arg-type]
        attempt_id=row["attempt_id"],  # type: ignore[arg-type]
        operation=row["operation"],  # type: ignore[arg-type]
        operation_sha256=row["operation_sha256"],  # type: ignore[arg-type]
        anchor_id=row["anchor_id"],  # type: ignore[arg-type]
        donor_anchor_id=donor,
        mutation_payload_json=row["mutation_payload_json"],  # type: ignore[arg-type]
        mutation_payload_sha256=row["mutation_payload_sha256"],  # type: ignore[arg-type]
        resulting_program_source_sha256=row["resulting_program_source_sha256"],  # type: ignore[arg-type]
        resulting_query_source_sha256=row["resulting_query_source_sha256"],  # type: ignore[arg-type]
        resulting_packet_sha256=row["resulting_packet_sha256"],  # type: ignore[arg-type]
        attempt_plan_sha256=row["attempt_plan_sha256"],  # type: ignore[arg-type]
    )


def parse_runtime_intervention_plan(
    value: Mapping[str, object],
) -> RuntimeInterventionPlan:
    row = _exact_mapping(value, _PLAN_KEYS, "runtime intervention plan")
    anchors = row["anchors"]
    donors = row["donor_derangements"]
    operations = row["operations"]
    attempts = row["attempts"]
    if (
        not isinstance(anchors, list)
        or not isinstance(donors, list)
        or not isinstance(operations, list)
        or not isinstance(attempts, list)
    ):
        raise ProtocolValidationError("CTAA runtime plan lists differ")
    return RuntimeInterventionPlan(
        schema=row["schema"],  # type: ignore[arg-type]
        bindings=_parse_bindings(row["bindings"]),
        anchors=tuple(_parse_anchor(item) for item in anchors),
        anchor_panel_sha256=row["anchor_panel_sha256"],  # type: ignore[arg-type]
        donor_derangements=tuple(_parse_derangement(item) for item in donors),
        donor_registry_sha256=row["donor_registry_sha256"],  # type: ignore[arg-type]
        operations=tuple(_parse_operation(item) for item in operations),
        attempts=tuple(_parse_attempt(item) for item in attempts),
        attempts_sha256=row["attempts_sha256"],  # type: ignore[arg-type]
        plan_sha256=row["plan_sha256"],  # type: ignore[arg-type]
    )


def _validate_hash(value: object, label: str) -> None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProtocolValidationError(f"CTAA {label} is not a canonical SHA-256")


def _validate_identifier(value: object, label: str) -> None:
    if not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None:
        raise ProtocolValidationError(f"CTAA {label} differs")


def _validate_bindings(value: PlanBindings) -> None:
    for key, item in bindings_to_dict(value).items():
        if key.endswith("sha256"):
            _validate_hash(item, key)
    _validate_identifier(value.arm_id, "arm id")
    if type(value.selection_seed) is not int or value.selection_seed < 0:
        raise ProtocolValidationError("CTAA selection seed differs")
    if type(value.training_seed) is not int or value.training_seed < 0:
        raise ProtocolValidationError("CTAA training seed differs")
    expected_kind = (
        "outer_product_control" if value.arm_id == "oprc_closure" else "closure_feature"
    )
    if value.core_kind != expected_kind:
        raise ProtocolValidationError("CTAA runtime core kind differs")
    if (
        not isinstance(value.batch_order, tuple)
        or len(value.batch_order) != RUNTIME_PANEL_SIZE
        or len(set(value.batch_order)) != RUNTIME_PANEL_SIZE
    ):
        raise ProtocolValidationError("CTAA runtime batch order differs")
    for anchor_id in value.batch_order:
        _validate_identifier(anchor_id, "batch-order anchor id")
    expected_batch_sha = _canonical_sha256(
        {
            "schema": "r12_ctaa_v2_runtime_batch_order_v1",
            "value": list(value.batch_order),
        }
    )
    if value.batch_order_sha256 != expected_batch_sha:
        raise ProtocolValidationError("CTAA runtime batch-order commitment differs")
    if value.scored_row_count != LOCKED_SCORED_ROW_COUNT:
        raise ProtocolValidationError("CTAA locked scored-row denominator differs")
    if value.runtime_panel_size != RUNTIME_PANEL_SIZE:
        raise ProtocolValidationError("CTAA runtime panel size differs")
    if value.runtime_attempts_affect_scored_denominator is not False:
        raise ProtocolValidationError(
            "CTAA runtime attempts alter scored-row denominator"
        )


def _anchor_sort_key(value: AnchorBinding) -> tuple[object, ...]:
    return (
        value.class_id,
        value.depth,
        value.renderer_index,
        value.query_state_cell_id,
        value.anchor_id,
    )


def _validate_anchors(
    anchors: tuple[AnchorBinding, ...], bindings: PlanBindings
) -> dict[str, AnchorBinding]:
    if len(anchors) != RUNTIME_PANEL_SIZE:
        raise ProtocolValidationError("CTAA runtime anchor count differs")
    if tuple(sorted(anchors, key=_anchor_sort_key)) != anchors:
        raise ProtocolValidationError("CTAA runtime anchors are not in canonical order")
    by_id: dict[str, AnchorBinding] = {}
    family_ids: set[str] = set()
    for anchor in anchors:
        _validate_identifier(anchor.anchor_id, "anchor id")
        _validate_identifier(anchor.family_id, "anchor family id")
        _validate_identifier(anchor.class_id, "anchor class id")
        _validate_identifier(anchor.query_state_cell_id, "query-state cell id")
        if anchor.anchor_id in by_id or anchor.family_id in family_ids:
            raise ProtocolValidationError("CTAA runtime anchor identity is duplicated")
        if anchor.shift_cell != "hhh":
            raise ProtocolValidationError("CTAA runtime anchor is not an hhh family")
        if anchor.partition is not bindings.partition:
            raise ProtocolValidationError("CTAA runtime anchor partition differs")
        if type(anchor.depth) is not int or anchor.depth <= 0:
            raise ProtocolValidationError("CTAA runtime anchor depth differs")
        if (
            type(anchor.renderer_index) is not int
            or not 0 <= anchor.renderer_index < RENDERER_COUNT
        ):
            raise ProtocolValidationError("CTAA runtime anchor renderer differs")
        if type(anchor.query_position) is not int or not 0 <= anchor.query_position < 3:
            raise ProtocolValidationError("CTAA runtime anchor query position differs")
        for key, item in anchor_to_dict(anchor).items():
            if key.endswith("sha256"):
                _validate_hash(item, f"anchor {key}")
        if len(anchor.action_card_sha256s) != 4:
            raise ProtocolValidationError("CTAA anchor action-card catalog differs")
        for item in anchor.action_card_sha256s:
            _validate_hash(item, "anchor action card")
        by_id[anchor.anchor_id] = anchor
        family_ids.add(anchor.family_id)
    if set(bindings.batch_order) != set(by_id):
        raise ProtocolValidationError("CTAA runtime batch order does not cover anchors")
    classes = sorted({item.class_id for item in anchors})
    depths = sorted({item.depth for item in anchors})
    cells = sorted({item.query_state_cell_id for item in anchors})
    if len(classes) != 3 or len(depths) != 2:
        raise ProtocolValidationError("CTAA runtime class-depth strata differ")
    if len(cells) != QUERY_STATE_CELL_COUNT:
        raise ProtocolValidationError("CTAA runtime query-state cells differ")
    for class_id in classes:
        for depth in depths:
            stratum = [
                item
                for item in anchors
                if item.class_id == class_id and item.depth == depth
            ]
            if len(stratum) != ANCHORS_PER_CLASS_DEPTH:
                raise ProtocolValidationError(
                    "CTAA runtime class-depth balance differs"
                )
            renderer_counts = {
                index: sum(item.renderer_index == index for item in stratum)
                for index in range(RENDERER_COUNT)
            }
            if set(renderer_counts.values()) != {ANCHORS_PER_RENDERER}:
                raise ProtocolValidationError("CTAA runtime renderer balance differs")
            cell_counts = {
                cell: sum(item.query_state_cell_id == cell for item in stratum)
                for cell in cells
            }
            if set(cell_counts.values()) != {ANCHORS_PER_QUERY_STATE_CELL}:
                raise ProtocolValidationError(
                    "CTAA runtime query-state balance differs"
                )
            position_counts = {
                position: sum(item.query_position == position for item in stratum)
                for position in range(3)
            }
            if set(position_counts.values()) != {ANCHORS_PER_QUERY_POSITION}:
                raise ProtocolValidationError(
                    "CTAA runtime query-position balance differs"
                )
    return by_id


def _validate_donor_pair(
    constraint: DonorConstraint, parent: AnchorBinding, donor: AnchorBinding
) -> None:
    if parent.anchor_id == donor.anchor_id:
        raise ProtocolValidationError("CTAA donor derangement contains a fixed point")
    midpoint_constraint = constraint in {
        DonorConstraint.MIDPOINT_STATE_DIFFERENT_MATCHED_SUFFIX,
        DonorConstraint.MIDPOINT_ACTION_DIFFERENT_MATCHED_SUFFIX,
    }
    if parent.depth != donor.depth or (
        not midpoint_constraint and parent.class_id != donor.class_id
    ):
        raise ProtocolValidationError("CTAA donor crosses a class-depth stratum")
    if constraint is DonorConstraint.RESIDUAL_EXACT_MASK:
        if parent.padding_mask_sha256 != donor.padding_mask_sha256:
            raise ProtocolValidationError("CTAA residual donor padding mask differs")
        if parent.packet_sha256 == donor.packet_sha256:
            raise ProtocolValidationError("CTAA residual donor packet aliases parent")
    elif constraint is DonorConstraint.LATE_QUERY_DIFFERENT:
        if (
            parent.query_source_sha256 == donor.query_source_sha256
            or parent.query_position == donor.query_position
        ):
            raise ProtocolValidationError("CTAA late-query donor does not change query")
    elif constraint in {
        DonorConstraint.MIDPOINT_STATE_DIFFERENT_MATCHED_SUFFIX,
        DonorConstraint.MIDPOINT_ACTION_DIFFERENT_MATCHED_SUFFIX,
    }:
        if parent.midpoint_suffix_sha256 != donor.midpoint_suffix_sha256:
            raise ProtocolValidationError("CTAA midpoint donor suffix differs")
        differs = (
            parent.midpoint_state_sha256 != donor.midpoint_state_sha256
            if constraint is DonorConstraint.MIDPOINT_STATE_DIFFERENT_MATCHED_SUFFIX
            else any(
                item != parent.midpoint_action_sha256
                for item in donor.action_card_sha256s
            )
        )
        if not differs:
            raise ProtocolValidationError(
                "CTAA midpoint donor aliases the injected register"
            )
    elif constraint is DonorConstraint.PACKET_DIFFERENT:
        if parent.packet_sha256 == donor.packet_sha256:
            raise ProtocolValidationError("CTAA packet donor aliases parent")
    else:  # pragma: no cover - called only for donor-requiring specs
        raise ProtocolValidationError("CTAA donor constraint differs")


def _validate_derangements(
    values: tuple[DonorDerangement, ...], anchors: dict[str, AnchorBinding]
) -> dict[str, DonorDerangement]:
    required = [
        spec.operation
        for spec in _SPECS
        if spec.expected.donor_constraint is not DonorConstraint.NONE
    ]
    if [item.operation for item in values] != required:
        raise ProtocolValidationError("CTAA frozen donor registry differs")
    by_operation: dict[str, DonorDerangement] = {}
    canonical_anchor_ids = sorted(anchors)
    for value in values:
        if value.schema != DERANGEMENT_SCHEMA:
            raise ProtocolValidationError("CTAA donor derangement schema differs")
        spec = OPERATION_SPECS[value.operation]
        if value.constraint is not spec.expected.donor_constraint:
            raise ProtocolValidationError("CTAA donor derangement constraint differs")
        if [pair.anchor_id for pair in value.pairs] != canonical_anchor_ids:
            raise ProtocolValidationError(
                "CTAA donor derangement parent coverage differs"
            )
        if any(pair.anchor_id == pair.donor_anchor_id for pair in value.pairs):
            raise ProtocolValidationError(
                "CTAA donor derangement contains a fixed point"
            )
        donor_ids = [pair.donor_anchor_id for pair in value.pairs]
        if sorted(donor_ids) != canonical_anchor_ids:
            raise ProtocolValidationError("CTAA donor derangement is not one-to-one")
        for pair in value.pairs:
            _validate_donor_pair(
                value.constraint,
                anchors[pair.anchor_id],
                anchors[pair.donor_anchor_id],
            )
        _validate_hash(value.derangement_sha256, "donor derangement")
        expected_sha = _canonical_sha256(derangement_to_dict(value, include_hash=False))
        if value.derangement_sha256 != expected_sha:
            raise ProtocolValidationError("CTAA donor derangement commitment differs")
        by_operation[value.operation] = value
    return by_operation


def _validate_operations(
    values: tuple[OperationCommitment, ...],
    panel_sha: str,
    donors: Mapping[str, DonorDerangement],
    attempts: Mapping[str, tuple[AnchorOperationCommitment, ...]],
) -> None:
    if [item.operation for item in values] != list(MANDATORY_OPERATIONS):
        raise ProtocolValidationError("CTAA mandatory operation order/set differs")
    for value in values:
        if value.schema != OPERATION_SCHEMA:
            raise ProtocolValidationError("CTAA operation schema differs")
        spec = OPERATION_SPECS[value.operation]
        expected_donor_sha = (
            donors[value.operation].derangement_sha256
            if spec.expected.donor_constraint is not DonorConstraint.NONE
            else None
        )
        if (
            value.kind is not spec.kind
            or value.stage is not spec.stage
            or value.relation != spec.relation
            or value.timing != spec.timing
            or value.expected != spec.expected
            or value.parameters != spec.parameters
        ):
            raise ProtocolValidationError("CTAA operation semantic contract differs")
        if value.anchor_panel_sha256 != panel_sha:
            raise ProtocolValidationError("CTAA operation anchor-panel binding differs")
        if value.donor_derangement_sha256 != expected_donor_sha:
            raise ProtocolValidationError("CTAA operation donor binding differs")
        expected_attempts = attempts[value.operation]
        expected_attempts_sha = _canonical_sha256(
            [attempt_to_dict(item) for item in expected_attempts]
        )
        if (
            value.attempt_count != RUNTIME_PANEL_SIZE
            or value.attempt_count != len(expected_attempts)
            or value.attempts_sha256 != expected_attempts_sha
        ):
            raise ProtocolValidationError("CTAA operation attempt registry differs")
        _validate_hash(value.operation_sha256, "operation commitment")
        expected_sha = _canonical_sha256(
            operation_to_dict(value, include_hash=False, include_attempts=False)
        )
        if value.operation_sha256 != expected_sha:
            raise ProtocolValidationError("CTAA operation commitment differs")


def _validate_attempts(
    values: tuple[AnchorOperationCommitment, ...],
    bindings: PlanBindings,
    operations: tuple[OperationCommitment, ...],
    donors: Mapping[str, DonorDerangement],
) -> dict[str, tuple[AnchorOperationCommitment, ...]]:
    expected_count = len(MANDATORY_OPERATIONS) * RUNTIME_PANEL_SIZE
    if len(values) != expected_count:
        raise ProtocolValidationError("CTAA runtime attempt count differs")
    operation_by_name = {item.operation: item for item in operations}
    donor_maps = {
        operation: {pair.anchor_id: pair.donor_anchor_id for pair in item.pairs}
        for operation, item in donors.items()
    }
    expected_pairs = [
        (operation, anchor_id)
        for operation in MANDATORY_OPERATIONS
        for anchor_id in bindings.batch_order
    ]
    actual_pairs = [(item.operation, item.anchor_id) for item in values]
    if actual_pairs != expected_pairs or len(set(actual_pairs)) != expected_count:
        raise ProtocolValidationError("CTAA runtime attempt coverage/order differs")
    result: dict[str, list[AnchorOperationCommitment]] = {
        operation: [] for operation in MANDATORY_OPERATIONS
    }
    for expected_index, value in enumerate(values):
        if value.schema != ATTEMPT_PLAN_SCHEMA or value.attempt_index != expected_index:
            raise ProtocolValidationError("CTAA runtime attempt identity differs")
        _validate_identifier(value.attempt_id, "runtime attempt id")
        if value.operation not in operation_by_name:
            raise ProtocolValidationError("CTAA runtime attempt operation differs")
        operation = operation_by_name[value.operation]
        if value.attempt_id != f"{value.operation}:{value.anchor_id}":
            raise ProtocolValidationError("CTAA runtime attempt id differs")
        if value.operation_sha256 != operation.operation_sha256:
            raise ProtocolValidationError(
                "CTAA runtime attempt operation binding differs"
            )
        expected_donor = donor_maps.get(value.operation, {}).get(value.anchor_id)
        if value.donor_anchor_id != expected_donor:
            raise ProtocolValidationError("CTAA runtime attempt donor binding differs")
        if not isinstance(value.mutation_payload_json, str):
            raise ProtocolValidationError("CTAA runtime attempt payload differs")
        try:
            payload = json.loads(
                value.mutation_payload_json,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ProtocolValidationError(
                "CTAA runtime attempt payload differs"
            ) from error
        if (
            not isinstance(payload, dict)
            or json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
            != value.mutation_payload_json
        ):
            raise ProtocolValidationError(
                "CTAA runtime attempt payload is not canonical"
            )
        expected_payload_sha = hashlib.sha256(
            value.mutation_payload_json.encode("ascii")
        ).hexdigest()
        if value.mutation_payload_sha256 != expected_payload_sha:
            raise ProtocolValidationError(
                "CTAA runtime attempt payload commitment differs"
            )
        for item in (
            value.operation_sha256,
            value.mutation_payload_sha256,
            value.resulting_program_source_sha256,
            value.resulting_query_source_sha256,
            value.resulting_packet_sha256,
            value.attempt_plan_sha256,
        ):
            if item is not None:
                _validate_hash(item, "runtime attempt")
        expected_attempt_sha = _canonical_sha256(
            attempt_to_dict(value, include_hash=False)
        )
        if value.attempt_plan_sha256 != expected_attempt_sha:
            raise ProtocolValidationError("CTAA runtime attempt commitment differs")
        result[value.operation].append(value)
    return {key: tuple(items) for key, items in result.items()}


def validate_runtime_intervention_plan(
    value: RuntimeInterventionPlan | Mapping[str, object],
) -> RuntimeInterventionPlan:
    """Validate and return a frozen runtime sidecar plan, or fail closed."""

    plan = (
        parse_runtime_intervention_plan(value) if isinstance(value, Mapping) else value
    )
    if not isinstance(plan, RuntimeInterventionPlan) or plan.schema != PLAN_SCHEMA:
        raise ProtocolValidationError("CTAA runtime intervention plan schema differs")
    _validate_bindings(plan.bindings)
    anchors = _validate_anchors(plan.anchors, plan.bindings)
    _validate_hash(plan.anchor_panel_sha256, "anchor panel")
    expected_panel_sha = _canonical_sha256(
        [anchor_to_dict(item) for item in plan.anchors]
    )
    if plan.anchor_panel_sha256 != expected_panel_sha:
        raise ProtocolValidationError("CTAA anchor-panel commitment differs")
    donors = _validate_derangements(plan.donor_derangements, anchors)
    _validate_hash(plan.donor_registry_sha256, "donor registry")
    expected_registry_sha = _canonical_sha256(
        [derangement_to_dict(item) for item in plan.donor_derangements]
    )
    if plan.donor_registry_sha256 != expected_registry_sha:
        raise ProtocolValidationError("CTAA donor-registry commitment differs")
    if [item.operation for item in plan.operations] != list(MANDATORY_OPERATIONS):
        raise ProtocolValidationError("CTAA mandatory operation order/set differs")
    attempts = _validate_attempts(plan.attempts, plan.bindings, plan.operations, donors)
    _validate_operations(plan.operations, plan.anchor_panel_sha256, donors, attempts)
    _validate_hash(plan.attempts_sha256, "runtime attempt registry")
    expected_attempts_sha = _canonical_sha256(
        [attempt_to_dict(item) for item in plan.attempts]
    )
    if plan.attempts_sha256 != expected_attempts_sha:
        raise ProtocolValidationError(
            "CTAA runtime attempt-registry commitment differs"
        )
    _validate_hash(plan.plan_sha256, "runtime intervention plan")
    expected_plan_sha = _canonical_sha256(plan_to_dict(plan, include_hash=False))
    if plan.plan_sha256 != expected_plan_sha:
        raise ProtocolValidationError(
            "CTAA runtime intervention plan commitment differs"
        )
    return plan
