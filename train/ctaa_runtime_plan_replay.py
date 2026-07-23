"""Deterministic, model-free replay of frozen CTAA runtime-plan recipes.

This module is deliberately narrower than the runtime executor.  It replays
the concrete source, query, and packet recipes committed by the plan builder,
and validates neural/runtime descriptors without evaluating a model.  It
accepts only already parsed CTAA sources so source interpretation remains
owned by the frozen typed parser in ``build_ctaa_runtime_intervention_plan``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch
from tokenizers import Tokenizer

from build_ctaa_runtime_intervention_plan import (
    MASK_SCHEMA,
    SEQUENCE_LIMIT,
    ParsedSource,
    _read_jsonl,
    _read_locked_bytes,
    _parse_program,
    _hash_order,
    _opaque_names,
    _packet_bytes,
    _permutation,
    _render_program,
    _tagged_sha256,
    _three_cycle,
)
from ctaa_intervention_protocol import (
    MANDATORY_OPERATIONS,
    OPERATION_SPECS,
    AnchorBinding,
    AnchorOperationCommitment,
    GateFamily,
    InterventionFamily,
    RuntimeInterventionPlan,
    validate_runtime_intervention_plan,
)
from ctaa_neural_core import CTAA_ACTION_COUNT, CTAA_MAX_STEPS, CTAA_WIDTH
from ctaa_packet_io import packet_body
from ctaa_runtime_interventions import (
    binding_only_counterfactual,
    card_only_counterfactual,
    card_storage_reindex,
    compensated_opcode_relabel,
    future_schedule_counterfactual,
    packet_transplant,
    post_stop_poison,
)
from ctaa_trunk_compiler import HardCTAAPacket, HardCTAAQuery
from pipeline.generate_ctaa_board import RENDERERS


PAYLOAD_SCHEMA = "r12_ctaa_v2_concrete_mutation_v2"


class ReplayValidationError(ValueError):
    """A concrete runtime recipe cannot be reproduced exactly."""


@dataclass(frozen=True)
class ReplayResult:
    attempt_index: int
    attempt_id: str
    operation: str
    anchor_id: str
    donor_anchor_id: str | None
    resulting_program_source_sha256: str | None
    resulting_query_source_sha256: str | None
    resulting_packet_sha256: str | None


@dataclass(frozen=True)
class RuntimePlanReplay:
    plan_sha256: str
    attempt_count: int
    results: tuple[ReplayResult, ...]


@dataclass(frozen=True)
class _Context:
    plan: RuntimeInterventionPlan
    anchors: Mapping[str, AnchorBinding]
    rows: Mapping[str, ParsedSource]
    donors: Mapping[str, Mapping[str, str]]


_BASE_FIELDS = frozenset({"schema", "operation", "anchor_id", "timing"})
_EXTRA_FIELDS: Mapping[str, frozenset[str]] = {
    InterventionFamily.H19_ZERO.value: frozenset(
        {
            "residual_layer",
            "token_start",
            "token_stop",
            "channel_start",
            "channel_stop",
            "padding_mask_sha256",
            "donor_anchor_id",
        }
    ),
    InterventionFamily.H19_BATCH_ROTATE.value: frozenset(
        {
            "residual_layer",
            "token_start",
            "token_stop",
            "channel_start",
            "channel_stop",
            "padding_mask_sha256",
            "donor_anchor_id",
        }
    ),
    InterventionFamily.H19_DONOR_TRANSPLANT.value: frozenset(
        {
            "residual_layer",
            "token_start",
            "token_stop",
            "channel_start",
            "channel_stop",
            "padding_mask_sha256",
            "donor_anchor_id",
        }
    ),
    InterventionFamily.H29_ZERO.value: frozenset(
        {
            "residual_layer",
            "token_start",
            "token_stop",
            "channel_start",
            "channel_stop",
            "padding_mask_sha256",
            "donor_anchor_id",
        }
    ),
    InterventionFamily.H29_BATCH_ROTATE.value: frozenset(
        {
            "residual_layer",
            "token_start",
            "token_stop",
            "channel_start",
            "channel_stop",
            "padding_mask_sha256",
            "donor_anchor_id",
        }
    ),
    InterventionFamily.H29_DONOR_TRANSPLANT.value: frozenset(
        {
            "residual_layer",
            "token_start",
            "token_stop",
            "channel_start",
            "channel_stop",
            "padding_mask_sha256",
            "donor_anchor_id",
        }
    ),
    InterventionFamily.ENTITY_RECODE.value: frozenset({"old_to_new"}),
    InterventionFamily.WITNESS_RECODE.value: frozenset({"old_to_new"}),
    InterventionFamily.OPCODE_RECODE.value: frozenset({"old_to_new"}),
    InterventionFamily.RENDERER_SUBSTITUTION.value: frozenset(
        {"parent_renderer", "target_renderer"}
    ),
    InterventionFamily.RULE_LINE_SHUFFLE.value: frozenset({"rule_order"}),
    InterventionFamily.CARD_ONLY_COUNTERFACTUAL.value: frozenset(
        {"card_address", "coordinate", "before", "after"}
    ),
    InterventionFamily.BINDING_ONLY_COUNTERFACTUAL.value: frozenset(
        {"old_to_new_opcode", "new_to_old_opcode"}
    ),
    InterventionFamily.COMPENSATED_OPCODE_RELABEL.value: frozenset(
        {"old_to_new_opcode"}
    ),
    InterventionFamily.CARD_STORAGE_REINDEX.value: frozenset(
        {"storage_order", "inverse"}
    ),
    InterventionFamily.WITNESS_CORRUPTION.value: frozenset(
        {"slot", "position", "before", "after"}
    ),
    InterventionFamily.PAIRED_SHUFFLED_LAW.value: frozenset({"law_order"}),
    InterventionFamily.SCHEDULE_ORDER_TWIN.value: frozenset({"swapped_active_slots"}),
    InterventionFamily.SOURCE_POISON.value: frozenset(
        {
            "replacement_offset",
            "replacement_length",
            "poison_bytes_hex",
            "poison_bytes_sha256",
        }
    ),
    InterventionFamily.FUTURE_MASK.value: frozenset(
        {"first_exposure_step", "changed_slots"}
    ),
    InterventionFamily.STOP_RELOCATION.value: frozenset(
        {"old_stop_index", "new_stop_index", "displaced_event"}
    ),
    InterventionFamily.LATE_QUERY_SWAP.value: frozenset(
        {"parent_query_position", "donor_query_position", "execution_policy"}
    ),
    InterventionFamily.POST_STOP_POISON.value: frozenset(
        {"stop_index", "changed_slots", "replacement_rule"}
    ),
    InterventionFamily.MIDPOINT_DONOR_STATE.value: frozenset(
        {"midpoint_step", "donor_state_sha256"}
    ),
    InterventionFamily.MIDPOINT_DONOR_ACTION.value: frozenset(
        {"midpoint_step", "donor_card_slot", "donor_action_sha256"}
    ),
    InterventionFamily.PACKET_TRANSPLANT.value: frozenset(
        {"literal_donor_packet_sha256"}
    ),
    GateFamily.SOURCE_DELETION.value: frozenset(
        {"probe_stage", "probe_targets", "required_result", "allowed_errno"}
    ),
    GateFamily.QUERY_ISOLATION.value: frozenset(
        {"probe_stage", "probe_target", "required_result", "disclosure_after"}
    ),
    GateFamily.ROUTE_AGREEMENT.value: frozenset(
        {"positions", "comparison", "required_tensor_shape"}
    ),
}
if tuple(_EXTRA_FIELDS) != MANDATORY_OPERATIONS:  # pragma: no cover
    raise RuntimeError("CTAA replay operation schema order differs")


_PROGRAM_RESULTS = frozenset(
    {
        InterventionFamily.ENTITY_RECODE.value,
        InterventionFamily.WITNESS_RECODE.value,
        InterventionFamily.OPCODE_RECODE.value,
        InterventionFamily.RENDERER_SUBSTITUTION.value,
        InterventionFamily.RULE_LINE_SHUFFLE.value,
        InterventionFamily.WITNESS_CORRUPTION.value,
        InterventionFamily.PAIRED_SHUFFLED_LAW.value,
        InterventionFamily.SCHEDULE_ORDER_TWIN.value,
        InterventionFamily.SOURCE_POISON.value,
        InterventionFamily.STOP_RELOCATION.value,
    }
)
_QUERY_RESULTS = frozenset({InterventionFamily.LATE_QUERY_SWAP.value})
_PACKET_RESULTS = frozenset(
    {
        InterventionFamily.RULE_LINE_SHUFFLE.value,
        InterventionFamily.CARD_ONLY_COUNTERFACTUAL.value,
        InterventionFamily.BINDING_ONLY_COUNTERFACTUAL.value,
        InterventionFamily.COMPENSATED_OPCODE_RELABEL.value,
        InterventionFamily.CARD_STORAGE_REINDEX.value,
        InterventionFamily.WITNESS_CORRUPTION.value,
        InterventionFamily.PAIRED_SHUFFLED_LAW.value,
        InterventionFamily.SCHEDULE_ORDER_TWIN.value,
        InterventionFamily.FUTURE_MASK.value,
        InterventionFamily.STOP_RELOCATION.value,
        InterventionFamily.POST_STOP_POISON.value,
        InterventionFamily.PACKET_TRANSPLANT.value,
    }
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayValidationError("CTAA replay payload contains duplicate keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ReplayValidationError(f"CTAA replay payload contains non-finite {value}")


def _payload(attempt: AnchorOperationCommitment) -> dict[str, object]:
    try:
        raw = attempt.mutation_payload_json.encode("ascii")
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeEncodeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayValidationError("CTAA replay payload JSON differs") from error
    if not isinstance(value, dict):
        raise ReplayValidationError("CTAA replay payload root differs")
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    if canonical != raw or _sha256(raw) != attempt.mutation_payload_sha256:
        raise ReplayValidationError("CTAA replay payload commitment differs")
    fields = _EXTRA_FIELDS.get(attempt.operation)
    if fields is None or set(value) != _BASE_FIELDS | fields:
        raise ReplayValidationError("CTAA replay payload schema differs")
    spec = OPERATION_SPECS[attempt.operation]
    expected_base = {
        "schema": PAYLOAD_SCHEMA,
        "operation": attempt.operation,
        "anchor_id": attempt.anchor_id,
        "timing": spec.timing,
    }
    if any(value[key] != expected for key, expected in expected_base.items()):
        raise ReplayValidationError("CTAA replay payload identity differs")
    return value


def _exact_int(value: object, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise ReplayValidationError(f"CTAA replay {label} differs")
    return value


def _exact(value: object, expected: object, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ReplayValidationError(f"CTAA replay {label} differs")


def _integer_list(value: object, expected: list[int], label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or any(type(item) is not int for item in value)
        or value != expected
    ):
        raise ReplayValidationError(f"CTAA replay {label} differs")
    return value


def _mapping(value: object, expected: Mapping[str, str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ReplayValidationError(f"CTAA replay {label} schema differs")
    if any(
        type(key) is not str or type(item) is not str for key, item in value.items()
    ):
        raise ReplayValidationError(f"CTAA replay {label} values differ")
    if value != dict(expected):
        raise ReplayValidationError(f"CTAA replay {label} differs")


def _hard_packet(row: ParsedSource) -> HardCTAAPacket:
    packet = HardCTAAPacket(
        action_cards=torch.tensor([row.cards], dtype=torch.uint8),
        initial_state=torch.tensor([row.initial_state], dtype=torch.uint8),
        opcode_schedule=torch.tensor([row.opcode_schedule], dtype=torch.uint8),
        opcode_to_card=torch.tensor([row.opcode_to_card], dtype=torch.uint8),
    )
    if packet_body(packet) != row.packet_bytes:
        raise ReplayValidationError("CTAA replay typed parent packet differs")
    return packet


def _packet_digest(
    cards: tuple[tuple[int, int, int], ...],
    initial: tuple[int, int, int],
    schedule: tuple[int, ...],
    opcode_to_card: tuple[int, int, int, int],
) -> tuple[str, bytes]:
    payload = _packet_bytes(cards, initial, schedule, opcode_to_card)
    return _sha256(payload), payload


def _validate_anchor_row(anchor: AnchorBinding, row: ParsedSource) -> None:
    packet = _hard_packet(row)
    del packet
    expected = {
        "family_id": row.family_id,
        "class_id": row.class_id,
        "depth": row.depth,
        "renderer_index": row.renderer_index,
        "query_state_cell_id": row.query_state_cell_id,
        "query_position": row.query_position,
        "program_source_sha256": _sha256(row.program_source.encode("utf-8")),
        "query_source_sha256": _sha256(row.query_source.encode("utf-8")),
        "packet_sha256": _sha256(row.packet_bytes),
        "padding_mask_sha256": _tagged_sha256(
            MASK_SCHEMA,
            [1] * row.token_count + [0] * (SEQUENCE_LIMIT - row.token_count),
        ),
        "midpoint_suffix_sha256": _sha256(row.midpoint_suffix_bytes),
        "midpoint_state_sha256": _sha256(row.midpoint_state_bytes),
        "midpoint_action_sha256": _sha256(row.midpoint_action_bytes),
        "action_card_sha256s": tuple(_sha256(bytes(card)) for card in row.cards),
    }
    for field, value in expected.items():
        if getattr(anchor, field) != value:
            raise ReplayValidationError(f"CTAA replay anchor {field} differs")


def _context(
    plan: RuntimeInterventionPlan | Mapping[str, object],
    rows_by_anchor: Mapping[str, ParsedSource],
) -> _Context:
    try:
        frozen = validate_runtime_intervention_plan(plan)
    except (TypeError, ValueError) as error:
        raise ReplayValidationError("CTAA replay plan validation failed") from error
    anchors = {anchor.anchor_id: anchor for anchor in frozen.anchors}
    if set(rows_by_anchor) != set(anchors):
        raise ReplayValidationError("CTAA replay source registry differs")
    rows: dict[str, ParsedSource] = {}
    for anchor_id, anchor in anchors.items():
        row = rows_by_anchor[anchor_id]
        if not isinstance(row, ParsedSource):
            raise ReplayValidationError("CTAA replay source type differs")
        _validate_anchor_row(anchor, row)
        rows[anchor_id] = row
    donors = {
        item.operation: {pair.anchor_id: pair.donor_anchor_id for pair in item.pairs}
        for item in frozen.donor_derangements
    }
    return _Context(frozen, anchors, rows, donors)


def _donor(ctx: _Context, attempt: AnchorOperationCommitment) -> ParsedSource | None:
    expected_id = ctx.donors.get(attempt.operation, {}).get(attempt.anchor_id)
    if attempt.donor_anchor_id != expected_id:
        raise ReplayValidationError("CTAA replay donor binding differs")
    if expected_id is None:
        return None
    if expected_id == attempt.anchor_id or expected_id not in ctx.rows:
        raise ReplayValidationError("CTAA replay donor identity differs")
    return ctx.rows[expected_id]


def _changed(value: bytes, parent: bytes, label: str) -> bytes:
    if value == parent:
        raise ReplayValidationError(f"CTAA replay {label} is a no-op")
    return value


def _residual_descriptor(
    operation: str,
    payload: Mapping[str, object],
    attempt: AnchorOperationCommitment,
    anchor: AnchorBinding,
    row: ParsedSource,
    donor: ParsedSource | None,
) -> None:
    layer = 19 if operation.startswith("h19_") else 29
    _exact_int(payload["residual_layer"], layer, "residual layer")
    _exact_int(payload["token_start"], 0, "residual token start")
    _exact_int(payload["token_stop"], row.token_count, "residual token stop")
    _exact_int(payload["channel_start"], 0, "residual channel start")
    _exact(payload["channel_stop"], "model_width", "residual channel stop")
    _exact(payload["padding_mask_sha256"], anchor.padding_mask_sha256, "padding mask")
    _exact(payload["donor_anchor_id"], attempt.donor_anchor_id, "residual donor")
    is_zero = operation.endswith("_zero")
    if is_zero != (donor is None):
        raise ReplayValidationError("CTAA replay residual donor requirement differs")
    if donor is not None:
        if _sha256(donor.packet_bytes) == anchor.packet_sha256:
            raise ReplayValidationError(
                "CTAA replay residual donor packet is unchanged"
            )
        if donor.token_count != row.token_count:
            raise ReplayValidationError("CTAA replay residual donor mask differs")


def _replay_one(ctx: _Context, attempt: AnchorOperationCommitment) -> ReplayResult:
    payload = _payload(attempt)
    anchor = ctx.anchors.get(attempt.anchor_id)
    row = ctx.rows.get(attempt.anchor_id)
    if anchor is None or row is None:
        raise ReplayValidationError("CTAA replay parent source differs")
    donor = _donor(ctx, attempt)
    operation = attempt.operation
    seed = ctx.plan.bindings.selection_seed
    partition = ctx.plan.bindings.partition.value
    program_bytes: bytes | None = None
    query_bytes: bytes | None = None
    packet_bytes: bytes | None = None

    if operation.startswith("h19_") or operation.startswith("h29_"):
        _residual_descriptor(operation, payload, attempt, anchor, row, donor)
    elif operation == InterventionFamily.ENTITY_RECODE.value:
        names = _opaque_names(seed, operation, attempt.anchor_id, "E", 3)
        _mapping(
            payload["old_to_new"],
            dict(zip(row.symbols, names, strict=True)),
            "entity recode",
        )
        program_bytes = _render_program(row, symbols=names).encode("utf-8")
    elif operation == InterventionFamily.WITNESS_RECODE.value:
        names = _opaque_names(seed, operation, attempt.anchor_id, "WQ", 4)
        expected = dict(zip(("W1", "W2", "W3", "W4"), names, strict=True))
        _mapping(payload["old_to_new"], expected, "witness recode")
        program_bytes = _render_program(row, addresses=names).encode("utf-8")
    elif operation == InterventionFamily.OPCODE_RECODE.value:
        names = _opaque_names(seed, operation, attempt.anchor_id, "OPQ", 4)
        _mapping(
            payload["old_to_new"],
            dict(zip(row.opcodes, names, strict=True)),
            "opcode recode",
        )
        program_bytes = _render_program(row, opcodes=names).encode("utf-8")
    elif operation == InterventionFamily.RENDERER_SUBSTITUTION.value:
        candidates = tuple(
            value
            for value in RENDERERS[partition]
            if value != row.renderer_value
            and ((value >> 5) & 1) == ((row.renderer_value >> 5) & 1)
        )
        target = candidates[
            int.from_bytes(_hash_order(seed, operation, attempt.anchor_id)[:8], "big")
            % len(candidates)
        ]
        _exact_int(payload["parent_renderer"], row.renderer_value, "parent renderer")
        _exact_int(payload["target_renderer"], target, "target renderer")
        program_bytes = _render_program(row, renderer_value=target).encode("utf-8")
    elif operation == InterventionFamily.RULE_LINE_SHUFFLE.value:
        order = _permutation(seed, operation, attempt.anchor_id, CTAA_ACTION_COUNT)
        if order == row.rule_order:
            order = order[1:] + order[:1]
        _integer_list(payload["rule_order"], list(order), "rule-line order")
        if order == row.rule_order:
            raise ReplayValidationError("CTAA replay rule-line shuffle is a no-op")
        program_bytes = _render_program(row, rule_order=order).encode("utf-8")
        _, packet_bytes = _packet_digest(
            row.cards,
            row.initial_state,
            row.schedule,
            order,  # type: ignore[arg-type]
        )
    elif operation == InterventionFamily.CARD_ONLY_COUNTERFACTUAL.value:
        card_address = row.schedule[0]
        coordinate = int.from_bytes(
            _hash_order(seed, operation, attempt.anchor_id, "coordinate")[:8],
            "big",
        ) % CTAA_WIDTH
        _exact_int(payload["card_address"], card_address, "card-only address")
        _exact_int(payload["coordinate"], coordinate, "card-only coordinate")
        parent_packet = _hard_packet(row)
        before = int(parent_packet.action_cards[0, card_address, coordinate].item())
        _exact_int(payload["before"], before, "card-only before")
        _exact_int(payload["after"], (before + 1) % CTAA_WIDTH, "card-only after")
        mutated = card_only_counterfactual(
            parent_packet,
            torch.tensor([card_address], dtype=torch.long),
            torch.tensor([coordinate], dtype=torch.long),
        ).packet
        packet_bytes = packet_body(mutated)
    elif operation == InterventionFamily.BINDING_ONLY_COUNTERFACTUAL.value:
        old_to_new = _three_cycle(
            seed,
            operation,
            attempt.anchor_id,
            row.opcode_schedule[0],
        )
        new_to_old = [0] * CTAA_ACTION_COUNT
        for old_slot, new_slot in enumerate(old_to_new):
            new_to_old[new_slot] = old_slot
        _integer_list(
            payload["old_to_new_opcode"],
            list(old_to_new),
            "binding-only old-to-new opcode",
        )
        _integer_list(
            payload["new_to_old_opcode"],
            new_to_old,
            "binding-only new-to-old opcode",
        )
        mutated = binding_only_counterfactual(
            _hard_packet(row),
            torch.tensor([new_to_old], dtype=torch.long),
        ).packet
        packet_bytes = packet_body(mutated)
    elif operation == InterventionFamily.COMPENSATED_OPCODE_RELABEL.value:
        old_to_new = _three_cycle(seed, operation, attempt.anchor_id)
        _integer_list(
            payload["old_to_new_opcode"],
            list(old_to_new),
            "compensated old-to-new opcode",
        )
        mutated = compensated_opcode_relabel(
            _hard_packet(row),
            torch.tensor([old_to_new], dtype=torch.long),
        ).packet
        packet_bytes = packet_body(mutated)
    elif operation == InterventionFamily.CARD_STORAGE_REINDEX.value:
        order = _permutation(seed, operation, attempt.anchor_id, CTAA_ACTION_COUNT)
        inverse = [0] * CTAA_ACTION_COUNT
        for new_slot, old_slot in enumerate(order):
            inverse[old_slot] = new_slot
        _integer_list(payload["storage_order"], list(order), "card-storage order")
        _integer_list(payload["inverse"], inverse, "card-storage inverse")
        mutated = card_storage_reindex(
            _hard_packet(row), torch.tensor([order], dtype=torch.long)
        ).packet
        packet_bytes = packet_body(mutated)
    elif operation == InterventionFamily.WITNESS_CORRUPTION.value:
        slot = (
            int.from_bytes(
                _hash_order(seed, operation, attempt.anchor_id, "slot")[:8], "big"
            )
            % CTAA_ACTION_COUNT
        )
        position = (
            int.from_bytes(
                _hash_order(seed, operation, attempt.anchor_id, "position")[:8], "big"
            )
            % CTAA_WIDTH
        )
        cards_list = [list(card) for card in row.cards]
        before = cards_list[slot][position]
        after = (before + 1) % CTAA_WIDTH
        cards_list[slot][position] = after
        _exact_int(payload["slot"], slot, "witness-corruption slot")
        _exact_int(payload["position"], position, "witness-corruption position")
        _exact_int(payload["before"], before, "witness-corruption before")
        _exact_int(payload["after"], after, "witness-corruption after")
        cards = tuple(tuple(card) for card in cards_list)
        program_bytes = _render_program(row, cards=cards).encode("utf-8")
        _, packet_bytes = _packet_digest(
            cards,
            row.initial_state,
            row.schedule,
            row.opcode_to_card,
        )
    elif operation == InterventionFamily.PAIRED_SHUFFLED_LAW.value:
        order = _permutation(seed, operation, attempt.anchor_id, CTAA_ACTION_COUNT)
        _integer_list(payload["law_order"], list(order), "shuffled-law order")
        cards = tuple(row.cards[slot] for slot in order)
        program_bytes = _render_program(row, cards=cards).encode("utf-8")
        _, packet_bytes = _packet_digest(
            cards,
            row.initial_state,
            row.schedule,
            row.opcode_to_card,
        )
    elif operation == InterventionFamily.SCHEDULE_ORDER_TWIN.value:
        pairs = [
            (left, right)
            for left in range(row.depth)
            for right in range(left + 1, row.depth)
            if row.schedule[left] != row.schedule[right]
        ]
        pairs.sort(
            key=lambda pair: _hash_order(seed, operation, attempt.anchor_id, *pair)
        )
        if not pairs:
            raise ReplayValidationError("CTAA replay schedule twin has no changed pair")
        left, right = pairs[0]
        _integer_list(payload["swapped_active_slots"], [left, right], "schedule swap")
        schedule_list = list(row.schedule)
        schedule_list[left], schedule_list[right] = (
            schedule_list[right],
            schedule_list[left],
        )
        schedule = tuple(schedule_list)
        program_bytes = _render_program(row, schedule=schedule).encode("utf-8")
        _, packet_bytes = _packet_digest(
            row.cards,
            row.initial_state,
            schedule,
            row.opcode_to_card,
        )
    elif operation == InterventionFamily.SOURCE_POISON.value:
        poison = b"SHOHIN-CTAA-SOURCE-POISON-v1\0" + _hash_order(
            seed, operation, attempt.anchor_id
        )
        _exact_int(payload["replacement_offset"], 0, "poison offset")
        _exact_int(
            payload["replacement_length"],
            len(row.program_source.encode("utf-8")),
            "poison length",
        )
        _exact(payload["poison_bytes_hex"], poison.hex(), "poison bytes")
        _exact(payload["poison_bytes_sha256"], _sha256(poison), "poison hash")
        program_bytes = poison
    elif operation == InterventionFamily.FUTURE_MASK.value:
        boundary = 1 + int.from_bytes(
            _hash_order(seed, operation, attempt.anchor_id)[:8], "big"
        ) % (row.depth - 1)
        _exact_int(payload["first_exposure_step"], boundary, "future boundary")
        _integer_list(
            payload["changed_slots"], list(range(boundary, row.depth)), "future slots"
        )
        mutated = future_schedule_counterfactual(
            _hard_packet(row), torch.tensor([boundary], dtype=torch.long)
        ).packet
        packet_bytes = packet_body(mutated)
    elif operation == InterventionFamily.STOP_RELOCATION.value:
        target = (
            int.from_bytes(_hash_order(seed, operation, attempt.anchor_id)[:8], "big")
            % row.depth
        )
        schedule_list = list(row.schedule)
        schedule_list[target], schedule_list[row.depth] = (
            schedule_list[row.depth],
            schedule_list[target],
        )
        _exact_int(payload["old_stop_index"], row.depth, "old STOP index")
        _exact_int(payload["new_stop_index"], target, "new STOP index")
        _exact_int(
            payload["displaced_event"], schedule_list[row.depth], "displaced event"
        )
        schedule = tuple(schedule_list)
        program_bytes = _render_program(row, schedule=schedule).encode("utf-8")
        _, packet_bytes = _packet_digest(
            row.cards,
            row.initial_state,
            schedule,
            row.opcode_to_card,
        )
    elif operation == InterventionFamily.LATE_QUERY_SWAP.value:
        if donor is None:
            raise ReplayValidationError("CTAA replay late-query donor is missing")
        _exact_int(
            payload["parent_query_position"],
            row.query_position,
            "parent query position",
        )
        _exact_int(
            payload["donor_query_position"],
            donor.query_position,
            "donor query position",
        )
        _exact(
            payload["execution_policy"],
            "reuse_immutable_parent_execution",
            "query execution policy",
        )
        parent_query = HardCTAAQuery(
            torch.tensor([row.query_position], dtype=torch.uint8)
        )
        donor_query = HardCTAAQuery(
            torch.tensor([donor.query_position], dtype=torch.uint8)
        )
        if bool(parent_query.position.eq(donor_query.position).any()):
            raise ReplayValidationError("CTAA replay late-query swap is a no-op")
        query_bytes = donor.query_source.encode("utf-8")
    elif operation == InterventionFamily.POST_STOP_POISON.value:
        _exact_int(payload["stop_index"], row.depth, "post-STOP index")
        _integer_list(
            payload["changed_slots"],
            list(range(row.depth + 1, CTAA_MAX_STEPS)),
            "post-STOP slots",
        )
        _exact(payload["replacement_rule"], "(event+1)%4", "post-STOP rule")
        packet_bytes = packet_body(post_stop_poison(_hard_packet(row)).packet)
    elif operation == InterventionFamily.MIDPOINT_DONOR_STATE.value:
        if donor is None:
            raise ReplayValidationError("CTAA replay midpoint-state donor is missing")
        midpoint = row.depth // 2
        _exact_int(payload["midpoint_step"], midpoint, "midpoint-state step")
        _exact(
            payload["donor_state_sha256"],
            _sha256(donor.midpoint_state_bytes),
            "midpoint donor-state hash",
        )
        if donor.midpoint_state_bytes == row.midpoint_state_bytes:
            raise ReplayValidationError("CTAA replay midpoint donor state is a no-op")
        if donor.midpoint_suffix_bytes != row.midpoint_suffix_bytes:
            raise ReplayValidationError("CTAA replay midpoint donor suffix differs")
    elif operation == InterventionFamily.MIDPOINT_DONOR_ACTION.value:
        if donor is None:
            raise ReplayValidationError("CTAA replay midpoint-action donor is missing")
        midpoint = row.depth // 2
        order = _permutation(seed, operation, attempt.anchor_id, CTAA_ACTION_COUNT)
        slot = next(
            (
                candidate
                for candidate in order
                if _sha256(bytes(donor.cards[candidate]))
                != anchor.midpoint_action_sha256
            ),
            None,
        )
        if slot is None:
            raise ReplayValidationError(
                "CTAA replay midpoint donor has no changed action"
            )
        _exact_int(payload["midpoint_step"], midpoint, "midpoint-action step")
        _exact_int(payload["donor_card_slot"], slot, "midpoint donor-card slot")
        _exact(
            payload["donor_action_sha256"],
            _sha256(bytes(donor.cards[slot])),
            "midpoint donor-action hash",
        )
        if donor.midpoint_suffix_bytes != row.midpoint_suffix_bytes:
            raise ReplayValidationError("CTAA replay midpoint donor suffix differs")
    elif operation == InterventionFamily.PACKET_TRANSPLANT.value:
        if donor is None:
            raise ReplayValidationError("CTAA replay packet donor is missing")
        _exact(
            payload["literal_donor_packet_sha256"],
            _sha256(donor.packet_bytes),
            "literal donor packet hash",
        )
        packet_bytes = packet_body(
            packet_transplant(_hard_packet(row), _hard_packet(donor)).packet
        )
    elif operation == GateFamily.SOURCE_DELETION.value:
        _exact(
            payload["probe_stage"], "source_blind_packet_executor", "source gate stage"
        )
        _exact(
            payload["probe_targets"],
            ["program_source", "board_root"],
            "source gate targets",
        )
        _exact(
            payload["required_result"], "all_open_attempts_denied", "source gate result"
        )
        _exact(payload["allowed_errno"], ["EACCES", "ENOENT"], "source gate errno")
    elif operation == GateFamily.QUERY_ISOLATION.value:
        _exact(
            payload["probe_stage"],
            "execution_before_receipt_commit",
            "query gate stage",
        )
        _exact(payload["probe_target"], "query_source", "query gate target")
        _exact(
            payload["required_result"], "all_open_attempts_denied", "query gate result"
        )
        _exact(
            payload["disclosure_after"],
            "validated_immutable_execution_receipt",
            "query disclosure boundary",
        )
    elif operation == GateFamily.ROUTE_AGREEMENT.value:
        _integer_list(
            payload["positions"], list(range(CTAA_MAX_STEPS + 1)), "route positions"
        )
        _exact(
            payload["comparison"],
            "exact_uint8_state_route_equals_composed_route",
            "route comparison",
        )
        _exact(
            payload["required_tensor_shape"],
            [CTAA_MAX_STEPS + 1, CTAA_WIDTH],
            "route tensor shape",
        )
    else:  # pragma: no cover - mandatory-operation import invariant
        raise ReplayValidationError("CTAA replay operation is unknown")

    parent_program = row.program_source.encode("utf-8")
    parent_query = row.query_source.encode("utf-8")
    if program_bytes is not None:
        _changed(program_bytes, parent_program, "program transformation")
    if query_bytes is not None:
        _changed(query_bytes, parent_query, "query transformation")
    if packet_bytes is not None:
        _changed(packet_bytes, row.packet_bytes, "packet transformation")
    program_sha = _sha256(program_bytes) if program_bytes is not None else None
    query_sha = _sha256(query_bytes) if query_bytes is not None else None
    packet_sha = _sha256(packet_bytes) if packet_bytes is not None else None
    expected_presence = (
        operation in _PROGRAM_RESULTS,
        operation in _QUERY_RESULTS,
        operation in _PACKET_RESULTS,
    )
    actual_presence = (
        program_bytes is not None,
        query_bytes is not None,
        packet_bytes is not None,
    )
    if actual_presence != expected_presence:
        raise ReplayValidationError("CTAA replay result-artifact contract differs")
    commitments = (
        attempt.resulting_program_source_sha256,
        attempt.resulting_query_source_sha256,
        attempt.resulting_packet_sha256,
    )
    recomputed = (program_sha, query_sha, packet_sha)
    if commitments != recomputed:
        raise ReplayValidationError("CTAA replay resulting artifact hash differs")
    return ReplayResult(
        attempt.attempt_index,
        attempt.attempt_id,
        operation,
        attempt.anchor_id,
        attempt.donor_anchor_id,
        program_sha,
        query_sha,
        packet_sha,
    )


def replay_runtime_plan(
    plan: RuntimeInterventionPlan | Mapping[str, object],
    rows_by_anchor: Mapping[str, ParsedSource],
) -> RuntimePlanReplay:
    """Validate and replay every committed attempt in frozen plan order."""

    ctx = _context(plan, rows_by_anchor)
    results = tuple(_replay_one(ctx, attempt) for attempt in ctx.plan.attempts)
    if len(results) != len(MANDATORY_OPERATIONS) * len(ctx.plan.anchors):
        raise ReplayValidationError("CTAA replay attempt count differs")
    return RuntimePlanReplay(ctx.plan.plan_sha256, len(results), results)


def replay_attempt(
    plan: RuntimeInterventionPlan | Mapping[str, object],
    rows_by_anchor: Mapping[str, ParsedSource],
    attempt_index: int,
) -> ReplayResult:
    """Replay one indexed attempt after validating the complete frozen plan."""

    ctx = _context(plan, rows_by_anchor)
    if type(attempt_index) is not int or not 0 <= attempt_index < len(
        ctx.plan.attempts
    ):
        raise ReplayValidationError("CTAA replay attempt index differs")
    return _replay_one(ctx, ctx.plan.attempts[attempt_index])


def load_runtime_replay_rows(
    *,
    plan: RuntimeInterventionPlan | Mapping[str, object],
    program_path: Path,
    query_path: Path,
    tokenizer_path: Path,
) -> dict[str, ParsedSource]:
    """Load immutable board sources and reconstruct exactly the selected panel."""

    frozen = validate_runtime_intervention_plan(plan)
    tokenizer_raw = _read_locked_bytes(tokenizer_path, "runtime replay tokenizer")
    if _sha256(tokenizer_raw) != frozen.bindings.tokenizer_sha256:
        raise ReplayValidationError("CTAA replay tokenizer binding differs")
    try:
        tokenizer = Tokenizer.from_str(tokenizer_raw.decode("utf-8"))
    except Exception as error:  # noqa: BLE001 - tokenizers exposes a generic error
        raise ReplayValidationError("CTAA replay tokenizer differs") from error
    program_rows = _read_jsonl(
        _read_locked_bytes(program_path, "runtime replay program source"),
        "runtime replay program source",
        frozenset({"family_id", "program_source"}),
    )
    query_rows = _read_jsonl(
        _read_locked_bytes(query_path, "runtime replay query source"),
        "runtime replay query source",
        frozenset({"family_id", "query_source"}),
    )
    query_by_id = {
        str(row["family_id"]): str(row["query_source"]) for row in query_rows
    }
    program_by_id = {
        str(row["family_id"]): str(row["program_source"]) for row in program_rows
    }
    anchor_by_family = {anchor.family_id: anchor for anchor in frozen.anchors}
    if not set(anchor_by_family).issubset(program_by_id) or not set(
        anchor_by_family
    ).issubset(query_by_id):
        raise ReplayValidationError("CTAA replay selected source set differs")
    rows = {
        anchor.anchor_id: _parse_program(
            anchor.family_id,
            program_by_id[anchor.family_id],
            query_by_id[anchor.family_id],
            tokenizer,
            frozen.bindings.partition.value,
        )
        for anchor in frozen.anchors
    }
    return rows
