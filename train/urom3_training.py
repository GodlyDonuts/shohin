"""Tokenization, supervision, and localization metrics for UROM-3."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from general_relational_object_machine import (
    APPLY_KIND,
    MAX_EVENTS,
    MAX_OBJECTS,
    MAX_RELATION_EDGES,
    MAX_RULES,
    DeletedRelationalProgram,
    TrunkRelationalObjectCompiler,
    rollout_hard_relational_program,
    rollout_relational_program,
)
from urom3_board import validate_row


@dataclass(frozen=True, slots=True)
class TokenizedUROMRow:
    row_sha256: str
    axis_cell: str
    family: str
    renderer: str
    program_ids: tuple[int, ...]
    query_ids: tuple[int, ...]
    pointer_targets: tuple[tuple[int, ...], ...]
    cardinality: int
    initial_edges: tuple[tuple[int, ...], ...]
    rule_edges: tuple[tuple[tuple[int, ...], ...], ...]
    rule_active: tuple[bool, ...]
    event_rule: tuple[int, ...]
    event_kind: tuple[int, ...]
    query_position: int
    state_trajectory: tuple[tuple[tuple[int, ...], ...], ...]
    terminal_state: tuple[tuple[int, ...], ...]
    answer_bits: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class UROMBatch:
    program_ids: torch.Tensor
    query_ids: torch.Tensor
    pointer_targets: torch.Tensor
    cardinality: torch.Tensor
    initial_edges: torch.Tensor
    rule_edges: torch.Tensor
    rule_active: torch.Tensor
    event_rule: torch.Tensor
    event_kind: torch.Tensor
    query_position: torch.Tensor
    state_trajectory: torch.Tensor
    terminal_state: torch.Tensor
    answer_bits: torch.Tensor


@dataclass(frozen=True, slots=True)
class UROMLossReceipt:
    total: torch.Tensor
    pointer: torch.Tensor
    cardinality: torch.Tensor
    initial: torch.Tensor
    rules: torch.Tensor
    rule_active: torch.Tensor
    event_rule: torch.Tensor
    event_kind: torch.Tensor
    query: torch.Tensor
    trajectory: torch.Tensor
    terminal: torch.Tensor
    answer: torch.Tensor


def _tensor_tuple(value: object, dimensions: int) -> tuple:
    if dimensions < 1 or not isinstance(value, list):
        raise ValueError("UROM nested target geometry differs")
    if dimensions == 1:
        return tuple(value)
    return tuple(_tensor_tuple(item, dimensions - 1) for item in value)


def _span_tokens(
    offsets: Sequence[tuple[int, int]],
    span: Sequence[int],
) -> tuple[int, ...]:
    if len(span) != 2:
        raise ValueError("UROM evidence span differs")
    start, end = (int(value) for value in span)
    selected = tuple(
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    )
    if not selected:
        raise ValueError("UROM evidence span has no token")
    return selected


def _pointer_targets(
    row: dict[str, object],
    offsets: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    evidence = row["evidence_spans"]
    if not isinstance(evidence, dict) or not isinstance(
        evidence.get("program"),
        dict,
    ):
        raise ValueError("UROM program evidence differs")
    spans: dict[str, object] = evidence["program"]  # type: ignore[assignment]
    control_count = 1
    declaration_start = control_count
    initial_start = declaration_start + MAX_OBJECTS
    rule_start = initial_start + MAX_OBJECTS
    rule_stride = 1 + 2 * MAX_RELATION_EDGES
    event_start = rule_start + MAX_RULES * rule_stride
    slot_count = event_start + MAX_EVENTS
    targets: list[tuple[int, ...]] = [tuple() for _ in range(slot_count)]

    compiler_targets = row["compiler_targets"]
    cardinality = int(compiler_targets["cardinality"])  # type: ignore[index]
    rule_active = compiler_targets["rule_active"]  # type: ignore[index]
    for index in range(cardinality):
        declaration = spans[f"declaration.{index}"]
        initial = spans[f"initial.{index}.source"]
        targets[declaration_start + index] = _span_tokens(
            offsets,
            declaration[0],  # type: ignore[index]
        )
        targets[initial_start + index] = _span_tokens(
            offsets,
            initial[0],  # type: ignore[index]
        )

    for rule in range(MAX_RULES):
        if not rule_active[rule]:
            continue
        start = rule_start + rule * rule_stride
        opcode = spans[f"rule.{rule}.opcode"]
        targets[start] = _span_tokens(offsets, opcode[0])  # type: ignore[index]
        edge = 0
        while f"rule.{rule}.edge.{edge}.source" in spans:
            if edge >= MAX_RELATION_EDGES:
                raise ValueError("UROM rule exceeds compiler edge slots")
            source = spans[f"rule.{rule}.edge.{edge}.source"]
            destination = spans[f"rule.{rule}.edge.{edge}.destination"]
            targets[start + 1 + edge] = _span_tokens(
                offsets,
                source[0],  # type: ignore[index]
            )
            targets[
                start + 1 + MAX_RELATION_EDGES + edge
            ] = _span_tokens(
                offsets,
                destination[0],  # type: ignore[index]
            )
            edge += 1

    for event in range(MAX_EVENTS):
        key = f"event.{event}.opcode"
        if key in spans:
            targets[event_start + event] = _span_tokens(
                offsets,
                spans[key][0],  # type: ignore[index]
            )
    return tuple(targets)


def parse_urom_row(
    value: object,
    tokenizer: Tokenizer,
    *,
    expected_split: str,
    max_length: int,
) -> TokenizedUROMRow:
    if not isinstance(value, dict):
        raise ValueError("UROM row must be an object")
    validate_row(value)
    if value.get("split") != expected_split:
        raise ValueError("UROM split differs")
    program_encoding = tokenizer.encode(str(value["program_text"]))
    query_encoding = tokenizer.encode(str(value["query_text"]))
    if (
        not program_encoding.ids
        or not query_encoding.ids
        or len(program_encoding.ids) > max_length
        or len(query_encoding.ids) > max_length
    ):
        raise ValueError("UROM token length differs")
    targets = value["compiler_targets"]
    query = value["late_query_target"]
    oracle = value["oracle"]
    return TokenizedUROMRow(
        row_sha256=str(value["row_sha256"]),
        axis_cell=str(value["axis_cell"]),
        family=str(value["family"]),
        renderer=str(value["renderer"]),
        program_ids=tuple(program_encoding.ids),
        query_ids=tuple(query_encoding.ids),
        pointer_targets=_pointer_targets(value, program_encoding.offsets),
        cardinality=int(targets["cardinality"]),  # type: ignore[index]
        initial_edges=_tensor_tuple(targets["initial_edges"], 2),  # type: ignore[index,arg-type]
        rule_edges=_tensor_tuple(targets["rule_edges"], 3),  # type: ignore[index,arg-type]
        rule_active=tuple(bool(item) for item in targets["rule_active"]),  # type: ignore[index]
        event_rule=tuple(int(item) for item in targets["event_rule"]),  # type: ignore[index]
        event_kind=tuple(int(item) for item in targets["event_kind"]),  # type: ignore[index]
        query_position=int(query["position"]),  # type: ignore[index]
        state_trajectory=_tensor_tuple(oracle["state_trajectory"], 3),  # type: ignore[index,arg-type]
        terminal_state=_tensor_tuple(oracle["terminal_state"], 2),  # type: ignore[index,arg-type]
        answer_bits=tuple(int(item) for item in oracle["answer_bits"]),  # type: ignore[index]
    )


def load_urom_rows(
    path: Path,
    tokenizer: Tokenizer,
    *,
    expected_split: str,
    max_length: int,
) -> list[TokenizedUROMRow]:
    rows = [
        parse_urom_row(
            json.loads(line),
            tokenizer,
            expected_split=expected_split,
            max_length=max_length,
        )
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("UROM split is empty")
    return rows


def collate_urom_rows(
    rows: Sequence[TokenizedUROMRow],
    *,
    padding_id: int = 1,
    device: torch.device | None = None,
) -> UROMBatch:
    if not rows:
        raise ValueError("UROM batch is empty")
    destination = device or torch.device("cpu")
    program_length = max(len(row.program_ids) for row in rows)
    query_length = max(len(row.query_ids) for row in rows)
    slot_count = len(rows[0].pointer_targets)
    program = torch.full(
        (len(rows), program_length),
        padding_id,
        dtype=torch.long,
    )
    query = torch.full(
        (len(rows), query_length),
        padding_id,
        dtype=torch.long,
    )
    pointer = torch.zeros(
        len(rows),
        slot_count,
        program_length,
        dtype=torch.bool,
    )
    for index, row in enumerate(rows):
        if len(row.pointer_targets) != slot_count:
            raise ValueError("UROM pointer slot count differs")
        program[index, : len(row.program_ids)] = torch.tensor(row.program_ids)
        query[index, : len(row.query_ids)] = torch.tensor(row.query_ids)
        for slot, indices in enumerate(row.pointer_targets):
            if indices:
                pointer[index, slot, torch.tensor(indices)] = True
    return UROMBatch(
        program_ids=program.to(destination),
        query_ids=query.to(destination),
        pointer_targets=pointer.to(destination),
        cardinality=torch.tensor(
            [row.cardinality for row in rows],
            dtype=torch.long,
            device=destination,
        ),
        initial_edges=torch.tensor(
            [row.initial_edges for row in rows],
            dtype=torch.float,
            device=destination,
        ),
        rule_edges=torch.tensor(
            [row.rule_edges for row in rows],
            dtype=torch.float,
            device=destination,
        ),
        rule_active=torch.tensor(
            [row.rule_active for row in rows],
            dtype=torch.long,
            device=destination,
        ),
        event_rule=torch.tensor(
            [row.event_rule for row in rows],
            dtype=torch.long,
            device=destination,
        ),
        event_kind=torch.tensor(
            [row.event_kind for row in rows],
            dtype=torch.long,
            device=destination,
        ),
        query_position=torch.tensor(
            [row.query_position for row in rows],
            dtype=torch.long,
            device=destination,
        ),
        state_trajectory=torch.tensor(
            [row.state_trajectory for row in rows],
            dtype=torch.float,
            device=destination,
        ),
        terminal_state=torch.tensor(
            [row.terminal_state for row in rows],
            dtype=torch.float,
            device=destination,
        ),
        answer_bits=torch.tensor(
            [row.answer_bits for row in rows],
            dtype=torch.float,
            device=destination,
        ),
    )


def _pointer_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != targets.shape or targets.dtype != torch.bool:
        raise ValueError("UROM pointer supervision differs")
    supervised = targets.any(-1)
    if not bool(supervised.any()):
        raise ValueError("UROM batch has no pointer supervision")
    log_probability = logits.float().log_softmax(-1)
    negative = torch.finfo(log_probability.dtype).min
    target_log_probability = torch.logsumexp(
        log_probability.masked_fill(~targets, negative),
        dim=-1,
    )
    return -target_log_probability[supervised].mean()


def _active_square(cardinality: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(MAX_OBJECTS, device=cardinality.device)
    active = positions[None] < cardinality[:, None]
    return active[:, :, None] & active[:, None, :]


def urom_loss(
    compiler: TrunkRelationalObjectCompiler,
    batch: UROMBatch,
    *,
    execution_weight: float = 1.0,
) -> UROMLossReceipt:
    compiled = compiler.compile_program(
        batch.program_ids,
        return_evidence=True,
    )
    if not isinstance(compiled, tuple):
        raise AssertionError("UROM compiler evidence is absent")
    program, evidence = compiled
    query = compiler.compile_late_query(batch.query_ids)
    active_square = _active_square(batch.cardinality)
    rule_mask = active_square[:, None] & batch.rule_active.bool()[:, :, None, None]
    pointer = _pointer_loss(evidence.pointer_logits, batch.pointer_targets)
    cardinality = F.cross_entropy(
        program.cardinality,
        batch.cardinality - 2,
    )
    initial_values = F.binary_cross_entropy_with_logits(
        program.initial_state,
        batch.initial_edges,
        reduction="none",
    )
    initial = initial_values[active_square].mean()
    rule_values = F.binary_cross_entropy_with_logits(
        program.rule_cards,
        batch.rule_edges,
        reduction="none",
    )
    rules = rule_values[rule_mask].mean()
    rule_active = F.cross_entropy(
        program.rule_active.reshape(-1, 2),
        batch.rule_active.reshape(-1),
    )
    event_rule_mask = batch.event_kind.eq(APPLY_KIND)
    event_rule_values = F.cross_entropy(
        program.event_rule.reshape(-1, MAX_RULES),
        batch.event_rule.reshape(-1),
        reduction="none",
    ).reshape_as(batch.event_rule)
    event_rule = event_rule_values[event_rule_mask].mean()
    event_kind = F.cross_entropy(
        program.event_kind.reshape(-1, 3),
        batch.event_kind.reshape(-1),
    )
    query_loss = F.cross_entropy(query.position, batch.query_position)

    rollout = rollout_relational_program(program, query)
    trajectory = torch.stack(rollout.state_trajectory, dim=1)
    trajectory_values = F.binary_cross_entropy(
        trajectory.clamp(1e-6, 1.0 - 1e-6),
        batch.state_trajectory,
        reduction="none",
    )
    trajectory_mask = active_square[:, None].expand_as(trajectory_values)
    trajectory_loss = trajectory_values[trajectory_mask].mean()
    terminal_values = F.binary_cross_entropy(
        rollout.final_state.clamp(1e-6, 1.0 - 1e-6),
        batch.terminal_state,
        reduction="none",
    )
    terminal = terminal_values[active_square].mean()
    answer_mask = (
        torch.arange(MAX_OBJECTS, device=batch.cardinality.device)[None]
        < batch.cardinality[:, None]
    )
    answer_values = F.binary_cross_entropy(
        rollout.answer_distribution.clamp(1e-6, 1.0 - 1e-6),
        batch.answer_bits,
        reduction="none",
    )
    answer = answer_values[answer_mask].mean()
    total = (
        pointer
        + cardinality
        + initial
        + rules
        + rule_active
        + event_rule
        + event_kind
        + query_loss
        + execution_weight * (trajectory_loss + terminal + answer)
    )
    return UROMLossReceipt(
        total=total,
        pointer=pointer,
        cardinality=cardinality,
        initial=initial,
        rules=rules,
        rule_active=rule_active,
        event_rule=event_rule,
        event_kind=event_kind,
        query=query_loss,
        trajectory=trajectory_loss,
        terminal=terminal,
        answer=answer,
    )


@torch.inference_mode()
def urom_metrics(
    compiler: TrunkRelationalObjectCompiler,
    batch: UROMBatch,
) -> dict[str, float]:
    compiled = compiler.compile_program(batch.program_ids)
    if not isinstance(compiled, DeletedRelationalProgram):
        raise AssertionError("UROM compiler returned evidence unexpectedly")
    hard = compiled.seal()
    query_logits = compiler.compile_late_query(batch.query_ids)
    query = query_logits.seal(hard.cardinality)
    rollout = rollout_hard_relational_program(hard, query)
    active_square = _active_square(batch.cardinality)
    active_objects = active_square.any(-1)
    program_fields = {
        "cardinality": hard.cardinality.long().eq(batch.cardinality),
        "initial": (
            hard.initial_edges.bool().eq(batch.initial_edges.bool())
            | ~active_square
        ).all(dim=(-1, -2)),
        "rules": (
            hard.rule_edges.bool().eq(batch.rule_edges.bool())
            | ~active_square[:, None]
            | ~batch.rule_active.bool()[:, :, None, None]
        ).all(dim=(-1, -2, -3)),
        "rule_active": hard.rule_active.eq(batch.rule_active.bool()).all(-1),
        "event_rule": (
            hard.event_rule.long().eq(batch.event_rule)
            | ~batch.event_kind.eq(APPLY_KIND)
        ).all(-1),
        "event_kind": hard.event_kind.long().eq(batch.event_kind).all(-1),
        "query": query.position.long().eq(batch.query_position),
    }
    terminal = (
        rollout.final_state.bool().eq(batch.terminal_state.bool())
        | ~active_square
    ).all(dim=(-1, -2))
    answer = (
        rollout.answer_distribution.bool().eq(batch.answer_bits.bool())
        | ~active_objects
    ).all(-1)
    trajectory = torch.stack(rollout.state_trajectory, dim=1)
    trajectory_exact = (
        trajectory.bool().eq(batch.state_trajectory.bool())
        | ~active_square[:, None]
    ).all(dim=(-1, -2, -3))
    joint = terminal & answer & trajectory_exact
    for value in program_fields.values():
        joint &= value
    metrics = {
        f"{name}_accuracy": float(value.float().mean())
        for name, value in program_fields.items()
    }
    metrics.update(
        {
            "trajectory_accuracy": float(trajectory_exact.float().mean()),
            "terminal_accuracy": float(terminal.float().mean()),
            "answer_accuracy": float(answer.float().mean()),
            "joint_accuracy": float(joint.float().mean()),
        }
    )
    return metrics
