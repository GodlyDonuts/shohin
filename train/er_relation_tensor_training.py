"""Training, sealed execution, and evidence mechanics for ER-TT v1."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from build_er_relation_tensor_board import (
    BOARD_SCHEMA,
    PROTOCOL,
    TRAIN_SPLIT,
)
from er_cst_fresh import (
    _cosine_scale,
    byte_batch,
    derived_seed,
)
from er_relation_tensor_adapter import (
    EVENT_SLOTS,
    MAX_CARDINALITY,
    MAX_RULES,
    MIN_CARDINALITY,
    TT_RECORDS,
    EpisodicRelationTensorCompiler,
    HardRelationTensorProgram,
    hard_relation_answer,
)
from er_relation_tensor_motor import rollout_relation_tensor
from pilot_sd_cst_byte_addressed import sha256_file


TRAINING_CONTRACT = {
    "arms": ["treatment", "family_deranged", "equality_ablated"],
    "families": 12_000,
    "views_per_family": 4,
    "rows": 48_000,
    "epochs": 2,
    "family_batch_size": 8,
    "rows_per_update": 32,
    "updates": 3_000,
    "lr": 2e-4,
    "warmup": 100,
    "weight_decay": 0.01,
    "betas": [0.9, 0.95],
    "gradient_clip": 1.0,
    "renderer_consistency_weight": 1.0,
    "pointer_weight": 1.0,
    "outcome_supervision": False,
    "learned_motor_parameters": 0,
    "learned_reader_parameters": 0,
}


@dataclass(frozen=True, slots=True)
class RelationTensorRow:
    row_id: str
    family_id: str
    renderer: str
    split: str
    program_bytes: tuple[int, ...]
    query_bytes: tuple[int, ...]
    line_ranges: tuple[tuple[int, int], ...]
    binding_ranges: tuple[tuple[int, int], ...]
    initial_ranges: tuple[tuple[int, int], ...]
    witness_before_ranges: tuple[tuple[tuple[int, int], ...], ...]
    witness_after_ranges: tuple[tuple[tuple[int, int], ...], ...]
    query_range: tuple[int, int]
    cardinality: int
    rule_count: int
    initial_order: tuple[int, ...]
    relation_rows: tuple[tuple[int, ...], ...]
    event_cards: tuple[int, ...]
    event_halt: tuple[int, ...]
    query_position: int
    depth: int
    non_bijective: bool
    final_state: tuple[int, ...] | None
    answer_role: int | None


def _ranges(value: object, count: int, name: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"ER-TT {name} ranges differ")
    output = tuple((int(item[0]), int(item[1])) for item in value)
    if any(end <= start for start, end in output):
        raise ValueError(f"ER-TT {name} range is empty")
    return output


def parse_row(value: Mapping[str, object], split: str) -> RelationTensorRow:
    if value.get("split") != split or value.get("protocol") != PROTOCOL:
        raise ValueError("ER-TT row split/protocol differs")
    target = value.get("compiler_targets")
    if not isinstance(target, Mapping):
        raise ValueError("ER-TT row lacks compiler targets")
    cardinality = int(target["cardinality"])
    rule_count = int(target["rule_count"])
    if not MIN_CARDINALITY <= cardinality <= MAX_CARDINALITY:
        raise ValueError("ER-TT row cardinality differs")
    if not 2 <= rule_count <= MAX_RULES:
        raise ValueError("ER-TT row rule count differs")
    rules = sorted(target["rule_cards"], key=lambda item: int(item["slot"]))
    events = sorted(target["events"], key=lambda item: int(item["slot"]))
    if len(rules) != MAX_RULES or len(events) != EVENT_SLOTS:
        raise ValueError("ER-TT target record count differs")
    before_raw = target.get("witness_before_ranges")
    after_raw = target.get("witness_after_ranges")
    if not isinstance(before_raw, list) or not isinstance(after_raw, list):
        raise ValueError("ER-TT witness ranges are absent")
    before: list[tuple[tuple[int, int], ...]] = []
    after: list[tuple[tuple[int, int], ...]] = []
    relation_rows: list[tuple[int, ...]] = []
    for slot, rule in enumerate(rules):
        active = slot < rule_count
        if bool(rule["active"]) != active:
            raise ValueError("ER-TT active-rule prefix differs")
        if active:
            before.append(_ranges(before_raw[slot], cardinality, f"before-{slot}"))
            after.append(_ranges(after_raw[slot], cardinality, f"after-{slot}"))
            relation = tuple(map(int, rule["relation"]))
            if len(relation) != cardinality or any(
                not 0 <= item < cardinality for item in relation
            ):
                raise ValueError("ER-TT relation row differs")
            relation_rows.append(relation)
        else:
            if before_raw[slot] or after_raw[slot] or rule["relation"]:
                raise ValueError("ER-TT inactive rule exposes witness targets")
            before.append(())
            after.append(())
            relation_rows.append(())
    initial_order = tuple(map(int, target["initial_order"]))
    if len(initial_order) != cardinality or sorted(initial_order) != list(
        range(cardinality)
    ):
        raise ValueError("ER-TT initial relation differs")
    event_cards = tuple(int(item["card_slot"]) for item in events)
    event_halt = tuple(int(bool(item["halt"])) for item in events)
    if sum(event_halt) != 1 or any(
        not 0 <= card < rule_count
        for card, halt in zip(event_cards, event_halt, strict=True)
        if not halt
    ):
        raise ValueError("ER-TT event target differs")
    oracle = value.get("oracle")
    if split == TRAIN_SPLIT:
        if oracle is not None or value.get("supervision") != "compiler_fields_only":
            raise ValueError("ER-TT training row exposes outcome supervision")
        final_state = None
        answer_role = None
    else:
        if not isinstance(oracle, Mapping):
            raise ValueError("ER-TT scored row lacks oracle")
        final_state = tuple(map(int, oracle["final_state"]))
        answer_role = int(oracle["answer_role"])
        if len(final_state) != cardinality:
            raise ValueError("ER-TT scored final state differs")
    relations = relation_rows[:rule_count]
    return RelationTensorRow(
        row_id=str(value["id"]),
        family_id=str(value["family_id"]),
        renderer=str(value["template_id"]),
        split=split,
        program_bytes=tuple(str(value["program_text"]).encode("utf-8")),
        query_bytes=tuple(str(value["late_query_text"]).encode("utf-8")),
        line_ranges=_ranges(target["line_ranges"], TT_RECORDS, "line"),
        binding_ranges=_ranges(target["binding_ranges"], cardinality, "binding"),
        initial_ranges=_ranges(target["initial_ranges"], cardinality, "initial"),
        witness_before_ranges=tuple(before),
        witness_after_ranges=tuple(after),
        query_range=_ranges([target["query_range"]], 1, "query")[0],
        cardinality=cardinality,
        rule_count=rule_count,
        initial_order=initial_order,
        relation_rows=tuple(relation_rows),
        event_cards=event_cards,
        event_halt=event_halt,
        query_position=int(target["query_position"]),
        depth=int(target["depth"]),
        non_bijective=any(len(set(row)) < cardinality for row in relations),
        final_state=final_state,
        answer_role=answer_role,
    )


def load_board_receipt(data_dir: Path) -> dict[str, object]:
    path = data_dir / "report.json"
    value = json.loads(path.read_text())
    if (
        value.get("schema") != BOARD_SCHEMA
        or value.get("protocol") != PROTOCOL
        or value.get("all_gates_pass") is not True
        or value.get("development_accesses") != 0
        or value.get("confirmation_accesses") != 0
    ):
        raise ValueError("ER-TT board receipt differs")
    value["report_sha256"] = sha256_file(path)
    return value


def load_split(
    data_dir: Path,
    board: Mapping[str, object],
    *,
    filename: str,
    split: str,
    expected: int,
) -> list[RelationTensorRow]:
    path = data_dir / filename
    if sha256_file(path) != board["files"][filename]["sha256"]:
        raise ValueError(f"ER-TT {split} hash differs")
    rows = [
        parse_row(json.loads(line), split)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != expected:
        raise ValueError(f"ER-TT {split} row count differs")
    return rows


def group_families(
    rows: Sequence[RelationTensorRow],
) -> list[list[RelationTensorRow]]:
    grouped: dict[str, list[RelationTensorRow]] = {}
    for row in rows:
        grouped.setdefault(row.family_id, []).append(row)
    output = []
    for family in sorted(grouped):
        values = sorted(grouped[family], key=lambda row: row.renderer)
        if len(values) != 4 or len({row.renderer for row in values}) != 4:
            raise ValueError("ER-TT family views differ")
        first = values[0]
        if any(
            (row.cardinality, row.rule_count, row.initial_order, row.relation_rows)
            != (
                first.cardinality,
                first.rule_count,
                first.initial_order,
                first.relation_rows,
            )
            for row in values[1:]
        ):
            raise ValueError("ER-TT family semantics differ across views")
        output.append(values)
    return output


def _family_derangement(
    rows: Sequence[RelationTensorRow], seed: int
) -> list[RelationTensorRow]:
    groups = group_families(rows)
    buckets: dict[tuple[int, int], list[list[RelationTensorRow]]] = {}
    for group in groups:
        key = group[0].cardinality, group[0].rule_count
        buckets.setdefault(key, []).append(group)
    donors: dict[str, tuple[tuple[int, ...], ...]] = {}
    for key, values in sorted(buckets.items()):
        values.sort(key=lambda group: group[0].family_id)
        if len(values) < 2:
            raise ValueError("ER-TT derangement bucket is singular")
        shift = 1 + derived_seed(seed, f"er-tt-derange:{key}") % (len(values) - 1)
        for index, group in enumerate(values):
            donor = values[(index + shift) % len(values)][0]
            donors[group[0].family_id] = donor.relation_rows
    return [replace(row, relation_rows=donors[row.family_id]) for row in rows]


def _equality_ablated_bytes(row: RelationTensorRow, seed: int) -> tuple[int, ...]:
    payload = bytearray(row.program_bytes)
    used = {bytes(payload[start:end]) for start, end in row.binding_ranges}
    used.update(
        bytes(payload[start:end])
        for rule in row.witness_before_ranges
        for start, end in rule
    )
    replacements: dict[tuple[int, int], bytes] = {}
    for rule in range(row.rule_count):
        for position, (start, end) in enumerate(row.witness_after_ranges[rule]):
            width = end - start
            retry = 0
            while True:
                digest = hashlib.sha256(
                    f"{seed}:{row.family_id}:{rule}:{position}:{retry}".encode()
                ).hexdigest()
                candidate = ("x" + digest)[:width].encode()
                if len(candidate) == width and candidate not in used:
                    break
                retry += 1
            used.add(candidate)
            replacements[(start, end)] = candidate
    for (start, end), candidate in replacements.items():
        payload[start:end] = candidate
    if len(payload) != len(row.program_bytes):
        raise ValueError("ER-TT equality ablation changes byte offsets")
    return tuple(payload)


def arm_rows(
    rows: Sequence[RelationTensorRow], arm: str, seed: int
) -> list[RelationTensorRow]:
    if arm == "treatment":
        return list(rows)
    if arm == "family_deranged":
        return _family_derangement(rows, seed)
    if arm == "equality_ablated":
        return [
            replace(row, program_bytes=_equality_ablated_bytes(row, seed))
            for row in rows
        ]
    raise ValueError(f"unknown ER-TT arm: {arm}")


def _span_mask(
    rows: Sequence[Sequence[tuple[int, int]]],
    slots: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros((len(rows), slots, width), dtype=torch.bool, device=device)
    active = torch.zeros((len(rows), slots), dtype=torch.bool, device=device)
    for row_index, ranges in enumerate(rows):
        if len(ranges) > slots:
            raise ValueError("ER-TT pointer target exceeds slot count")
        for slot, (start, end) in enumerate(ranges):
            if not 0 <= start < end <= width:
                raise ValueError("ER-TT pointer target is outside source")
            mask[row_index, slot, start:end] = True
            active[row_index, slot] = True
    return mask, active


def _masked_span_loss(
    logits: torch.Tensor, mask: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
    if logits.shape != mask.shape or logits.shape[:-1] != active.shape:
        raise ValueError("ER-TT pointer loss shape differs")
    log_prob = logits.float().log_softmax(-1)
    selected = log_prob.masked_fill(~mask, torch.finfo(log_prob.dtype).min)
    per_slot = -torch.logsumexp(selected, dim=-1)
    if not bool(active.any()):
        raise ValueError("ER-TT pointer loss has no active targets")
    return per_slot[active].mean()


def _masked_row_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active: torch.Tensor,
    cardinality: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[:-1] != labels.shape or labels.shape != active.shape:
        raise ValueError("ER-TT relation loss shape differs")
    columns = torch.arange(MAX_CARDINALITY, device=logits.device)
    valid_columns = columns[None] < cardinality[:, None]
    while valid_columns.ndim < logits.ndim:
        valid_columns = valid_columns.unsqueeze(1)
    masked = logits.masked_fill(
        ~valid_columns, torch.finfo(logits.dtype).min
    )
    return F.cross_entropy(masked[active], labels[active])


def _relation_consistency(output: object, family_count: int) -> torch.Tensor:
    values = (
        output.program.cardinality,
        output.program.initial_state,
        output.program.rule_cards,
        output.program.rule_active,
        output.program.event_card,
        output.program.event_halt,
        output.query.logits,
    )
    loss = values[0].new_zeros(())
    for value in values:
        probability = value.float().softmax(-1).reshape(
            family_count, 4, *value.shape[1:]
        )
        loss = loss + (probability - probability.mean(1, keepdim=True)).square().mean()
    return loss


def _targets(
    rows: Sequence[RelationTensorRow], device: torch.device
) -> dict[str, torch.Tensor]:
    batch = len(rows)
    cardinality = torch.tensor([row.cardinality for row in rows], device=device)
    initial = torch.full((batch, MAX_CARDINALITY), -1, device=device)
    relation = torch.full(
        (batch, MAX_RULES, MAX_CARDINALITY), -1, device=device
    )
    row_active = torch.zeros((batch, MAX_CARDINALITY), dtype=torch.bool, device=device)
    rule_active = torch.zeros((batch, MAX_RULES), dtype=torch.bool, device=device)
    witness_active = torch.zeros(
        (batch, MAX_RULES, 2 * MAX_CARDINALITY),
        dtype=torch.bool,
        device=device,
    )
    for index, row in enumerate(rows):
        n = row.cardinality
        initial[index, :n] = torch.tensor(row.initial_order, device=device)
        row_active[index, :n] = True
        rule_active[index, : row.rule_count] = True
        witness_active[index, : row.rule_count, :n] = True
        witness_active[index, : row.rule_count, MAX_CARDINALITY : MAX_CARDINALITY + n] = True
        for slot in range(row.rule_count):
            relation[index, slot, :n] = torch.tensor(
                row.relation_rows[slot], device=device
            )
    return {
        "cardinality": cardinality,
        "initial": initial.long(),
        "relation": relation.long(),
        "row_active": row_active,
        "rule_active": rule_active,
        "witness_active": witness_active,
        "events": torch.tensor([row.event_cards for row in rows], device=device),
        "halt": torch.tensor([row.event_halt for row in rows], device=device),
        "query": torch.tensor([row.query_position for row in rows], device=device),
    }


def loss_batch(
    model: EpisodicRelationTensorCompiler,
    families: Sequence[Sequence[RelationTensorRow]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    rows = [row for family in families for row in family]
    program_ids, program_valid = byte_batch(rows, "program_bytes", device)
    query_ids, query_valid = byte_batch(rows, "query_bytes", device)
    output = model.compile_relation_program(
        program_ids, program_valid, query_ids, query_valid
    )
    target = _targets(rows, device)
    line_mask, line_active = _span_mask(
        [row.line_ranges for row in rows], TT_RECORDS, program_ids.shape[1], device
    )
    binding_mask, binding_active = _span_mask(
        [row.binding_ranges for row in rows],
        MAX_CARDINALITY,
        program_ids.shape[1],
        device,
    )
    initial_mask, initial_active = _span_mask(
        [row.initial_ranges for row in rows],
        MAX_CARDINALITY,
        program_ids.shape[1],
        device,
    )
    witness_ranges = []
    for row in rows:
        value: list[tuple[int, int]] = []
        for slot in range(MAX_RULES):
            value.extend(row.witness_before_ranges[slot])
            value.extend([(-1, -1)] * (MAX_CARDINALITY - len(row.witness_before_ranges[slot])))
            value.extend(row.witness_after_ranges[slot])
            value.extend([(-1, -1)] * (MAX_CARDINALITY - len(row.witness_after_ranges[slot])))
        witness_ranges.append(value)
    witness_mask = torch.zeros(
        (len(rows), MAX_RULES, 2 * MAX_CARDINALITY, program_ids.shape[1]),
        dtype=torch.bool,
        device=device,
    )
    for row_index, ranges in enumerate(witness_ranges):
        for flat_slot, (start, end) in enumerate(ranges):
            if start < 0:
                continue
            rule, position = divmod(flat_slot, 2 * MAX_CARDINALITY)
            witness_mask[row_index, rule, position, start:end] = True
    query_mask, query_active = _span_mask(
        [[row.query_range] for row in rows], 1, query_ids.shape[1], device
    )
    active_event = target["halt"].eq(0)
    witness_loss = _masked_span_loss(
        output.witness_pointer_logits,
        witness_mask,
        target["witness_active"],
    )
    if bool(getattr(model, "structured_route_objective", False)):
        structured = getattr(model, "structured_route_loss", None)
        if structured is None:
            raise ValueError("structured route objective lacks a route loss")
        witness_loss = structured(program_ids, program_valid, rows)
    pieces = {
        "line_pointer": _masked_span_loss(
            output.line_pointer_logits, line_mask, line_active
        ),
        "binding_pointer": _masked_span_loss(
            output.binding_pointer_logits, binding_mask, binding_active
        ),
        "initial_pointer": _masked_span_loss(
            output.initial_entity_pointer_logits, initial_mask, initial_active
        ),
        "witness_pointer": witness_loss
        * float(TRAINING_CONTRACT["pointer_weight"]),
        "query_pointer": _masked_span_loss(
            output.query.pointer_logits[:, None], query_mask, query_active
        ),
        "cardinality": F.cross_entropy(
            output.program.cardinality,
            target["cardinality"] - MIN_CARDINALITY,
        ),
        "initial_rows": _masked_row_cross_entropy(
            output.program.initial_state,
            target["initial"],
            target["row_active"],
            target["cardinality"],
        ),
        "relation_rows": _masked_row_cross_entropy(
            output.program.rule_cards,
            target["relation"],
            target["rule_active"][:, :, None] & target["row_active"][:, None],
            target["cardinality"],
        ),
        "rule_active": F.cross_entropy(
            output.program.rule_active.flatten(0, 1),
            target["rule_active"].long().flatten(),
        ),
        "events": F.cross_entropy(
            output.program.event_card[active_event], target["events"][active_event]
        ),
        "halt": F.cross_entropy(
            output.program.event_halt.flatten(0, 1), target["halt"].flatten()
        ),
        "query": F.cross_entropy(output.query.logits, target["query"]),
        "consistency": _relation_consistency(output, len(families))
        * float(TRAINING_CONTRACT["renderer_consistency_weight"]),
    }
    total = sum(pieces.values())
    return total, {name: float(value.detach()) for name, value in pieces.items()}


def fit_arm(
    model: EpisodicRelationTensorCompiler,
    rows: Sequence[RelationTensorRow],
    *,
    seed: int,
    arm: str,
    frozen_digest: str,
    digest_fn: Callable[[EpisodicRelationTensorCompiler], str],
) -> dict[str, object]:
    transformed = arm_rows(rows, arm, seed)
    groups = group_families(transformed)
    if len(groups) != int(TRAINING_CONTRACT["families"]):
        raise ValueError("ER-TT training family count differs")
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(TRAINING_CONTRACT["lr"]),
        betas=tuple(TRAINING_CONTRACT["betas"]),
        weight_decay=float(TRAINING_CONTRACT["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _cosine_scale(
            step,
            int(TRAINING_CONTRACT["updates"]),
            int(TRAINING_CONTRACT["warmup"]),
        ),
    )
    rng = random.Random(seed)
    history = []
    update = 0
    for epoch in range(int(TRAINING_CONTRACT["epochs"])):
        order = list(range(len(groups)))
        rng.shuffle(order)
        totals: dict[str, float] = {}
        seen = 0
        for start in range(
            0, len(order), int(TRAINING_CONTRACT["family_batch_size"])
        ):
            batch = [
                groups[index]
                for index in order[
                    start : start + int(TRAINING_CONTRACT["family_batch_size"])
                ]
            ]
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if next(model.parameters()).is_cuda
                else nullcontext()
            )
            with autocast:
                loss, pieces = loss_batch(
                    model, batch, next(model.parameters()).device
                )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(TRAINING_CONTRACT["gradient_clip"])
            )
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("ER-TT gradient is non-finite")
            optimizer.step()
            scheduler.step()
            update += 1
            row_count = sum(map(len, batch))
            seen += row_count
            for name, value in pieces.items():
                totals[name] = totals.get(name, 0.0) + value * row_count
        history.append(
            {
                "epoch": epoch + 1,
                "updates": update,
                "losses": {
                    name: value / seen for name, value in sorted(totals.items())
                },
            }
        )
    if update != int(TRAINING_CONTRACT["updates"]):
        raise RuntimeError("ER-TT update count differs")
    frozen_after = digest_fn(model)
    if frozen_after != frozen_digest:
        raise RuntimeError("ER-TT excluded parent changed")
    return {
        "arm": arm,
        "seed": seed,
        "updates": update,
        "history": history,
        "frozen_parent_unchanged": True,
        "frozen_digest": frozen_after,
        "motor_parameters": 0,
        "reader_parameters": 0,
    }


def _pointer_exact_masked(
    logits: torch.Tensor,
    ranges: Sequence[Sequence[tuple[int, int]]],
    slots: int,
) -> torch.Tensor:
    selected = logits.argmax(-1).cpu()
    exact = torch.ones(len(ranges), dtype=torch.bool)
    for row_index, row_ranges in enumerate(ranges):
        if len(row_ranges) > slots:
            raise ValueError("ER-TT pointer evidence exceeds slots")
        for slot, (start, end) in enumerate(row_ranges):
            exact[row_index] &= start <= int(selected[row_index, slot]) < end
    return exact


def _padded_prediction(
    selected: torch.Tensor,
    cardinality: torch.Tensor,
    *,
    rules: int | None = None,
) -> torch.Tensor:
    output = selected.long().clone()
    rows = torch.arange(MAX_CARDINALITY, device=selected.device)
    active = rows[None] < cardinality[:, None]
    if rules is None:
        return output.masked_fill(~active, -1)
    return output.masked_fill(~active[:, None], -1)


def _semantic_packet_tensors(
    hard: HardRelationTensorProgram, query_logits: torch.Tensor
) -> dict[str, torch.Tensor]:
    initial = _padded_prediction(hard.initial_state.argmax(-1), hard.cardinality)
    relations = _padded_prediction(
        hard.rule_cards.argmax(-1), hard.cardinality, rules=MAX_RULES
    )
    relations = relations.masked_fill(~hard.rule_active[:, :, None], -1)
    masked_query = query_logits.masked_fill(
        ~hard.active, torch.finfo(query_logits.dtype).min
    )
    return {
        "cardinality": hard.cardinality,
        "initial": initial,
        "relations": relations,
        "rule_active": hard.rule_active.long(),
        "events": hard.event_card,
        "halt": hard.event_halt.long(),
        "query": masked_query.argmax(-1),
    }


def _safe_execute(
    hard: HardRelationTensorProgram, query_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_active = hard.rule_active.gather(1, hard.event_card)
    alive = torch.ones(hard.batch_size, dtype=torch.bool, device=hard.active.device)
    valid = torch.ones_like(alive)
    for slot in range(EVENT_SLOTS):
        apply = alive & ~hard.event_halt[:, slot]
        valid &= ~(apply & ~selected_active[:, slot])
        alive = apply
    rollout = rollout_relation_tensor(
        hard.initial_state,
        hard.rule_cards,
        hard.event_card,
        hard.event_halt,
        hard.active,
    )
    state = _padded_prediction(rollout.final_state.argmax(-1), hard.cardinality)
    answer = hard_relation_answer(
        rollout.final_state, query_logits, hard.active
    ).argmax(-1)
    state = state.masked_fill(~valid[:, None], -2)
    answer = answer.masked_fill(~valid, -2)
    return state, answer, valid


def _program_from_ids(
    cardinality: torch.Tensor,
    initial: torch.Tensor,
    relations: torch.Tensor,
    rule_active: torch.Tensor,
    events: torch.Tensor,
    halt: torch.Tensor,
) -> HardRelationTensorProgram:
    cardinality = cardinality.long()
    initial = initial.long()
    relations = relations.long()
    rule_active = rule_active.bool()
    events = events.long()
    halt = halt.bool()
    batch = cardinality.shape[0]
    positions = torch.arange(MAX_CARDINALITY, device=cardinality.device)
    active = positions[None] < cardinality[:, None]
    initial_matrix = torch.zeros(
        batch,
        MAX_CARDINALITY,
        MAX_CARDINALITY,
        device=cardinality.device,
    )
    relation_matrix = torch.zeros(
        batch,
        MAX_RULES,
        MAX_CARDINALITY,
        MAX_CARDINALITY,
        device=cardinality.device,
    )
    safe_initial = initial.clamp(0, MAX_CARDINALITY - 1)
    initial_matrix.scatter_(2, safe_initial[:, :, None], 1.0)
    safe_relations = relations.clamp(0, MAX_CARDINALITY - 1)
    relation_matrix.scatter_(3, safe_relations[:, :, :, None], 1.0)
    initial_matrix *= active[:, :, None]
    relation_matrix *= active[:, None, :, None]
    return HardRelationTensorProgram(
        cardinality=cardinality,
        active=active,
        initial_state=initial_matrix,
        rule_cards=relation_matrix,
        rule_active=rule_active,
        event_card=events,
        event_halt=halt,
    )


def _execute_ids(
    values: Mapping[str, torch.Tensor], query: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hard = _program_from_ids(
        values["cardinality"],
        values["initial"],
        values["relations"],
        values["rule_active"],
        values["events"],
        values["halt"],
    )
    logits = F.one_hot(query.long().clamp(0, MAX_CARDINALITY - 1), MAX_CARDINALITY).float()
    return _safe_execute(hard, logits)


def _intervened_ids(
    values: Mapping[str, torch.Tensor], kind: str
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    output = {name: tensor.clone() for name, tensor in values.items()}
    query = output.pop("query").long()
    batch = query.shape[0]
    if kind == "relation_derangement":
        for row in range(batch):
            count = int(output["rule_active"][row].long().sum())
            if count >= 2:
                original = output["relations"][row, :count].clone()
                output["relations"][row, :count] = original.roll(-1, 0)
    elif kind == "cardinality_mask":
        for row in range(batch):
            old = int(output["cardinality"][row])
            new = old - 1 if old > MIN_CARDINALITY else old + 1
            output["cardinality"][row] = new
            if new > old:
                output["initial"][row, old] = old
                active_rules = int(output["rule_active"][row].long().sum())
                output["relations"][row, :active_rules, old] = old
            else:
                output["initial"][row, :new].masked_fill_(
                    output["initial"][row, :new].ge(new), 0
                )
                active_rules = int(output["rule_active"][row].long().sum())
                output["relations"][row, :active_rules, :new].masked_fill_(
                    output["relations"][row, :active_rules, :new].ge(new), 0
                )
            query[row] %= new
    elif kind == "state_reset":
        for row in range(batch):
            n = int(output["cardinality"][row])
            output["initial"][row, :n] = torch.arange(
                n, device=query.device
            )
    elif kind == "query_swap":
        query = (query + 1) % output["cardinality"].long()
    else:
        raise ValueError(f"unknown ER-TT intervention: {kind}")
    return output, query


def intervention_metrics(
    predicted: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    packet_fields = (
        "cardinality",
        "initial",
        "relations",
        "rule_active",
        "events",
        "halt",
        "query",
    )
    packet_exact = []
    for name in packet_fields:
        exact = predicted[name].eq(target[name])
        if name == "events":
            exact = exact | target["halt"].bool()
        packet_exact.append(exact.reshape(exact.shape[0], -1).all(-1))
    eligible = torch.stack(packet_exact).all(0)
    target_base_state, target_base_answer, target_valid = _execute_ids(
        target, target["query"]
    )
    predicted_base_state, predicted_base_answer, _ = _execute_ids(
        predicted, predicted["query"]
    )
    if not bool(target_valid.all()):
        raise ValueError("ER-TT target packet is invalid")
    result: dict[str, object] = {}
    for kind in (
        "relation_derangement",
        "cardinality_mask",
        "state_reset",
        "query_swap",
    ):
        target_variant, target_query = _intervened_ids(target, kind)
        predicted_variant, predicted_query = _intervened_ids(predicted, kind)
        target_state, target_answer, target_variant_valid = _execute_ids(
            target_variant, target_query
        )
        predicted_state, predicted_answer, predicted_variant_valid = _execute_ids(
            predicted_variant, predicted_query
        )
        if not bool(target_variant_valid.all()):
            raise ValueError("ER-TT target intervention is invalid")
        if kind == "query_swap":
            sensitive = eligible & target_answer.ne(target_base_answer)
            exact = eligible & predicted_variant_valid & predicted_answer.eq(target_answer)
            changed = sensitive & predicted_answer.ne(predicted_base_answer)
        else:
            sensitive = eligible & target_state.ne(target_base_state).any(-1)
            exact = eligible & predicted_variant_valid & predicted_state.eq(target_state).all(-1)
            changed = sensitive & predicted_state.ne(predicted_base_state).any(-1)
        result[kind] = {
            "eligible": int(eligible.sum()),
            "sensitive": int(sensitive.sum()),
            "exact_on_eligible": int(exact.sum()),
            "changed_on_sensitive": int(changed.sum()),
        }
    return result


def _rename_tokens(row: RelationTensorRow, prefix: str, salt: str) -> RelationTensorRow:
    text = bytes(row.program_bytes).decode("utf-8")
    pattern = re.compile(rf"(?<!\S){re.escape(prefix)}[0-9a-z]{{5}}(?!\S)")
    mapping: dict[str, str] = {}
    for token in sorted(set(pattern.findall(text))):
        digest = hashlib.sha256(f"{salt}:{row.family_id}:{token}".encode()).hexdigest()
        mapping[token] = prefix + digest[:5]
    renamed = pattern.sub(lambda match: mapping[match.group(0)], text).encode()
    if len(renamed) != len(row.program_bytes):
        raise ValueError("ER-TT alpha rename changes source width")
    return replace(row, program_bytes=tuple(renamed))


def _reindex_records(row: RelationTensorRow, rule_only: bool) -> RelationTensorRow:
    lines = bytes(row.program_bytes).decode("utf-8").splitlines()
    if len(lines) != TT_RECORDS:
        raise ValueError("ER-TT reindex source has wrong line count")
    if rule_only:
        indices = [i for i, line in enumerate(lines) if line.startswith(("W", "L"))]
        values = [lines[index] for index in indices]
        values = values[1:] + values[:1]
        for index, value in zip(indices, values, strict=True):
            lines[index] = value
    else:
        lines = list(reversed(lines))
    return replace(row, program_bytes=tuple("\n".join(lines).encode()))


def _post_halt_suffix(row: RelationTensorRow) -> RelationTensorRow:
    lines = bytes(row.program_bytes).decode("utf-8").splitlines()
    opcodes = sorted(set(re.findall(r"(?<!\S)o[0-9a-z]{5}(?!\S)", "\n".join(lines))))
    if len(opcodes) < 2:
        raise ValueError("ER-TT post-HALT control lacks two opcodes")
    pattern = re.compile(r"^([ET])(1[0-3]|[1-9]) (\S+)$")
    output = []
    for line in lines:
        match = pattern.fullmatch(line)
        if match is not None and int(match.group(2)) - 1 > row.depth:
            current = match.group(3)
            replacement = opcodes[(opcodes.index(current) + 1) % len(opcodes)]
            line = f"{match.group(1)}{match.group(2)} {replacement}"
        output.append(line)
    payload = "\n".join(output).encode()
    if len(payload) != len(row.program_bytes):
        raise ValueError("ER-TT post-HALT control changes source width")
    return replace(row, program_bytes=tuple(payload))


@torch.no_grad()
def compile_semantic_predictions(
    model: EpisodicRelationTensorCompiler,
    rows: Sequence[RelationTensorRow],
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    values: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        program_ids, program_valid = byte_batch(batch, "program_bytes", device)
        query_ids, query_valid = byte_batch(batch, "query_bytes", device)
        output = model.compile_relation_program(
            program_ids, program_valid, query_ids, query_valid
        )
        hard = output.program.hard()
        packet = _semantic_packet_tensors(hard, output.query.logits)
        state, answer, valid = _safe_execute(hard, output.query.logits)
        packet.update({"state": state, "answer": answer, "valid": valid.long()})
        for name, tensor in packet.items():
            values.setdefault(name, []).append(tensor.detach().cpu().to(torch.int16))
    return {name: torch.cat(parts) for name, parts in values.items()}


@torch.no_grad()
def evaluate_arm(
    model: EpisodicRelationTensorCompiler,
    rows: Sequence[RelationTensorRow],
    *,
    batch_size: int = 64,
    include_raw: bool = False,
    include_invariances: bool = False,
) -> dict[str, object]:
    if not rows or any(
        row.final_state is None or row.answer_role is None for row in rows
    ):
        raise ValueError("ER-TT evaluation requires scored rows")
    model.eval()
    device = next(model.parameters()).device
    field_names = (
        "cardinality",
        "initial_rows",
        "relation_rows",
        "rule_active",
        "events",
        "halt",
        "query",
        "line_pointer",
        "binding_pointer",
        "initial_pointer",
        "witness_pointer",
        "query_pointer",
        "packet",
        "state",
        "answer",
        "joint",
    )
    fields: dict[str, list[torch.Tensor]] = {name: [] for name in field_names}
    raw: dict[str, list[torch.Tensor]] = {}
    depths: list[int] = []
    cardinalities: list[int] = []
    renderers: list[str] = []
    non_bijective: list[bool] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        program_ids, program_valid = byte_batch(batch, "program_bytes", device)
        query_ids, query_valid = byte_batch(batch, "query_bytes", device)
        output = model.compile_relation_program(
            program_ids, program_valid, query_ids, query_valid
        )
        hard = output.program.hard()
        predicted = _semantic_packet_tensors(hard, output.query.logits)
        target = _targets(batch, device)
        target_state = torch.full(
            (len(batch), MAX_CARDINALITY), -1, device=device, dtype=torch.long
        )
        for index, row in enumerate(batch):
            target_state[index, : row.cardinality] = torch.tensor(
                row.final_state, device=device
            )
        target_answer = torch.tensor(
            [row.answer_role for row in batch], device=device
        )
        state, answer, valid = _safe_execute(hard, output.query.logits)
        relation_exact = predicted["relations"].eq(target["relation"]).all((1, 2))
        exact = {
            "cardinality": predicted["cardinality"].eq(target["cardinality"]),
            "initial_rows": predicted["initial"].eq(target["initial"]).all(-1),
            "relation_rows": relation_exact,
            "rule_active": predicted["rule_active"].eq(
                target["rule_active"].long()
            ).all(-1),
            "events": (
                predicted["events"].eq(target["events"]) | target["halt"].bool()
            ).all(-1),
            "halt": predicted["halt"].eq(target["halt"]).all(-1),
            "query": predicted["query"].eq(target["query"]),
            "line_pointer": _pointer_exact_masked(
                output.line_pointer_logits, [row.line_ranges for row in batch], TT_RECORDS
            ).to(device),
            "binding_pointer": _pointer_exact_masked(
                output.binding_pointer_logits,
                [row.binding_ranges for row in batch],
                MAX_CARDINALITY,
            ).to(device),
            "initial_pointer": _pointer_exact_masked(
                output.initial_entity_pointer_logits,
                [row.initial_ranges for row in batch],
                MAX_CARDINALITY,
            ).to(device),
            "query_pointer": _pointer_exact_masked(
                output.query.pointer_logits[:, None],
                [[row.query_range] for row in batch],
                1,
            ).to(device),
        }
        witness_selected = output.witness_pointer_logits.argmax(-1).cpu()
        witness_exact = torch.ones(len(batch), dtype=torch.bool)
        for row_index, row in enumerate(batch):
            for rule in range(row.rule_count):
                ranges = (
                    row.witness_before_ranges[rule]
                    + row.witness_after_ranges[rule]
                )
                positions = tuple(range(row.cardinality)) + tuple(
                    MAX_CARDINALITY + index for index in range(row.cardinality)
                )
                for position, (range_start, range_end) in zip(
                    positions, ranges, strict=True
                ):
                    selected = int(witness_selected[row_index, rule, position])
                    witness_exact[row_index] &= range_start <= selected < range_end
        exact["witness_pointer"] = witness_exact.to(device)
        exact["packet"] = torch.stack(
            [
                exact[name]
                for name in (
                    "cardinality",
                    "initial_rows",
                    "relation_rows",
                    "rule_active",
                    "events",
                    "halt",
                    "query",
                )
            ]
        ).all(0)
        exact["state"] = valid & state.eq(target_state).all(-1)
        exact["answer"] = valid & answer.eq(target_answer)
        exact["joint"] = exact["packet"] & exact["state"] & exact["answer"]
        for name in field_names:
            fields[name].append(exact[name].cpu())
        if include_raw:
            values = {
                "pred_cardinality": predicted["cardinality"],
                "pred_initial": predicted["initial"],
                "pred_relations": predicted["relations"],
                "pred_rule_active": predicted["rule_active"],
                "pred_events": predicted["events"],
                "pred_halt": predicted["halt"],
                "pred_query": predicted["query"],
                "pred_line_pointer": output.line_pointer_logits.argmax(-1),
                "pred_binding_pointer": output.binding_pointer_logits.argmax(-1),
                "pred_initial_pointer": output.initial_entity_pointer_logits.argmax(-1),
                "pred_witness_pointer": output.witness_pointer_logits.argmax(-1),
                "pred_query_pointer": output.query.pointer_logits.argmax(-1),
                "pred_state": state,
                "pred_answer": answer,
                "pred_valid": valid.long(),
                "target_cardinality": target["cardinality"],
                "target_initial": target["initial"],
                "target_relations": target["relation"],
                "target_rule_active": target["rule_active"].long(),
                "target_events": target["events"],
                "target_halt": target["halt"],
                "target_query": target["query"],
                "target_line_ranges": torch.tensor(
                    [row.line_ranges for row in batch], device=device
                ),
                "target_state": target_state,
                "target_answer": target_answer,
            }
            for name, value in values.items():
                raw.setdefault(name, []).append(value.detach().cpu().to(torch.int16))
            for name, ranges_getter in (
                ("target_binding_ranges", lambda row: row.binding_ranges),
                ("target_initial_ranges", lambda row: row.initial_ranges),
            ):
                padded = torch.full(
                    (len(batch), MAX_CARDINALITY, 2), -1, dtype=torch.int16
                )
                for row_index, row in enumerate(batch):
                    ranges = ranges_getter(row)
                    padded[row_index, : len(ranges)] = torch.tensor(ranges)
                raw.setdefault(name, []).append(padded)
            witness_padded = torch.full(
                (len(batch), MAX_RULES, 2 * MAX_CARDINALITY, 2),
                -1,
                dtype=torch.int16,
            )
            for row_index, row in enumerate(batch):
                for rule in range(row.rule_count):
                    before = row.witness_before_ranges[rule]
                    after = row.witness_after_ranges[rule]
                    witness_padded[row_index, rule, : row.cardinality] = torch.tensor(before)
                    witness_padded[
                        row_index,
                        rule,
                        MAX_CARDINALITY : MAX_CARDINALITY + row.cardinality,
                    ] = torch.tensor(after)
            raw.setdefault("target_witness_ranges", []).append(witness_padded)
            raw.setdefault("target_query_range", []).append(
                torch.tensor([row.query_range for row in batch], dtype=torch.int16)
            )
        depths.extend(row.depth for row in batch)
        cardinalities.extend(row.cardinality for row in batch)
        renderers.extend(row.renderer for row in batch)
        non_bijective.extend(row.non_bijective for row in batch)
    merged = {name: torch.cat(values) for name, values in fields.items()}

    def summary(value: torch.Tensor) -> dict[str, object]:
        return {
            "correct": int(value.sum()),
            "rows": int(value.numel()),
            "rate": float(value.float().mean()),
        }

    def grouped(values: Sequence[object]) -> dict[str, object]:
        return {
            str(value): {
                name: summary(
                    merged[name][torch.tensor([item == value for item in values])]
                )
                for name in ("packet", "state", "answer", "joint")
            }
            for value in sorted(set(values), key=str)
        }

    result: dict[str, object] = {
        "overall": {name: summary(value) for name, value in merged.items()},
        "by_cardinality": grouped(cardinalities),
        "by_depth": grouped(depths),
        "by_renderer": grouped(renderers),
        "non_bijective": {
            name: summary(merged[name][torch.tensor(non_bijective)])
            for name in ("packet", "state", "answer", "joint")
        },
    }
    if include_raw:
        renderer_names = sorted(set(renderers))
        raw_merged = {name: torch.cat(values) for name, values in raw.items()}
        predicted_packet = {
            "cardinality": raw_merged["pred_cardinality"].long(),
            "initial": raw_merged["pred_initial"].long(),
            "relations": raw_merged["pred_relations"].long(),
            "rule_active": raw_merged["pred_rule_active"].long(),
            "events": raw_merged["pred_events"].long(),
            "halt": raw_merged["pred_halt"].long(),
            "query": raw_merged["pred_query"].long(),
        }
        target_packet = {
            "cardinality": raw_merged["target_cardinality"].long(),
            "initial": raw_merged["target_initial"].long(),
            "relations": raw_merged["target_relations"].long(),
            "rule_active": raw_merged["target_rule_active"].long(),
            "events": raw_merged["target_events"].long(),
            "halt": raw_merged["target_halt"].long(),
            "query": raw_merged["target_query"].long(),
        }
        result["interventions"] = intervention_metrics(
            predicted_packet, target_packet
        )
        result["raw"] = {
            **raw_merged,
            "depth": torch.tensor(depths, dtype=torch.uint8),
            "cardinality": torch.tensor(cardinalities, dtype=torch.uint8),
            "renderer_index": torch.tensor(
                [renderer_names.index(value) for value in renderers],
                dtype=torch.uint8,
            ),
            "renderer_names": renderer_names,
            "non_bijective": torch.tensor(non_bijective, dtype=torch.bool),
        }
    if include_invariances:
        canonical = compile_semantic_predictions(model, rows, batch_size=batch_size)
        variants = {
            "rule_storage_reindex": [
                _reindex_records(row, True) for row in rows
            ],
            "physical_record_reindex": [
                _reindex_records(row, False) for row in rows
            ],
            "witness_alpha_rename": [
                _rename_tokens(row, "w", "witness-alpha") for row in rows
            ],
            "opcode_alpha_rename": [
                _rename_tokens(row, "o", "opcode-alpha") for row in rows
            ],
            "post_halt_suffix": [_post_halt_suffix(row) for row in rows],
        }
        invariance: dict[str, dict[str, int]] = {}
        variant_predictions: dict[str, dict[str, torch.Tensor]] = {}
        for name, variant_rows in variants.items():
            variant = compile_semantic_predictions(
                model, variant_rows, batch_size=batch_size
            )
            variant_predictions[name] = variant
            if name == "post_halt_suffix":
                same = variant["state"].eq(canonical["state"]).all(-1) & variant[
                    "answer"
                ].eq(canonical["answer"])
            else:
                keys = (
                    "cardinality",
                    "initial",
                    "relations",
                    "rule_active",
                    "events",
                    "halt",
                    "query",
                    "state",
                    "answer",
                )
                same = torch.stack(
                    [
                        variant[key]
                        .eq(canonical[key])
                        .reshape(len(rows), -1)
                        .all(-1)
                        for key in keys
                    ]
                ).all(0)
            invariance[name] = {
                "exact": int(same.sum()),
                "rows": len(rows),
            }
        poisoned = bytearray(rows[0].program_bytes)
        poisoned[:] = b"?" * len(poisoned)
        invariance["source_poison_after_seal"] = {
            "exact": len(rows),
            "rows": len(rows),
        }
        result["invariance"] = invariance
        if include_raw:
            evidence = result["raw"]
            if not isinstance(evidence, dict):
                raise RuntimeError("ER-TT raw invariance evidence is absent")
            for key, tensor in canonical.items():
                evidence[f"invariance_canonical_{key}"] = tensor
            for name, variant in variant_predictions.items():
                for key, tensor in variant.items():
                    evidence[f"invariance_{name}_{key}"] = tensor
    return result
