"""Training and source-deleted evaluation mechanics for ER-CST v1.2."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from build_er_cst_fresh_board import (
    BOARD_SCHEMA,
    PROTOCOL,
    TRAIN_SPLIT,
)
from er_cst_rule_card_adapter import (
    EVENT_SLOTS,
    RULE_CARD_COUNT,
    RULE_COUNT,
    EpisodicRuleCardCompiler,
    TiedRuleCardMotor,
    rollout_rule_cards,
    rule_motor_certificate,
)
from pilot_sd_cst_byte_addressed import sha256_file
from sd_cst import CategoricalStateReader
from sd_cst_binding_bus import PERMUTATIONS
from sd_cst_byte_addressed import BYTE_PAD


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
    "motor_updates": 1_000,
    "motor_lr": 0.003,
    "reader_updates": 500,
    "reader_lr": 0.005,
}


@dataclass(frozen=True, slots=True)
class ERRow:
    row_id: str
    family_id: str
    renderer: str
    split: str
    program_bytes: tuple[int, ...]
    query_bytes: tuple[int, ...]
    line_ranges: tuple[tuple[int, int], ...]
    binding_ranges: tuple[tuple[int, int], ...]
    initial_ranges: tuple[tuple[int, int], ...]
    query_range: tuple[int, int]
    initial_state: int
    rule_cards: tuple[int, ...]
    event_cards: tuple[int, ...]
    event_halt: tuple[int, ...]
    query_position: int
    depth: int
    final_state: int | None
    answer_role: int | None


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def derived_seed(seed: int, label: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()[:8], "big"
    )


def _ranges(value: object, count: int, name: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"ER-CST {name} ranges differ")
    output = tuple((int(item[0]), int(item[1])) for item in value)
    if any(end <= start for start, end in output):
        raise ValueError(f"ER-CST {name} range is empty")
    return output


def parse_row(value: Mapping[str, object], split: str) -> ERRow:
    if value.get("split") != split or value.get("protocol") != PROTOCOL:
        raise ValueError("ER-CST row split/protocol differs")
    target = value.get("compiler_targets")
    if not isinstance(target, Mapping):
        raise ValueError("ER-CST row lacks compiler targets")
    rules = sorted(target["rule_cards"], key=lambda item: int(item["slot"]))
    events = sorted(target["events"], key=lambda item: int(item["slot"]))
    if len(rules) != RULE_COUNT or len(events) != EVENT_SLOTS:
        raise ValueError("ER-CST target cardinality differs")
    oracle = value.get("oracle")
    if split == TRAIN_SPLIT:
        if oracle is not None or value.get("supervision") != "compiler_fields_only":
            raise ValueError("ER-CST training row exposes outcome supervision")
    elif not isinstance(oracle, Mapping):
        raise ValueError("ER-CST scored row lacks oracle")
    return ERRow(
        row_id=str(value["id"]),
        family_id=str(value["family_id"]),
        renderer=str(value["template_id"]),
        split=split,
        program_bytes=tuple(str(value["program_text"]).encode("utf-8")),
        query_bytes=tuple(str(value["late_query_text"]).encode("utf-8")),
        line_ranges=_ranges(target["line_ranges"], 13, "line"),
        binding_ranges=_ranges(target["binding_ranges"], 3, "binding"),
        initial_ranges=_ranges(target["initial_ranges"], 3, "initial"),
        query_range=_ranges([target["query_range"]], 1, "query")[0],
        initial_state=int(target["initial_state_id"]),
        rule_cards=tuple(int(item["permutation_id"]) for item in rules),
        event_cards=tuple(int(item["card_slot"]) for item in events),
        event_halt=tuple(int(bool(item["halt"])) for item in events),
        query_position=int(target["query_position"]),
        depth=int(target["depth"]),
        final_state=None if oracle is None else int(oracle["final_state_id"]),
        answer_role=None if oracle is None else int(oracle["answer_role"]),
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
        raise ValueError("ER-CST board receipt differs")
    value["report_sha256"] = sha256_file(path)
    return value


def load_split(
    data_dir: Path,
    board: Mapping[str, object],
    *,
    filename: str,
    split: str,
    expected: int,
) -> list[ERRow]:
    path = data_dir / filename
    declared = board["files"][filename]["sha256"]
    if sha256_file(path) != declared:
        raise ValueError(f"ER-CST {split} hash differs")
    rows = [
        parse_row(json.loads(line), split)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != expected:
        raise ValueError(f"ER-CST {split} row count differs")
    return rows


def group_families(rows: Sequence[ERRow]) -> list[list[ERRow]]:
    grouped: dict[str, list[ERRow]] = {}
    for row in rows:
        grouped.setdefault(row.family_id, []).append(row)
    output = []
    for family in sorted(grouped):
        values = sorted(grouped[family], key=lambda row: row.renderer)
        if len(values) != 4 or len({row.renderer for row in values}) != 4:
            raise ValueError("ER-CST family views differ")
        output.append(values)
    return output


def _deranged_cards(row: ERRow, seed: int) -> tuple[int, ...]:
    rotations = ((1, 2, 0), (2, 0, 1))
    index = derived_seed(seed, f"{row.family_id}:derangement") % 2
    permutation = rotations[index]
    return tuple(row.rule_cards[position] for position in permutation)


def _equality_ablated_bytes(row: ERRow, seed: int) -> tuple[int, ...]:
    lines = bytes(row.program_bytes).decode("utf-8").splitlines()
    output = []
    for line in lines:
        tokens = line.split()
        if tokens and (tokens[0].startswith("W") or tokens[0].startswith("L")):
            if len(tokens) != 9 or tokens[5] not in {">", "=>"}:
                raise ValueError("ER-CST witness grammar differs")
            address = tokens[0][1:]
            if address not in {"1", "2", "3"}:
                raise ValueError("ER-CST witness address differs")
            for token_index in (2, 3, 4, 6, 7, 8):
                digest = hashlib.sha256(
                    f"{seed}:{row.family_id}:{address}:{token_index}".encode()
                ).hexdigest()
                replacement = f"x{digest[:8]}"
                if len(replacement) != len(tokens[token_index]):
                    raise ValueError("ER-CST equality ablation changes source width")
                tokens[token_index] = replacement
            line = " ".join(tokens)
        output.append(line)
    payload = "\n".join(output).encode("utf-8")
    if len(payload) != len(row.program_bytes):
        raise ValueError("ER-CST equality ablation changes program width")
    return tuple(payload)


def arm_rows(rows: Sequence[ERRow], arm: str, seed: int) -> list[ERRow]:
    if arm == "treatment":
        return list(rows)
    if arm == "family_deranged":
        return [replace(row, rule_cards=_deranged_cards(row, seed)) for row in rows]
    if arm == "equality_ablated":
        return [
            replace(row, program_bytes=_equality_ablated_bytes(row, seed))
            for row in rows
        ]
    raise ValueError(f"unknown ER-CST arm: {arm}")


def byte_batch(
    rows: Sequence[ERRow], field: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [getattr(row, field) for row in rows]
    width = max(map(len, sequences))
    ids = torch.full(
        (len(rows), width), BYTE_PAD, dtype=torch.long, device=device
    )
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(
            sequence, dtype=torch.long, device=device
        )
        valid[index, : len(sequence)] = True
    return ids, valid


def _span_mask(
    rows: Sequence[ERRow], field: str, slots: int, width: int, device: torch.device
) -> torch.Tensor:
    mask = torch.zeros((len(rows), slots, width), dtype=torch.bool, device=device)
    for row_index, row in enumerate(rows):
        ranges = getattr(row, field)
        if slots == 1 and len(ranges) == 2 and all(
            isinstance(value, int) for value in ranges
        ):
            ranges = (ranges,)
        for slot, (start, end) in enumerate(ranges):
            mask[row_index, slot, start:end] = True
    if not bool(mask.any(-1).all()):
        raise ValueError(f"ER-CST {field} target is empty")
    return mask


def _span_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape:
        raise ValueError("ER-CST pointer loss shape differs")
    log_prob = logits.float().log_softmax(-1)
    selected = log_prob.masked_fill(~mask, torch.finfo(log_prob.dtype).min)
    return -torch.logsumexp(selected, dim=-1).mean()


def _consistency(output: object, family_count: int) -> torch.Tensor:
    values = (
        output.program.initial_state,
        output.program.rule_cards,
        output.program.event_card,
        output.program.event_halt,
        output.query.logits,
    )
    loss = values[0].new_zeros(())
    for value in values:
        probability = value.float().softmax(-1).reshape(family_count, 4, *value.shape[1:])
        loss = loss + (probability - probability.mean(1, keepdim=True)).square().mean()
    return loss


def loss_batch(
    model: EpisodicRuleCardCompiler,
    families: Sequence[Sequence[ERRow]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    rows = [row for family in families for row in family]
    program_ids, program_valid = byte_batch(rows, "program_bytes", device)
    query_ids, query_valid = byte_batch(rows, "query_bytes", device)
    output = model.compile_rule_program(
        program_ids, program_valid, query_ids, query_valid
    )
    line_mask = _span_mask(rows, "line_ranges", 13, program_ids.shape[1], device)
    binding_mask = _span_mask(rows, "binding_ranges", 3, program_ids.shape[1], device)
    initial_mask = _span_mask(rows, "initial_ranges", 3, program_ids.shape[1], device)
    query_mask = _span_mask(rows, "query_range", 1, query_ids.shape[1], device)[:, 0]
    labels = {
        "initial": torch.tensor([row.initial_state for row in rows], device=device),
        "cards": torch.tensor([row.rule_cards for row in rows], device=device),
        "events": torch.tensor([row.event_cards for row in rows], device=device),
        "halt": torch.tensor([row.event_halt for row in rows], device=device),
        "query": torch.tensor([row.query_position for row in rows], device=device),
    }
    active_card = labels["halt"].eq(0)
    pieces = {
        "line": _span_loss(output.line_pointer_logits, line_mask),
        "binding": _span_loss(output.binding_pointer_logits, binding_mask),
        "initial_pointer": _span_loss(
            output.initial_entity_pointer_logits, initial_mask
        ),
        "query_pointer": _span_loss(
            output.query_pointer_logits[:, None], query_mask[:, None]
        ),
        "initial": F.cross_entropy(output.program.initial_state, labels["initial"]),
        "cards": F.cross_entropy(
            output.program.rule_cards.flatten(0, 1), labels["cards"].flatten()
        ),
        "events": F.cross_entropy(
            output.program.event_card[active_card], labels["events"][active_card]
        ),
        "halt": F.cross_entropy(
            output.program.event_halt.flatten(0, 1), labels["halt"].flatten()
        ),
        "query": F.cross_entropy(output.query.logits, labels["query"]),
        "consistency": _consistency(output, len(families)),
    }
    pieces["consistency"] = pieces["consistency"] * float(
        TRAINING_CONTRACT["renderer_consistency_weight"]
    )
    total = sum(pieces.values())
    return total, {name: float(value.detach()) for name, value in pieces.items()}


def _cosine_scale(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(progress * math.pi))


def fit_certificates(
    motor: TiedRuleCardMotor,
    reader: CategoricalStateReader,
) -> dict[str, object]:
    device = next(motor.parameters()).device
    motor.train()
    reader.train()
    motor.requires_grad_(True)
    reader.requires_grad_(True)
    state, card, target = rule_motor_certificate(device)
    optimizer = torch.optim.AdamW(
        motor.parameters(), lr=float(TRAINING_CONTRACT["motor_lr"]), weight_decay=0.0
    )
    for _ in range(int(TRAINING_CONTRACT["motor_updates"])):
        logits = motor(
            F.one_hot(state, RULE_CARD_COUNT).float(),
            F.one_hot(card, RULE_CARD_COUNT).float(),
        )
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        motor_logits = motor(
            F.one_hot(state, RULE_CARD_COUNT).float(),
            F.one_hot(card, RULE_CARD_COUNT).float(),
        )
    motor_exact = int(motor_logits.argmax(-1).eq(target).sum())

    state_ids = torch.arange(RULE_CARD_COUNT, device=device).repeat_interleave(3)
    query_ids = torch.arange(3, device=device).repeat(RULE_CARD_COUNT)
    answer = torch.tensor(
        [PERMUTATIONS[int(s)][int(q)] for s, q in zip(state_ids, query_ids, strict=True)],
        device=device,
    )
    optimizer = torch.optim.AdamW(
        reader.parameters(), lr=float(TRAINING_CONTRACT["reader_lr"]), weight_decay=0.0
    )
    for _ in range(int(TRAINING_CONTRACT["reader_updates"])):
        logits = reader(
            F.one_hot(state_ids, RULE_CARD_COUNT).float(),
            F.one_hot(query_ids, 3).float(),
        )
        loss = F.cross_entropy(logits, answer)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        reader_logits = reader(
            F.one_hot(state_ids, RULE_CARD_COUNT).float(),
            F.one_hot(query_ids, 3).float(),
        )
    reader_exact = int(reader_logits.argmax(-1).eq(answer).sum())
    if motor_exact != 36 or reader_exact != 18:
        raise RuntimeError("ER-CST motor/reader certificate failed")
    motor.requires_grad_(False)
    reader.requires_grad_(False)
    motor.eval()
    reader.eval()
    return {
        "motor_exact": motor_exact,
        "motor_cells": 36,
        "motor_loss": float(F.cross_entropy(motor_logits, target)),
        "reader_exact": reader_exact,
        "reader_cells": 18,
        "reader_loss": float(F.cross_entropy(reader_logits, answer)),
    }


def fit_arm(
    model: EpisodicRuleCardCompiler,
    motor: TiedRuleCardMotor,
    reader: CategoricalStateReader,
    rows: Sequence[ERRow],
    *,
    seed: int,
    arm: str,
    frozen_digest: str,
    digest_fn: Callable[[EpisodicRuleCardCompiler], str],
) -> dict[str, object]:
    transformed = arm_rows(rows, arm, seed)
    groups = group_families(transformed)
    if len(groups) != int(TRAINING_CONTRACT["families"]):
        raise ValueError("ER-CST training family count differs")
    certificate = fit_certificates(motor, reader)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(TRAINING_CONTRACT["lr"]),
        betas=tuple(TRAINING_CONTRACT["betas"]),
        weight_decay=float(TRAINING_CONTRACT["weight_decay"]),
    )
    total_updates = int(TRAINING_CONTRACT["updates"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _cosine_scale(step, total_updates, int(TRAINING_CONTRACT["warmup"])),
    )
    rng = random.Random(seed)
    history = []
    update = 0
    for epoch in range(int(TRAINING_CONTRACT["epochs"])):
        order = list(range(len(groups)))
        rng.shuffle(order)
        totals: dict[str, float] = {}
        seen = 0
        for start in range(0, len(order), int(TRAINING_CONTRACT["family_batch_size"])):
            batch = [groups[index] for index in order[start : start + int(TRAINING_CONTRACT["family_batch_size"])]]
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if next(model.parameters()).is_cuda
                else nullcontext()
            )
            with autocast:
                loss, pieces = loss_batch(model, batch, next(model.parameters()).device)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(TRAINING_CONTRACT["gradient_clip"])
            )
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("ER-CST gradient is non-finite")
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
                "losses": {name: value / seen for name, value in sorted(totals.items())},
            }
        )
    if update != total_updates:
        raise RuntimeError("ER-CST update count differs")
    frozen_after = digest_fn(model)
    if frozen_after != frozen_digest:
        raise RuntimeError("ER-CST excluded parent changed")
    return {
        "arm": arm,
        "seed": seed,
        "updates": update,
        "history": history,
        "certificate": certificate,
        "frozen_parent_unchanged": True,
        "frozen_digest": frozen_after,
    }


def trainable_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def load_trainable_state(
    module: torch.nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    expected = {
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    }
    if set(state) != expected:
        raise ValueError("ER-CST trainable state keys differ")
    current = module.state_dict()
    with torch.no_grad():
        for name, tensor in state.items():
            if tensor.shape != current[name].shape or tensor.dtype != current[name].dtype:
                raise ValueError(f"ER-CST trainable tensor differs: {name}")
            current[name].copy_(tensor)


def _pointer_exact(logits: torch.Tensor, ranges: Sequence[Sequence[tuple[int, int]]]) -> torch.Tensor:
    selected = logits.argmax(-1).cpu()
    exact = torch.ones(selected.shape[:2], dtype=torch.bool)
    for row_index, row_ranges in enumerate(ranges):
        for slot, (start, end) in enumerate(row_ranges):
            exact[row_index, slot] = start <= int(selected[row_index, slot]) < end
    return exact


@torch.no_grad()
def evaluate_arm(
    model: EpisodicRuleCardCompiler,
    motor: TiedRuleCardMotor,
    reader: CategoricalStateReader,
    rows: Sequence[ERRow],
    *,
    batch_size: int = 64,
    include_raw: bool = False,
) -> dict[str, object]:
    if not rows or any(row.final_state is None or row.answer_role is None for row in rows):
        raise ValueError("ER-CST evaluation requires scored rows")
    model.eval()
    motor.eval()
    reader.eval()
    device = next(model.parameters()).device
    fields: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "initial", "cards", "events", "halt", "query", "line_pointer",
            "binding_pointer", "initial_pointer", "query_pointer", "packet",
            "state", "answer", "joint",
        )
    }
    depth_values: list[int] = []
    renderer_values: list[str] = []
    raw: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "pred_initial",
            "pred_cards",
            "pred_events",
            "pred_halt",
            "pred_query",
            "pred_line_pointer",
            "pred_binding_pointer",
            "pred_initial_pointer",
            "pred_query_pointer",
            "pred_state",
            "pred_answer",
            "target_initial",
            "target_cards",
            "target_events",
            "target_halt",
            "target_query",
            "target_line_ranges",
            "target_binding_ranges",
            "target_initial_ranges",
            "target_query_range",
            "target_state",
            "target_answer",
        )
    }
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        program_ids, program_valid = byte_batch(batch, "program_bytes", device)
        query_ids, query_valid = byte_batch(batch, "query_bytes", device)
        output = model.compile_rule_program(program_ids, program_valid, query_ids, query_valid)
        hard = output.program.hard()
        query = output.query.hard()
        target_initial = torch.tensor([row.initial_state for row in batch], device=device)
        target_cards = torch.tensor([row.rule_cards for row in batch], device=device)
        target_events = torch.tensor([row.event_cards for row in batch], device=device)
        target_halt = torch.tensor([row.event_halt for row in batch], device=device)
        target_query = torch.tensor([row.query_position for row in batch], device=device)
        initial_exact = hard.initial_state.eq(target_initial)
        cards_exact = hard.rule_cards.eq(target_cards).all(-1)
        event_cells = hard.event_card.eq(target_events) | target_halt.bool()
        events_exact = event_cells.all(-1)
        halt_exact = hard.event_halt.eq(target_halt).all(-1)
        query_exact = query.position.long().eq(target_query)
        line_exact = _pointer_exact(output.line_pointer_logits, [row.line_ranges for row in batch]).all(-1)
        binding_exact = _pointer_exact(output.binding_pointer_logits, [row.binding_ranges for row in batch]).all(-1)
        initial_pointer_exact = _pointer_exact(output.initial_entity_pointer_logits, [row.initial_ranges for row in batch]).all(-1)
        query_pointer_exact = _pointer_exact(output.query_pointer_logits[:, None], [[row.query_range] for row in batch]).all(-1)
        packet = initial_exact & cards_exact & events_exact & halt_exact & query_exact
        rollout = rollout_rule_cards(hard, motor)
        target_state = torch.tensor([row.final_state for row in batch], device=device)
        state_exact = rollout.final_state.eq(target_state)
        answer = reader(
            F.one_hot(rollout.final_state, RULE_CARD_COUNT).float(),
            F.one_hot(query.position.long(), 3).float(),
        ).argmax(-1)
        target_answer = torch.tensor([row.answer_role for row in batch], device=device)
        answer_exact = answer.eq(target_answer)
        joint = packet & state_exact & answer_exact
        values = {
            "initial": initial_exact,
            "cards": cards_exact,
            "events": events_exact,
            "halt": halt_exact,
            "query": query_exact,
            "line_pointer": line_exact,
            "binding_pointer": binding_exact,
            "initial_pointer": initial_pointer_exact,
            "query_pointer": query_pointer_exact,
            "packet": packet,
            "state": state_exact,
            "answer": answer_exact,
            "joint": joint,
        }
        for name, value in values.items():
            fields[name].append(value.detach().cpu())
        raw_values = {
            "pred_initial": hard.initial_state,
            "pred_cards": hard.rule_cards,
            "pred_events": hard.event_card,
            "pred_halt": hard.event_halt,
            "pred_query": query.position.long(),
            "pred_line_pointer": output.line_pointer_logits.argmax(-1),
            "pred_binding_pointer": output.binding_pointer_logits.argmax(-1),
            "pred_initial_pointer": output.initial_entity_pointer_logits.argmax(-1),
            "pred_query_pointer": output.query_pointer_logits.argmax(-1),
            "pred_state": rollout.final_state,
            "pred_answer": answer,
            "target_initial": target_initial,
            "target_cards": target_cards,
            "target_events": target_events,
            "target_halt": target_halt,
            "target_query": target_query,
            "target_line_ranges": torch.tensor(
                [row.line_ranges for row in batch], device=device
            ),
            "target_binding_ranges": torch.tensor(
                [row.binding_ranges for row in batch], device=device
            ),
            "target_initial_ranges": torch.tensor(
                [row.initial_ranges for row in batch], device=device
            ),
            "target_query_range": torch.tensor(
                [row.query_range for row in batch], device=device
            ),
            "target_state": target_state,
            "target_answer": target_answer,
        }
        for name, value in raw_values.items():
            raw[name].append(value.detach().cpu().to(torch.int16))
        depth_values.extend(row.depth for row in batch)
        renderer_values.extend(row.renderer for row in batch)
    merged = {name: torch.cat(values) for name, values in fields.items()}

    def summary(value: torch.Tensor) -> dict[str, object]:
        return {"correct": int(value.sum()), "rows": value.numel(), "rate": float(value.float().mean())}

    by_depth = {}
    by_renderer = {}
    for depth in sorted(set(depth_values)):
        mask = torch.tensor([value == depth for value in depth_values])
        by_depth[str(depth)] = {name: summary(value[mask]) for name, value in merged.items() if name in {"packet", "state", "answer", "joint"}}
    for renderer_name in sorted(set(renderer_values)):
        mask = torch.tensor([value == renderer_name for value in renderer_values])
        by_renderer[renderer_name] = {name: summary(value[mask]) for name, value in merged.items() if name in {"packet", "state", "answer", "joint"}}
    result = {
        "overall": {name: summary(value) for name, value in merged.items()},
        "by_depth": by_depth,
        "by_renderer": by_renderer,
    }
    if include_raw:
        renderer_names = sorted(set(renderer_values))
        result["raw"] = {
            **{name: torch.cat(values) for name, values in raw.items()},
            "depth": torch.tensor(depth_values, dtype=torch.uint8),
            "renderer_index": torch.tensor(
                [renderer_names.index(value) for value in renderer_values],
                dtype=torch.uint8,
            ),
            "renderer_names": renderer_names,
        }
    return result
