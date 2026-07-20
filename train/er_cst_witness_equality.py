"""Training and evaluation mechanics for the ER-CST witness-equality bus."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from build_er_cst_witness_equality_board import (
    BOARD_SCHEMA,
    PROTOCOL,
    TRAIN_SPLIT,
)
from er_cst_fresh import (
    _consistency,
    _cosine_scale,
    _pointer_exact,
    _span_loss,
    arm_rows,
    byte_batch,
    fit_certificates,
    group_families,
)
from er_cst_rule_card_adapter import (
    EVENT_SLOTS,
    RULE_CARD_COUNT,
    RULE_COUNT,
    TiedRuleCardMotor,
    rollout_rule_cards,
)
from er_cst_witness_equality_bus import WitnessEqualityBusCompiler
from pilot_sd_cst_byte_addressed import sha256_file
from sd_cst import CategoricalStateReader


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
    "witness_pointer_weight": 1.0,
    "motor_updates": 1_000,
    "motor_lr": 0.003,
    "reader_updates": 500,
    "reader_lr": 0.005,
}


@dataclass(frozen=True, slots=True)
class WitnessERRow:
    row_id: str
    family_id: str
    renderer: str
    split: str
    program_bytes: tuple[int, ...]
    query_bytes: tuple[int, ...]
    line_ranges: tuple[tuple[int, int], ...]
    binding_ranges: tuple[tuple[int, int], ...]
    initial_ranges: tuple[tuple[int, int], ...]
    witness_ranges: tuple[tuple[tuple[int, int], ...], ...]
    query_range: tuple[int, int]
    initial_state: int
    rule_cards: tuple[int, ...]
    event_cards: tuple[int, ...]
    event_halt: tuple[int, ...]
    query_position: int
    depth: int
    final_state: int | None
    answer_role: int | None


def _ranges(value: object, count: int, name: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"ER-CST witness {name} ranges differ")
    output = tuple((int(item[0]), int(item[1])) for item in value)
    if any(end <= start for start, end in output):
        raise ValueError(f"ER-CST witness {name} range is empty")
    return output


def parse_row(value: Mapping[str, object], split: str) -> WitnessERRow:
    if value.get("split") != split or value.get("protocol") != PROTOCOL:
        raise ValueError("ER-CST witness row split/protocol differs")
    target = value.get("compiler_targets")
    if not isinstance(target, Mapping):
        raise ValueError("ER-CST witness row lacks compiler targets")
    rules = sorted(target["rule_cards"], key=lambda item: int(item["slot"]))
    events = sorted(target["events"], key=lambda item: int(item["slot"]))
    if len(rules) != RULE_COUNT or len(events) != EVENT_SLOTS:
        raise ValueError("ER-CST witness target cardinality differs")
    before = target.get("witness_before_ranges")
    after = target.get("witness_after_ranges")
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("ER-CST witness ranges are absent")
    witness_ranges = tuple(
        _ranges(before[slot], 3, f"before-{slot}")
        + _ranges(after[slot], 3, f"after-{slot}")
        for slot in range(RULE_COUNT)
    )
    oracle = value.get("oracle")
    if split == TRAIN_SPLIT:
        if oracle is not None or value.get("supervision") != "compiler_fields_only":
            raise ValueError("ER-CST witness training row exposes outcome supervision")
    elif not isinstance(oracle, Mapping):
        raise ValueError("ER-CST witness scored row lacks oracle")
    return WitnessERRow(
        row_id=str(value["id"]),
        family_id=str(value["family_id"]),
        renderer=str(value["template_id"]),
        split=split,
        program_bytes=tuple(str(value["program_text"]).encode("utf-8")),
        query_bytes=tuple(str(value["late_query_text"]).encode("utf-8")),
        line_ranges=_ranges(target["line_ranges"], 13, "line"),
        binding_ranges=_ranges(target["binding_ranges"], 3, "binding"),
        initial_ranges=_ranges(target["initial_ranges"], 3, "initial"),
        witness_ranges=witness_ranges,
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
        raise ValueError("ER-CST witness board receipt differs")
    value["report_sha256"] = sha256_file(path)
    return value


def load_split(
    data_dir: Path,
    board: Mapping[str, object],
    *,
    filename: str,
    split: str,
    expected: int,
) -> list[WitnessERRow]:
    path = data_dir / filename
    if sha256_file(path) != board["files"][filename]["sha256"]:
        raise ValueError(f"ER-CST witness {split} hash differs")
    rows = [
        parse_row(json.loads(line), split)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != expected:
        raise ValueError(f"ER-CST witness {split} row count differs")
    return rows


def _span_mask(
    ranges: Sequence[Sequence[tuple[int, int]]],
    width: int,
    device: torch.device,
) -> torch.Tensor:
    slots = len(ranges[0])
    mask = torch.zeros((len(ranges), slots, width), dtype=torch.bool, device=device)
    for row_index, row_ranges in enumerate(ranges):
        if len(row_ranges) != slots:
            raise ValueError("ER-CST witness pointer cardinality differs")
        for slot, (start, end) in enumerate(row_ranges):
            mask[row_index, slot, start:end] = True
    if not bool(mask.any(-1).all()):
        raise ValueError("ER-CST witness pointer target is empty")
    return mask


def loss_batch(
    model: WitnessEqualityBusCompiler,
    families: Sequence[Sequence[WitnessERRow]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    rows = [row for family in families for row in family]
    program_ids, program_valid = byte_batch(rows, "program_bytes", device)
    query_ids, query_valid = byte_batch(rows, "query_bytes", device)
    output = model.compile_rule_program(program_ids, program_valid, query_ids, query_valid)
    line_mask = _span_mask([row.line_ranges for row in rows], program_ids.shape[1], device)
    binding_mask = _span_mask([row.binding_ranges for row in rows], program_ids.shape[1], device)
    initial_mask = _span_mask([row.initial_ranges for row in rows], program_ids.shape[1], device)
    witness_mask = _span_mask(
        [tuple(value for rule in row.witness_ranges for value in rule) for row in rows],
        program_ids.shape[1],
        device,
    ).reshape(len(rows), RULE_COUNT, 6, program_ids.shape[1])
    query_mask = _span_mask([[row.query_range] for row in rows], query_ids.shape[1], device)[:, 0]
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
        "initial_pointer": _span_loss(output.initial_entity_pointer_logits, initial_mask),
        "witness_pointer": _span_loss(
            output.witness_pointer_logits.flatten(1, 2), witness_mask.flatten(1, 2)
        ) * float(TRAINING_CONTRACT["witness_pointer_weight"]),
        "query_pointer": _span_loss(output.query_pointer_logits[:, None], query_mask[:, None]),
        "initial": F.cross_entropy(output.program.initial_state, labels["initial"]),
        "cards": F.cross_entropy(output.program.rule_cards.flatten(0, 1), labels["cards"].flatten()),
        "events": F.cross_entropy(output.program.event_card[active_card], labels["events"][active_card]),
        "halt": F.cross_entropy(output.program.event_halt.flatten(0, 1), labels["halt"].flatten()),
        "query": F.cross_entropy(output.query.logits, labels["query"]),
        "consistency": _consistency(output, len(families))
        * float(TRAINING_CONTRACT["renderer_consistency_weight"]),
    }
    total = sum(pieces.values())
    return total, {name: float(value.detach()) for name, value in pieces.items()}


def fit_arm(
    model: WitnessEqualityBusCompiler,
    motor: TiedRuleCardMotor,
    reader: CategoricalStateReader,
    rows: Sequence[WitnessERRow],
    *,
    seed: int,
    arm: str,
    frozen_digest: str,
    digest_fn: Callable[[WitnessEqualityBusCompiler], str],
) -> dict[str, object]:
    transformed = arm_rows(rows, arm, seed)
    groups = group_families(transformed)
    if len(groups) != int(TRAINING_CONTRACT["families"]):
        raise ValueError("ER-CST witness training family count differs")
    certificate = fit_certificates(motor, reader)
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
        for start in range(0, len(order), int(TRAINING_CONTRACT["family_batch_size"])):
            batch = [
                groups[index]
                for index in order[start : start + int(TRAINING_CONTRACT["family_batch_size"])]
            ]
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
                raise RuntimeError("ER-CST witness gradient is non-finite")
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
    if update != int(TRAINING_CONTRACT["updates"]):
        raise RuntimeError("ER-CST witness update count differs")
    frozen_after = digest_fn(model)
    if frozen_after != frozen_digest:
        raise RuntimeError("ER-CST witness excluded parent changed")
    return {
        "arm": arm,
        "seed": seed,
        "updates": update,
        "history": history,
        "certificate": certificate,
        "frozen_parent_unchanged": True,
        "frozen_digest": frozen_after,
    }


@torch.no_grad()
def evaluate_arm(
    model: WitnessEqualityBusCompiler,
    motor: TiedRuleCardMotor,
    reader: CategoricalStateReader,
    rows: Sequence[WitnessERRow],
    *,
    batch_size: int = 64,
    include_raw: bool = False,
) -> dict[str, object]:
    if not rows or any(row.final_state is None or row.answer_role is None for row in rows):
        raise ValueError("ER-CST witness evaluation requires scored rows")
    model.eval()
    motor.eval()
    reader.eval()
    device = next(model.parameters()).device
    field_names = (
        "initial", "cards", "events", "halt", "query", "line_pointer",
        "binding_pointer", "initial_pointer", "witness_pointer", "query_pointer",
        "packet", "state", "answer", "joint",
    )
    fields: dict[str, list[torch.Tensor]] = {name: [] for name in field_names}
    raw: dict[str, list[torch.Tensor]] = {}
    depths: list[int] = []
    renderers: list[str] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        program_ids, program_valid = byte_batch(batch, "program_bytes", device)
        query_ids, query_valid = byte_batch(batch, "query_bytes", device)
        output = model.compile_rule_program(program_ids, program_valid, query_ids, query_valid)
        hard = output.program.hard()
        query = output.query.hard()
        targets = {
            "initial": torch.tensor([row.initial_state for row in batch], device=device),
            "cards": torch.tensor([row.rule_cards for row in batch], device=device),
            "events": torch.tensor([row.event_cards for row in batch], device=device),
            "halt": torch.tensor([row.event_halt for row in batch], device=device),
            "query": torch.tensor([row.query_position for row in batch], device=device),
            "state": torch.tensor([row.final_state for row in batch], device=device),
            "answer": torch.tensor([row.answer_role for row in batch], device=device),
        }
        witness_ranges = [
            tuple(value for rule in row.witness_ranges for value in rule) for row in batch
        ]
        witness_exact = _pointer_exact(
            output.witness_pointer_logits.flatten(1, 2), witness_ranges
        ).all(-1)
        exact = {
            "initial": hard.initial_state.eq(targets["initial"]),
            "cards": hard.rule_cards.eq(targets["cards"]).all(-1),
            "events": (hard.event_card.eq(targets["events"]) | targets["halt"].bool()).all(-1),
            "halt": hard.event_halt.eq(targets["halt"]).all(-1),
            "query": query.position.long().eq(targets["query"]),
            "line_pointer": _pointer_exact(output.line_pointer_logits, [row.line_ranges for row in batch]).all(-1),
            "binding_pointer": _pointer_exact(output.binding_pointer_logits, [row.binding_ranges for row in batch]).all(-1),
            "initial_pointer": _pointer_exact(output.initial_entity_pointer_logits, [row.initial_ranges for row in batch]).all(-1),
            "witness_pointer": witness_exact,
            "query_pointer": _pointer_exact(output.query_pointer_logits[:, None], [[row.query_range] for row in batch]).all(-1),
        }
        exact["packet"] = torch.stack(
            [exact[name] for name in ("initial", "cards", "events", "halt", "query")]
        ).all(0)
        rollout = rollout_rule_cards(hard, motor)
        exact["state"] = rollout.final_state.eq(targets["state"])
        answer = reader(
            F.one_hot(rollout.final_state, RULE_CARD_COUNT).float(),
            F.one_hot(query.position.long(), 3).float(),
        ).argmax(-1)
        exact["answer"] = answer.eq(targets["answer"])
        exact["joint"] = exact["packet"] & exact["state"] & exact["answer"]
        for name in field_names:
            fields[name].append(exact[name].cpu())
        if include_raw:
            values = {
                "pred_initial": hard.initial_state,
                "pred_cards": hard.rule_cards,
                "pred_events": hard.event_card,
                "pred_halt": hard.event_halt,
                "pred_query": query.position.long(),
                "pred_line_pointer": output.line_pointer_logits.argmax(-1),
                "pred_binding_pointer": output.binding_pointer_logits.argmax(-1),
                "pred_initial_pointer": output.initial_entity_pointer_logits.argmax(-1),
                "pred_witness_pointer": output.witness_pointer_logits.argmax(-1),
                "pred_query_pointer": output.query_pointer_logits.argmax(-1),
                "pred_state": rollout.final_state,
                "pred_answer": answer,
                "target_initial": targets["initial"],
                "target_cards": targets["cards"],
                "target_events": targets["events"],
                "target_halt": targets["halt"],
                "target_query": targets["query"],
                "target_line_ranges": torch.tensor([row.line_ranges for row in batch], device=device),
                "target_binding_ranges": torch.tensor([row.binding_ranges for row in batch], device=device),
                "target_initial_ranges": torch.tensor([row.initial_ranges for row in batch], device=device),
                "target_witness_ranges": torch.tensor([row.witness_ranges for row in batch], device=device),
                "target_query_range": torch.tensor([row.query_range for row in batch], device=device),
                "target_state": targets["state"],
                "target_answer": targets["answer"],
            }
            for name, value in values.items():
                raw.setdefault(name, []).append(value.detach().cpu().to(torch.int16))
        depths.extend(row.depth for row in batch)
        renderers.extend(row.renderer for row in batch)
    merged = {name: torch.cat(values) for name, values in fields.items()}

    def summary(value: torch.Tensor) -> dict[str, object]:
        return {"correct": int(value.sum()), "rows": value.numel(), "rate": float(value.float().mean())}

    by_depth = {}
    by_renderer = {}
    for depth in sorted(set(depths)):
        mask = torch.tensor([value == depth for value in depths])
        by_depth[str(depth)] = {
            name: summary(merged[name][mask]) for name in ("packet", "state", "answer", "joint")
        }
    for renderer in sorted(set(renderers)):
        mask = torch.tensor([value == renderer for value in renderers])
        by_renderer[renderer] = {
            name: summary(merged[name][mask]) for name in ("packet", "state", "answer", "joint")
        }
    result: dict[str, object] = {
        "overall": {name: summary(value) for name, value in merged.items()},
        "by_depth": by_depth,
        "by_renderer": by_renderer,
    }
    if include_raw:
        renderer_names = sorted(set(renderers))
        result["raw"] = {
            **{name: torch.cat(values) for name, values in raw.items()},
            "depth": torch.tensor(depths, dtype=torch.uint8),
            "renderer_index": torch.tensor(
                [renderer_names.index(value) for value in renderers], dtype=torch.uint8
            ),
            "renderer_names": renderer_names,
        }
    return result
