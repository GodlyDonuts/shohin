"""Train-only row parsing, tokenization, and loss for the CTAA compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from ctaa_trunk_compiler import TrunkCausalCTAACompiler


TRAIN_ROW_KEYS = {
    "family_id",
    "program_source",
    "query_source",
    "action_cards",
    "opcode_to_card",
    "initial_state",
    "opcode_schedule",
    "schedule",
    "query_position",
    "renderer",
}


@dataclass(frozen=True)
class TokenizedCompilerRow:
    program_ids: tuple[int, ...]
    query_ids: tuple[int, ...]
    action_cards: tuple[tuple[int, int, int], ...]
    opcode_to_card: tuple[int, int, int, int]
    initial_state: tuple[int, int, int]
    opcode_schedule: tuple[int, ...]
    schedule: tuple[int, ...]
    query_position: int


@dataclass(frozen=True)
class CompilerBatch:
    program_ids: torch.Tensor
    query_ids: torch.Tensor
    action_cards: torch.Tensor
    opcode_to_card: torch.Tensor
    initial_state: torch.Tensor
    opcode_schedule: torch.Tensor
    schedule: torch.Tensor
    query_position: torch.Tensor


@dataclass(frozen=True)
class CompilerLossReceipt:
    total: torch.Tensor
    cards: torch.Tensor
    binding: torch.Tensor
    initial: torch.Tensor
    opcode_schedule: torch.Tensor
    query: torch.Tensor


def parse_train_row(value: object, tokenizer: Tokenizer, max_length: int) -> TokenizedCompilerRow:
    if not isinstance(value, dict) or set(value) != TRAIN_ROW_KEYS:
        raise ValueError("CTAA compiler training row schema differs")
    program = tokenizer.encode(value["program_source"]).ids
    query = tokenizer.encode(value["query_source"]).ids
    if not program or not query or len(program) > max_length or len(query) > max_length:
        raise ValueError("CTAA compiler training token length differs")
    cards = tuple(tuple(int(item) for item in row) for row in value["action_cards"])
    binding = tuple(int(item) for item in value["opcode_to_card"])
    initial = tuple(int(item) for item in value["initial_state"])
    opcode_schedule = tuple(int(item) for item in value["opcode_schedule"])
    schedule = tuple(int(item) for item in value["schedule"])
    query_position = int(value["query_position"])
    if (
        len(cards) != 4
        or any(len(card) != 3 for card in cards)
        or sorted(binding) != list(range(4))
        or len(initial) != 3
        or len(opcode_schedule) != 41
        or opcode_schedule.count(4) != 1
        or len(schedule) != 41
        or schedule.count(4) != 1
        or not 0 <= query_position < 3
    ):
        raise ValueError("CTAA compiler training target geometry differs")
    resolved = tuple(
        4 if event == 4 else binding[event] for event in opcode_schedule
    )
    if resolved != schedule:
        raise ValueError("CTAA compiler binding does not resolve to schedule")
    return TokenizedCompilerRow(
        program_ids=tuple(program),
        query_ids=tuple(query),
        action_cards=cards,  # type: ignore[arg-type]
        opcode_to_card=binding,  # type: ignore[arg-type]
        initial_state=initial,  # type: ignore[arg-type]
        opcode_schedule=opcode_schedule,
        schedule=schedule,
        query_position=query_position,
    )


def collate_compiler_rows(
    rows: Sequence[TokenizedCompilerRow],
    *,
    padding_id: int = 1,
    device: torch.device | None = None,
) -> CompilerBatch:
    if not rows:
        raise ValueError("CTAA compiler batch is empty")
    destination = device or torch.device("cpu")
    program_length = max(len(row.program_ids) for row in rows)
    query_length = max(len(row.query_ids) for row in rows)
    program = torch.full((len(rows), program_length), padding_id, dtype=torch.long)
    query = torch.full((len(rows), query_length), padding_id, dtype=torch.long)
    for index, row in enumerate(rows):
        program[index, : len(row.program_ids)] = torch.tensor(row.program_ids)
        query[index, : len(row.query_ids)] = torch.tensor(row.query_ids)
    return CompilerBatch(
        program_ids=program.to(destination),
        query_ids=query.to(destination),
        action_cards=torch.tensor([row.action_cards for row in rows], dtype=torch.long, device=destination),
        opcode_to_card=torch.tensor(
            [row.opcode_to_card for row in rows],
            dtype=torch.long,
            device=destination,
        ),
        initial_state=torch.tensor([row.initial_state for row in rows], dtype=torch.long, device=destination),
        opcode_schedule=torch.tensor(
            [row.opcode_schedule for row in rows],
            dtype=torch.long,
            device=destination,
        ),
        schedule=torch.tensor([row.schedule for row in rows], dtype=torch.long, device=destination),
        query_position=torch.tensor([row.query_position for row in rows], dtype=torch.long, device=destination),
    )


def compiler_loss(
    compiler: TrunkCausalCTAACompiler,
    batch: CompilerBatch,
) -> CompilerLossReceipt:
    program = compiler.compile_program(batch.program_ids)
    query_logits = compiler.compile_query(batch.query_ids)
    cards = F.cross_entropy(program.action_cards.reshape(-1, 3), batch.action_cards.reshape(-1))
    binding = F.cross_entropy(
        program.opcode_to_card.reshape(-1, 4),
        batch.opcode_to_card.reshape(-1),
    )
    initial = F.cross_entropy(program.initial_state.reshape(-1, 3), batch.initial_state.reshape(-1))
    opcode_schedule = F.cross_entropy(
        program.opcode_schedule.reshape(-1, 5),
        batch.opcode_schedule.reshape(-1),
    )
    query = F.cross_entropy(query_logits, batch.query_position)
    return CompilerLossReceipt(
        total=cards + binding + initial + opcode_schedule + query,
        cards=cards,
        binding=binding,
        initial=initial,
        opcode_schedule=opcode_schedule,
        query=query,
    )


@torch.inference_mode()
def compiler_batch_metrics(
    compiler: TrunkCausalCTAACompiler,
    batch: CompilerBatch,
) -> dict[str, float]:
    program = compiler.compile_program(batch.program_ids)
    query = compiler.compile_query(batch.query_ids)
    hard_binding = compiler.materialize_binding(program.opcode_to_card).long()
    hard_opcode_schedule = program.opcode_schedule.argmax(-1)
    resolved_schedule = hard_binding.gather(
        1, hard_opcode_schedule.clamp_max(3)
    )
    resolved_schedule = torch.where(
        hard_opcode_schedule.eq(4),
        hard_opcode_schedule,
        resolved_schedule,
    )
    return {
        "cards_exact": float(program.action_cards.argmax(-1).eq(batch.action_cards).flatten(1).all(1).float().mean()),
        "independent_binding_exact": float(
            hard_binding
            .eq(batch.opcode_to_card)
            .all(1)
            .float()
            .mean()
        ),
        "initial_exact": float(program.initial_state.argmax(-1).eq(batch.initial_state).all(1).float().mean()),
        "opcode_schedule_exact": float(
            program.opcode_schedule.argmax(-1)
            .eq(batch.opcode_schedule)
            .all(1)
            .float()
            .mean()
        ),
        "schedule_exact": float(
            resolved_schedule.eq(batch.schedule).all(1).float().mean()
        ),
        "query_exact": float(query.argmax(-1).eq(batch.query_position).float().mean()),
    }
