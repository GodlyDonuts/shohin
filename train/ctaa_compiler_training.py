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
    "initial_state",
    "schedule",
    "query_position",
    "renderer",
}


@dataclass(frozen=True)
class TokenizedCompilerRow:
    program_ids: tuple[int, ...]
    query_ids: tuple[int, ...]
    action_cards: tuple[tuple[int, int, int], ...]
    initial_state: tuple[int, int, int]
    schedule: tuple[int, ...]
    query_position: int


@dataclass(frozen=True)
class CompilerBatch:
    program_ids: torch.Tensor
    query_ids: torch.Tensor
    action_cards: torch.Tensor
    initial_state: torch.Tensor
    schedule: torch.Tensor
    query_position: torch.Tensor


@dataclass(frozen=True)
class CompilerLossReceipt:
    total: torch.Tensor
    cards: torch.Tensor
    initial: torch.Tensor
    schedule: torch.Tensor
    query: torch.Tensor


def parse_train_row(value: object, tokenizer: Tokenizer, max_length: int) -> TokenizedCompilerRow:
    if not isinstance(value, dict) or set(value) != TRAIN_ROW_KEYS:
        raise ValueError("CTAA compiler training row schema differs")
    program = tokenizer.encode(value["program_source"]).ids
    query = tokenizer.encode(value["query_source"]).ids
    if not program or not query or len(program) > max_length or len(query) > max_length:
        raise ValueError("CTAA compiler training token length differs")
    cards = tuple(tuple(int(item) for item in row) for row in value["action_cards"])
    initial = tuple(int(item) for item in value["initial_state"])
    schedule = tuple(int(item) for item in value["schedule"])
    query_position = int(value["query_position"])
    if (
        len(cards) != 4
        or any(len(card) != 3 for card in cards)
        or len(initial) != 3
        or len(schedule) != 41
        or schedule.count(4) != 1
        or not 0 <= query_position < 3
    ):
        raise ValueError("CTAA compiler training target geometry differs")
    return TokenizedCompilerRow(
        program_ids=tuple(program),
        query_ids=tuple(query),
        action_cards=cards,  # type: ignore[arg-type]
        initial_state=initial,  # type: ignore[arg-type]
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
        initial_state=torch.tensor([row.initial_state for row in rows], dtype=torch.long, device=destination),
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
    initial = F.cross_entropy(program.initial_state.reshape(-1, 3), batch.initial_state.reshape(-1))
    schedule = F.cross_entropy(program.schedule.reshape(-1, 5), batch.schedule.reshape(-1))
    query = F.cross_entropy(query_logits, batch.query_position)
    return CompilerLossReceipt(
        total=cards + initial + schedule + query,
        cards=cards,
        initial=initial,
        schedule=schedule,
        query=query,
    )


@torch.inference_mode()
def compiler_batch_metrics(
    compiler: TrunkCausalCTAACompiler,
    batch: CompilerBatch,
) -> dict[str, float]:
    program = compiler.compile_program(batch.program_ids)
    query = compiler.compile_query(batch.query_ids)
    return {
        "cards_exact": float(program.action_cards.argmax(-1).eq(batch.action_cards).flatten(1).all(1).float().mean()),
        "initial_exact": float(program.initial_state.argmax(-1).eq(batch.initial_state).all(1).float().mean()),
        "schedule_exact": float(program.schedule.argmax(-1).eq(batch.schedule).all(1).float().mean()),
        "query_exact": float(query.argmax(-1).eq(batch.query_position).float().mean()),
    }
