from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer

from ctaa_compiler_training import (
    collate_compiler_rows,
    compiler_loss,
    parse_train_row,
)
from ctaa_trunk_compiler import TrunkCausalCTAACompiler
from model import GPT, GPTConfig


def tiny_compiler() -> TrunkCausalCTAACompiler:
    model = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=3,
            n_head=3,
            n_kv_head=1,
            d_model=24,
            d_ff=48,
            seq_len=64,
            zloss=0.0,
        )
    )
    return TrunkCausalCTAACompiler(
        model,
        compiler_width=24,
        heads=3,
        encoder_layers=1,
        encoder_feedforward=48,
        decoder_layers=1,
        decoder_feedforward=48,
        early_layer=1,
        late_layer=2,
    )


def tokenizer() -> Tokenizer:
    tokenizer = Tokenizer.from_str(
        '{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"A":2,"B":3,"Q":4},"unk_token":"[UNK]"}}'
    )
    return tokenizer


def row() -> dict[str, object]:
    return {
        "family_id": "T1",
        "program_source": "A B",
        "query_source": "Q",
        "action_cards": [[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1]],
        "opcode_to_card": [2, 0, 3, 1],
        "initial_state": [2, 0, 1],
        "opcode_schedule": [1, 3, 4, *([2] * 38)],
        "schedule": [0, 1, 4, *([3] * 38)],
        "query_position": 2,
        "renderer": 0,
    }


def test_parser_and_collator_reject_outcome_leak_and_right_pad() -> None:
    parsed = parse_train_row(row(), tokenizer(), 64)
    batch = collate_compiler_rows([parsed, parsed])
    assert batch.program_ids.shape == (2, 2)
    assert batch.query_ids.shape == (2, 1)
    assert batch.action_cards.shape == (2, 4, 3)
    assert batch.opcode_to_card.shape == (2, 4)
    assert batch.opcode_schedule.shape == (2, 41)
    leaked = {**row(), "answer": 1}
    with pytest.raises(ValueError, match="schema"):
        parse_train_row(leaked, tokenizer(), 64)


def test_compiler_loss_reaches_all_adapter_families() -> None:
    compiler = tiny_compiler()
    parsed = parse_train_row(row(), tokenizer(), 64)
    receipt = compiler_loss(compiler, collate_compiler_rows([parsed, parsed]))
    receipt.total.backward()
    assert torch.isfinite(receipt.total)
    families = {
        name.split(".", 1)[0]
        for name, parameter in compiler.named_parameters()
        if not name.startswith("model.") and parameter.grad is not None
    }
    assert families == {
        "early_memory_norm",
        "early_memory_projection",
        "late_memory_norm",
        "late_memory_projection",
        "memory_encoder",
        "program_queries",
        "query_query",
        "decoder",
        "decoder_norm",
        "tuple_head",
        "binding_head",
        "event_head",
        "query_head",
    }
    assert all(parameter.grad is None for parameter in compiler.model.parameters())
