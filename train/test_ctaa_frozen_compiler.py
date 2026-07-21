from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ctaa_frozen_compiler import load_source_rows, token_batches
from run_ctaa_program_compiler import compile_program_sources
from run_ctaa_query_compiler import compile_query_sources


class FakeTokenizer:
    def encode(self, source: str):
        return SimpleNamespace(ids=[ord(character) % 13 + 2 for character in source])


class FakeCompiler:
    padding_id = 1
    model = SimpleNamespace(cfg=SimpleNamespace(seq_len=32))

    def compile_program(self, ids: torch.Tensor):
        rows = ids.shape[0]
        cards = torch.zeros((rows, 4, 3, 3))
        cards[..., 0] = 1
        initial = torch.zeros((rows, 3, 3))
        initial[..., 1] = 1
        schedule = torch.zeros((rows, 41, 5))
        schedule[..., 0] = 1
        schedule[:, 2, 4] = 2
        return SimpleNamespace(action_cards=cards, initial_state=initial, schedule=schedule)

    def compile_query(self, ids: torch.Tensor):
        logits = torch.zeros((ids.shape[0], 3))
        logits[:, 2] = 1
        return logits


def test_source_loader_rejects_any_target_or_outcome_field(tmp_path) -> None:
    clean = tmp_path / "clean.jsonl"
    clean.write_text('{"family_id":"f0","program_source":"abc"}\n')
    assert load_source_rows(clean, "program_source") == (["f0"], ["abc"])
    poisoned = tmp_path / "poisoned.jsonl"
    poisoned.write_text(
        '{"family_id":"f0","program_source":"abc","answer":2}\n'
    )
    with pytest.raises(ValueError, match="schema"):
        load_source_rows(poisoned, "program_source")


def test_source_batches_are_monotonic_right_padded() -> None:
    batches = list(
        token_batches(
            ["a", "abcd"],
            FakeTokenizer(),
            batch_size=2,
            max_length=32,
            padding_id=1,
            device=torch.device("cpu"),
        )
    )
    assert batches[0].shape == (2, 4)
    assert batches[0][0].tolist()[1:] == [1, 1, 1]


def test_program_and_query_compilers_materialize_raw_predictions() -> None:
    compiler = FakeCompiler()
    tokenizer = FakeTokenizer()
    cards, initial, schedule = compile_program_sources(
        compiler,
        tokenizer,
        ["abc", "def"],
        batch_size=1,
        device=torch.device("cpu"),
    )
    positions = compile_query_sources(
        compiler,
        tokenizer,
        ["q0", "q1"],
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert cards.shape == (2, 4, 3)
    assert initial.unique().tolist() == [1]
    assert schedule[:, 2].tolist() == [4, 4]
    assert positions.tolist() == [2, 2]

