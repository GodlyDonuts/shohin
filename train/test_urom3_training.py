from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from general_relational_object_machine import (
    MAX_EVENTS,
    MAX_OBJECTS,
    MAX_RELATION_EDGES,
    MAX_RULES,
    TrunkRelationalObjectCompiler,
)
from urom3_board import generate_rows
from urom3_training import (
    collate_urom_rows,
    parse_urom_row,
    urom_loss,
    urom_metrics,
)


class _TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width)

    def forward(
        self,
        hidden: torch.Tensor,
        _cosine: torch.Tensor,
        _sine: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return self.projection(hidden), None


class _TinyShohin(nn.Module):
    def __init__(self, *, vocabulary: int, sequence_length: int) -> None:
        super().__init__()
        width = 16
        self.cfg = SimpleNamespace(
            n_loop=1,
            d_model=width,
            vocab_size=vocabulary,
            seq_len=sequence_length,
        )
        self.tok = nn.Embedding(vocabulary, width)
        self.blocks = nn.ModuleList([_TinyBlock(width) for _ in range(4)])
        self.register_buffer("cos", torch.zeros(sequence_length, 1))
        self.register_buffer("sin", torch.zeros(sequence_length, 1))


def _tokenizer() -> Tokenizer:
    path = (
        __file__.rsplit("/train/", 1)[0]
        + "/artifacts/tokenizer/tokenizer.json"
    )
    return Tokenizer.from_file(path)


def _compiler(tokenizer: Tokenizer) -> TrunkRelationalObjectCompiler:
    return TrunkRelationalObjectCompiler(
        _TinyShohin(
            vocabulary=tokenizer.get_vocab_size(),
            sequence_length=2_048,
        ),
        compiler_width=32,
        compiler_heads=4,
        encoder_layers=1,
        encoder_feedforward=64,
        decoder_layers=1,
        decoder_feedforward=64,
        identity_width=12,
        early_layer=1,
        late_layer=2,
    )


def test_real_tokenizer_maps_every_semantic_occurrence_slot() -> None:
    tokenizer = _tokenizer()
    row = generate_rows(split="train", count=1, seed=9103)[0]
    parsed = parse_urom_row(
        row,
        tokenizer,
        expected_split="train",
        max_length=2_048,
    )
    rule_stride = 1 + 2 * MAX_RELATION_EDGES
    expected_slots = (
        1
        + MAX_OBJECTS
        + MAX_OBJECTS
        + MAX_RULES * rule_stride
        + MAX_EVENTS
    )
    assert len(parsed.pointer_targets) == expected_slots == 441

    cardinality = int(row["compiler_targets"]["cardinality"])  # type: ignore[index]
    rule_active = row["compiler_targets"]["rule_active"]  # type: ignore[index]
    evidence = row["evidence_spans"]["program"]  # type: ignore[index]
    expected_supervised = 2 * cardinality
    for rule, active in enumerate(rule_active):
        if active:
            edge_count = sum(
                key.startswith(f"rule.{rule}.edge.")
                and key.endswith(".source")
                for key in evidence
            )
            expected_supervised += 1 + 2 * edge_count
    expected_supervised += sum(
        f"event.{event}.opcode" in evidence for event in range(MAX_EVENTS)
    )
    assert sum(bool(target) for target in parsed.pointer_targets) == (
        expected_supervised
    )
    for indices in parsed.pointer_targets:
        assert all(0 <= index < len(parsed.program_ids) for index in indices)


def test_collation_loss_and_frozen_trunk_gradient_contract() -> None:
    torch.manual_seed(29)
    tokenizer = _tokenizer()
    parsed = [
        parse_urom_row(
            row,
            tokenizer,
            expected_split="train",
            max_length=2_048,
        )
        for row in generate_rows(split="train", count=2, seed=481)
    ]
    batch = collate_urom_rows(parsed)
    compiler = _compiler(tokenizer)

    assert batch.pointer_targets.shape[:2] == (
        2,
        compiler.object_slot_count,
    )
    assert batch.rule_edges.shape == (
        2,
        MAX_RULES,
        MAX_OBJECTS,
        MAX_OBJECTS,
    )
    assert batch.state_trajectory.shape == (
        2,
        MAX_EVENTS,
        MAX_OBJECTS,
        MAX_OBJECTS,
    )

    receipt = urom_loss(compiler, batch)
    losses = (
        receipt.total,
        receipt.pointer,
        receipt.cardinality,
        receipt.initial,
        receipt.rules,
        receipt.rule_active,
        receipt.event_rule,
        receipt.event_kind,
        receipt.query,
        receipt.trajectory,
        receipt.terminal,
        receipt.answer,
    )
    assert all(bool(torch.isfinite(value)) for value in losses)
    receipt.total.backward()

    assert all(
        parameter.grad is None
        for parameter in compiler.backbone.model.parameters()
    )
    trainable = [
        parameter
        for parameter in compiler.parameters()
        if parameter.requires_grad
    ]
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in trainable
    )

    metrics = urom_metrics(compiler, batch)
    assert {
        "cardinality_accuracy",
        "initial_accuracy",
        "rules_accuracy",
        "rule_active_accuracy",
        "event_rule_accuracy",
        "event_kind_accuracy",
        "query_accuracy",
        "trajectory_accuracy",
        "terminal_accuracy",
        "answer_accuracy",
        "joint_accuracy",
    } == set(metrics)
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
