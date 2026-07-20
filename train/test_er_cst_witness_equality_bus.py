from __future__ import annotations

import torch

from build_er_cst_witness_equality_board import TRAIN_SPLIT, build_board
from er_cst_fresh import byte_batch
from er_cst_rule_card_adapter import TiedRuleCardMotor
from er_cst_witness_equality import loss_batch, parse_row
from er_cst_witness_equality_bus import (
    WITNESS_POSITIONS,
    WitnessEqualityBusCompiler,
    freeze_to_witness_equality_adaptive,
)
from sd_cst import CategoricalStateReader
from sd_cst_binding_bus import PERMUTATIONS


def _small_model() -> WitnessEqualityBusCompiler:
    return WitnessEqualityBusCompiler(
        width=32,
        heads=4,
        encoder_layers=1,
        slot_layers=1,
        ff=64,
        slot_ff=64,
        max_bytes=512,
        fingerprint_width=16,
        orbit_width=32,
        orbit_heads=4,
        orbit_layers=1,
        orbit_ff=64,
        native_slot_layers=1,
        native_slot_heads=4,
        native_slot_ff=64,
        record_width=32,
        record_heads=4,
        record_layers=1,
        record_set_layers=1,
        record_ff=64,
        max_line_bytes=96,
        sinkhorn_steps=4,
        occurrence_ff=64,
        equality_width=16,
    )


def _pointer_logits(
    sources: list[bytes], ranges: list[list[tuple[int, int]]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(map(len, sources))
    ids = torch.full((len(sources), width), 256, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    logits = torch.full((len(sources), WITNESS_POSITIONS, width), -20.0)
    for row, source in enumerate(sources):
        ids[row, : len(source)] = torch.tensor(tuple(source))
        valid[row, : len(source)] = True
        for slot, (start, end) in enumerate(ranges[row]):
            logits[row, slot, start:end] = 20.0
    return ids, valid, logits


def test_finite_equality_assignment_recovers_all_s3_cards() -> None:
    torch.manual_seed(10)
    model = _small_model()
    sources = []
    ranges = []
    for card_id, permutation in enumerate(PERMUTATIONS):
        before = [f"a{card_id}{index}xyz" for index in range(3)]
        after = [before[index] for index in permutation]
        words = before + after
        source = " ".join(words).encode()
        row_ranges = []
        cursor = 0
        for word in words:
            start = source.index(word.encode(), cursor)
            row_ranges.append((start, start + len(word)))
            cursor = start + len(word)
        sources.append(source)
        ranges.append(row_ranges)
    ids, valid, logits = _pointer_logits(sources, ranges)
    witness = logits[:, None].expand(-1, 3, -1, -1).contiguous()
    cards, equality = model._equality_card_logits(ids, valid, witness)
    assert cards.shape == (6, 3, 6)
    assert equality.shape == (6, 3, 3, 3)
    prediction = cards.argmax(-1)
    expected = torch.arange(6)[:, None].expand(-1, 3)
    assert torch.equal(prediction, expected)


def test_witness_path_does_not_backpropagate_into_shared_records() -> None:
    torch.manual_seed(11)
    model = _small_model().train()
    batch = 2
    records = torch.randn(batch, 13, 32, requires_grad=True)
    memory = torch.randn(batch, 13, 96, 32, requires_grad=True)
    valid = torch.ones(batch, 13, 96, dtype=torch.bool)
    source_indices = torch.arange(96)[None, None].expand(batch, 13, -1)
    assignment = torch.eye(13)[None].expand(batch, -1, -1).requires_grad_()
    logits = model._global_witness_logits(
        records, memory, valid, source_indices, assignment, 96
    )
    (-logits[..., 0].mean()).backward()
    assert records.grad is None
    assert memory.grad is None
    assert assignment.grad is None
    assert model.er_witness_queries.grad is not None


def test_public_compiler_emits_witness_evidence_without_direct_card_head() -> None:
    splits, report = build_board(
        seed=5_113_901,
        families={
            "er_cst_train": 8,
            "er_cst_development": 8,
            "er_cst_confirmation": 8,
        },
    )
    assert report["all_gates_pass"] is True
    row = splits[TRAIN_SPLIT][0]
    model = _small_model().eval()
    program_ids, program_valid = byte_batch(
        [type("Row", (), {"program_bytes": tuple(row["program_text"].encode())})()],
        "program_bytes",
        torch.device("cpu"),
    )
    query_ids, query_valid = byte_batch(
        [type("Row", (), {"query_bytes": tuple(row["late_query_text"].encode())})()],
        "query_bytes",
        torch.device("cpu"),
    )
    output = model.compile_rule_program(
        program_ids, program_valid, query_ids, query_valid
    )
    assert output.witness_pointer_logits.shape[:3] == (1, 3, 6)
    assert output.equality_logits.shape == (1, 3, 3, 3)
    assert output.program.rule_cards.shape == (1, 3, 6)
    assert not hasattr(model, "er_rule_permutation_head")


def test_default_system_is_strictly_below_200m() -> None:
    model = WitnessEqualityBusCompiler()
    motor = TiedRuleCardMotor()
    reader = CategoricalStateReader()
    freeze_to_witness_equality_adaptive(model)
    complete = (
        125_081_664
        + model.parameter_count()
        + sum(parameter.numel() for parameter in motor.parameters())
        + sum(parameter.numel() for parameter in reader.parameters())
    )
    assert complete < 200_000_000
    assert 190_000_000 < complete


def test_real_family_loss_reaches_witness_and_equality_parameters() -> None:
    splits, report = build_board(
        seed=7_301_191,
        families={
            "er_cst_train": 8,
            "er_cst_development": 8,
            "er_cst_confirmation": 8,
        },
    )
    assert report["all_gates_pass"] is True
    family = [parse_row(value, TRAIN_SPLIT) for value in splits[TRAIN_SPLIT][:4]]
    model = _small_model().train()
    declared = set(freeze_to_witness_equality_adaptive(model))
    loss, pieces = loss_batch(model, [family], torch.device("cpu"))
    assert torch.isfinite(loss)
    assert "witness_pointer" in pieces
    loss.backward()
    leaked = [
        name
        for name, parameter in model.named_parameters()
        if name not in declared and parameter.grad is not None
    ]
    assert leaked == []
    assert model.er_witness_queries.grad is not None
    assert model.er_equality_projection.weight.grad is not None
    assert model.er_event_card_query_projection.weight.grad is not None
