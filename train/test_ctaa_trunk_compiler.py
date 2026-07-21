from __future__ import annotations

from dataclasses import fields

import pytest
import torch
import torch.nn.functional as F

from ctaa_neural_core import ClosureTiedPointerCore
from ctaa_trunk_compiler import (
    HardCTAAPacket,
    HardCTAAQuery,
    TrunkCausalCTAACompiler,
)
from model import GPT, GPTConfig
from referential_literal_pointer_compiler import OrdinaryTokenTaggerCompiler


def tiny_model() -> GPT:
    return GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=3,
            n_head=3,
            n_kv_head=1,
            d_model=24,
            d_ff=48,
            seq_len=32,
            zloss=0.0,
        )
    )


def tiny_compiler(model: GPT | None = None) -> TrunkCausalCTAACompiler:
    return TrunkCausalCTAACompiler(
        model or tiny_model(),
        compiler_width=24,
        heads=3,
        encoder_layers=1,
        encoder_feedforward=48,
        decoder_layers=1,
        decoder_feedforward=48,
        early_layer=1,
        padding_id=1,
    )


def inputs() -> torch.Tensor:
    return torch.tensor(
        [
            [7, 9, 11, 13, 15, 1, 1, 1],
            [8, 10, 12, 14, 16, 18, 1, 1],
        ],
        dtype=torch.long,
    )


def test_compiler_emits_materialized_program_without_query_or_source() -> None:
    compiler = tiny_compiler()
    output = compiler.compile_program(inputs())
    assert output.action_cards.shape == (2, 4, 3, 3)
    assert output.initial_state.shape == (2, 3, 3)
    assert output.schedule.shape == (2, 41, 5)

    output.schedule.fill_(-20.0)
    output.schedule[:, :, 0] = 20.0
    output.schedule[:, 4, 0] = -20.0
    output.schedule[:, 4, 4] = 20.0
    packet = compiler.materialize_program(output)
    assert {field.name for field in fields(packet)} == {
        "action_cards",
        "initial_state",
        "schedule",
    }
    assert packet.bytes_per_row == 56
    assert all(value.dtype == torch.uint8 for value in packet.__dict__.values())


def test_late_query_is_a_separate_source_and_one_byte_commit() -> None:
    compiler = tiny_compiler()
    query_logits = compiler.compile_query(inputs())
    query = compiler.materialize_query(query_logits)
    assert query_logits.shape == (2, 3)
    assert query.position.shape == (2,)
    assert query.position.dtype == torch.uint8


def test_early_and_late_trunk_interventions_are_independent() -> None:
    torch.manual_seed(31)
    compiler = tiny_compiler().eval()
    bundle = compiler.encode_source(inputs())
    native = compiler.compile_program_from_residuals(bundle)
    zero_early = compiler.compile_program_from_residuals(
        bundle,
        intervention="zero_early",
    )
    with torch.no_grad():
        compiler.late_memory_projection.weight.normal_(std=0.02)
    late_native = compiler.compile_program_from_residuals(bundle)
    zero_late = compiler.compile_program_from_residuals(
        bundle,
        intervention="zero_late",
    )
    assert not torch.allclose(native.action_cards, zero_early.action_cards)
    assert not torch.allclose(late_native.schedule, zero_late.schedule)


def test_frozen_trunk_has_no_gradient_and_every_adapter_family_does() -> None:
    torch.manual_seed(41)
    compiler = tiny_compiler()
    program = compiler.compile_program(inputs())
    query = compiler.compile_query(inputs())
    loss = (
        F.cross_entropy(
            program.action_cards.reshape(-1, 3),
            torch.arange(program.action_cards.numel() // 3) % 3,
        )
        + F.cross_entropy(
            program.initial_state.reshape(-1, 3),
            torch.arange(program.initial_state.numel() // 3) % 3,
        )
        + F.cross_entropy(
            program.schedule.reshape(-1, 5),
            torch.arange(program.schedule.numel() // 5) % 5,
        )
        + F.cross_entropy(query, torch.tensor([0, 2]))
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in compiler.model.parameters())
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
        "event_head",
        "query_head",
    }


def test_qualified_memory_warm_start_is_exact_and_late_path_starts_zero() -> None:
    model = tiny_model()
    qualified = OrdinaryTokenTaggerCompiler(
        model,
        layer=1,
        width=24,
        heads=3,
        encoder_layers=1,
        ff=48,
    )
    compiler = tiny_compiler(tiny_model())
    loaded = compiler.initialize_qualified_memory(qualified.state_dict())
    assert loaded
    assert torch.equal(
        compiler.early_memory_projection.weight,
        qualified.memory_projection.weight,
    )
    assert all(
        torch.equal(
            compiler.state_dict()[name],
            qualified.state_dict()[name.replace("early_", "")],
        )
        for name in loaded
        if name.startswith("early_")
    )
    assert torch.count_nonzero(compiler.late_memory_projection.weight) == 0


def test_hard_program_executes_before_late_query_is_disclosed() -> None:
    core = ClosureTiedPointerCore(width=3)
    with torch.no_grad():
        core.address_logits.fill_(-20.0)
        core.address_logits.diagonal().fill_(20.0)
    packet = HardCTAAPacket(
        action_cards=torch.tensor(
            [[[1, 0, 2], [2, 2, 0], [0, 1, 1], [2, 0, 1]]],
            dtype=torch.uint8,
        ),
        initial_state=torch.tensor([[0, 1, 2]], dtype=torch.uint8),
        schedule=torch.tensor(
            [[0, 1, 4, *([0] * 38)]],
            dtype=torch.uint8,
        ),
    )
    trace = packet.execute(core)
    query = HardCTAAQuery(position=torch.tensor([1], dtype=torch.uint8))
    assert query.answer(trace).tolist() == [2]
    assert trace.states[0, -1].tolist() == [2, 2, 1]


def test_compiler_derives_padding_mask_and_rejects_bad_donor() -> None:
    compiler = tiny_compiler()
    empty = torch.ones(2, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="empty row"):
        compiler.compile_program(empty)
    bundle = compiler.encode_source(inputs())
    with pytest.raises(ValueError, match="donor residual geometry"):
        compiler.compile_program_from_residuals(
            bundle,
            intervention="donor",
            donor=None,
        )
    interior_pad = inputs().clone()
    interior_pad[0, 2] = 1
    with pytest.raises(ValueError, match="monotonic right padding"):
        compiler.compile_program(interior_pad)
    mismatched = compiler.encode_source(inputs()[:, :6])
    with pytest.raises(ValueError, match="donor residual geometry"):
        compiler.compile_program_from_residuals(
            bundle,
            intervention="donor_early",
            donor=mismatched,
        )
