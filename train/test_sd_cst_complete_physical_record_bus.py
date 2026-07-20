from __future__ import annotations

import torch

from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    MOTOR_PARAMETERS,
    READER_PARAMETERS,
)
from sd_cst_byte_addressed import BYTE_PAD
from sd_cst_complete_physical_record_bus import (
    CompletePhysicalRecordBusCompiler,
    complete_record_trainable_names,
    freeze_to_complete_record_bus,
    freeze_to_local_completion,
    local_completion_trainable_names,
)
from sd_cst_physical_record_bus import physical_record_trainable_names


def _small_model() -> CompletePhysicalRecordBusCompiler:
    return CompletePhysicalRecordBusCompiler(
        width=32,
        heads=4,
        encoder_layers=1,
        slot_layers=1,
        ff=64,
        slot_ff=64,
        max_bytes=256,
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
        max_line_bytes=32,
        sinkhorn_steps=4,
    )


def _batch(texts: list[bytes]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, texts))
    ids = torch.full((len(texts), width), BYTE_PAD, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for row, text in enumerate(texts):
        values = torch.tensor(list(text), dtype=torch.long)
        ids[row, : len(text)] = values
        valid[row, : len(text)] = True
    return ids, valid


def test_complete_local_front_end_shapes_and_probability_mass() -> None:
    torch.manual_seed(5)
    model = _small_model().eval()

    def forbidden_parent_path(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("global parent encoder was called")

    model._encode = forbidden_parent_path  # type: ignore[method-assign]
    model._orbit_encode = forbidden_parent_path  # type: ignore[method-assign]
    program = b"\n".join(f"record {index} zed".encode() for index in range(9))
    program_ids, program_valid = _batch([program])
    query_ids, query_valid = _batch([b"report slot 2"])
    with torch.no_grad():
        output = model.compile_program(program_ids, program_valid)
        query = model.compile_query_with_evidence(query_ids, query_valid)
    assert output.tape.initial_state.shape == (1, 6)
    assert output.tape.event_kind.shape == (1, 8, 3)
    assert output.tape.event_identity.shape == (1, 8, 3)
    assert output.tape.amount.shape == (1, 8, 2)
    assert output.binding_pointer_logits.shape == (1, 3, program_ids.shape[1])
    assert output.initial_entity_pointer_logits.shape == (
        1,
        3,
        program_ids.shape[1],
    )
    assert output.event_entity_pointer_logits.shape == (
        1,
        8,
        program_ids.shape[1],
    )
    assert query.query.logits.shape == (1, 3)
    assert query.pointer_logits.shape == query_ids.shape
    assert torch.allclose(
        output.binding_pointer_logits.exp().sum(-1),
        torch.ones(1, 3),
        atol=1e-5,
    )
    assert torch.allclose(
        output.initial_entity_pointer_logits.exp().sum(-1),
        torch.ones(1, 3),
        atol=1e-5,
    )


def test_completion_only_freeze_has_local_gradients_and_no_parent_gradients() -> None:
    torch.manual_seed(7)
    model = _small_model()
    local = local_completion_trainable_names(model)
    declared = freeze_to_local_completion(model)
    assert set(declared) == local
    assert local
    assert local.isdisjoint(physical_record_trainable_names(model))
    program = b"\n".join(f"record {index} zed".encode() for index in range(9))
    program_ids, program_valid = _batch([program])
    query_ids, query_valid = _batch([b"report slot 2"])
    output = model.compile_program(program_ids, program_valid)
    query = model.compile_query_with_evidence(query_ids, query_valid)
    loss = (
        output.tape.initial_state.square().sum()
        + output.tape.event_identity.square().sum()
        + output.binding_pointer_logits.square().mean()
        + query.query.logits.square().sum()
        + query.pointer_logits.square().mean()
    )
    loss.backward()
    assert any(
        parameter.grad is not None and bool(parameter.grad.ne(0).any())
        for name, parameter in model.named_parameters()
        if name in local
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name not in local
    )


def test_complete_freeze_and_exact_parameter_certificate() -> None:
    model = CompletePhysicalRecordBusCompiler()
    local = local_completion_trainable_names(model)
    record = physical_record_trainable_names(model)
    complete_names = complete_record_trainable_names(model)
    assert complete_names == local | record
    assert len(local) == 10
    assert len(complete_names) == 98
    actual = freeze_to_complete_record_bus(model)
    assert set(actual) == complete_names
    compiler = model.parameter_count()
    complete = BASE_PARAMETERS + compiler + MOTOR_PARAMETERS + READER_PARAMETERS
    assert compiler == 66_426_124
    assert sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name in local
    ) == 594_435
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 11_701_265
    assert complete == 191_527_829
    assert complete < 200_000_000
