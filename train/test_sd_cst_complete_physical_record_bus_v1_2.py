from __future__ import annotations

import torch

from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    MOTOR_PARAMETERS,
    READER_PARAMETERS,
)
from sd_cst_byte_addressed import BYTE_PAD
from sd_cst_complete_physical_record_bus_v1_2 import (
    CompletePhysicalRecordBusCompilerV1_2,
    freeze_to_occurrence_head,
    occurrence_head_trainable_names,
)


def _small_model() -> CompletePhysicalRecordBusCompilerV1_2:
    return CompletePhysicalRecordBusCompilerV1_2(
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
        occurrence_ff=64,
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


def test_v1_2_uses_only_local_program_and_query_paths() -> None:
    torch.manual_seed(3)
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
    assert output.binding_pointer_logits.shape == (1, 3, program_ids.shape[1])
    assert output.initial_entity_pointer_logits.shape == (
        1,
        3,
        program_ids.shape[1],
    )
    assert query.query.logits.shape == (1, 3)


def test_v1_2_freeze_and_gradients_are_exact() -> None:
    torch.manual_seed(5)
    model = _small_model()
    expected = occurrence_head_trainable_names(model)
    actual = freeze_to_occurrence_head(model)
    assert set(actual) == expected
    program = b"\n".join(f"record {index} zed".encode() for index in range(9))
    program_ids, program_valid = _batch([program])
    output = model.compile_program(program_ids, program_valid)
    loss = (
        output.binding_pointer_logits.square().mean()
        + output.initial_entity_pointer_logits.square().mean()
        + output.tape.initial_state.square().mean()
    )
    loss.backward()
    assert all(
        parameter.grad is not None and bool(parameter.grad.ne(0).any())
        for name, parameter in model.named_parameters()
        if name in expected
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name not in expected
    )


def test_v1_2_exact_parameter_certificate() -> None:
    model = CompletePhysicalRecordBusCompilerV1_2()
    freeze_to_occurrence_head(model)
    compiler = model.parameter_count()
    complete = BASE_PARAMETERS + compiler + MOTOR_PARAMETERS + READER_PARAMETERS
    assert compiler == 67_027_474
    assert sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ) == 601_350
    assert complete == 192_129_179
    assert complete < 200_000_000
