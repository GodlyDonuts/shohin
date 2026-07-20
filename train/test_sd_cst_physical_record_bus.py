from __future__ import annotations

import pytest
import torch

from pilot_sd_cst_physical_record_bus import (
    _load_trainable_state,
    _state_digest,
    _trainable_state,
)
from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    MOTOR_PARAMETERS,
    READER_PARAMETERS,
)
from sd_cst_byte_addressed import BYTE_PAD, PROGRAM_SLOTS
from sd_cst_physical_record_bus import (
    PhysicalRecordBusCompiler,
    freeze_to_physical_record_bus,
    physical_record_trainable_names,
)


def _small_model(*, constrained_assignment: bool = True) -> PhysicalRecordBusCompiler:
    return PhysicalRecordBusCompiler(
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
        sinkhorn_steps=16,
        constrained_assignment=constrained_assignment,
    )


def _batch(lines: list[bytes]) -> tuple[torch.Tensor, torch.Tensor]:
    source = b"\n".join(lines)
    ids = torch.full((1, len(source) + 5), BYTE_PAD, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    values = torch.tensor(list(source), dtype=torch.long)
    ids[0, : len(source)] = values
    valid[0, : len(source)] = True
    return ids, valid


def test_physical_line_pack_is_exact_and_padding_is_inert() -> None:
    model = _small_model()
    lines = [f"r{index} value".encode() for index in range(PROGRAM_SLOTS)]
    ids, valid = _batch(lines)
    local_ids, local_valid, source_indices, masks = model._pack_physical_lines(
        ids,
        valid,
    )
    assert masks.shape == (1, PROGRAM_SLOTS, ids.shape[1])
    assert local_ids.shape == (1, PROGRAM_SLOTS, model.max_line_bytes)
    assert torch.equal(local_valid.sum(-1), masks.sum(-1))
    for index, line in enumerate(lines):
        expected = line + (b"\n" if index < PROGRAM_SLOTS - 1 else b"")
        recovered = bytes(local_ids[0, index, local_valid[0, index]].tolist())
        assert recovered == expected
    assert torch.equal(
        local_ids.masked_select(~local_valid),
        torch.full_like(local_ids.masked_select(~local_valid), BYTE_PAD),
    )
    assert bool(source_indices.lt(ids.shape[1]).all())


def test_physical_line_contract_rejects_wrong_count_and_long_line() -> None:
    model = _small_model()
    ids, valid = _batch([b"short"] * (PROGRAM_SLOTS - 1))
    with pytest.raises(ValueError, match="exactly nine"):
        model._physical_line_masks(ids, valid)

    ids, valid = _batch([b"x" * 33] + [b"short"] * (PROGRAM_SLOTS - 1))
    with pytest.raises(ValueError, match="exceeds"):
        model._physical_line_masks(ids, valid)


def test_sinkhorn_and_greedy_assignments_obey_one_to_one_contract() -> None:
    torch.manual_seed(7)
    model = _small_model()
    logits = torch.randn(3, PROGRAM_SLOTS, PROGRAM_SLOTS)
    soft = model._soft_assignment(logits)
    assert torch.allclose(
        soft.sum(-1),
        torch.ones_like(soft.sum(-1)),
        atol=2e-4,
    )
    assert torch.allclose(
        soft.sum(-2),
        torch.ones_like(soft.sum(-2)),
        atol=2e-4,
    )
    hard = model._greedy_one_to_one(logits)
    assert torch.equal(hard.sum(-1), torch.ones_like(hard.sum(-1)))
    assert torch.equal(hard.sum(-2), torch.ones_like(hard.sum(-2)))


def test_independent_control_does_not_enforce_physical_exclusivity() -> None:
    model = _small_model(constrained_assignment=False).eval()
    logits = torch.zeros(1, PROGRAM_SLOTS, PROGRAM_SLOTS)
    logits[:, 0] = 10.0
    assignment = model._assignment(logits)
    assert torch.equal(
        assignment[:, 0].sum(-1),
        torch.full((1,), PROGRAM_SLOTS, dtype=assignment.dtype),
    )
    assert torch.equal(assignment.sum(1), torch.ones_like(assignment.sum(1)))


def test_matched_arm_initialization_is_exact_and_digest_bound() -> None:
    torch.manual_seed(17)
    treatment = _small_model(constrained_assignment=True)
    state = _trainable_state(treatment)
    digest = _state_digest(state)
    control = _small_model(constrained_assignment=False)
    _load_trainable_state(control, state)
    control_state = _trainable_state(control)
    assert _state_digest(control_state) == digest
    first = next(iter(sorted(control_state)))
    control_state[first].view(-1)[0].add_(1)
    assert _state_digest(control_state) != digest


def test_compile_shapes_gradients_freeze_and_parameter_cap() -> None:
    torch.manual_seed(11)
    model = _small_model()
    expected = physical_record_trainable_names(model)
    actual = freeze_to_physical_record_bus(model)
    assert set(actual) == expected
    assert all(
        parameter.requires_grad == (name in expected)
        for name, parameter in model.named_parameters()
    )
    ids, valid = _batch([f"record {index} zed".encode() for index in range(9)])
    output = model.compile_program(ids, valid)
    assert output.tape.initial_state.shape == (1, 6)
    assert output.tape.event_kind.shape == (1, 8, 3)
    assert output.tape.event_identity.shape == (1, 8, 3)
    assert output.tape.amount.shape == (1, 8, 2)
    assert output.line_pointer_logits.shape == (1, 9, ids.shape[1])
    assert output.event_entity_pointer_logits.shape == (1, 8, ids.shape[1])
    loss = (
        output.tape.event_kind.sum()
        + output.tape.event_identity.sum()
        + output.tape.amount.sum()
        + output.line_pointer_logits.sum()
    )
    loss.backward()
    assert expected
    assert any(
        parameter.grad is not None and bool(parameter.grad.ne(0).any())
        for name, parameter in model.named_parameters()
        if name in expected
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name not in expected
    )

    default = PhysicalRecordBusCompiler()
    compiler = default.parameter_count()
    complete = BASE_PARAMETERS + compiler + MOTOR_PARAMETERS + READER_PARAMETERS
    trainable = sum(
        parameter.numel()
        for name, parameter in default.named_parameters()
        if name in physical_record_trainable_names(default)
    )
    assert compiler == 65_831_689
    assert trainable == 11_106_830
    assert complete == 190_933_394
    assert complete < 200_000_000
