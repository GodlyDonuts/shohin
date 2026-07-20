from __future__ import annotations

import torch

from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    MOTOR_PARAMETERS,
    READER_PARAMETERS,
)
from pilot_sd_cst_renderer_native_program import frozen_state_digest
from sd_cst_renderer_native_program import (
    RendererNativeProgramCompiler,
    freeze_to_renderer_native_joint,
    freeze_to_renderer_native_program,
    renderer_native_joint_trainable_names,
    renderer_native_program_trainable_names,
)


def test_renderer_native_program_shapes_freeze_and_budget() -> None:
    torch.manual_seed(7)
    model = RendererNativeProgramCompiler()
    ids = torch.randint(0, 128, (2, 96), dtype=torch.long)
    ids[:, [11, 21, 31, 41, 51, 61, 71, 81]] = 10
    valid = torch.ones_like(ids, dtype=torch.bool)
    output = model.compile_program(ids, valid)
    query = model.compile_query_with_evidence(ids, valid)
    assert output.tape.initial_state.shape == (2, 6)
    assert output.tape.event_kind.shape == (2, 8, 3)
    assert output.tape.event_identity.shape == (2, 8, 3)
    assert output.tape.amount.shape == (2, 8, 2)
    assert output.line_pointer_logits.shape == (2, 9, 96)
    assert output.event_entity_pointer_logits.shape == (2, 8, 96)
    assert query.query.logits.shape == (2, 3)

    names = renderer_native_program_trainable_names(model)
    frozen_before = frozen_state_digest(model, names)
    declared = freeze_to_renderer_native_program(model)
    assert set(declared) == names
    assert names
    assert all(
        parameter.requires_grad == (name in names)
        for name, parameter in model.named_parameters()
    )
    with torch.no_grad():
        name = next(iter(sorted(names)))
        dict(model.named_parameters())[name].view(-1)[0].add_(1)
    assert frozen_state_digest(model, names) == frozen_before

    complete = (
        BASE_PARAMETERS + model.parameter_count() + MOTOR_PARAMETERS + READER_PARAMETERS
    )
    assert 172_723_071 < complete < 200_000_000


def test_renderer_orbit_component_refactor_preserves_combined_encoding() -> None:
    torch.manual_seed(11)
    model = RendererNativeProgramCompiler().eval()
    ids = torch.randint(0, 128, (1, 64), dtype=torch.long)
    valid = torch.ones_like(ids, dtype=torch.bool)
    combined, orbit = model._encode_components(ids, valid)
    assert torch.equal(combined, model._encode(ids, valid))
    assert combined.shape == (1, 64, model.width)
    assert orbit.shape == (1, 64, model.orbit_width)


def test_renderer_native_joint_unfreezes_only_shared_memory_and_decoder() -> None:
    model = RendererNativeProgramCompiler()
    declared = renderer_native_joint_trainable_names(model)
    frozen = freeze_to_renderer_native_joint(model)
    assert set(frozen) == declared
    assert renderer_native_program_trainable_names(model) < declared
    assert "orbit_byte_embedding.weight" in declared
    assert "orbit_position_embedding.weight" in declared
    assert "orbit_encoder.layers.0.self_attn.in_proj_weight" in declared
    assert "orbit_norm.weight" in declared
    assert "orbit_to_parent.weight" not in declared
    assert "orbit_residual_scale" not in declared
    assert "ordinal_head.weight" not in declared
    assert "binding_query_projection.weight" not in declared
