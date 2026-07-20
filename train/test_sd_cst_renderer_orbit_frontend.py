from __future__ import annotations

import torch

from sd_cst_renderer_orbit_frontend import (
    RendererOrbitGroundedCompiler,
    freeze_to_renderer_orbit_front_end,
    renderer_orbit_trainable_names,
)


def _batch(texts: list[bytes]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, texts))
    ids = torch.full((len(texts), width), 256, dtype=torch.long)
    valid = torch.zeros((len(texts), width), dtype=torch.bool)
    for row, text in enumerate(texts):
        values = torch.tensor(list(text), dtype=torch.long)
        ids[row, : len(text)] = values
        valid[row, : len(text)] = True
    return ids, valid


def test_parameter_accounting_stays_below_global_cap() -> None:
    model = RendererOrbitGroundedCompiler()
    compiler = model.parameter_count()
    complete = 125_081_664 + compiler + 2_781 + 17_260
    assert compiler > 20_955_890
    assert complete < 200_000_000


def test_front_end_freeze_is_exact() -> None:
    model = RendererOrbitGroundedCompiler(
        orbit_width=64,
        orbit_heads=4,
        orbit_layers=2,
        orbit_ff=128,
    )
    expected = renderer_orbit_trainable_names(model)
    actual = freeze_to_renderer_orbit_front_end(model)
    assert set(actual) == set(expected)
    assert expected
    assert all(
        parameter.requires_grad == (name in expected)
        for name, parameter in model.named_parameters()
    )


def test_query_value_path_is_position_free_after_pointer_selection() -> None:
    torch.manual_seed(7)
    model = RendererOrbitGroundedCompiler(
        orbit_width=64,
        orbit_heads=4,
        orbit_layers=2,
        orbit_ff=128,
    ).eval()
    ids, valid = _batch([b"ask position 2", b"different words 2 please"])
    with torch.no_grad():
        first = model.compile_query_with_evidence(ids, valid)
        forced = torch.full_like(first.pointer_logits, -1000.0)
        forced[0, 13] = 1000.0
        forced[1, 16] = 1000.0
        weights = forced.softmax(-1).to(model.orbit_byte_embedding.weight.dtype)
        raw = model.ordinal_value_projection(model.orbit_byte_embedding(ids))
        selected = torch.einsum("bl,blw->bw", weights, raw)
        logits = model.ordinal_head(model.ordinal_norm(selected))
    assert torch.equal(ids[0, 13], ids[1, 16])
    assert torch.equal(logits[0], logits[1])


def test_query_pointer_and_program_outputs_have_declared_shapes() -> None:
    torch.manual_seed(11)
    model = RendererOrbitGroundedCompiler(
        orbit_width=64,
        orbit_heads=4,
        orbit_layers=2,
        orbit_ff=128,
    ).eval()
    query_ids, query_valid = _batch([b"report position 1", b"slot 3?"])
    program_ids, program_valid = _batch(
        [
            b"bindings a b c\nevent 1 stop",
            b"registry c b a\naction 1 stop",
        ]
    )
    with torch.no_grad():
        query = model.compile_query_with_evidence(query_ids, query_valid)
        program = model.compile_program(program_ids, program_valid)
    assert query.query.logits.shape == (2, 3)
    assert query.pointer_logits.shape == query_ids.shape
    assert program.tape.initial_state.shape == (2, 6)
    assert program.tape.event_kind.shape == (2, 8, 3)
    assert program.tape.event_identity.shape == (2, 8, 3)
    assert program.tape.amount.shape == (2, 8, 2)
