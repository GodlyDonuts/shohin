from __future__ import annotations

import re

import torch

from er_dual_stream_relation_adapter import (
    OPAQUE_CANONICAL_BYTE,
    DualStreamRelationCompiler,
    dual_stream_parameter_report,
    freeze_to_dual_stream,
)
from er_relation_tensor_adapter import MAX_CARDINALITY, MAX_RULES


def _small_model() -> DualStreamRelationCompiler:
    return DualStreamRelationCompiler(
        width=32,
        heads=4,
        encoder_layers=1,
        slot_layers=1,
        ff=64,
        slot_ff=64,
        max_bytes=1024,
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


def _batch(source: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([tuple(source)], dtype=torch.long)
    valid = torch.ones_like(ids, dtype=torch.bool)
    return ids, valid


def _program() -> bytes:
    lines = [
        "D3 x00000 x00001 x00002 ; I x00001 x00000 x00002",
        "W1 x10000 x20000 x20001 x20002 > x20002 x20001 x20002",
        "W2 x10001 x30000 x30001 x30002 > x30000 x30002 x30000",
        "W3 OFF",
        "W4 OFF",
        "E1 x10000",
        "E2 HALT",
    ]
    lines.extend(f"E{slot} x10001" for slot in range(3, 14))
    return "\n".join(lines).encode()


def test_structural_view_is_exactly_alpha_invariant() -> None:
    source = _program()
    renamed = re.sub(
        rb"(?<!\S)x[0-9]{5}(?!\S)",
        lambda match: b"z" + bytes((ord("9") - byte % 10 for byte in match.group(0)[1:])),
        source,
    )
    ids, valid = _batch(source)
    renamed_ids, renamed_valid = _batch(renamed)
    structural, starts = DualStreamRelationCompiler.structural_view(ids, valid)
    renamed_structural, renamed_starts = DualStreamRelationCompiler.structural_view(
        renamed_ids, renamed_valid
    )
    assert torch.equal(starts, renamed_starts)
    assert torch.equal(structural, renamed_structural)
    assert int(starts.sum()) == 32
    assert bool(structural[starts].eq(OPAQUE_CANONICAL_BYTE).all())


def test_whole_symbol_identity_recovers_non_bijective_relation() -> None:
    torch.manual_seed(410)
    model = _small_model().eval()
    before = [f"x{index:05d}" for index in range(6)]
    relation = (5, 0, 0, 3, 5, 1)
    words = before + [before[index] for index in relation]
    source = " ".join(words).encode()
    ids, valid = _batch(source)
    _, starts = model.structural_view(ids, valid)
    positions = starts[0].nonzero().flatten()
    logits = torch.full((1, 12, len(source)), -20.0)
    logits[0, torch.arange(12), positions] = 20.0
    equality = model._marginal_identity_equality(
        ids,
        starts,
        logits[:, 6:],
        logits[:, :6],
    )
    assert equality.shape == (1, 6, 6)
    assert torch.equal(equality.argmax(-1), torch.tensor(relation)[None])


def test_untrained_hard_packet_is_alpha_invariant_by_construction() -> None:
    torch.manual_seed(413)
    model = _small_model().eval()
    source = _program()
    tokens = sorted(set(re.findall(rb"(?<!\S)x[0-9]{5}(?!\S)", source)))
    mapping = {
        token: f"z{index:05d}".encode()
        for index, token in enumerate(reversed(tokens))
    }
    renamed = re.sub(
        rb"(?<!\S)x[0-9]{5}(?!\S)",
        lambda match: mapping[match.group(0)],
        source,
    )
    ids, valid = _batch(source)
    renamed_ids, renamed_valid = _batch(renamed)
    query_ids, query_valid = _batch(b"Q2")
    first = model.compile_relation_program(ids, valid, query_ids, query_valid).program.hard()
    second = model.compile_relation_program(
        renamed_ids, renamed_valid, query_ids, query_valid
    ).program.hard()
    for field in (
        "cardinality",
        "active",
        "initial_state",
        "rule_cards",
        "rule_active",
        "event_card",
        "event_halt",
    ):
        assert torch.equal(getattr(first, field), getattr(second, field))


def test_opcode_identity_binding_is_alpha_invariant() -> None:
    torch.manual_seed(411)
    model = _small_model().eval()
    source = b"x10000 x10001 x10002 x10002 x10000 x10001"
    ids, valid = _batch(source)
    _, starts = model.structural_view(ids, valid)
    positions = starts[0].nonzero().flatten()
    logits = torch.full((1, 6, len(source)), -20.0)
    logits[0, torch.arange(6), positions] = 20.0
    scores = model._marginal_identity_equality(
        ids,
        starts,
        logits[:, 3:],
        logits[:, :3],
    )
    assert scores.argmax(-1).tolist() == [[2, 0, 1]]


def test_marginal_exact_equality_recovers_oracle_routes_and_dense_gradients() -> None:
    torch.manual_seed(415)
    model = _small_model().train()
    source = b"x00000 x00001 x00002 x00002 x00000 x00001"
    ids, valid = _batch(source)
    _, starts = model.structural_view(ids, valid)
    positions = starts[0].nonzero().flatten()
    left = torch.full((1, 3, len(source)), -8.0, requires_grad=True)
    right = torch.full((1, 3, len(source)), -8.0, requires_grad=True)
    with torch.no_grad():
        left[0, torch.arange(3), positions[3:]] = 8.0
        right[0, torch.arange(3), positions[:3]] = 8.0
    equality = model._marginal_identity_equality(ids, starts, left, right)
    assert equality.argmax(-1).tolist() == [[2, 0, 1]]
    loss = -equality[0, torch.arange(3), torch.tensor([2, 0, 1])].mean()
    loss.backward()
    assert left.grad is not None and bool(left.grad.abs().gt(0).any())
    assert right.grad is not None and bool(right.grad.abs().gt(0).any())


def test_ordered_witness_lattice_learns_opcode_exclusion_and_preserves_order() -> None:
    logits = torch.zeros(1, 1, 2 * MAX_CARDINALITY, 13, requires_grad=True)
    candidates = torch.zeros(1, 1, 13, dtype=torch.bool)
    candidate_positions = torch.arange(0, 13, 2)
    candidates[0, 0, candidate_positions] = True
    active_slots = (0, 1, 2, 6, 7, 8)
    with torch.no_grad():
        for ordinal, slot in enumerate(active_slots):
            logits[0, 0, slot, candidate_positions[ordinal + 1]] = 8.0
    cardinality = torch.tensor([[8.0, -8.0, -8.0, -8.0]])
    probability = DualStreamRelationCompiler._ordered_route_probability(
        logits, candidates, cardinality
    )
    selected = probability[0, 0, list(active_slots)].argmax(-1)
    assert torch.equal(selected, candidate_positions[1:])
    assert torch.allclose(
        probability[0, 0, list(active_slots)].sum(-1),
        torch.ones(len(active_slots)),
        atol=1e-5,
    )
    loss = -probability[0, 0, list(active_slots), candidate_positions[1:]].log().mean()
    loss.backward()
    assert logits.grad is not None and bool(logits.grad.abs().gt(0).any())


def test_ordered_witness_lattice_maps_direct_candidate_sequence_exactly() -> None:
    logits = torch.randn(1, 1, 2 * MAX_CARDINALITY, 12)
    candidates = torch.zeros(1, 1, 12, dtype=torch.bool)
    candidate_positions = torch.tensor([0, 2, 4, 6, 8, 10])
    candidates[0, 0, candidate_positions] = True
    cardinality = torch.tensor([[8.0, -8.0, -8.0, -8.0]])
    probability = DualStreamRelationCompiler._ordered_route_probability(
        logits, candidates, cardinality
    )
    active_slots = (0, 1, 2, 6, 7, 8)
    assert torch.equal(
        probability[0, 0, list(active_slots)].argmax(-1), candidate_positions
    )


def test_compiler_emits_source_deleted_packet_and_routes_gradients() -> None:
    torch.manual_seed(412)
    model = _small_model().train()
    declared = set(freeze_to_dual_stream(model))
    ids, valid = _batch(_program())
    query_ids, query_valid = _batch(b"Q2")
    output = model.compile_relation_program(ids, valid, query_ids, query_valid)
    assert output.program.rule_cards.shape == (1, MAX_RULES, 6, 6)
    assert output.program.event_card.shape == (1, 13, MAX_RULES)
    assert output.program.initial_state.shape == (1, MAX_CARDINALITY, 6)
    loss = (
        output.program.rule_cards.square().mean()
        + output.program.event_card.square().mean()
        + output.program.initial_state.square().mean()
    )
    loss.backward()
    assert model.er_ds_declaration_queries.grad is not None
    assert model.er_ds_witness_queries.grad is not None
    assert model.er_ds_rule_opcode_query.grad is not None
    assert model.er_ds_event_opcode_query.grad is not None
    leaked = [
        name
        for name, parameter in model.named_parameters()
        if name not in declared and parameter.grad is not None
    ]
    assert leaked == []
    assert not hasattr(output.program, "source")


def test_pointer_only_gradient_reaches_role_assignment_not_record_encoder() -> None:
    torch.manual_seed(414)
    model = _small_model().train()
    freeze_to_dual_stream(model)
    ids, valid = _batch(_program())
    query_ids, query_valid = _batch(b"Q2")
    output = model.compile_relation_program(ids, valid, query_ids, query_valid)
    loss = (
        output.line_pointer_logits.square().mean()
        + output.binding_pointer_logits.square().mean()
        + output.witness_pointer_logits.square().mean()
    )
    loss.backward()
    assert model.er_tt_record_role_head.weight.grad is not None
    assert model.er_ds_router_query.weight.grad is not None
    assert model.er_ds_router_key.weight.grad is not None
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name.startswith(("record_line_encoder.", "record_set_encoder."))
    )


def test_default_system_remains_below_200m() -> None:
    model = DualStreamRelationCompiler()
    freeze_to_dual_stream(model)
    report = dual_stream_parameter_report(model)
    assert report["motor"] == 0
    assert report["reader"] == 0
    assert report["complete_system"] < 200_000_000
    assert report["headroom_below_200m"] > 0
    assert report["complete_system"] == 125_081_664 + model.parameter_count()
