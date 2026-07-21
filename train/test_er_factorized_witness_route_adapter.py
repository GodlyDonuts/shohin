from __future__ import annotations

import json
from pathlib import Path

import torch

from build_er_relation_tensor_board import TRAIN_SPLIT
from er_cst_fresh import derived_seed
from er_dual_stream_relation_adapter import DualStreamRelationCompiler
from er_factorized_witness_route_adapter import (
    MAX_RECORD_SYMBOLS,
    FactorizedWitnessRouteCompiler,
)
from er_relation_tensor_training import loss_batch, parse_row
from pilot_er_dual_stream_train_canary import (
    alpha_predictions,
    alpha_recode_row,
)
from pilot_er_factorized_witness_route_adapter import (
    EXPECTED_PARAMETERS,
    initialize_factorized_witness_route,
)


ROOT = Path(__file__).resolve().parents[1]


def _kwargs() -> dict[str, int]:
    return {
        "width": 32,
        "heads": 4,
        "encoder_layers": 1,
        "slot_layers": 1,
        "ff": 64,
        "slot_ff": 64,
        "max_bytes": 1024,
        "fingerprint_width": 16,
        "orbit_width": 32,
        "orbit_heads": 4,
        "orbit_layers": 1,
        "orbit_ff": 64,
        "native_slot_layers": 1,
        "native_slot_heads": 4,
        "native_slot_ff": 64,
        "record_width": 32,
        "record_heads": 4,
        "record_layers": 1,
        "record_set_layers": 1,
        "record_ff": 64,
        "max_line_bytes": 96,
        "sinkhorn_steps": 4,
        "occurrence_ff": 64,
        "equality_width": 16,
    }


def _program() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lines = [
        "D3 e00000 e00001 e00002 ; I e00001 e00000 e00002",
        "W1 o00000 w00000 w00001 w00002 > w00002 w00001 w00002",
        "W2 OFF",
        "W3 OFF",
        "W4 OFF",
        "E1 o00000",
        "E2 HALT",
    ]
    lines.extend(f"E{slot} o00000" for slot in range(3, 14))
    source = torch.tensor([tuple("\n".join(lines).encode())], dtype=torch.long)
    query = torch.tensor([tuple(b"Q2")], dtype=torch.long)
    return (
        source,
        torch.ones_like(source, dtype=torch.bool),
        query,
        torch.ones_like(query, dtype=torch.bool),
    )


def test_zero_bias_preserves_all_dual_stream_outputs() -> None:
    seed = derived_seed(1709, "factorized-zero-equivalence")
    torch.manual_seed(seed)
    control = DualStreamRelationCompiler(**_kwargs()).eval()
    torch.manual_seed(seed)
    treatment = FactorizedWitnessRouteCompiler(**_kwargs()).eval()
    common = {
        name: tensor
        for name, tensor in treatment.state_dict().items()
        if name in control.state_dict()
    }
    assert all(
        torch.equal(control.state_dict()[name], tensor)
        for name, tensor in common.items()
    )
    assert torch.count_nonzero(treatment.er_fw_witness_gate) == 0
    args = _program()
    with torch.no_grad():
        left = control.compile_relation_program(*args)
        right = treatment.compile_relation_program(*args)
    assert torch.equal(left.witness_pointer_logits, right.witness_pointer_logits)
    assert torch.equal(left.binding_pointer_logits, right.binding_pointer_logits)
    assert torch.equal(left.program.rule_cards, right.program.rule_cards)
    assert torch.equal(left.program.event_card, right.program.event_card)


def test_factorized_gate_and_bias_receive_witness_route_gradient() -> None:
    torch.manual_seed(2107)
    model = FactorizedWitnessRouteCompiler(**_kwargs()).train()
    source, source_valid, query, query_valid = _program()
    output = model.compile_relation_program(source, source_valid, query, query_valid)
    loss = -output.witness_pointer_logits[0, 0, 0, 3]
    loss.backward()
    gate_gradient = model.er_fw_witness_gate.grad
    assert gate_gradient is not None
    assert bool(gate_gradient.abs().gt(0).any())
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.er_fw_witness_gate.fill_(0.1)
    output = model.compile_relation_program(source, source_valid, query, query_valid)
    (-output.witness_pointer_logits[0, 0, 0, 3]).backward()
    bias_gradient = model.er_fw_witness_address_bias.grad
    assert bias_gradient is not None
    assert bool(bias_gradient.abs().gt(0).any())


def test_nonzero_bias_changes_only_witness_routes() -> None:
    seed = derived_seed(2309, "factorized-route-isolation")
    torch.manual_seed(seed)
    control = DualStreamRelationCompiler(**_kwargs()).eval()
    torch.manual_seed(seed)
    treatment = FactorizedWitnessRouteCompiler(**_kwargs()).eval()
    with torch.no_grad():
        treatment.er_fw_witness_gate.fill_(0.5)
        left = control.compile_relation_program(*_program())
        right = treatment.compile_relation_program(*_program())
    assert not torch.equal(left.witness_pointer_logits, right.witness_pointer_logits)
    assert torch.equal(left.binding_pointer_logits, right.binding_pointer_logits)
    assert torch.equal(
        left.initial_entity_pointer_logits,
        right.initial_entity_pointer_logits,
    )
    assert torch.equal(left.program.event_card, right.program.event_card)


def test_candidate_addresses_are_identity_free_and_local() -> None:
    candidates = torch.zeros((1, 2, 16), dtype=torch.bool)
    candidates[0, 0, (1, 4, 9)] = True
    candidates[0, 1, (2, 3, 7, 12)] = True
    ordinal, count = FactorizedWitnessRouteCompiler.candidate_addresses(candidates)
    assert count.tolist() == [[3, 4]]
    assert ordinal[0, 0, (1, 4, 9)].tolist() == [0, 1, 2]
    assert int(ordinal.max()) <= MAX_RECORD_SYMBOLS


def test_actual_parent_reconstructs_and_real_rows_reach_every_trainable() -> None:
    model, parameters, _, receipt = initialize_factorized_witness_route(
        joint_checkpoint=ROOT
        / "train/sd_cst_renderer_native_joint_pilot_6795424534800881443/compiler.pt",
        physical_checkpoint=ROOT
        / "train/sd_cst_physical_record_bus_pilot_8959672499628717158/compiler.pt",
        v1_checkpoint=ROOT
        / "train/sd_cst_complete_physical_record_bus_pilot_4564290739472553435/compiler.pt",
        v1_2_checkpoint=ROOT
        / "train/sd_cst_complete_physical_record_bus_v1_2_pilot_7097310278885678337/compiler.pt",
        confirmed_checkpoint=ROOT
        / "train/sd_cst_complete_physical_fresh_run_8920874392524997882_8446904969546017898/compiler.pt",
        confirmation_assessment=ROOT
        / "train/sd_cst_complete_physical_confirmation_run_8920874392524997882_8446904969546017898/confirmation_assessment.json",
        witness_checkpoint=ROOT
        / "train/er_cst_witness_equality_2244518911844010727_2262748995832026278/compiler.pt",
        witness_confirmation_assessment=ROOT
        / "train/er_cst_witness_equality_confirmation_2244518911844010727_2262748995832026278/confirmation_assessment.json",
        seed=73,
        device=torch.device("cpu"),
    )
    assert parameters == EXPECTED_PARAMETERS
    assert receipt["witness_bias_is_count_role_ordinal_factorized"] is True
    assert receipt["witness_bias_is_centered_and_bounded"] is True
    assert receipt["uses_rejected_canary_weights"] is False
    assert torch.count_nonzero(model.er_fw_witness_gate) == 0

    board = (
        ROOT / "artifacts/r12/er_relation_tensor_board_1209366536012979338/train.jsonl"
    )
    with board.open() as handle:
        rows = [parse_row(json.loads(next(handle)), TRAIN_SPLIT) for _ in range(4)]
    recoded = [alpha_recode_row(row, "factorized-test-alpha") for row in rows]
    original_prediction = alpha_predictions(model, rows, batch_size=4)
    recoded_prediction = alpha_predictions(model, recoded, batch_size=4)
    assert all(
        torch.equal(original_prediction[name], recoded_prediction[name])
        for name in original_prediction
    )
    model.train()
    loss, pieces = loss_batch(model, [rows], torch.device("cpu"))
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.tensor(list(pieces.values()))).all()
    loss.backward()
    declared = set(receipt["trainable_names"])
    leaked = [
        name
        for name, parameter in model.named_parameters()
        if name not in declared and parameter.grad is not None
    ]
    assert leaked == []
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []
