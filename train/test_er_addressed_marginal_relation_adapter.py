from __future__ import annotations

import json
from pathlib import Path

import torch

from build_er_relation_tensor_board import TRAIN_SPLIT
from er_addressed_marginal_relation_adapter import (
    MAX_RECORD_SYMBOLS,
    AddressedMarginalRelationCompiler,
)
from er_dual_stream_relation_adapter import dual_stream_parameter_report
from er_relation_tensor_training import loss_batch, parse_row
from pilot_er_dual_stream_train_canary import (
    alpha_predictions,
    alpha_recode_row,
)
from pilot_er_addressed_marginal_relation_adapter import (
    EXPECTED_PARAMETERS,
    initialize_addressed_marginal_relation,
)


ROOT = Path(__file__).resolve().parents[1]


def _small_model() -> AddressedMarginalRelationCompiler:
    return AddressedMarginalRelationCompiler(
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


def test_candidate_addresses_are_identity_free_and_record_local() -> None:
    candidates = torch.zeros((2, 2, 16), dtype=torch.bool)
    candidates[0, 0, (1, 4, 9)] = True
    candidates[0, 1, (2, 3, 7, 12)] = True
    candidates[1, 0, (5,)] = True
    candidates[1, 1, (0, 15)] = True
    ordinal, count = AddressedMarginalRelationCompiler.candidate_addresses(
        candidates
    )
    assert count.tolist() == [[3, 4], [1, 2]]
    assert ordinal[0, 0, (1, 4, 9)].tolist() == [0, 1, 2]
    assert ordinal[0, 1, (2, 3, 7, 12)].tolist() == [0, 1, 2, 3]
    assert int(ordinal.max()) <= MAX_RECORD_SYMBOLS


def test_address_embeddings_receive_route_gradient() -> None:
    torch.manual_seed(719)
    model = _small_model().train()
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
    source = "\n".join(lines).encode()
    ids = torch.tensor([tuple(source)], dtype=torch.long)
    valid = torch.ones_like(ids, dtype=torch.bool)
    query = torch.tensor([tuple(b"Q2")], dtype=torch.long)
    query_valid = torch.ones_like(query, dtype=torch.bool)
    output = model.compile_relation_program(ids, valid, query, query_valid)
    loss = (
        output.binding_pointer_logits.square().mean()
        + output.witness_pointer_logits.square().mean()
        + output.program.rule_cards.square().mean()
    )
    loss.backward()
    assert model.er_am_candidate_ordinal_embedding.weight.grad is not None
    assert model.er_am_candidate_count_embedding.weight.grad is not None
    assert bool(model.er_am_candidate_ordinal_embedding.weight.grad.abs().gt(0).any())
    assert bool(model.er_am_candidate_count_embedding.weight.grad.abs().gt(0).any())


def test_default_addressed_system_parameter_certificate() -> None:
    model = AddressedMarginalRelationCompiler()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    report = dual_stream_parameter_report(model)
    assert report["complete_system"] < 200_000_000
    assert report["compiler"] == EXPECTED_PARAMETERS["compiler"]


def test_actual_confirmed_parent_reconstructs_without_failed_canary_weights() -> None:
    model, parameters, _, receipt = initialize_addressed_marginal_relation(
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
    assert receipt["candidate_ordinal_address_is_identity_free"] is True
    assert receipt["candidate_count_address_is_identity_free"] is True
    assert receipt["uses_rejected_canary_weights"] is False
    assert model.er_am_candidate_ordinal_embedding.weight.requires_grad
    assert model.er_am_candidate_count_embedding.weight.requires_grad

    board = ROOT / "artifacts/r12/er_relation_tensor_board_1209366536012979338/train.jsonl"
    with board.open() as handle:
        rows = [parse_row(json.loads(next(handle)), TRAIN_SPLIT) for _ in range(4)]
    recoded = [alpha_recode_row(row, "addressed-test-alpha") for row in rows]
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
