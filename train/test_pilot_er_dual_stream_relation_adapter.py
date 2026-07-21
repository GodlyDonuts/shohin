from __future__ import annotations

import json
from pathlib import Path

import torch

from build_er_relation_tensor_board import TRAIN_SPLIT
from er_relation_tensor_training import loss_batch, parse_row
from pilot_er_dual_stream_relation_adapter import (
    EXPECTED_PARAMETERS,
    REMOVED_V1_STATE,
    initialize_dual_stream_relation,
)


ROOT = Path(__file__).resolve().parents[1]
WITNESS_RUN = (
    ROOT
    / "train/er_cst_witness_equality_2244518911844010727_2262748995832026278"
)
WITNESS_CONFIRMATION = (
    ROOT
    / "train/er_cst_witness_equality_confirmation_2244518911844010727_2262748995832026278"
)


def test_actual_confirmed_parent_reconstructs_into_dual_stream() -> None:
    model, parameters, _, receipt = initialize_dual_stream_relation(
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
        witness_checkpoint=WITNESS_RUN / "compiler.pt",
        witness_confirmation_assessment=WITNESS_CONFIRMATION
        / "confirmation_assessment.json",
        seed=41,
        device=torch.device("cpu"),
    )
    assert parameters == EXPECTED_PARAMETERS
    assert set(receipt["removed_v1_state"]) == REMOVED_V1_STATE
    assert receipt["structural_stream_alpha_canonical"] is True
    assert receipt["whole_symbol_exact_equality"] is True
    assert receipt["event_binding_uses_identity_equality"] is True
    assert receipt["routing_assignment_receives_pointer_gradients"] is True
    assert receipt["routing_assignment_detaches_record_features"] is True
    assert receipt["identity_equality_is_exact_route_marginal"] is True
    assert receipt["dead_v1_identity_parameters_removed"] is True
    assert not hasattr(model, "er_tt_occurrence_head")
    assert not hasattr(model, "er_event_card_query_projection")
    assert not hasattr(model, "er_equality_projection")
    assert not hasattr(model, "bigram_embedding")

    board = ROOT / "artifacts/r12/er_relation_tensor_board_1209366536012979338/train.jsonl"
    with board.open() as handle:
        rows = [parse_row(json.loads(next(handle)), TRAIN_SPLIT) for _ in range(4)]
    assert len({row.family_id for row in rows}) == 1
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
    for name in (
        "er_ds_declaration_queries",
        "er_ds_witness_queries",
        "er_ds_rule_opcode_query",
        "er_ds_event_opcode_query",
        "er_ds_router_norm.weight",
        "er_ds_router_query.weight",
        "er_ds_router_key.weight",
    ):
        assert model.get_parameter(name).grad is not None
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []
