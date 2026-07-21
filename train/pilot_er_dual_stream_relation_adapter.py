"""Confirmed-parent reconstruction and parameter certificate for dual-stream ER-TT."""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from er_cst_fresh import derived_seed
from er_dual_stream_relation_adapter import (
    DualStreamRelationCompiler,
    dual_stream_parameter_report,
    freeze_to_dual_stream,
)
from pilot_er_cst_rule_card_adapter import state_dict_digest
from pilot_er_relation_tensor_adapter import initialize_er_relation_tensor
from pilot_sd_cst_renderer_native_program import frozen_state_digest


REMOVED_V1_STATE = frozenset(
    {
        "bigram_embedding.weight",
        "er_equality_projection.weight",
        "er_equality_scale",
        "er_event_card_query_projection.weight",
        "er_rule_card_key_projection.weight",
        "er_tt_occurrence_head.bias",
        "er_tt_occurrence_head.weight",
        "er_tt_witness_position_embedding",
        "er_tt_witness_side_embedding",
        "er_witness_key_projection.weight",
        "er_witness_norm.bias",
        "er_witness_norm.weight",
        "er_witness_query_projection.weight",
        "fingerprint_projection.weight",
        "local_occurrence_hidden.bias",
        "local_occurrence_hidden.weight",
        "local_occurrence_norm.bias",
        "local_occurrence_norm.weight",
        "logit_scale",
    }
)
EXPECTED_PARAMETERS = {
    "base": 125_081_664,
    "compiler": 60_450_632,
    "motor": 0,
    "reader": 0,
    "complete_system": 185_532_296,
    "headroom_below_200m": 14_467_704,
    "trainable": 11_129_504,
}


def initialize_dual_stream_relation(
    *,
    joint_checkpoint: Path,
    physical_checkpoint: Path,
    v1_checkpoint: Path,
    v1_2_checkpoint: Path,
    confirmed_checkpoint: Path,
    confirmation_assessment: Path,
    witness_checkpoint: Path,
    witness_confirmation_assessment: Path,
    seed: int,
    device: torch.device,
) -> tuple[DualStreamRelationCompiler, dict[str, int], str, dict[str, object]]:
    parent, parent_parameters, _, parent_receipt = initialize_er_relation_tensor(
        joint_checkpoint=joint_checkpoint,
        physical_checkpoint=physical_checkpoint,
        v1_checkpoint=v1_checkpoint,
        v1_2_checkpoint=v1_2_checkpoint,
        confirmed_checkpoint=confirmed_checkpoint,
        confirmation_assessment=confirmation_assessment,
        witness_checkpoint=witness_checkpoint,
        witness_confirmation_assessment=witness_confirmation_assessment,
        seed=seed,
        device=torch.device("cpu"),
    )
    parent_state = parent.state_dict()
    parent_digest = state_dict_digest(parent_state)

    torch.manual_seed(derived_seed(seed, "er-dual-stream-compiler"))
    model = DualStreamRelationCompiler()
    model_state = model.state_dict()
    missing = set(parent_state) - set(model_state)
    if missing != REMOVED_V1_STATE:
        raise ValueError(f"dual-stream removed v1 state differs: {sorted(missing)}")
    shared = set(parent_state) & set(model_state)
    with torch.no_grad():
        for name in shared:
            source = parent_state[name]
            destination = model_state[name]
            if source.shape != destination.shape or source.dtype != destination.dtype:
                raise ValueError(f"dual-stream inherited tensor differs: {name}")
            destination.copy_(source)
    copied = {
        name: tensor for name, tensor in model.state_dict().items() if name in shared
    }
    expected = {name: tensor for name, tensor in parent_state.items() if name in shared}
    copied_digest = state_dict_digest(copied)
    if copied_digest != state_dict_digest(expected):
        raise ValueError("dual-stream inherited copy is not byte-identical")
    del parent
    gc.collect()

    declared = freeze_to_dual_stream(model)
    excluded_digest = frozen_state_digest(model, frozenset(declared))
    parameters = dual_stream_parameter_report(model)
    if parameters != EXPECTED_PARAMETERS:
        raise ValueError("dual-stream exact parameter certificate differs")
    model.to(device)
    receipt: dict[str, object] = {
        "v1_untrained_parent_parameters": parent_parameters,
        "v1_untrained_parent_receipt": parent_receipt,
        "v1_untrained_parent_state_sha256": parent_digest,
        "copied_parent_subset_sha256": copied_digest,
        "removed_v1_state": sorted(REMOVED_V1_STATE),
        "parameter_certificate": parameters,
        "trainable_names": list(declared),
        "structural_stream_alpha_canonical": True,
        "whole_symbol_exact_equality": True,
        "event_binding_uses_identity_equality": True,
        "routing_assignment_receives_pointer_gradients": True,
        "routing_assignment_detaches_record_features": True,
        "identity_equality_is_exact_route_marginal": True,
        "witness_routes_use_cardinality_marginalized_monotone_lattice": True,
        "witness_opcode_exclusion_is_model_scored": True,
        "dead_v1_identity_parameters_removed": True,
        "learned_motor_parameters": 0,
        "learned_reader_parameters": 0,
    }
    return model, parameters, excluded_digest, receipt
