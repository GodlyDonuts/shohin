"""Confirmed-parent reconstruction for factorized ER-TT witness routing."""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from er_cst_fresh import derived_seed
from er_dual_stream_relation_adapter import (
    dual_stream_parameter_report,
    freeze_to_dual_stream,
)
from er_factorized_witness_route_adapter import FactorizedWitnessRouteCompiler
from pilot_er_cst_rule_card_adapter import state_dict_digest
from pilot_er_dual_stream_relation_adapter import REMOVED_V1_STATE
from pilot_er_relation_tensor_adapter import initialize_er_relation_tensor
from pilot_sd_cst_renderer_native_program import frozen_state_digest


EXPECTED_PARAMETERS = {
    "base": 125_081_664,
    "compiler": 60_452_996,
    "motor": 0,
    "reader": 0,
    "complete_system": 185_534_660,
    "headroom_below_200m": 14_465_340,
    "trainable": 11_131_868,
}


def initialize_factorized_witness_route(
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
) -> tuple[FactorizedWitnessRouteCompiler, dict[str, int], str, dict[str, object]]:
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
    model = FactorizedWitnessRouteCompiler()
    model_state = model.state_dict()
    missing = set(parent_state) - set(model_state)
    if missing != REMOVED_V1_STATE:
        raise ValueError(
            f"factorized witness removed v1 state differs: {sorted(missing)}"
        )
    shared = set(parent_state) & set(model_state)
    with torch.no_grad():
        for name in shared:
            source = parent_state[name]
            destination = model_state[name]
            if source.shape != destination.shape or source.dtype != destination.dtype:
                raise ValueError(f"factorized witness inherited tensor differs: {name}")
            destination.copy_(source)
    copied = {
        name: tensor for name, tensor in model.state_dict().items() if name in shared
    }
    expected = {name: tensor for name, tensor in parent_state.items() if name in shared}
    copied_digest = state_dict_digest(copied)
    if copied_digest != state_dict_digest(expected):
        raise ValueError("factorized witness inherited copy is not byte-identical")
    del parent
    gc.collect()

    declared = freeze_to_dual_stream(model)
    excluded_digest = frozen_state_digest(model, frozenset(declared))
    parameters = dual_stream_parameter_report(model)
    if parameters != EXPECTED_PARAMETERS:
        raise ValueError("factorized witness parameter certificate differs")
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
        "witness_bias_is_count_role_ordinal_factorized": True,
        "witness_gate_is_zero_initialized": True,
        "witness_bias_is_centered_and_bounded": True,
        "non_witness_route_logits_are_unchanged": True,
        "starts_from_reconstructed_confirmed_parent": True,
        "uses_rejected_canary_weights": False,
        "learned_motor_parameters": 0,
        "learned_reader_parameters": 0,
    }
    return model, parameters, excluded_digest, receipt
