"""Exact parent reconstruction and parameter certificate for ER-CST-WEB."""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from er_cst_fresh import derived_seed
from er_cst_rule_card_adapter import TiedRuleCardMotor, rule_card_parameter_report
from er_cst_witness_equality_bus import (
    WitnessEqualityBusCompiler,
    freeze_to_witness_equality_adaptive,
)
from pilot_er_cst_rule_card_adapter import initialize_er_cst, state_dict_digest
from pilot_sd_cst_byte_addressed import BASE_PARAMETERS, READER_PARAMETERS
from pilot_sd_cst_renderer_native_program import frozen_state_digest


def initialize_er_cst_witness_equality(
    *,
    joint_checkpoint: Path,
    physical_checkpoint: Path,
    v1_checkpoint: Path,
    v1_2_checkpoint: Path,
    confirmed_checkpoint: Path,
    confirmation_assessment: Path,
    seed: int,
    device: torch.device,
) -> tuple[
    WitnessEqualityBusCompiler,
    TiedRuleCardMotor,
    dict[str, int],
    str,
    dict[str, object],
]:
    old_model, motor, _, _, old_receipt = initialize_er_cst(
        joint_checkpoint=joint_checkpoint,
        physical_checkpoint=physical_checkpoint,
        v1_checkpoint=v1_checkpoint,
        v1_2_checkpoint=v1_2_checkpoint,
        confirmed_checkpoint=confirmed_checkpoint,
        confirmation_assessment=confirmation_assessment,
        seed=seed,
        device=torch.device("cpu"),
    )
    torch.manual_seed(derived_seed(seed, "er-cst-compiler"))
    model = WitnessEqualityBusCompiler()
    old_state = old_model.state_dict()
    state = model.state_dict()
    missing = set(old_state) - set(state) - {
        "er_rule_permutation_head.weight",
        "er_rule_permutation_head.bias",
    }
    if missing:
        raise ValueError("ER-CST witness model omits inherited tensors")
    with torch.no_grad():
        for name in set(old_state) & set(state):
            if old_state[name].shape != state[name].shape or old_state[name].dtype != state[name].dtype:
                raise ValueError(f"ER-CST witness inherited tensor differs: {name}")
            state[name].copy_(old_state[name])
    inherited = {name: tensor for name, tensor in model.state_dict().items() if name in old_state}
    expected_inherited = {name: tensor for name, tensor in old_state.items() if name in inherited}
    if state_dict_digest(inherited) != state_dict_digest(expected_inherited):
        raise ValueError("ER-CST witness inherited copy is not byte-identical")
    del old_model
    gc.collect()

    declared = freeze_to_witness_equality_adaptive(model)
    motor.requires_grad_(True)
    excluded_digest = frozen_state_digest(model, frozenset(declared))
    parameters = rule_card_parameter_report(
        model,
        motor,
        base_parameters=BASE_PARAMETERS,
        reader_parameters=READER_PARAMETERS,
    )
    if parameters["complete_system"] >= 200_000_000:
        raise ValueError("ER-CST witness system exceeds 200M")
    model.to(device)
    motor.to(device)
    receipt: dict[str, object] = {
        **{
            key: value
            for key, value in old_receipt.items()
            if key not in {"parameter_certificate", "trainable_names"}
        },
        "parameter_certificate": parameters,
        "trainable_names": list(declared),
        "direct_rule_classifier_removed": True,
        "card_path_shared_record_gradient": False,
        "witness_occurrences_per_rule": 6,
        "finite_card_assignments": 6,
    }
    return model, motor, parameters, excluded_digest, receipt
