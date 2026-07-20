"""Confirmed-parent reconstruction and parameter certificate for ER-TT."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Mapping

import torch

from er_cst_fresh import derived_seed, load_trainable_state
from er_relation_tensor_adapter import (
    EpisodicRelationTensorCompiler,
    freeze_to_relation_tensor_adaptive,
    relation_tensor_parameter_report,
)
from pilot_er_cst_rule_card_adapter import state_dict_digest
from pilot_er_cst_witness_equality_bus import initialize_er_cst_witness_equality
from pilot_sd_cst_byte_addressed import sha256_file
from pilot_sd_cst_renderer_native_program import frozen_state_digest


WITNESS_CHECKPOINT_SHA256 = (
    "917c1a1fce67c02258d0f90f04398ab433d18ba63c2dca92450cc5856c022ae7"
)
WITNESS_CONFIRMATION_ASSESSMENT_SHA256 = (
    "4a0fb47233d86887bb46aa853560bf81d319840610d62abe4f1dfaa899671310"
)
WITNESS_CHECKPOINT_SCHEMA = "r12_er_cst_witness_equality_checkpoint_v1_1"
WITNESS_CONFIRMATION_SCHEMA = (
    "r12_er_cst_witness_equality_confirmation_assessment_v1_1"
)
WITNESS_PROTOCOL = "R12-ER-CST-WEB-v1.1"
WITNESS_TRAINING_SEED = 2_262_748_995_832_026_278
WITNESS_COMPLETE_PARAMETERS = 192_726_827
REMOVED_PARENT_STATE = frozenset(
    {
        "er_record_role_embedding",
        "er_record_role_head.bias",
        "er_record_role_head.weight",
        "er_witness_queries",
        "local_occurrence_head.bias",
        "local_occurrence_head.weight",
        "local_query_head.bias",
        "local_query_head.weight",
        "permutations",
    }
)


def _validated_witness_checkpoint(
    checkpoint_path: Path,
    confirmation_assessment_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    if sha256_file(checkpoint_path) != WITNESS_CHECKPOINT_SHA256:
        raise ValueError("ER-TT witness parent checkpoint hash differs")
    if (
        sha256_file(confirmation_assessment_path)
        != WITNESS_CONFIRMATION_ASSESSMENT_SHA256
    ):
        raise ValueError("ER-TT witness confirmation assessment hash differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assessment = json.loads(confirmation_assessment_path.read_text())
    if (
        checkpoint.get("schema") != WITNESS_CHECKPOINT_SCHEMA
        or checkpoint.get("protocol") != WITNESS_PROTOCOL
        or checkpoint.get("training_seed") != WITNESS_TRAINING_SEED
        or checkpoint.get("parameters", {}).get("complete_system")
        != WITNESS_COMPLETE_PARAMETERS
        or checkpoint.get("development_accesses") != 0
        or checkpoint.get("confirmation_accesses") != 0
    ):
        raise ValueError("ER-TT witness parent checkpoint receipt differs")
    if (
        assessment.get("schema") != WITNESS_CONFIRMATION_SCHEMA
        or assessment.get("decision") != "confirm_er_cst_witness_equality_v1_1"
        or assessment.get("all_gates_pass") is not True
        or assessment.get("custody")
        != {"development_accesses": 1, "confirmation_accesses": 1}
        or assessment.get("parameters", {}).get("complete_system")
        != WITNESS_COMPLETE_PARAMETERS
    ):
        raise ValueError("ER-TT witness parent confirmation differs")
    treatment = checkpoint.get("arms", {}).get("treatment")
    if not isinstance(treatment, Mapping) or not isinstance(
        treatment.get("compiler_trainable_state"),
        Mapping,
    ):
        raise ValueError("ER-TT witness treatment compiler state is absent")
    return checkpoint, assessment


def initialize_er_relation_tensor(
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
) -> tuple[
    EpisodicRelationTensorCompiler,
    dict[str, int],
    str,
    dict[str, object],
]:
    checkpoint, assessment = _validated_witness_checkpoint(
        witness_checkpoint,
        witness_confirmation_assessment,
    )
    parent, motor, parent_parameters, _, parent_receipt = (
        initialize_er_cst_witness_equality(
            joint_checkpoint=joint_checkpoint,
            physical_checkpoint=physical_checkpoint,
            v1_checkpoint=v1_checkpoint,
            v1_2_checkpoint=v1_2_checkpoint,
            confirmed_checkpoint=confirmed_checkpoint,
            confirmation_assessment=confirmation_assessment,
            seed=WITNESS_TRAINING_SEED,
            device=torch.device("cpu"),
        )
    )
    if parent_parameters["complete_system"] != WITNESS_COMPLETE_PARAMETERS:
        raise ValueError("ER-TT reconstructed witness parent parameters differ")
    if checkpoint["parent_receipt"] != parent_receipt:
        raise ValueError("ER-TT reconstructed witness parent receipt differs")
    treatment = checkpoint["arms"]["treatment"]
    load_trainable_state(parent, treatment["compiler_trainable_state"])
    parent_state = parent.state_dict()
    parent_digest = state_dict_digest(parent_state)

    torch.manual_seed(derived_seed(seed, "er-tt-compiler"))
    model = EpisodicRelationTensorCompiler()
    model_state = model.state_dict()
    missing = set(parent_state) - set(model_state)
    if missing != REMOVED_PARENT_STATE:
        raise ValueError(f"ER-TT removed parent state differs: {sorted(missing)}")
    shared = set(parent_state) & set(model_state)
    with torch.no_grad():
        for name in shared:
            source = parent_state[name]
            destination = model_state[name]
            if source.shape != destination.shape or source.dtype != destination.dtype:
                raise ValueError(f"ER-TT inherited tensor differs: {name}")
            destination.copy_(source)
    copied = {name: tensor for name, tensor in model.state_dict().items() if name in shared}
    expected = {name: tensor for name, tensor in parent_state.items() if name in shared}
    copied_digest = state_dict_digest(copied)
    if copied_digest != state_dict_digest(expected):
        raise ValueError("ER-TT inherited parent copy is not byte-identical")
    del parent, motor
    gc.collect()

    declared = freeze_to_relation_tensor_adaptive(model)
    excluded_digest = frozen_state_digest(model, frozenset(declared))
    parameters = relation_tensor_parameter_report(model)
    expected_parameters = {
        "base": 125_081_664,
        "compiler": 67_659_190,
        "motor": 0,
        "reader": 0,
        "complete_system": 192_740_854,
        "headroom_below_200m": 7_259_146,
        "trainable": 12_037_293,
    }
    if parameters != expected_parameters:
        raise ValueError("ER-TT exact parameter certificate differs")
    model.to(device)
    receipt: dict[str, object] = {
        "witness_checkpoint_sha256": WITNESS_CHECKPOINT_SHA256,
        "witness_confirmation_assessment_sha256": (
            WITNESS_CONFIRMATION_ASSESSMENT_SHA256
        ),
        "witness_parent_state_sha256": parent_digest,
        "copied_parent_subset_sha256": copied_digest,
        "witness_confirmation_decision": assessment["decision"],
        "removed_parent_state": sorted(REMOVED_PARENT_STATE),
        "parameter_certificate": parameters,
        "trainable_names": list(declared),
        "finite_permutation_buffer_removed": True,
        "learned_motor_parameters": 0,
        "learned_reader_parameters": 0,
    }
    return model, parameters, excluded_digest, receipt
