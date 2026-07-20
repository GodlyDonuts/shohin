"""Exact parent reconstruction and parameter certificate for ER-CST."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from er_cst_rule_card_adapter import (
    EpisodicRuleCardCompiler,
    TiedRuleCardMotor,
    freeze_to_er_adaptive,
    rule_card_parameter_report,
)
from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    READER_PARAMETERS,
    sha256_file,
)
from pilot_sd_cst_renderer_native_program import frozen_state_digest
from sd_cst_complete_physical_fresh import (
    derived_seed,
    initialize_model as initialize_confirmed_parent,
    load_trainable_state,
)


CONFIRMED_CHECKPOINT_SHA256 = (
    "a5888d88541904cfa186a6686012c13c7b555f7d186ba1e3e73f71dbaca462d8"
)
CONFIRMATION_ASSESSMENT_SHA256 = (
    "4629a745f6eed2e388eb6e1f78b29dff346ee6939e21275ae6ff1d66719d3cb9"
)
CONFIRMED_SCHEMA = "r12_sd_cst_complete_physical_fresh_checkpoint_v1_3"
CONFIRMED_PROTOCOL = "r12_sd_cst_complete_physical_fresh_v1_3"
CONFIRMATION_SCHEMA = (
    "r12_sd_cst_complete_physical_fresh_confirmation_assessment_v1_3"
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(canonical_json(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _validate_confirmation(path: Path) -> dict[str, object]:
    if sha256_file(path) != CONFIRMATION_ASSESSMENT_SHA256:
        raise ValueError("ER-CST confirmation assessment hash differs")
    value = json.loads(path.read_text())
    if (
        value.get("schema") != CONFIRMATION_SCHEMA
        or value.get("decision") != "confirm_complete_physical_fresh_v1_3"
        or value.get("all_gates_pass") is not True
        or value.get("custody")
        != {"development_accesses": 1, "confirmation_accesses": 1}
        or value.get("parameters", {}).get("complete_system") != 192_129_179
    ):
        raise ValueError("ER-CST parent confirmation differs")
    return value


def initialize_er_cst(
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
    EpisodicRuleCardCompiler,
    TiedRuleCardMotor,
    dict[str, int],
    str,
    dict[str, object],
]:
    assessment = _validate_confirmation(confirmation_assessment)
    if sha256_file(confirmed_checkpoint) != CONFIRMED_CHECKPOINT_SHA256:
        raise ValueError("ER-CST confirmed compiler hash differs")
    checkpoint = torch.load(
        confirmed_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if (
        checkpoint.get("schema") != CONFIRMED_SCHEMA
        or checkpoint.get("protocol") != CONFIRMED_PROTOCOL
        or checkpoint.get("development_accesses") != 0
        or checkpoint.get("confirmation_accesses") != 0
        or checkpoint.get("parameters", {}).get("complete_system") != 192_129_179
    ):
        raise ValueError("ER-CST confirmed checkpoint receipt differs")
    treatment = checkpoint.get("arms", {}).get("treatment")
    if not isinstance(treatment, Mapping) or not isinstance(
        treatment.get("trainable_state"), Mapping
    ):
        raise ValueError("ER-CST confirmed treatment state is absent")

    parent, parent_parameters, _ = initialize_confirmed_parent(
        joint_checkpoint,
        physical_checkpoint,
        v1_checkpoint,
        v1_2_checkpoint,
        torch.device("cpu"),
    )
    load_trainable_state(parent, treatment["trainable_state"])
    if parent_parameters.get("complete_system") != 192_129_179:
        raise ValueError("ER-CST reconstructed parent parameters differ")
    parent_state = parent.state_dict()
    parent_digest = state_dict_digest(parent_state)

    torch.manual_seed(derived_seed(seed, "er-cst-compiler"))
    model = EpisodicRuleCardCompiler()
    model_state = model.state_dict()
    if not set(parent_state).issubset(model_state):
        raise ValueError("ER-CST model does not contain the confirmed parent")
    with torch.no_grad():
        for name, tensor in parent_state.items():
            if (
                tensor.shape != model_state[name].shape
                or tensor.dtype != model_state[name].dtype
            ):
                raise ValueError(f"ER-CST parent tensor differs: {name}")
            model_state[name].copy_(tensor)
    copied_parent = {
        name: tensor for name, tensor in model.state_dict().items() if name in parent_state
    }
    if state_dict_digest(copied_parent) != parent_digest:
        raise ValueError("ER-CST parent copy is not byte-identical")

    torch.manual_seed(derived_seed(seed, "er-cst-motor"))
    motor = TiedRuleCardMotor()
    declared = freeze_to_er_adaptive(model)
    motor.requires_grad_(True)
    excluded_digest = frozen_state_digest(model, frozenset(declared))
    parameters = rule_card_parameter_report(
        model,
        motor,
        base_parameters=BASE_PARAMETERS,
        reader_parameters=READER_PARAMETERS,
    )
    if (
        parameters["compiler"] != 67_336_999
        or parameters["motor"] != 2_438
        or parameters["complete_system"] != 192_421_936
        or parameters["headroom_below_200m"] != 7_578_064
        or parameters["trainable"] != 11_716_385
    ):
        raise ValueError("ER-CST exact parameter certificate differs")
    model.to(device)
    motor.to(device)
    receipt: dict[str, object] = {
        "confirmed_checkpoint_sha256": CONFIRMED_CHECKPOINT_SHA256,
        "confirmation_assessment_sha256": CONFIRMATION_ASSESSMENT_SHA256,
        "confirmed_parent_state_sha256": parent_digest,
        "confirmation_decision": assessment["decision"],
        "parameter_certificate": parameters,
        "trainable_names": list(declared),
        "motor_trainable_names": sorted(
            f"motor.{name}" for name, _ in motor.named_parameters()
        ),
    }
    return model, motor, parameters, excluded_digest, receipt
