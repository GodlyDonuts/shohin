from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pilot_er_relation_tensor_adapter import (
    REMOVED_PARENT_STATE,
    _validated_witness_checkpoint,
    initialize_er_relation_tensor,
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


def test_local_witness_checkpoint_and_confirmation_are_exact() -> None:
    checkpoint, assessment = _validated_witness_checkpoint(
        WITNESS_RUN / "compiler.pt",
        WITNESS_CONFIRMATION / "confirmation_assessment.json",
    )
    assert checkpoint["parameters"]["complete_system"] == 192_726_827
    assert assessment["decision"] == "confirm_er_cst_witness_equality_v1_1"


def test_wrong_witness_checkpoint_fails_before_load(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="checkpoint hash"):
        _validated_witness_checkpoint(
            bad,
            WITNESS_CONFIRMATION / "confirmation_assessment.json",
        )


def test_actual_confirmed_parent_reconstructs_into_er_tt() -> None:
    model, parameters, _, receipt = initialize_er_relation_tensor(
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
        seed=11,
        device=torch.device("cpu"),
    )
    assert parameters["complete_system"] == 192_740_854
    assert parameters["headroom_below_200m"] == 7_259_146
    assert set(receipt["removed_parent_state"]) == REMOVED_PARENT_STATE
    assert receipt["finite_permutation_buffer_removed"] is True
    assert not hasattr(model, "permutations")
