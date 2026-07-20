from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import torch

from run_sd_cst_hard_packets import (
    _load_weights_only,
    _validate_core,
    _validate_packets,
    main,
)
from sd_cst import (
    CategoricalStateReader,
    HardLateQuery,
    HardProgramTape,
    TiedCategoricalMotor,
    rollout_hard_categorical,
)


def _core() -> dict[str, object]:
    return {
        "schema": "r12_sd_cst_projected_execution_core_v1",
        "motor": TiedCategoricalMotor().state_dict(),
        "reader": CategoricalStateReader().state_dict(),
        "seed": 1,
        "motor_seed": 2,
        "reader_seed": 3,
        "compiler_checkpoint_sha256": "a" * 64,
        "score_eligible": False,
    }


def _arm(batch: int = 3) -> dict[str, object]:
    event_kind = torch.zeros((batch, 8), dtype=torch.uint8)
    event_kind[:, -1] = 2
    return {
        "initial_state": torch.zeros(batch, dtype=torch.uint8),
        "event_kind": event_kind,
        "event_identity": torch.zeros((batch, 8), dtype=torch.uint8),
        "amount": torch.zeros((batch, 8), dtype=torch.uint8),
        "query": torch.zeros(batch, dtype=torch.uint8),
        "control": "normal",
        "force_alive": False,
        "state_swap": None,
        "swap_after_step": 0,
    }


def _packets() -> dict[str, object]:
    return {
        "schema": "r12_sd_cst_hard_packet_bundle_v1",
        "arms": {"canonical": _arm()},
    }


def test_accepts_exact_current_mechanics_contract():
    core = _validate_core(_core())
    arms = _validate_packets(_packets())
    assert set(core) == {
        "schema",
        "motor",
        "reader",
        "seed",
        "motor_seed",
        "reader_seed",
        "compiler_checkpoint_sha256",
        "score_eligible",
    }
    assert set(arms) == {"canonical"}


@pytest.mark.parametrize("target", ["core", "packets", "arm", "motor", "reader"])
def test_rejects_extra_keys_at_every_mapping_boundary(target: str):
    core = _core()
    packets = _packets()
    if target == "core":
        core["extra"] = 1
        validator, value = _validate_core, core
    elif target == "packets":
        packets["extra"] = 1
        validator, value = _validate_packets, packets
    elif target == "arm":
        packets["arms"]["canonical"]["extra"] = torch.tensor(1)  # type: ignore[index]
        validator, value = _validate_packets, packets
    else:
        core[target]["extra"] = torch.tensor(1)  # type: ignore[index]
        validator, value = _validate_core, core
    with pytest.raises(SystemExit, match="keys do not match"):
        validator(value)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("initial_state", torch.zeros(3, dtype=torch.int64)),
        ("event_kind", torch.zeros((3, 7), dtype=torch.uint8)),
        ("event_identity", torch.zeros((3, 8, 1), dtype=torch.uint8)),
        ("amount", torch.zeros((3, 8), dtype=torch.float32)),
        ("query", torch.zeros((3, 1), dtype=torch.uint8)),
        ("state_swap", torch.tensor([0, 0, 2], dtype=torch.int64)),
    ],
)
def test_rejects_wrong_packet_tensor_contract(field: str, replacement: torch.Tensor):
    packets = _packets()
    arm = packets["arms"]["canonical"]  # type: ignore[index]
    arm[field] = replacement
    with pytest.raises(SystemExit):
        _validate_packets(packets)


def test_rejects_wrong_core_tensor_dtype_shape_and_nonfinite():
    for mutation in ("dtype", "shape", "nonfinite"):
        core = _core()
        if mutation == "dtype":
            core["motor"]["network.0.weight"] = torch.zeros(  # type: ignore[index]
                (128, 14),
                dtype=torch.float64,
            )
        elif mutation == "shape":
            core["reader"]["network.2.bias"] = torch.zeros(4)  # type: ignore[index]
        else:
            core["motor"]["network.0.bias"][0] = float("nan")  # type: ignore[index]
        with pytest.raises(SystemExit):
            _validate_core(core)


class _UnsafePayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return os.system, (f"touch {self.marker}",)


def test_weights_only_loader_never_deserializes_objects(tmp_path: Path):
    marker = tmp_path / "deserialized"
    payload = tmp_path / "unsafe.pt"
    torch.save(_UnsafePayload(marker), payload)
    with pytest.raises(SystemExit, match="refusing unsafe or malformed"):
        _load_weights_only(payload, "test payload")
    assert not marker.exists()


def test_main_preserves_current_mechanics_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(17)
    core = _core()
    packets = _packets()
    core_path = tmp_path / "core.pt"
    packets_path = tmp_path / "packets.pt"
    output_path = tmp_path / "outputs.pt"
    torch.save(core, core_path)
    torch.save(packets, packets_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sd_cst_hard_packets.py",
            "--packets",
            str(packets_path),
            "--execution-core",
            str(core_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    output = torch.load(output_path, map_location="cpu", weights_only=True)
    assert set(output) == {"schema", "outputs"}
    assert output["schema"] == "r12_sd_cst_hard_packet_outputs_v1"
    actual = output["outputs"]["canonical"]
    arm = packets["arms"]["canonical"]
    motor = TiedCategoricalMotor()
    reader = CategoricalStateReader()
    motor.load_state_dict(core["motor"], strict=True)
    reader.load_state_dict(core["reader"], strict=True)
    expected = rollout_hard_categorical(
        motor,
        reader,
        HardProgramTape(
            arm["initial_state"],
            arm["event_kind"],
            arm["event_identity"],
            arm["amount"],
        ),
        HardLateQuery(arm["query"]),
    )
    assert torch.equal(actual["final_state"], expected.final_state)
    assert torch.equal(actual["answer"], expected.answer_logits.argmax(-1))
    assert torch.equal(
        actual["state_trajectory"],
        torch.stack(expected.state_trajectory, dim=1),
    )
    assert torch.equal(
        actual["alive_trajectory"],
        torch.stack(expected.alive_trajectory, dim=1),
    )
