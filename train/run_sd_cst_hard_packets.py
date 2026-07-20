#!/usr/bin/env python3
"""Source-blind executor for sealed SD-CST categorical packets."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from sd_cst import (
    CategoricalStateReader,
    HardLateQuery,
    HardProgramTape,
    StateSwap,
    TiedCategoricalMotor,
    rollout_hard_categorical,
)


PACKET_SCHEMA = "r12_sd_cst_hard_packet_bundle_v1"
CORE_SCHEMA = "r12_sd_cst_projected_execution_core_v1"
PACKET_KEYS = frozenset({"schema", "arms"})
CORE_KEYS = frozenset(
    {
        "schema",
        "motor",
        "reader",
        "seed",
        "motor_seed",
        "reader_seed",
        "compiler_checkpoint_sha256",
        "score_eligible",
    }
)
ARM_KEYS = frozenset(
    {
        "initial_state",
        "event_kind",
        "event_identity",
        "amount",
        "query",
        "control",
        "force_alive",
        "state_swap",
        "swap_after_step",
    }
)
MOTOR_TENSORS = {
    "network.0.weight": ((128, 14), torch.float32),
    "network.0.bias": ((128,), torch.float32),
    "network.2.weight": ((128, 128), torch.float32),
    "network.2.bias": ((128,), torch.float32),
    "network.4.weight": ((6, 128), torch.float32),
    "network.4.bias": ((6,), torch.float32),
}
READER_TENSORS = {
    "network.0.weight": ((64, 9), torch.float32),
    "network.0.bias": ((64,), torch.float32),
    "network.2.weight": ((3, 64), torch.float32),
    "network.2.bias": ((3,), torch.float32),
}


def _die(message: str) -> None:
    raise SystemExit(message)


def _load_weights_only(path: Path, label: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        _die(f"{label} requires a PyTorch build with weights_only loading: {error}")
    except Exception as error:
        _die(f"refusing unsafe or malformed {label}: {error}")


def _require_exact_keys(value: object, expected: frozenset[str], label: str) -> Mapping:
    if not isinstance(value, Mapping):
        _die(f"{label} must be a mapping")
    if set(value) != expected:
        _die(f"{label} keys do not match the sealed schema")
    return value


def _require_tensor(
    value: object,
    *,
    label: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        _die(f"{label} must be a tensor")
    if value.dtype != dtype or value.ndim != len(shape) or tuple(value.shape) != shape:
        _die(f"{label} must have dtype {dtype} and shape {shape}")
    if value.layout != torch.strided:
        _die(f"{label} must use strided tensor layout")
    return value


def _validate_state_dict(
    value: object,
    expected: dict[str, tuple[tuple[int, ...], torch.dtype]],
    label: str,
) -> Mapping[str, torch.Tensor]:
    state = _require_exact_keys(value, frozenset(expected), label)
    for name, (shape, dtype) in expected.items():
        tensor = _require_tensor(
            state[name],
            label=f"{label}.{name}",
            dtype=dtype,
            shape=shape,
        )
        if not bool(torch.isfinite(tensor).all()):
            _die(f"{label}.{name} contains a non-finite value")
    return state


def _require_plain_int(value: object, label: str) -> int:
    if type(value) is not int:
        _die(f"{label} must be an integer")
    return value


def _validate_core(value: object) -> Mapping:
    core = _require_exact_keys(value, CORE_KEYS, "execution core")
    if core["schema"] != CORE_SCHEMA:
        _die("execution core schema mismatch")
    for name in ("seed", "motor_seed", "reader_seed"):
        if _require_plain_int(core[name], f"execution core {name}") < 0:
            _die(f"execution core {name} must be non-negative")
    digest = core["compiler_checkpoint_sha256"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        _die("execution core compiler checkpoint digest must be lowercase SHA-256")
    if core["score_eligible"] is not False:
        _die("execution core must be explicitly score-ineligible")
    _validate_state_dict(core["motor"], MOTOR_TENSORS, "execution core motor")
    _validate_state_dict(core["reader"], READER_TENSORS, "execution core reader")
    return core


def _require_category_range(tensor: torch.Tensor, upper: int, label: str) -> None:
    if tensor.numel() and int(tensor.max()) >= upper:
        _die(f"{label} is outside its categorical range")


def _validate_arm(value: object, name: str) -> Mapping:
    arm = _require_exact_keys(value, ARM_KEYS, f"packet arm {name}")
    event_kind = arm["event_kind"]
    if not isinstance(event_kind, torch.Tensor) or event_kind.ndim != 2:
        _die(f"packet arm {name}.event_kind must be a rank-2 tensor")
    batch = int(event_kind.shape[0])
    if batch <= 0:
        _die(f"packet arm {name} must contain at least one row")
    tensors = {
        "initial_state": ((batch,), 6),
        "event_kind": ((batch, 8), 3),
        "event_identity": ((batch, 8), 3),
        "amount": ((batch, 8), 2),
        "query": ((batch,), 3),
    }
    for field, (shape, upper) in tensors.items():
        tensor = _require_tensor(
            arm[field],
            label=f"packet arm {name}.{field}",
            dtype=torch.uint8,
            shape=shape,
        )
        _require_category_range(tensor, upper, f"packet arm {name}.{field}")
    if type(arm["control"]) is not str or arm["control"] not in {
        "normal",
        "reset",
        "freeze",
    }:
        _die(f"packet arm {name}.control is invalid")
    if type(arm["force_alive"]) is not bool:
        _die(f"packet arm {name}.force_alive must be boolean")
    swap_after_step = _require_plain_int(
        arm["swap_after_step"],
        f"packet arm {name}.swap_after_step",
    )
    state_swap = arm["state_swap"]
    if state_swap is None:
        if swap_after_step != 0:
            _die(f"packet arm {name} has a swap step without a state swap")
    else:
        permutation = _require_tensor(
            state_swap,
            label=f"packet arm {name}.state_swap",
            dtype=torch.int64,
            shape=(batch,),
        )
        if not 0 <= swap_after_step < 8:
            _die(f"packet arm {name}.swap_after_step is outside [0,7]")
        if not torch.equal(torch.sort(permutation).values, torch.arange(batch)):
            _die(f"packet arm {name}.state_swap must be a full batch permutation")
    return arm


def _validate_packets(value: object) -> dict[str, Mapping]:
    packets = _require_exact_keys(value, PACKET_KEYS, "hard packet bundle")
    if packets["schema"] != PACKET_SCHEMA:
        _die("hard packet bundle schema mismatch")
    arms = packets["arms"]
    if not isinstance(arms, Mapping) or not arms:
        _die("hard packet bundle arms must be a non-empty mapping")
    validated = {}
    for name, arm in arms.items():
        if type(name) is not str or not name:
            _die("hard packet arm names must be non-empty strings")
        validated[name] = _validate_arm(arm, name)
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--execution-core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing executor output: {args.output}")
    packets = _load_weights_only(args.packets, "hard packet bundle")
    core = _validate_core(_load_weights_only(args.execution_core, "execution core"))
    arms = _validate_packets(packets)

    motor = TiedCategoricalMotor()
    reader = CategoricalStateReader()
    motor.load_state_dict(core["motor"], strict=True)
    reader.load_state_dict(core["reader"], strict=True)
    motor.eval()
    reader.eval()
    outputs = {}
    for name, arm in arms.items():
        tape = HardProgramTape(
            arm["initial_state"],
            arm["event_kind"],
            arm["event_identity"],
            arm["amount"],
        )
        query = HardLateQuery(arm["query"])
        state_swap = None
        if arm["state_swap"] is not None:
            state_swap = StateSwap(
                after_step=int(arm["swap_after_step"]),
                batch_permutation=arm["state_swap"].long(),
            )
        result = rollout_hard_categorical(
            motor,
            reader,
            tape,
            query,
            control=str(arm["control"]),
            state_swap=state_swap,
            force_alive=bool(arm["force_alive"]),
        )
        outputs[name] = {
            "final_state": result.final_state.cpu(),
            "answer": result.answer_logits.argmax(-1).cpu(),
            "state_trajectory": torch.stack(result.state_trajectory, dim=1).cpu(),
            "alive_trajectory": torch.stack(result.alive_trajectory, dim=1).cpu(),
        }
    torch.save(
        {
            "schema": "r12_sd_cst_hard_packet_outputs_v1",
            "outputs": outputs,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
