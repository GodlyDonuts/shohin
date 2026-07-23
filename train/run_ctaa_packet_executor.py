#!/usr/bin/env python3
"""Execute one source-deleted CTAA packet artifact in a fresh process."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import torch

from ctaa_neural_core import ClosureFeatureTransitionCore, OuterProductTransitionControl
from ctaa_neural_core import CTAA_MAX_STEPS, CTAA_WIDTH
from ctaa_packet_io import (
    _read_immutable_bytes,
    read_packet_bytes,
    read_packet_file,
    sha256_file,
)


CORE_SCHEMA = "ctaa_recurrent_core_v1"
EXECUTION_SCHEMA = "ctaa_source_blind_execution_v2"
EXECUTION_KEYS = {
    "schema",
    "core_kind",
    "packet_sha256",
    "core_sha256",
    "state_route",
    "halted",
    "composed_cards",
    "composed_states",
}


def _load_core_payload(raw: bytes):
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != CORE_SCHEMA:
        raise ValueError("CTAA recurrent-core checkpoint schema differs")
    kind = payload.get("kind")
    if kind == "closure_feature":
        core = ClosureFeatureTransitionCore()
    elif kind == "outer_product_control":
        core = OuterProductTransitionControl()
    else:
        raise ValueError("CTAA recurrent-core kind differs")
    core.load_state_dict(payload.get("state", {}), strict=True)
    return core.eval(), str(kind)


def load_core(path: Path):
    return _load_core_payload(_read_immutable_bytes(path))


@torch.inference_mode()
def validate_execution_artifact(
    execution_path: Path,
    packet_path: Path,
    core_path: Path,
) -> dict[str, object]:
    """Validate and replay one immutable execution before query disclosure."""

    execution_raw = _read_immutable_bytes(execution_path)
    packet_raw = _read_immutable_bytes(packet_path)
    core_raw = _read_immutable_bytes(core_path)
    execution = torch.load(
        io.BytesIO(execution_raw), map_location="cpu", weights_only=True
    )
    if (
        not isinstance(execution, dict)
        or set(execution) != EXECUTION_KEYS
        or execution.get("schema") != EXECUTION_SCHEMA
    ):
        raise ValueError("CTAA execution artifact schema differs")
    packet = read_packet_bytes(packet_raw)
    core, kind = _load_core_payload(core_raw)
    batch = packet.opcode_schedule.shape[0]
    tensor_contract = {
        "state_route": (torch.uint8, (batch, CTAA_MAX_STEPS + 1, CTAA_WIDTH)),
        "halted": (torch.bool, (batch, CTAA_MAX_STEPS + 1)),
        "composed_cards": (
            torch.uint8,
            (batch, CTAA_MAX_STEPS + 1, CTAA_WIDTH),
        ),
        "composed_states": (
            torch.uint8,
            (batch, CTAA_MAX_STEPS + 1, CTAA_WIDTH),
        ),
    }
    for name, (dtype, shape) in tensor_contract.items():
        value = execution.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != dtype
            or tuple(value.shape) != shape
            or value.device.type != "cpu"
        ):
            raise ValueError(f"CTAA execution {name} tensor differs")
        if dtype == torch.uint8 and value.numel() and int(value.max()) >= CTAA_WIDTH:
            raise ValueError(f"CTAA execution {name} leaves the categorical domain")
    if (
        execution.get("core_kind") != kind
        or execution.get("packet_sha256") != hashlib.sha256(packet_raw).hexdigest()
        or execution.get("core_sha256") != hashlib.sha256(core_raw).hexdigest()
    ):
        raise ValueError("CTAA execution artifact source binding differs")
    replay = packet.execute_dual(core)
    expected = {
        "state_route": replay.state_route.states.to(torch.uint8).cpu(),
        "halted": replay.state_route.halted.cpu(),
        "composed_cards": replay.composed_cards.to(torch.uint8).cpu(),
        "composed_states": replay.composed_states.to(torch.uint8).cpu(),
    }
    if any(not torch.equal(execution[name], value) for name, value in expected.items()):
        raise ValueError("CTAA execution artifact deterministic replay differs")
    if not torch.equal(execution["state_route"], execution["composed_states"]):
        raise ValueError("CTAA execution routes disagree before query disclosure")
    return execution


def write_execution_once(path: Path, payload: dict[str, object]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing existing CTAA execution artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing CTAA execution temporary: {temporary}")
    try:
        torch.save(payload, temporary)
        temporary.chmod(0o444)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.chmod(0o644)
            temporary.unlink()
    path.chmod(0o444)
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packet = read_packet_file(args.packet)
    core, kind = load_core(args.core)
    with torch.inference_mode():
        trace = packet.execute_dual(core)
    payload = {
        "schema": EXECUTION_SCHEMA,
        "core_kind": kind,
        "packet_sha256": sha256_file(args.packet),
        "core_sha256": sha256_file(args.core),
        "state_route": trace.state_route.states.to(torch.uint8).cpu(),
        "halted": trace.state_route.halted.cpu(),
        "composed_cards": trace.composed_cards.to(torch.uint8).cpu(),
        "composed_states": trace.composed_states.to(torch.uint8).cpu(),
    }
    digest = write_execution_once(args.output, payload)
    print(f"execution_sha256={digest}")


if __name__ == "__main__":
    main()
