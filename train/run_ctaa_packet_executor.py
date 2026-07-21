#!/usr/bin/env python3
"""Execute one source-deleted CTAA packet artifact in a fresh process."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ctaa_neural_core import ClosureFeatureTransitionCore, OuterProductTransitionControl
from ctaa_packet_io import read_packet_file, sha256_file


CORE_SCHEMA = "ctaa_recurrent_core_v1"
EXECUTION_SCHEMA = "ctaa_source_blind_execution_v1"


def load_core(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
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
