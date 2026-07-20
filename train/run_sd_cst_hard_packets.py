#!/usr/bin/env python3
"""Source-blind executor for sealed SD-CST categorical packets."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sd_cst import (
    CategoricalStateReader,
    HardLateQuery,
    HardProgramTape,
    StateSwap,
    TiedCategoricalMotor,
    rollout_hard_categorical,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--execution-core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing executor output: {args.output}")
    packets = torch.load(args.packets, map_location="cpu", weights_only=False)
    core = torch.load(args.execution_core, map_location="cpu", weights_only=False)
    if packets.get("schema") != "r12_sd_cst_hard_packet_bundle_v1":
        raise SystemExit("hard packet bundle schema mismatch")
    if core.get("schema") != "r12_sd_cst_projected_execution_core_v1":
        raise SystemExit("execution core schema mismatch")
    allowed_packet_keys = {"schema", "arms"}
    if set(packets) != allowed_packet_keys:
        raise SystemExit("packet bundle contains a forbidden side channel")

    motor = TiedCategoricalMotor()
    reader = CategoricalStateReader()
    motor.load_state_dict(core["motor"], strict=True)
    reader.load_state_dict(core["reader"], strict=True)
    motor.eval()
    reader.eval()
    outputs = {}
    for name, arm in packets["arms"].items():
        allowed_arm_keys = {
            "initial_state", "event_kind", "event_identity", "amount",
            "query", "control", "force_alive", "state_swap", "swap_after_step",
        }
        if set(arm) != allowed_arm_keys:
            raise SystemExit(f"packet arm {name} contains a forbidden side channel")
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
    torch.save({
        "schema": "r12_sd_cst_hard_packet_outputs_v1",
        "outputs": outputs,
    }, args.output)


if __name__ == "__main__":
    main()
