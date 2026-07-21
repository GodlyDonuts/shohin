#!/usr/bin/env python3
"""Evaluate a frozen neural CTAA core on exact finite semantic axes."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import torch

from ctaa_evaluation_io import sha256_file, write_json_once
from pipeline.generate_ctaa_board import semantic_splits
from run_ctaa_packet_executor import load_core


SCHEMA = "r12_ctaa_v2_finite_core_evaluation_v1"


def _apply(action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return state.gather(1, action)


@torch.inference_mode()
def score_core(core, device: torch.device) -> dict[str, object]:
    states = torch.tensor(tuple(product(range(3), repeat=3)), dtype=torch.long)
    axes = semantic_splits()
    reports = {}
    for axis in ("train", "development", "confirmation"):
        actions = torch.tensor(sorted(axes[axis]), dtype=torch.long)
        atomic_action = actions[:, None].expand(-1, 27, -1).reshape(-1, 3)
        atomic_state = states[None].expand(9, -1, -1).reshape(-1, 3)
        atomic_target = _apply(atomic_action, atomic_state)
        atomic_prediction = core(
            atomic_action.to(device),
            atomic_state.to(device),
        ).argmax(-1).cpu()

        first = actions[:, None, None].expand(-1, 9, 27, -1).reshape(-1, 3)
        second = actions[None, :, None].expand(9, -1, 27, -1).reshape(-1, 3)
        initial = states[None, None].expand(9, 9, -1, -1).reshape(-1, 3)
        state_one = _apply(first, initial)
        two_target = _apply(second, state_one)
        state_one_prediction = core(first.to(device), initial.to(device)).argmax(-1)
        two_prediction = core(second.to(device), state_one_prediction).argmax(-1).cpu()
        composed_target = first.gather(1, second)
        composed_prediction = core(second.to(device), first.to(device)).argmax(-1).cpu()
        composed_state_prediction = core(
            composed_prediction.to(device),
            initial.to(device),
        ).argmax(-1).cpu()
        reports[axis] = {
            "atomic_cases": 243,
            "two_action_cases": 2_187,
            "atomic_exact": float(
                atomic_prediction.eq(atomic_target).all(1).float().mean()
            ),
            "two_action_exact": float(
                two_prediction.eq(two_target).all(1).float().mean()
            ),
            "composition_exact": float(
                composed_prediction.eq(composed_target).all(1).float().mean()
            ),
            "route_agreement": float(
                composed_state_prediction.eq(two_prediction).all(1).float().mean()
            ),
        }
    return reports


def evaluate(core_path: Path, output_path: Path, device_name: str) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CTAA finite-core evaluation requires available CUDA")
    core, kind = load_core(core_path)
    core.to(device).eval()
    axes = score_core(core, device)
    gates = {
        axis: (
            report["atomic_exact"] == 1.0
            and report["two_action_exact"] == 1.0
            and report["composition_exact"] == 1.0
            and report["route_agreement"] == 1.0
        )
        for axis, report in axes.items()
    }
    report = {
        "schema": SCHEMA,
        "core_sha256": sha256_file(core_path),
        "core_kind": kind,
        "device": str(device),
        "axes": axes,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "board_access": 0,
    }
    report_sha = write_json_once(output_path, report)
    return {**report, "report_sha256": report_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    report = evaluate(args.core, args.output, args.device)
    print(json.dumps(report, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit("CTAA finite-core exactness gate failed")


if __name__ == "__main__":
    main()
