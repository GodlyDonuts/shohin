#!/usr/bin/env python3
"""Scoreless optimization/capacity preflight for matched CTAA recurrent cores."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from ctaa_neural_core import ClosureFeatureTransitionCore, OuterProductTransitionControl


SCHEMA = "ctaa_matched_core_preflight_v1"


def finite_pairs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tuples = tuple(product(range(3), repeat=3))
    left = torch.tensor(
        [first for first in tuples for _second in tuples],
        dtype=torch.long,
        device=device,
    )
    right = torch.tensor(
        [second for _first in tuples for second in tuples],
        dtype=torch.long,
        device=device,
    )
    return left, right


def closure_targets(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return right.gather(1, left)


def arbitrary_targets(seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    targets = torch.randint(0, 3, (729, 3), generator=generator)
    return targets.to(device)


def exact_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(logits.argmax(-1).eq(targets).all(-1).float().mean().item())


def optimize(
    model: torch.nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
    targets: torch.Tensor,
    *,
    learning_rate: float,
    max_steps: int,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    reached_step = None
    final_loss = None
    for step in range(1, max_steps + 1):
        logits = model(left, right)
        loss = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
        if step == 1 or step % 10 == 0 or step == max_steps:
            with torch.no_grad():
                accuracy = exact_accuracy(model(left, right), targets)
            if accuracy == 1.0:
                reached_step = step
                break
    with torch.no_grad():
        logits = model(left, right)
        exact = exact_accuracy(logits, targets)
        coordinate = float(logits.argmax(-1).eq(targets).float().mean().item())
    return {
        "exact_accuracy": exact,
        "coordinate_accuracy": coordinate,
        "final_loss": final_loss,
        "first_checked_exact_step": reached_step,
        "optimizer_steps": step,
    }


def run(seed: int, device: torch.device, max_steps: int, learning_rate: float) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    left, right = finite_pairs(device)
    treatment = ClosureFeatureTransitionCore().to(device)
    treatment_result = optimize(
        treatment,
        left,
        right,
        closure_targets(left, right),
        learning_rate=learning_rate,
        max_steps=max_steps,
    )
    torch.manual_seed(seed + 1)
    control = OuterProductTransitionControl().to(device)
    control_features = control.features(left, right)
    unique_control_features = int(torch.unique(control_features, dim=0).shape[0])
    control_result = optimize(
        control,
        left,
        right,
        arbitrary_targets(seed + 2, device),
        learning_rate=learning_rate,
        max_steps=max_steps,
    )
    gates = {
        "parameters_exactly_matched": treatment.unique_parameters
        == control.unique_parameters
        == 107_753,
        "control_features_separate_all_pairs": unique_control_features == 729,
        "closure_treatment_optimizes_exactly": treatment_result["exact_accuracy"] == 1.0,
        "arbitrary_control_table_optimizes_exactly": control_result["exact_accuracy"] == 1.0,
    }
    return {
        "schema": SCHEMA,
        "seed": seed,
        "device": str(device),
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "treatment_parameters": treatment.unique_parameters,
        "control_parameters": control.unique_parameters,
        "unique_control_features": unique_control_features,
        "treatment": treatment_result,
        "control": control_result,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": "scoreless finite optimization preflight; no board or reasoning result",
    }


def write_once(path: Path, report: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing CTAA preflight report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing CTAA preflight temporary: {temporary}")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.chmod(0o444)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.chmod(0o644)
            temporary.unlink()
    path.chmod(0o444)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_steps < 1 or args.learning_rate <= 0:
        raise ValueError("CTAA preflight optimizer geometry differs")
    report = run(
        args.seed,
        torch.device(args.device),
        args.max_steps,
        args.learning_rate,
    )
    write_once(args.output, report)
    if not report["all_gates_pass"]:
        raise SystemExit("CTAA matched-core optimization preflight failed")


if __name__ == "__main__":
    main()
