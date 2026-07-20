#!/usr/bin/env python3
"""Multi-seed pre-board certificate canary for SD-CST atomic components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sd_cst import (
    CategoricalStateReader,
    TiedCategoricalMotor,
)
from train_sd_cst import (
    fit_motor_certificate,
    fit_reader_certificate,
    sha256_file,
)


class AtomicSystem(torch.nn.Module):
    """Only the two fully supervised components needed by the canary."""

    def __init__(self) -> None:
        super().__init__()
        self.motor = TiedCategoricalMotor(128)
        self.reader = CategoricalStateReader(64)


def run(seed_start: int, seeds: int, device: torch.device) -> dict[str, object]:
    rows = []
    for offset in range(seeds):
        seed = seed_start + offset
        system = AtomicSystem().to(device)
        motor = fit_motor_certificate(
            system, seed=seed ^ 0xA70C, lr=0.003, max_updates=1000,
        )
        reader = fit_reader_certificate(
            system, seed=seed ^ 0x18EA, lr=0.005, max_updates=500,
        )
        rows.append({
            "seed": seed,
            "motor_correct": motor["state_action_correct"] + motor["stop_correct"],
            "motor_loss": motor["final_loss"],
            "reader_correct": reader["correct"],
            "reader_loss": reader["final_loss"],
        })
    return {
        "schema": "r12_sd_cst_v1_1_atomic_optimizer_canary_v1",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "seed_start": seed_start,
        "seeds": seeds,
        "all_exact": all(
            row["motor_correct"] == 78 and row["reader_correct"] == 18
            for row in rows
        ),
        "rows": rows,
        "settings": {
            "motor": {"lr": 0.003, "updates": 1000},
            "reader": {"lr": 0.005, "updates": 500},
        },
        "forbidden_inputs": (
            "no base checkpoint, tokenizer, source board, development, or confirmation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.seeds < 64 and not args.cpu:
        raise SystemExit("production H100 canary requires at least 64 seeds")
    if args.out.exists():
        raise SystemExit(f"refusing existing atomic canary: {args.out}")
    device = torch.device("cpu" if args.cpu else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("atomic optimizer canary requires CUDA")
    result = run(args.seed_start, args.seeds, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_exact": result["all_exact"],
        "out": str(args.out.resolve()),
        "sha256": sha256_file(args.out),
    }, sort_keys=True))
    if not result["all_exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
