#!/usr/bin/env python3
"""Fit S7 true/false generators and the favorable ordinary transformer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from s6_contextual_affine_law_inducer import ContextualAffineLawInducer
from s7_learned_cayley_generator import LearnedCayleyGenerator, PRIMARY_MODULI


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(data_dir: Path, name: str, report: dict[str, object]) -> list[dict[str, object]]:
    path = data_dir / name
    if _sha256(path) != report["files"][name]["sha256"]:
        raise SystemExit(f"S7 hash mismatch: {name}")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _fit_generator(
    model: LearnedCayleyGenerator,
    rows: list[dict[str, object]],
    target_field: str,
    updates: int,
    learning_rate: float,
) -> dict[str, object]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    losses: list[float] = []
    for _ in range(updates):
        terms: list[torch.Tensor] = []
        for modulus in PRIMARY_MODULI:
            selected = [row for row in rows if int(row["modulus"]) == modulus]
            current = torch.tensor(
                [int(row["current_symbol"]) for row in selected],
                dtype=torch.long,
                device=model.successor(modulus).device,
            )
            target = torch.tensor(
                [int(row[target_field]) for row in selected],
                dtype=torch.long,
                device=model.successor(modulus).device,
            )
            terms.append(F.cross_entropy(model.successor(modulus).index_select(0, current), target))
            zero_target = torch.tensor(
                [int(selected[0]["zero_symbol"])],
                dtype=torch.long,
                device=model.zero(modulus).device,
            )
            terms.append(F.cross_entropy(model.zero(modulus).unsqueeze(0), zero_target))
        loss = torch.stack(terms).sum()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite S7 generator loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    successor_correct = 0
    successor_total = 0
    zero_correct = 0
    with torch.no_grad():
        for modulus in PRIMARY_MODULI:
            selected = [row for row in rows if int(row["modulus"]) == modulus]
            predicted = model.successor(modulus).argmax(-1).detach().cpu().tolist()
            expected = [int(row[target_field]) for row in selected]
            successor_correct += sum(int(a == b) for a, b in zip(predicted, expected, strict=True))
            successor_total += len(expected)
            zero_correct += int(
                model.discrete_zero(modulus) == int(selected[0]["zero_symbol"])
            )
    return {
        "target_field": target_field,
        "updates": updates,
        "learning_rate": learning_rate,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "successor_correct": successor_correct,
        "successor_total": successor_total,
        "successor_accuracy": successor_correct / successor_total,
        "zero_correct": zero_correct,
        "zero_total": len(PRIMARY_MODULI),
        "zero_accuracy": zero_correct / len(PRIMARY_MODULI),
    }


def _fit_transformer(
    model: ContextualAffineLawInducer,
    rows: list[dict[str, object]],
    updates: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    device = next(model.parameters()).device
    names = ("modulus", "card_y0", "card_y1", "current_location", "destination")
    tensors = {
        name: torch.tensor([int(row[name]) for row in rows], dtype=torch.long, device=device)
        for name in names
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    losses: list[float] = []
    model.train()
    for _ in range(updates):
        indices = torch.randint(
            len(rows), (batch_size,), generator=generator, device="cpu"
        ).to(device)
        logits = model(
            tensors["modulus"].index_select(0, indices),
            tensors["card_y0"].index_select(0, indices),
            tensors["card_y1"].index_select(0, indices),
            tensors["current_location"].index_select(0, indices),
        )
        loss = F.cross_entropy(logits, tensors["destination"].index_select(0, indices))
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite S7 transformer loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite S7 transformer gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        predictions = model(
            tensors["modulus"],
            tensors["card_y0"],
            tensors["card_y1"],
            tensors["current_location"],
        ).argmax(-1)
        correct = int((predictions == tensors["destination"]).sum().item())
    return {
        "updates": updates,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "atomic_train_correct": correct,
        "atomic_train_total": len(rows),
        "atomic_train_accuracy": correct / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generator-updates", type=int, default=1000)
    parser.add_argument("--transformer-updates", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S7 checkpoint: {args.out}")
    if (
        args.generator_updates != 1000
        or args.transformer_updates != 4000
        or args.batch_size != 256
    ):
        raise SystemExit("S7 frozen optimization schedule mismatch")

    report = json.loads((args.data_dir / "report.json").read_text())
    if report.get("decision") != "admit_s7_learned_cayley_board":
        raise SystemExit("S7 board is not admitted")
    generator_rows = _load_rows(args.data_dir, "generator_train.jsonl", report)
    transformer_rows = _load_rows(
        args.data_dir, "transformer_atomic_train.jsonl", report
    )
    if len(generator_rows) != 23:
        raise SystemExit("S7 generator row count mismatch")

    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("S7 training requires allocated CUDA")
        torch.cuda.manual_seed_all(args.seed)

    treatment = LearnedCayleyGenerator().to(device)
    false_generator = LearnedCayleyGenerator().to(device)
    transformer = ContextualAffineLawInducer().to(device)
    treatment_fit = _fit_generator(treatment, generator_rows, "next_symbol", 1000, 0.05)
    false_fit = _fit_generator(
        false_generator, generator_rows, "false_next_symbol", 1000, 0.05
    )
    transformer_fit = _fit_transformer(
        transformer, transformer_rows, 4000, 256, 5e-4, args.seed + 1
    )
    for label, fit in (("treatment", treatment_fit), ("false", false_fit)):
        if fit["successor_accuracy"] != 1.0 or fit["zero_accuracy"] != 1.0:
            raise SystemExit(f"S7 {label} generator failed exact fit")
    if transformer_fit["atomic_train_accuracy"] < 0.99:
        raise SystemExit("S7 favorable transformer failed frozen fit gate")

    checkpoint = {
        "schema": "r12_s7_learned_cayley_checkpoint_v1",
        "seed": args.seed,
        "board_seed": report["seed"],
        "board_source_commit": report["source_commit"],
        "board_report_sha256": _sha256(args.data_dir / "report.json"),
        "generator_train_sha256": report["files"]["generator_train.jsonl"]["sha256"],
        "transformer_train_sha256": report["files"]["transformer_atomic_train.jsonl"]["sha256"],
        "config": {
            "generator_updates": 1000,
            "generator_learning_rate": 0.05,
            "transformer_updates": 4000,
            "transformer_batch_size": 256,
            "transformer_learning_rate": 5e-4,
        },
        "treatment_parameters": treatment.num_params(),
        "treatment_total_system_parameters": treatment.total_system_params(),
        "ordinary_transformer_parameters": transformer.num_params(),
        "treatment_fit": treatment_fit,
        "false_generator_fit": false_fit,
        "ordinary_transformer_fit": transformer_fit,
        "treatment_state": {
            name: tensor.detach().cpu() for name, tensor in treatment.state_dict().items()
        },
        "false_generator_state": {
            name: tensor.detach().cpu()
            for name, tensor in false_generator.state_dict().items()
        },
        "ordinary_transformer_state": {
            name: tensor.detach().cpu() for name, tensor in transformer.state_dict().items()
        },
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "training_contract": (
            "treatment/false see 23 successor cells plus three zero anchors; "
            "ordinary transformer sees train-law atomic cells; zero recurrent, "
            "answer, development-law, or confirmation-law supervision"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    print(
        json.dumps(
            {
                "checkpoint": str(args.out),
                "treatment_fit": treatment_fit,
                "false_generator_fit": false_fit,
                "ordinary_transformer_fit": transformer_fit,
                "treatment_parameters": treatment.num_params(),
                "treatment_total_system_parameters": treatment.total_system_params(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
