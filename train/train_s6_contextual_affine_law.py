#!/usr/bin/env python3
"""Fit the sole S6 treatment and favorable law-ID control on atomic cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from s6_contextual_affine_law_inducer import (
    ContextualAffineLawInducer,
    LawIdMemorizer,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_atomic_rows(data_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    report = json.loads((data_dir / "report.json").read_text())
    if report.get("decision") != "admit_s6_development_board":
        raise SystemExit("S6 board is not admitted")
    path = data_dir / "atomic_train.jsonl"
    expected = report["files"]["atomic_train.jsonl"]["sha256"]
    if _sha256(path) != expected:
        raise SystemExit("S6 atomic training hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != report["audit"]["atomic_training_rows"]:
        raise SystemExit("S6 atomic row count mismatch")
    forbidden = {"slope", "intercept", "a", "b", "final_state", "answer"}
    if any(set(row) & forbidden for row in rows):
        raise SystemExit("S6 atomic training row exposes forbidden fields")
    if any(row.get("supervision") != "atomic_destination_only" for row in rows):
        raise SystemExit("S6 training contains non-atomic supervision")
    return rows, report


def _tensorize(rows: list[dict[str, object]], device: torch.device) -> dict[str, torch.Tensor]:
    names = (
        "modulus",
        "card_y0",
        "card_y1",
        "current_location",
        "destination",
        "control_law_id",
    )
    return {
        name: torch.tensor([int(row[name]) for row in rows], dtype=torch.long, device=device)
        for name in names
    }


def _fit_model(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    arm: str,
    updates: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, object]:
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    losses: list[float] = []
    row_count = tensors["modulus"].shape[0]
    for update in range(updates):
        indices = torch.randint(
            row_count, (batch_size,), generator=generator, device="cpu"
        ).to(tensors["modulus"].device)
        if arm == "treatment":
            logits = model(
                tensors["modulus"].index_select(0, indices),
                tensors["card_y0"].index_select(0, indices),
                tensors["card_y1"].index_select(0, indices),
                tensors["current_location"].index_select(0, indices),
            )
        elif arm == "law_id":
            logits = model(
                tensors["modulus"].index_select(0, indices),
                tensors["control_law_id"].index_select(0, indices),
                tensors["current_location"].index_select(0, indices),
            )
        else:
            raise ValueError(f"unknown S6 arm: {arm}")
        targets = tensors["destination"].index_select(0, indices)
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite S6 {arm} loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite S6 {arm} gradient at update {update}")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        if arm == "treatment":
            logits = model(
                tensors["modulus"],
                tensors["card_y0"],
                tensors["card_y1"],
                tensors["current_location"],
            )
        else:
            logits = model(
                tensors["modulus"],
                tensors["control_law_id"],
                tensors["current_location"],
            )
        predictions = logits.argmax(-1)
        correct = int((predictions == tensors["destination"]).sum().item())
    return {
        "arm": arm,
        "updates": updates,
        "batch_size": batch_size,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "atomic_train_correct": correct,
        "atomic_train_total": row_count,
        "atomic_train_accuracy": correct / row_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S6 checkpoint: {args.out}")
    if args.updates != 4000 or args.batch_size != 256:
        raise SystemExit("S6 frozen optimization schedule mismatch")

    rows, board_report = _load_atomic_rows(args.data_dir)
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("S6 training requires allocated CUDA")
        torch.cuda.manual_seed_all(args.seed)

    treatment = ContextualAffineLawInducer().to(device)
    train_law_count = max(int(row["control_law_id"]) for row in rows) + 1
    law_id_control = LawIdMemorizer(train_law_count=train_law_count).to(device)
    if treatment.num_params() >= 8_000_000:
        raise SystemExit("S6 treatment exceeds module parameter cap")
    if treatment.total_system_params() >= 150_000_000:
        raise SystemExit("S6 complete system exceeds strict parameter cap")

    tensors = _tensorize(rows, device)
    treatment_fit = _fit_model(
        treatment,
        tensors,
        "treatment",
        args.updates,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.seed + 1,
    )
    control_fit = _fit_model(
        law_id_control,
        tensors,
        "law_id",
        args.updates,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.seed + 1,
    )
    if treatment_fit["atomic_train_accuracy"] < 0.99:
        raise SystemExit("S6 treatment failed frozen atomic fit gate")
    if control_fit["atomic_train_accuracy"] < 0.99:
        raise SystemExit("S6 law-ID control failed favorable atomic fit gate")

    checkpoint = {
        "schema": "r12_s6_contextual_affine_law_checkpoint_v1",
        "seed": args.seed,
        "board_source_commit": board_report["source_commit"],
        "board_seed": board_report["seed"],
        "board_report_sha256": _sha256(args.data_dir / "report.json"),
        "atomic_train_sha256": board_report["files"]["atomic_train.jsonl"]["sha256"],
        "config": {
            "width": 256,
            "layers": 6,
            "heads": 8,
            "feedforward": 1024,
            "dropout": 0.0,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "treatment_parameters": treatment.num_params(),
        "total_system_parameters": treatment.total_system_params(),
        "law_id_control_parameters": law_id_control.num_params(),
        "train_law_count": train_law_count,
        "treatment_fit": treatment_fit,
        "law_id_control_fit": control_fit,
        "treatment_state": {
            name: tensor.detach().cpu() for name, tensor in treatment.state_dict().items()
        },
        "law_id_control_state": {
            name: tensor.detach().cpu()
            for name, tensor in law_id_control.state_dict().items()
        },
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "training_contract": (
            "atomic destination cells from train laws only; zero recurrent, "
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
                "law_id_control_fit": control_fit,
                "treatment_parameters": treatment.num_params(),
                "total_system_parameters": treatment.total_system_params(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

