#!/usr/bin/env python3
"""Train-only UROM-3 canary with immutable base and adapter-only output."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Iterable, Sequence

import torch
from tokenizers import Tokenizer

from general_relational_object_machine import TrunkRelationalObjectCompiler
from model import GPT, GPTConfig
from urom3_training import (
    TokenizedUROMRow,
    collate_urom_rows,
    load_urom_rows,
    urom_loss,
    urom_metrics,
)


CHECKPOINT_SCHEMA = "urom3_train_only_canary_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine_scale(step: int, updates: int, warmup: int) -> float:
    if updates < 1 or warmup < 0 or warmup >= updates:
        raise ValueError("UROM learning-rate schedule differs")
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, updates - warmup - 1)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def adapter_state(
    compiler: TrunkRelationalObjectCompiler,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in compiler.state_dict().items()
        if not name.startswith("backbone.model.")
    }


def trainable_parameters(
    compiler: TrunkRelationalObjectCompiler,
) -> list[torch.nn.Parameter]:
    parameters = [
        parameter
        for parameter in compiler.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("UROM compiler has no trainable parameters")
    base_ids = {
        id(parameter) for parameter in compiler.backbone.model.parameters()
    }
    if any(id(parameter) in base_ids for parameter in parameters):
        raise ValueError("UROM raw trunk is unexpectedly trainable")
    return parameters


def batches(
    rows: Sequence[TokenizedUROMRow],
    *,
    batch_size: int,
    seed: int,
) -> Iterable[list[TokenizedUROMRow]]:
    if batch_size < 1 or not rows:
        raise ValueError("UROM batching contract differs")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [rows[index] for index in indices[start : start + batch_size]]


@torch.inference_mode()
def evaluate_rows(
    compiler: TrunkRelationalObjectCompiler,
    rows: Sequence[TokenizedUROMRow],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    compiler.eval()
    for selected in batches(rows, batch_size=batch_size, seed=0):
        batch = collate_urom_rows(selected, device=device)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with context:
            metrics = urom_metrics(compiler, batch)
        weight = len(selected)
        count += weight
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + weight * value
    return {name: value / count for name, value in totals.items()}


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing existing UROM checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing existing UROM report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compiler-width", type=int, default=384)
    parser.add_argument("--compiler-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=5)
    parser.add_argument("--encoder-feedforward", type=int, default=1_408)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--decoder-feedforward", type=int, default=1_024)
    parser.add_argument("--identity-width", type=int, default=128)
    parser.add_argument("--early-layer", type=int, default=19)
    parser.add_argument("--late-layer", type=int, default=29)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.updates < 1
        or args.batch_size < 1
        or args.max_rows < args.batch_size
        or args.learning_rate <= 0
        or args.clip <= 0
        or args.log_every < 1
    ):
        raise SystemExit("UROM canary arguments differ")
    if sha256_file(args.base) != args.base_sha256:
        raise SystemExit("UROM protected base hash differs")
    if not args.tokenizer.is_file() or not args.train.is_file():
        raise SystemExit("UROM canary input is absent")
    if args.out.exists() or args.report.exists():
        raise SystemExit("UROM canary output already exists")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise SystemExit("UROM canary requires bf16 CUDA")
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    base_payload = torch.load(
        args.base,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if (
        not isinstance(base_payload, dict)
        or set(("cfg", "model", "step")) - set(base_payload)
        or int(base_payload["step"]) != 300_000
    ):
        raise SystemExit("UROM protected base payload differs")
    model = GPT(GPTConfig(**base_payload["cfg"]))
    model.load_state_dict(base_payload["model"], strict=True)
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=model_dtype).eval()
    compiler_config = {
        "compiler_width": args.compiler_width,
        "compiler_heads": args.compiler_heads,
        "encoder_layers": args.encoder_layers,
        "encoder_feedforward": args.encoder_feedforward,
        "decoder_layers": args.decoder_layers,
        "decoder_feedforward": args.decoder_feedforward,
        "identity_width": args.identity_width,
        "early_layer": args.early_layer,
        "late_layer": args.late_layer,
    }
    compiler = TrunkRelationalObjectCompiler(
        model,
        **compiler_config,
    ).to(device)
    report = compiler.parameter_report()

    rows = load_urom_rows(
        args.train,
        tokenizer,
        expected_split="train",
        max_length=model.cfg.seq_len,
    )
    if len(rows) < args.max_rows:
        raise SystemExit("UROM train-only canary has too few rows")
    selected_indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(selected_indices)
    rows = [rows[index] for index in selected_indices[: args.max_rows]]
    initial_metrics = evaluate_rows(
        compiler,
        rows,
        batch_size=args.batch_size,
        device=device,
    )

    parameters = trainable_parameters(compiler)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    compiler.train()
    start_time = time.monotonic()
    losses: list[float] = []
    row_cursor = 0
    order = list(range(len(rows)))
    random.Random(args.seed + 1).shuffle(order)
    for update in range(args.updates):
        if row_cursor + args.batch_size > len(order):
            random.Random(args.seed + 2 + update).shuffle(order)
            row_cursor = 0
        selected = [
            rows[index]
            for index in order[row_cursor : row_cursor + args.batch_size]
        ]
        row_cursor += args.batch_size
        batch = collate_urom_rows(selected, device=device)
        scale = cosine_scale(update, args.updates, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * scale
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with context:
            receipt = urom_loss(compiler, batch)
        if not bool(torch.isfinite(receipt.total)):
            raise RuntimeError("UROM canary loss became non-finite")
        receipt.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("UROM canary gradient became non-finite")
        optimizer.step()
        losses.append(float(receipt.total.detach()))
        if (update + 1) % args.log_every == 0 or update == 0:
            print(
                json.dumps(
                    {
                        "update": update + 1,
                        "loss": losses[-1],
                        "mean_recent_loss": sum(losses[-args.log_every :])
                        / min(len(losses), args.log_every),
                        "gradient_norm": float(gradient_norm),
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    final_metrics = evaluate_rows(
        compiler,
        rows,
        batch_size=args.batch_size,
        device=device,
    )
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "base_sha256": args.base_sha256,
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "train_sha256": sha256_file(args.train),
        "seed": args.seed,
        "updates": args.updates,
        "compiler_config": compiler_config,
        "parameter_report": report,
        "adapter": adapter_state(compiler),
    }
    atomic_torch_save(checkpoint, args.out)
    receipt = {
        "schema": "urom3_train_only_canary_report_v1",
        "base_sha256": args.base_sha256,
        "tokenizer_sha256": checkpoint["tokenizer_sha256"],
        "train_sha256": checkpoint["train_sha256"],
        "checkpoint_sha256": sha256_file(args.out),
        "seed": args.seed,
        "rows": len(rows),
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "compiler_config": compiler_config,
        "parameter_report": report,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "initial_loss_mean": sum(losses[: min(10, len(losses))])
        / min(10, len(losses)),
        "final_loss_mean": sum(losses[-min(10, len(losses)) :])
        / min(10, len(losses)),
        "elapsed_seconds": time.monotonic() - start_time,
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    atomic_json(receipt, args.report)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
