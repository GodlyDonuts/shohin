#!/usr/bin/env python3
"""Train the CTAA source compiler from outcome-free training rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from ctaa_artifact_loader import (
    TOKENIZER_SHA256,
    load_qualified_memory_state,
    load_raw_trunk,
    require_sha256,
    verify_complete_system_parameters,
)
from ctaa_compiler_training import (
    TokenizedCompilerRow,
    collate_compiler_rows,
    compiler_batch_metrics,
    compiler_loss,
    parse_train_row,
)
from ctaa_neural_core import ClosureFeatureTransitionCore
from ctaa_trunk_compiler import TrunkCausalCTAACompiler


SCHEMA = "r12_ctaa_v2_compiler_training_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_train_rows(
    path: Path,
    tokenizer: Tokenizer,
    max_length: int,
) -> list[TokenizedCompilerRow]:
    result = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                result.append(parse_train_row(json.loads(line), tokenizer, max_length))
            except Exception as error:
                raise ValueError(f"CTAA compiler train row {line_number} failed") from error
    if not result:
        raise ValueError("CTAA compiler train file is empty")
    return result


def write_once(path: Path, payload: dict[str, object]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing existing CTAA compiler checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing CTAA compiler temporary: {temporary}")
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


def train(
    *,
    base_path: Path,
    qualified_path: Path,
    tokenizer_path: Path,
    train_path: Path,
    output: Path,
    seed: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    device_name: str,
) -> dict[str, object]:
    if updates < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("CTAA compiler training configuration differs")
    require_sha256(tokenizer_path, TOKENIZER_SHA256, "tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    trunk, base_receipt = load_raw_trunk(base_path)
    qualified = load_qualified_memory_state(qualified_path)
    compiler = TrunkCausalCTAACompiler(trunk)
    loaded = compiler.initialize_qualified_memory(qualified)
    core = ClosureFeatureTransitionCore()
    ledger = verify_complete_system_parameters(
        trunk,
        compiler.adapter_num_parameters,
        core.unique_parameters,
    )
    rows = load_train_rows(train_path, tokenizer, trunk.cfg.seq_len)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CTAA compiler training requires available CUDA")
    compiler.to(device).train()
    parameters = list(compiler.adapter_parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    last = None
    for _ in range(updates):
        indices = torch.randint(len(rows), (batch_size,), generator=generator).tolist()
        batch = collate_compiler_rows(
            [rows[index] for index in indices],
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            receipt = compiler_loss(compiler, batch)
        if not torch.isfinite(receipt.total):
            raise FloatingPointError("CTAA compiler loss is not finite")
        receipt.total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        last = {
            "total": float(receipt.total.detach()),
            "cards": float(receipt.cards.detach()),
            "initial": float(receipt.initial.detach()),
            "query": float(receipt.query.detach()),
        }
    compiler.eval()
    audit_indices = torch.randperm(len(rows), generator=generator)[: min(512, len(rows))].tolist()
    audit_batch = collate_compiler_rows(
        [rows[index] for index in audit_indices],
        device=device,
    )
    train_metrics = compiler_batch_metrics(compiler, audit_batch)
    adapter_state = {
        name: value.detach().cpu()
        for name, value in compiler.state_dict().items()
        if not name.startswith("model.")
    }
    payload = {
        "schema": SCHEMA,
        "adapter_state": adapter_state,
        "training": {
            "seed": seed,
            "updates": updates,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "rows": len(rows),
            "base_sha256": base_receipt.sha256,
            "base_step": base_receipt.step,
            "qualified_compiler_sha256": sha256_file(qualified_path),
            "qualified_memory_tensors": len(loaded),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "train_sha256": sha256_file(train_path),
            "adapter_parameters": compiler.adapter_num_parameters,
            "parameter_ledger": ledger,
            "last_loss": last,
            "train_sample_metrics": train_metrics,
            "development_access": 0,
            "confirmation_access": 0,
        },
    }
    digest = write_once(output, payload)
    return {"checkpoint_sha256": digest, **payload["training"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--qualified-compiler", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = train(
        base_path=args.base,
        qualified_path=args.qualified_compiler,
        tokenizer_path=args.tokenizer,
        train_path=args.train,
        output=args.output,
        seed=args.seed,
        updates=args.updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device_name=args.device,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
