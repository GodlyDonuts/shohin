#!/usr/bin/env python3
"""Train one matched CTAA recurrent-core arm from train-only finite rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch

from ctaa_core_training import (
    ARMS,
    Arm,
    AtomicBatch,
    ClosureBatch,
    closure_label_derangement,
    make_core,
    matched_core_loss,
)
from run_ctaa_packet_executor import CORE_SCHEMA


CHECKPOINT_SCHEMA = "r12_ctaa_v2_core_training_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"CTAA train row {line_number} differs")
            rows.append(value)
    if not rows:
        raise ValueError("CTAA train file is empty")
    return rows


def load_atomic(path: Path) -> AtomicBatch:
    rows = _read_jsonl(path)
    expected = {"action", "state", "context", "output"}
    if any(set(row) != expected for row in rows):
        raise ValueError("CTAA atomic training schema differs")
    return AtomicBatch(
        action=torch.tensor([row["action"] for row in rows], dtype=torch.long),
        state=torch.tensor([row["state"] for row in rows], dtype=torch.long),
        output=torch.tensor([row["output"] for row in rows], dtype=torch.long),
    )


def load_closure(path: Path) -> ClosureBatch:
    rows = _read_jsonl(path)
    expected = {"first", "second", "state", "context", "composed", "output"}
    if any(set(row) != expected for row in rows):
        raise ValueError("CTAA closure training schema differs")
    return ClosureBatch(
        first=torch.tensor([row["first"] for row in rows], dtype=torch.long),
        second=torch.tensor([row["second"] for row in rows], dtype=torch.long),
        state=torch.tensor([row["state"] for row in rows], dtype=torch.long),
        composed=torch.tensor([row["composed"] for row in rows], dtype=torch.long),
        output=torch.tensor([row["output"] for row in rows], dtype=torch.long),
    )


def _slice(batch, indices: torch.Tensor, device: torch.device):
    return type(batch)(
        **{
            name: value.index_select(0, indices).to(device)
            for name, value in batch.__dict__.items()
        }
    )


def _unique_rows(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    joined = torch.cat(values, dim=1)
    indices = torch.unique(joined, dim=0, return_inverse=False, return_counts=False)
    widths = [value.shape[1] for value in values]
    return tuple(indices.split(widths, dim=1))


@torch.inference_mode()
def train_domain_metrics(core, atomic: AtomicBatch, closure: ClosureBatch, device: torch.device) -> dict[str, float]:
    action, state, output = _unique_rows(atomic.action, atomic.state, atomic.output)
    atomic_prediction = core(action.to(device), state.to(device)).argmax(-1).cpu()
    first, second, closure_state, composed, closure_output = _unique_rows(
        closure.first,
        closure.second,
        closure.state,
        closure.composed,
        closure.output,
    )
    composition_prediction = core(second.to(device), first.to(device)).argmax(-1).cpu()
    state_one = closure_state.gather(1, first)
    sequential_prediction = core(second.to(device), state_one.to(device)).argmax(-1).cpu()
    return {
        "atomic_exact": float(atomic_prediction.eq(output).all(1).float().mean()),
        "composition_exact": float(composition_prediction.eq(composed).all(1).float().mean()),
        "two_step_exact": float(sequential_prediction.eq(closure_output).all(1).float().mean()),
        "atomic_unique_cases": int(action.shape[0]),
        "closure_unique_cases": int(first.shape[0]),
    }


def write_checkpoint_once(path: Path, payload: dict[str, object]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing existing CTAA core checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing CTAA core temporary: {temporary}")
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
    arm: Arm,
    atomic_path: Path,
    closure_path: Path,
    output: Path,
    seed: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    device_name: str,
) -> dict[str, object]:
    if arm not in ARMS or updates < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("CTAA core training configuration differs")
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CTAA core training requires available CUDA")
    atomic = load_atomic(atomic_path)
    closure = load_closure(closure_path)
    core = make_core(arm).to(device)
    mapping = (
        closure_label_derangement(seed, closure.composed)
        if arm == "ctaa_shuffled_closure"
        else None
    )
    optimizer = torch.optim.AdamW(core.parameters(), lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    last = None
    for _ in range(updates):
        atomic_indices = torch.randint(
            len(atomic.action),
            (batch_size,),
            generator=generator,
        )
        closure_indices = torch.randint(
            len(closure.first),
            (batch_size,),
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            receipt = matched_core_loss(
                core,
                arm,
                _slice(atomic, atomic_indices, device),
                _slice(closure, closure_indices, device),
                shuffled_mapping=mapping if arm == "ctaa_shuffled_closure" else None,
            )
        if not torch.isfinite(receipt.total):
            raise FloatingPointError("CTAA core loss is not finite")
        receipt.total.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        optimizer.step()
        last = float(receipt.total.detach())
    core.eval()
    metrics = train_domain_metrics(core, atomic, closure, device)
    kind = "outer_product_control" if arm == "oprc_closure" else "closure_feature"
    payload = {
        "schema": CORE_SCHEMA,
        "kind": kind,
        "state": {name: value.detach().cpu() for name, value in core.state_dict().items()},
        "training": {
            "schema": CHECKPOINT_SCHEMA,
            "arm": arm,
            "seed": seed,
            "updates": updates,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "transition_calls_per_closure_row": 4,
            "atomic_sha256": sha256_file(atomic_path),
            "closure_sha256": sha256_file(closure_path),
            "last_loss": last,
            "train_domain_metrics": metrics,
            "development_access": 0,
            "confirmation_access": 0,
        },
    }
    digest = write_checkpoint_once(output, payload)
    return {"checkpoint_sha256": digest, **payload["training"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--atomic", type=Path, required=True)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = train(
        arm=args.arm,
        atomic_path=args.atomic,
        closure_path=args.closure,
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
