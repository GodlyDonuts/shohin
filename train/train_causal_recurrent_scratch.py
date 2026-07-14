#!/usr/bin/env python3
"""Train a frozen-base source-visible recurrent scratch adapter.

The recurrent and reset arms use the same base checkpoint, adapter
initialization, examples, batches, optimizer, and number of cell executions.
The reset arm restarts the slots before every execution and therefore cannot
accumulate a multi-step state.  Only adapter parameters are optimized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch
from tokenizers import Tokenizer

from causal_recurrent_scratch import CausalRecurrentScratch
from latent_rollout_train import (
    bucketed_batches,
    limit_complete_batches,
    load_examples,
    lr_scale,
    make_batch,
)
from model import GPT, GPTConfig


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_state(adapter):
    return {
        name: value.detach().cpu()
        for name, value in adapter.state_dict().items()
        if not name.startswith("model.")
    }


def hash_adapter_state(adapter) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(adapter_state(adapter).items()):
        tensor = value.detach().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("recurrent", "reset"), required=True)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--workspace-topk", type=int, default=8)
    parser.add_argument("--workspace-temperature", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--eos", default="<|endoftext|>")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps <= 0 or args.epochs <= 0 or args.batch_size <= 0 or args.max_examples < 0:
        raise SystemExit("steps, epochs, and batch size must be positive; max-examples non-negative")
    if not torch.cuda.is_available():
        raise SystemExit("causal recurrent scratch training requires a CUDA allocation")
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit("refusing non-empty output directory: {}".format(args.out))
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos)
    if eos_id is None:
        raise SystemExit("tokenizer EOS token is missing")
    examples, skipped = load_examples(args.data, tokenizer, eos_id, 0)
    batches, batch_report = bucketed_batches(examples, args.batch_size, args.seed)
    batches = limit_complete_batches(batches, args.max_examples, args.batch_size)
    if not batches:
        raise SystemExit("selected zero complete batches")
    batch_report["selected_batches"] = len(batches)
    batch_report["selected_examples"] = len(batches) * args.batch_size
    total_steps = args.epochs * len(batches)

    checkpoint = torch.load(args.init, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    model = GPT(cfg).to("cuda")
    model.load_state_dict(checkpoint["model"])
    adapter = CausalRecurrentScratch(
        model, layer=args.layer, slots=args.slots, width=args.width,
        workspace_topk=args.workspace_topk,
        workspace_temperature=args.workspace_temperature,
    ).to("cuda")
    initial_adapter_sha256 = hash_adapter_state(adapter)
    trainable = list(adapter.adapter_parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    recurrent = args.mode == "recurrent"
    metadata = {
        "protocol": "causal_recurrent_scratch_v1",
        "mode": args.mode,
        "base_checkpoint": os.path.realpath(args.init),
        "base_sha256": sha256_file(args.init),
        "base_step": checkpoint.get("step"),
        "data": os.path.realpath(args.data),
        "data_sha256": sha256_file(args.data),
        "seed": args.seed,
        "layer": args.layer,
        "slots": args.slots,
        "width": args.width,
        "steps": args.steps,
        "workspace_topk": args.workspace_topk,
        "workspace_temperature": args.workspace_temperature,
        "workspace_basis": "frozen_normalized_unembedding_topk_v1" if args.workspace_topk else "unconstrained_linear",
        "batch_size": args.batch_size,
        "selected_examples_per_epoch": len(batches) * args.batch_size,
        "updates": total_steps,
        "adapter_parameters": adapter.adapter_num_params(),
        "initial_adapter_sha256": initial_adapter_sha256,
        "reset_aggregation": "mean_of_identical_independent_executions_v1",
        "base_parameters_trainable": 0,
        "source_present_at_decode": True,
        "answer_tokens_visible_to_scratch": False,
        "claim_boundary": (
            "Frozen-base mechanics and causal-state experiment. A positive fit is not a broad reasoning claim."
        ),
    }
    print(json.dumps({
        "causal_recurrent_scratch": metadata,
        "examples": len(examples),
        "skipped": skipped,
        "batch_report": batch_report,
    }, sort_keys=True), flush=True)

    started, step = time.time(), 0
    adapter.train()
    for epoch in range(args.epochs):
        epoch_batches, _ = bucketed_batches(examples, args.batch_size, args.seed + epoch)
        epoch_batches = limit_complete_batches(epoch_batches, args.max_examples, args.batch_size)
        for indices in epoch_batches:
            prompts, answers = make_batch(examples, indices, "cuda")
            scale = lr_scale(step, total_steps, args.warmup)
            optimizer.param_groups[0]["lr"] = args.lr * scale
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _, _ = adapter.supervised_loss(
                    prompts, answers, eos_id, args.steps, recurrent=recurrent,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite scratch loss at step {}".format(step))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite scratch gradient at step {}".format(step))
            optimizer.step()
            if step % args.log_every == 0:
                rate = args.batch_size * (step + 1) / max(1e-6, time.time() - started)
                print(
                    "[scratch] mode={} epoch={} step={}/{} loss={:.4f} gnorm={:.3f} "
                    "gate={:.5f} lr={:.6f} ex/s={:.1f}".format(
                        args.mode, epoch, step, total_steps, loss.item(), float(grad_norm),
                        float(torch.tanh(adapter.readout_gate).detach()), args.lr * scale, rate,
                    ),
                    flush=True,
                )
            step += 1

    metadata["updates"] = step
    output = os.path.join(args.out, "scratch_adapter_ep1.pt")
    if os.path.exists(output):
        raise RuntimeError("refusing existing output: {}".format(output))
    torch.save({
        "adapter_state": adapter_state(adapter),
        "causal_recurrent_scratch": metadata,
        "step": "scratch_adapter_ep1",
    }, output)
    print("[scratch] saved {}".format(output), flush=True)
    print("[scratch] done {} updates in {:.0f}s".format(step, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
