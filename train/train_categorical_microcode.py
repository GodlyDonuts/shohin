#!/usr/bin/env python3
"""Train a frozen-base semantic compiler and learned categorical micro-ALU."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import time

import torch
from tokenizers import Tokenizer

from categorical_microcode import CategoricalMicrocodeCompiler, compile_example, sha256_file
from model import GPT, GPTConfig


def load_examples(path, tokenizer, seq_len):
    examples, skipped = [], collections.Counter()
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                example = compile_example(row, tokenizer)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid compiler row {}: {}".format(line_number, exc)) from exc
            if len(example.ids) > seq_len:
                skipped["overlength"] += 1
                continue
            examples.append(example)
    if not examples:
        raise ValueError("no compiler examples")
    return examples, dict(sorted(skipped.items()))


def bucketed_batches(examples, batch_size, seed):
    buckets = collections.defaultdict(list)
    for index, example in enumerate(examples):
        buckets[len(example.ids)].append(index)
    rng = random.Random(seed)
    batches, dropped = [], 0
    for length in sorted(buckets):
        indices = buckets[length]
        rng.shuffle(indices)
        usable = len(indices) // batch_size * batch_size
        dropped += len(indices) - usable
        batches.extend(indices[offset:offset + batch_size] for offset in range(0, usable, batch_size))
    rng.shuffle(batches)
    return batches, {"buckets": len(buckets), "full_batches": len(batches), "dropped": dropped}


def limit_batches(batches, max_examples, batch_size):
    if max_examples <= 0:
        return batches
    return batches[:max_examples // batch_size]


def lr_scale(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def adapter_state(compiler):
    return {
        name: value.detach().cpu()
        for name, value in compiler.state_dict().items()
        if not name.startswith("model.")
    }


def hash_adapter_state(compiler):
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter_state(compiler).items()):
        tensor = tensor.contiguous()
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def flatten_operation_positions(examples, indices, device):
    batch_indices, positions, targets = [], [], []
    for local, index in enumerate(indices):
        example = examples[index]
        for position, target in zip(example.operation_positions, example.operation_targets):
            batch_indices.append(local)
            positions.append(position)
            targets.append(target)
    return (
        torch.tensor(batch_indices, dtype=torch.long, device=device),
        torch.tensor(positions, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--basis-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.epochs <= 0 or args.max_examples < 0:
        raise SystemExit("invalid batch, epoch, or example limit")
    if not torch.cuda.is_available():
        raise SystemExit("categorical microcode training requires CUDA")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing output directory: {}".format(args.out))
    os.makedirs(args.out)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    with open(args.admission) as source:
        admission = json.load(source)
    data_sha256 = sha256_file(args.data)
    tokenizer_sha256 = sha256_file(args.tokenizer)
    if not admission.get("all_checks_pass"):
        raise SystemExit("categorical microcode admission did not pass")
    if admission.get("train_sha256") != data_sha256:
        raise SystemExit("categorical microcode admission does not bind training data")
    if admission.get("tokenizer_sha256") != tokenizer_sha256:
        raise SystemExit("categorical microcode admission does not bind tokenizer")
    checkpoint = torch.load(args.init, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    examples, skipped = load_examples(args.data, tokenizer, cfg.seq_len)
    batches, batch_report = bucketed_batches(examples, args.batch_size, args.seed)
    batches = limit_batches(batches, args.max_examples, args.batch_size)
    if not batches:
        raise SystemExit("selected zero complete compiler batches")
    batch_report["selected_batches"] = len(batches)
    batch_report["selected_examples"] = len(batches) * args.batch_size
    total_steps = len(batches) * args.epochs

    model = GPT(cfg).to("cuda")
    model.load_state_dict(checkpoint["model"])
    compiler = CategoricalMicrocodeCompiler(model, layer=args.layer, hidden=args.hidden).to("cuda")
    initial_hash = hash_adapter_state(compiler)
    trainable = list(compiler.adapter_parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    metadata = {
        "protocol": "causal_microcode_bottleneck_v1",
        "base_checkpoint": os.path.realpath(args.init),
        "base_sha256": sha256_file(args.init),
        "base_step": checkpoint.get("step"),
        "data": os.path.realpath(args.data),
        "data_sha256": data_sha256,
        "admission": os.path.realpath(args.admission),
        "admission_sha256": sha256_file(args.admission),
        "admission_eval_sha256": admission["eval_sha256"],
        "seed": args.seed,
        "layer": args.layer,
        "hidden": args.hidden,
        "batch_size": args.batch_size,
        "selected_examples": len(batches) * args.batch_size,
        "updates": total_steps,
        "adapter_parameters": compiler.adapter_num_params(),
        "base_parameters_trainable": 0,
        "initial_adapter_sha256": initial_hash,
        "lexical_frontend": "standalone_integer_extraction_and_line_segmentation_v1",
        "executor": "learned_exhaustive_decimal_transition_table_400_contexts_v1",
        "claim_boundary": (
            "Narrow semantic compilation plus learned categorical execution; numeric lexing is deterministic, "
            "and a pass is not broad language reasoning."
        ),
    }
    print(json.dumps({
        "categorical_microcode": metadata, "examples": len(examples),
        "skipped": skipped, "batch_report": batch_report,
    }, sort_keys=True), flush=True)

    started, step = time.time(), 0
    compiler.train()
    for epoch in range(args.epochs):
        epoch_batches, _ = bucketed_batches(examples, args.batch_size, args.seed + epoch)
        epoch_batches = limit_batches(epoch_batches, args.max_examples, args.batch_size)
        for indices in epoch_batches:
            ids = torch.tensor([examples[index].ids for index in indices], dtype=torch.long, device="cuda")
            op_batch, op_positions, op_targets = flatten_operation_positions(examples, indices, "cuda")
            query_positions = torch.tensor(
                [examples[index].query_position for index in indices], dtype=torch.long, device="cuda",
            )
            query_targets = torch.tensor(
                [examples[index].query_target for index in indices], dtype=torch.long, device="cuda",
            )
            scale = lr_scale(step, total_steps, args.warmup)
            optimizer.param_groups[0]["lr"] = args.lr * scale
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = compiler.encode(ids)
                operation_logits = compiler.classify_positions(
                    hidden, op_batch, op_positions, "operation",
                )
                query_logits = compiler.classify_positions(
                    hidden, torch.arange(len(indices), device="cuda"), query_positions, "query",
                )
                operation_loss = torch.nn.functional.cross_entropy(operation_logits.float(), op_targets)
                query_loss = torch.nn.functional.cross_entropy(query_logits.float(), query_targets)
                basis_loss = compiler.basis_loss()
                loss = operation_loss + query_loss + args.basis_weight * basis_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite compiler loss at step {}".format(step))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite compiler gradient at step {}".format(step))
            optimizer.step()
            if step % args.log_every == 0:
                op_acc = operation_logits.argmax(-1).eq(op_targets).float().mean().item()
                query_acc = query_logits.argmax(-1).eq(query_targets).float().mean().item()
                print(
                    "[microcode] step={}/{} loss={:.4f} op={:.4f} query={:.4f} basis={:.4f} "
                    "op_acc={:.3f} query_acc={:.3f} gnorm={:.3f} ex/s={:.1f}".format(
                        step, total_steps, loss.item(), operation_loss.item(), query_loss.item(),
                        basis_loss.item(), op_acc, query_acc, float(grad_norm),
                        args.batch_size * (step + 1) / max(1e-6, time.time() - started),
                    ), flush=True,
                )
            step += 1

    metadata["updates"] = step
    output = os.path.join(args.out, "microcode_adapter_ep1.pt")
    torch.save({
        "adapter_state": adapter_state(compiler),
        "categorical_microcode": metadata,
        "step": "microcode_adapter_ep1",
    }, output)
    print("[microcode] saved {}".format(output), flush=True)
    print("[microcode] done {} updates in {:.0f}s".format(step, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
