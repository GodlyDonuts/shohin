#!/usr/bin/env python3
"""Train matched paired-view microcode compilers with optional equivalence loss."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from categorical_microcode import CategoricalMicrocodeCompiler, compile_example, sha256_file
from model import GPT, GPTConfig
from train_categorical_microcode import adapter_state, hash_adapter_state


def load_pairs(path, tokenizer, seq_len):
    grouped = {}
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            equivalence_id = row.get("equivalence_id")
            view = row.get("equivalence_view")
            if not isinstance(equivalence_id, str) or view not in (0, 1):
                raise ValueError("invalid pair metadata at row {}".format(line_number))
            example = compile_example(row, tokenizer)
            if len(example.ids) > seq_len:
                raise ValueError("overlength pair row {}".format(line_number))
            views = grouped.setdefault(equivalence_id, {})
            if view in views:
                raise ValueError("duplicate pair view at row {}".format(line_number))
            views[view] = example
    pairs = []
    for equivalence_id in sorted(grouped):
        views = grouped[equivalence_id]
        if set(views) != {0, 1}:
            raise ValueError("incomplete pair {}".format(equivalence_id))
        left, right = views[0], views[1]
        signature_left = (
            left.operation_targets, left.operation_values, left.query_target,
            left.initial_values, left.answer,
        )
        signature_right = (
            right.operation_targets, right.operation_values, right.query_target,
            right.initial_values, right.answer,
        )
        if signature_left != signature_right:
            raise ValueError("semantic mismatch in pair {}".format(equivalence_id))
        pairs.append((equivalence_id, left, right))
    if not pairs:
        raise ValueError("no equivalence pairs")
    return pairs


def make_batches(pairs, batch_pairs, max_pairs, seed):
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    if max_pairs > 0:
        indices = indices[:max_pairs]
    usable = len(indices) // batch_pairs * batch_pairs
    return [indices[offset:offset + batch_pairs] for offset in range(0, usable, batch_pairs)]


def pad_ids(examples, device):
    length = max(len(example.ids) for example in examples)
    return torch.tensor(
        [list(example.ids) + [0] * (length - len(example.ids)) for example in examples],
        dtype=torch.long, device=device,
    )


def flatten_operations(examples, device):
    batch_indices, positions, targets, slices = [], [], [], []
    cursor = 0
    for local, example in enumerate(examples):
        count = len(example.operation_positions)
        batch_indices.extend([local] * count)
        positions.extend(example.operation_positions)
        targets.extend(example.operation_targets)
        slices.append(slice(cursor, cursor + count))
        cursor += count
    return (
        torch.tensor(batch_indices, dtype=torch.long, device=device),
        torch.tensor(positions, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
        slices,
    )


def symmetric_kl(left, right):
    left_log = F.log_softmax(left.float(), dim=-1)
    right_log = F.log_softmax(right.float(), dim=-1)
    left_prob, right_prob = left_log.exp(), right_log.exp()
    return 0.5 * (
        F.kl_div(left_log, right_prob, reduction="batchmean")
        + F.kl_div(right_log, left_prob, reduction="batchmean")
    )


def lr_scale(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-pairs", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=48000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--basis-weight", type=float, default=1.0)
    parser.add_argument("--equivalence-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if args.batch_pairs <= 0 or args.max_pairs <= 0 or args.equivalence_weight < 0:
        raise SystemExit("invalid pair batch, pair limit, or equivalence weight")
    if not torch.cuda.is_available():
        raise SystemExit("paired microcode training requires CUDA")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing output directory: {}".format(args.out))
    os.makedirs(args.out)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    admission = json.load(open(args.admission))
    data_sha256 = sha256_file(args.data)
    tokenizer_sha256 = sha256_file(args.tokenizer)
    if not admission.get("all_checks_pass"):
        raise SystemExit("equivalence admission did not pass")
    if admission.get("train_sha256") != data_sha256:
        raise SystemExit("equivalence admission does not bind training data")
    if admission.get("tokenizer_sha256") != tokenizer_sha256:
        raise SystemExit("equivalence admission does not bind tokenizer")

    checkpoint = torch.load(args.init, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    pairs = load_pairs(args.data, tokenizer, cfg.seq_len)
    batches = make_batches(pairs, args.batch_pairs, args.max_pairs, args.seed)
    if not batches:
        raise SystemExit("selected zero pair batches")
    total_steps = len(batches)

    model = GPT(cfg).to("cuda")
    model.load_state_dict(checkpoint["model"])
    compiler = CategoricalMicrocodeCompiler(model, layer=args.layer, hidden=args.hidden).to("cuda")
    initial_hash = hash_adapter_state(compiler)
    trainable = list(compiler.adapter_parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    metadata = {
        "protocol": "causal_microcode_bottleneck_equivalence_v2",
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
        "batch_pairs": args.batch_pairs,
        "selected_pairs": len(batches) * args.batch_pairs,
        "selected_examples": len(batches) * args.batch_pairs * 2,
        "updates": total_steps,
        "equivalence_weight": args.equivalence_weight,
        "adapter_parameters": compiler.adapter_num_params(),
        "base_parameters_trainable": 0,
        "initial_adapter_sha256": initial_hash,
        "padding_contract": "right-padding only; causal read positions precede padding",
        "claim_boundary": (
            "Paired surface-invariance test for a narrow semantic compiler; numeric lexing and the "
            "categorical executor remain supplied structure."
        ),
    }
    print(json.dumps({"categorical_microcode_equivalence": metadata, "available_pairs": len(pairs)},
                     sort_keys=True), flush=True)

    compiler.train()
    started = time.time()
    for step, batch in enumerate(batches):
        first = [pairs[index][1] for index in batch]
        second = [pairs[index][2] for index in batch]
        examples = first + second
        ids = pad_ids(examples, "cuda")
        op_batch, op_positions, op_targets, op_slices = flatten_operations(examples, "cuda")
        query_positions = torch.tensor(
            [example.query_position for example in examples], dtype=torch.long, device="cuda",
        )
        query_targets = torch.tensor(
            [example.query_target for example in examples], dtype=torch.long, device="cuda",
        )
        optimizer.param_groups[0]["lr"] = args.lr * lr_scale(step, total_steps, args.warmup)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = compiler.encode(ids)
            operation_logits = compiler.classify_positions(
                hidden, op_batch, op_positions, "operation",
            )
            query_logits = compiler.classify_positions(
                hidden, torch.arange(len(examples), device="cuda"), query_positions, "query",
            )
            operation_loss = F.cross_entropy(operation_logits.float(), op_targets)
            query_loss = F.cross_entropy(query_logits.float(), query_targets)
            pair_losses = []
            for local in range(len(batch)):
                pair_losses.append(symmetric_kl(
                    operation_logits[op_slices[local]],
                    operation_logits[op_slices[len(batch) + local]],
                ))
            pair_losses.append(symmetric_kl(query_logits[:len(batch)], query_logits[len(batch):]))
            equivalence_loss = torch.stack(pair_losses).mean()
            basis_loss = compiler.basis_loss()
            loss = (
                operation_loss + query_loss + args.basis_weight * basis_loss
                + args.equivalence_weight * equivalence_loss
            )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite paired compiler loss at step {}".format(step))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite paired compiler gradient at step {}".format(step))
        optimizer.step()
        if step % args.log_every == 0:
            op_acc = operation_logits.argmax(-1).eq(op_targets).float().mean().item()
            query_acc = query_logits.argmax(-1).eq(query_targets).float().mean().item()
            print(
                "[microcode-equiv] step={}/{} loss={:.4f} op={:.4f} query={:.4f} basis={:.4f} "
                "equiv={:.4f} op_acc={:.3f} query_acc={:.3f} gnorm={:.3f}".format(
                    step, total_steps, loss.item(), operation_loss.item(), query_loss.item(),
                    basis_loss.item(), equivalence_loss.item(), op_acc, query_acc, float(grad_norm),
                ), flush=True,
            )

    output = os.path.join(args.out, "microcode_adapter_ep1.pt")
    torch.save({
        "adapter_state": adapter_state(compiler),
        "categorical_microcode": metadata,
        "step": "microcode_adapter_ep1",
    }, output)
    print("[microcode-equiv] saved {}".format(output), flush=True)
    print("[microcode-equiv] done {} updates in {:.0f}s".format(total_steps, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
