#!/usr/bin/env python3
"""Train only S4 v2 event-relative pointer heads over the frozen v1 parser."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from s4_event_relative_pointer import (
    POINTER_PREFIXES,
    EventRelativePointerParser,
    event_relative_pointer_loss,
)
from self_delimiting_event_tape import (
    adapter_hash,
    adapter_state,
    load_examples,
    make_batches,
    pad_batch,
    sha256_file,
    shuffle_supervision,
)


def lr_scale(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--v1-parser", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026071905)
    parser.add_argument("--shuffle-supervision", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("S4 v2 training requires CUDA")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing S4 v2 output")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    report = json.load(open(args.report))
    if not report.get("all_gates_pass"):
        raise SystemExit("S4 training corpus report did not pass")
    if report["artifacts"]["train"]["sha256"] != sha256_file(args.data):
        raise SystemExit("S4 report does not bind training data")
    if report.get("confirmation_access") != 0:
        raise SystemExit("S4 corpus accessed confirmation")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint = torch.load(args.base, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    examples = load_examples(args.data, tokenizer, "s4_event_tape_train", cfg.seq_len)
    if args.shuffle_supervision:
        examples = shuffle_supervision(examples, args.seed ^ 0xE7A21)

    v1 = torch.load(args.v1_parser, map_location="cpu")
    v1_metadata = v1.get("parser", {})
    if v1_metadata.get("protocol") != "r12_s4_self_delimiting_event_parser_treatment_v1":
        raise SystemExit("invalid frozen S4 v1 parser")
    if v1_metadata.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("S4 v1/base mismatch")
    if v1_metadata.get("data_sha256") != sha256_file(args.data):
        raise SystemExit("S4 v1/training-data mismatch")
    model = GPT(cfg).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    event_parser = EventRelativePointerParser(
        model,
        layer=int(v1_metadata["layer"]),
        width=int(v1_metadata["width"]),
        heads=int(v1_metadata["heads"]),
        encoder_layers=int(v1_metadata["encoder_layers"]),
        ff=int(v1_metadata["ff"]),
    ).to("cuda")
    initialized = event_parser.initialize_v1(v1["adapter_state"])
    trainable = list(event_parser.pointer_parameters())
    if not trainable or any(not parameter.requires_grad for parameter in trainable):
        raise SystemExit("invalid S4 v2 pointer parameter set")
    if any(parameter.requires_grad for name, parameter in event_parser.named_parameters()
           if not name.startswith(POINTER_PREFIXES)):
        raise SystemExit("non-pointer S4 parameter is trainable")
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    adapter_parameters = event_parser.adapter_num_params()
    pointer_parameters = event_parser.pointer_num_params()
    if base_parameters + adapter_parameters >= 150_000_000:
        raise SystemExit("S4 v2 exceeds strict 150M cap")

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    batches_by_epoch = [
        make_batches(examples, args.batch_size, args.seed + epoch)
        for epoch in range(args.epochs)
    ]
    total_steps = sum(len(batches) for batches in batches_by_epoch)
    metadata = {
        "protocol": (
            "r12_s4_event_relative_pointer_shuffled_v2"
            if args.shuffle_supervision else
            "r12_s4_event_relative_pointer_treatment_v2"
        ),
        "base_sha256": sha256_file(args.base),
        "base_step": checkpoint.get("step"),
        "v1_parser_sha256": sha256_file(args.v1_parser),
        "v1_adapter_sha256": v1_metadata["final_adapter_sha256"],
        "initialized_v1_keys": len(initialized),
        "data_sha256": sha256_file(args.data),
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "layer": int(v1_metadata["layer"]),
        "width": int(v1_metadata["width"]),
        "heads": int(v1_metadata["heads"]),
        "encoder_layers": int(v1_metadata["encoder_layers"]),
        "ff": int(v1_metadata["ff"]),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "examples": len(examples),
        "updates": total_steps,
        "learning_rate": args.lr,
        "warmup": args.warmup,
        "clip": args.clip,
        "seed": args.seed,
        "shuffle_supervision": args.shuffle_supervision,
        "adapter_parameters": adapter_parameters,
        "pointer_parameters": pointer_parameters,
        "base_parameters": base_parameters,
        "total_parameters": base_parameters + adapter_parameters,
        "trainable_parameters": pointer_parameters,
        "development_access": 0,
        "confirmation_access": 0,
        "inference_inputs": "whole-source token IDs and source-length mask only",
    }
    print(json.dumps({"s4_v2_training": metadata}, sort_keys=True), flush=True)
    started = time.time()
    step = 0
    event_parser.train()
    event_parser.model.eval()
    for epoch, batches in enumerate(batches_by_epoch):
        for indices in batches:
            selected, ids, valid, _ = pad_batch(examples, indices, "cuda")
            optimizer.param_groups[0]["lr"] = args.lr * lr_scale(step, total_steps, args.warmup)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = event_parser(ids, valid)
                loss, components = event_relative_pointer_loss(
                    event_parser, outputs, selected, valid,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite S4 v2 loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite S4 v2 gradient")
            optimizer.step()
            if step % args.log_every == 0:
                elapsed = max(1e-6, time.time() - started)
                print(json.dumps({
                    "update": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    **{"{}_loss".format(name): float(value.item()) for name, value in components.items()},
                    "grad_norm": float(grad_norm.item()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "examples_per_second": (step + 1) * args.batch_size / elapsed,
                }, sort_keys=True), flush=True)
            step += 1
    metadata["elapsed_seconds"] = time.time() - started
    metadata["final_adapter_sha256"] = adapter_hash(event_parser)
    output = {
        "parser": metadata,
        "adapter_state": adapter_state(event_parser),
        "kind_lexicon": v1["kind_lexicon"],
    }
    os.makedirs(os.path.dirname(os.path.realpath(args.out)), exist_ok=True)
    torch.save(output, args.out)
    print(json.dumps({
        "saved": os.path.realpath(args.out),
        "adapter_sha256": metadata["final_adapter_sha256"],
        "elapsed_seconds": metadata["elapsed_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
