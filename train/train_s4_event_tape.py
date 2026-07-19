#!/usr/bin/env python3
"""Train the public S4 self-delimiting event parser."""

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
from self_delimiting_event_tape import (
    SelfDelimitingEventTapeParser,
    adapter_hash,
    adapter_state,
    build_kind_lexicon,
    load_examples,
    make_batches,
    pad_batch,
    parser_loss,
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
    parser.add_argument("--initializer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=5)
    parser.add_argument("--ff", type=int, default=1408)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--role-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026071904)
    parser.add_argument("--shuffle-supervision", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("S4 parser training requires CUDA")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing output {}".format(args.out))
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("invalid training schedule")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    report = json.load(open(args.report))
    if not report.get("all_gates_pass"):
        raise SystemExit("S4 corpus report did not pass")
    if report.get("schema") != "r12_s4_self_delimiting_event_tape_corpus_report_v1":
        raise SystemExit("unexpected S4 corpus schema")
    if report["artifacts"]["train"]["sha256"] != sha256_file(args.data):
        raise SystemExit("S4 report does not bind training data")
    if report["tokenizer_sha256"] != sha256_file(args.tokenizer):
        raise SystemExit("S4 report does not bind tokenizer")
    if report.get("confirmation_access") != 0:
        raise SystemExit("S4 corpus accessed confirmation")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint = torch.load(args.base, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    examples = load_examples(args.data, tokenizer, "s4_event_tape_train", cfg.seq_len)
    lexicon = build_kind_lexicon(examples)
    if args.shuffle_supervision:
        examples = shuffle_supervision(examples, args.seed ^ 0x5A4FF1E)
    model = GPT(cfg).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    event_parser = SelfDelimitingEventTapeParser(
        model,
        layer=args.layer,
        width=args.width,
        heads=args.heads,
        encoder_layers=args.encoder_layers,
        ff=args.ff,
    ).to("cuda")
    initializer = torch.load(args.initializer, map_location="cpu")
    initializer_metadata = initializer.get("compiler", {})
    if initializer_metadata.get("protocol") != (
        "r12_referential_literal_pointer_compiler_ordinary_tagger_factorized_development"
    ):
        raise SystemExit("invalid S4 memory initializer")
    if initializer_metadata.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("initializer/base mismatch")
    initialized_keys = event_parser.initialize_memory_encoder(initializer["adapter_state"])
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    adapter_parameters = event_parser.adapter_num_params()
    if base_parameters + adapter_parameters >= 150_000_000:
        raise SystemExit("S4 parser exceeds strict 150M cap")
    initial_adapter_sha256 = adapter_hash(event_parser)
    trainable = list(event_parser.adapter_parameters())
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
            "r12_s4_self_delimiting_event_parser_shuffled_control_v1"
            if args.shuffle_supervision else
            "r12_s4_self_delimiting_event_parser_treatment_v1"
        ),
        "base_sha256": sha256_file(args.base),
        "base_step": checkpoint.get("step"),
        "initializer_sha256": sha256_file(args.initializer),
        "initializer_adapter_sha256": initializer_metadata["final_adapter_sha256"],
        "initialized_memory_keys": len(initialized_keys),
        "data_sha256": sha256_file(args.data),
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "layer": args.layer,
        "width": args.width,
        "heads": args.heads,
        "encoder_layers": args.encoder_layers,
        "ff": args.ff,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "examples": len(examples),
        "updates": total_steps,
        "learning_rate": args.lr,
        "warmup": args.warmup,
        "clip": args.clip,
        "role_weight": args.role_weight,
        "semantic_weight": args.semantic_weight,
        "seed": args.seed,
        "shuffle_supervision": args.shuffle_supervision,
        "shuffle_contract": (
            "all target token positions are independently permuted within each source while "
            "preserving the target inventory and semantic labels"
            if args.shuffle_supervision else None
        ),
        "adapter_parameters": adapter_parameters,
        "base_parameters": base_parameters,
        "total_parameters": base_parameters + adapter_parameters,
        "base_parameters_trainable": 0,
        "initial_adapter_sha256": initial_adapter_sha256,
        "development_access": 0,
        "confirmation_access": 0,
        "inference_inputs": "whole-source token IDs and source-length mask only",
        "claim_boundary": (
            "Public variable-event parser development. No confirmation, unseen action "
            "semantics, planning, free-form reasoning, or novelty claim."
        ),
    }
    print(json.dumps({"s4_training": metadata}, sort_keys=True), flush=True)
    started = time.time()
    global_step = 0
    event_parser.train()
    event_parser.model.eval()
    for epoch, batches in enumerate(batches_by_epoch):
        for indices in batches:
            selected, ids, valid, roles = pad_batch(examples, indices, "cuda")
            optimizer.param_groups[0]["lr"] = args.lr * lr_scale(
                global_step, total_steps, args.warmup,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = event_parser(ids, valid)
                loss, components = parser_loss(
                    outputs,
                    selected,
                    roles,
                    role_weight=args.role_weight,
                    semantic_weight=args.semantic_weight,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite S4 loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite S4 gradient")
            optimizer.step()
            if global_step % args.log_every == 0:
                elapsed = max(1e-6, time.time() - started)
                print(json.dumps({
                    "update": global_step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "role_loss": float(components["role"].item()),
                    "kind_loss": float(components["kind"].item()),
                    "amount_loss": float(components["amount"].item()),
                    "query_loss": float(components["query"].item()),
                    "grad_norm": float(grad_norm.item()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "examples_per_second": (global_step + 1) * args.batch_size / elapsed,
                }, sort_keys=True), flush=True)
            global_step += 1
    metadata["elapsed_seconds"] = time.time() - started
    metadata["final_adapter_sha256"] = adapter_hash(event_parser)
    output = {
        "parser": metadata,
        "adapter_state": adapter_state(event_parser),
        "kind_lexicon": lexicon,
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
