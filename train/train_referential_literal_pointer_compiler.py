#!/usr/bin/env python3
"""Fit the development-only complete source-pointer compiler pilot."""

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
from referential_literal_pointer_compiler import (
    CompletePointerCompiler,
    TARGET_LABELS,
    adapter_hash,
    adapter_state,
    compiler_loss,
    load_examples,
    make_batches,
    pad_batch,
    role_supervision_loss,
    sha256_file,
)


CORPUS_SCHEMAS = {
    "r12_referential_literal_pointer_corpus_v1": "",
    "r12_referential_literal_pointer_factorized_corpus_v1": "_factorized",
}


def lr_scale(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--ff", type=int, default=1024)
    parser.add_argument("--encoder-layers", type=int, default=0)
    parser.add_argument("--role-supervision", action="store_true")
    parser.add_argument("--role-weight", type=float, default=0.0)
    parser.add_argument("--separate-kind-decoder", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--kind-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026071803)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("complete compiler training requires CUDA")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing output {}".format(args.out))
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("invalid epochs or batch size")
    if args.encoder_layers < 0 or args.role_weight < 0:
        raise SystemExit("invalid encoder layers or role weight")
    if bool(args.role_weight) != bool(args.role_supervision):
        raise SystemExit("role supervision and positive role weight must be enabled together")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    report = json.load(open(args.report))
    data_sha256 = sha256_file(args.data)
    tokenizer_sha256 = sha256_file(args.tokenizer)
    if not report.get("all_gates_pass"):
        raise SystemExit("corpus report did not pass")
    if report.get("artifacts", {}).get("train", {}).get("sha256") != data_sha256:
        raise SystemExit("corpus report does not bind training bytes")
    if report.get("tokenizer_sha256") != tokenizer_sha256:
        raise SystemExit("corpus report does not bind tokenizer")
    corpus_schema = report.get("schema")
    if corpus_schema not in CORPUS_SCHEMAS:
        raise SystemExit("unsupported complete-pointer corpus schema")
    split_rows = report.get("splits", {})
    expected_pointer_labels = sum(
        int(split.get("rows", 0)) for split in split_rows.values()
    ) * len(TARGET_LABELS)
    if report.get("acquisition_ledger", {}).get("target_pointer_labels") != expected_pointer_labels:
        raise SystemExit("complete-pointer label ledger does not match split rows")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint = torch.load(args.base, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    examples = load_examples(
        args.data, tokenizer, "train", cfg.seq_len, keep_evidence=False,
        limit=args.max_examples,
    )
    model = GPT(cfg).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    compiler = CompletePointerCompiler(
        model,
        layer=args.layer,
        width=args.width,
        heads=args.heads,
        decoder_layers=args.decoder_layers,
        ff=args.ff,
        encoder_layers=args.encoder_layers,
        role_supervision=args.role_supervision,
        separate_kind_decoder=args.separate_kind_decoder,
    ).to("cuda")
    if compiler.adapter_num_params() + sum(parameter.numel() for parameter in model.parameters()) >= 150_000_000:
        raise SystemExit("compiler exceeds strict 150M total-parameter cap")
    initial_adapter_sha256 = adapter_hash(compiler)
    trainable = list(compiler.adapter_parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    epoch_batches = [make_batches(examples, args.batch_size, args.seed + epoch) for epoch in range(args.epochs)]
    total_steps = sum(len(batches) for batches in epoch_batches)
    protocol = (
            "r12_referential_literal_pointer_compiler_v1_3_islands_development"
            if args.separate_kind_decoder else
            "r12_referential_literal_pointer_compiler_v1_2_structured_development"
            if args.encoder_layers or args.role_supervision else
            "r12_referential_literal_pointer_compiler_v1_1_development"
        )
    if CORPUS_SCHEMAS[corpus_schema]:
        protocol = protocol.replace(
            "_development", CORPUS_SCHEMAS[corpus_schema] + "_development",
        )
    metadata = {
        "protocol": protocol,
        "corpus_schema": corpus_schema,
        "base": os.path.realpath(args.base),
        "base_sha256": sha256_file(args.base),
        "base_step": checkpoint.get("step"),
        "data": os.path.realpath(args.data),
        "data_sha256": data_sha256,
        "report": os.path.realpath(args.report),
        "report_sha256": sha256_file(args.report),
        "tokenizer": os.path.realpath(args.tokenizer),
        "tokenizer_sha256": tokenizer_sha256,
        "layer": args.layer,
        "width": args.width,
        "heads": args.heads,
        "decoder_layers": args.decoder_layers,
        "ff": args.ff,
        "encoder_layers": args.encoder_layers,
        "role_supervision": args.role_supervision,
        "role_weight": args.role_weight,
        "separate_kind_decoder": args.separate_kind_decoder,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "examples": len(examples),
        "updates": total_steps,
        "learning_rate": args.lr,
        "warmup": args.warmup,
        "clip": args.clip,
        "kind_weight": args.kind_weight,
        "seed": args.seed,
        "adapter_parameters": compiler.adapter_num_params(),
        "base_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "total_parameters": compiler.adapter_num_params() + sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "base_parameters_trainable": 0,
        "initial_adapter_sha256": initial_adapter_sha256,
        "confirmation_access": 0,
        "development_evaluation_access": 0,
        "inference_inputs": "source token IDs and source-length mask only",
        "claim_boundary": (
            "Development-only complete pointer-compiler feasibility pilot. No confirmation, "
            "executor, halt, native-reasoning, or novelty claim."
        ),
    }
    print(json.dumps({"compiler_training": metadata}, sort_keys=True), flush=True)

    started = time.time()
    global_step = 0
    compiler.train()
    compiler.model.eval()
    for epoch, batches in enumerate(epoch_batches):
        for indices in batches:
            selected, ids, valid = pad_batch(examples, indices, "cuda")
            optimizer.param_groups[0]["lr"] = args.lr * lr_scale(global_step, total_steps, args.warmup)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = compiler(ids, valid)
                loss, pointer, kind, _ = compiler_loss(outputs, selected, args.kind_weight)
                role = (
                    role_supervision_loss(outputs, selected)
                    if args.role_supervision else loss.new_zeros(())
                )
                loss = loss + args.role_weight * role
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss at update {}".format(global_step))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite gradient at update {}".format(global_step))
            optimizer.step()
            if global_step % args.log_every == 0:
                elapsed = max(1e-6, time.time() - started)
                print(json.dumps({
                    "update": global_step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "pointer_loss": float(pointer.item()),
                    "kind_loss": float(kind.item()),
                    "role_loss": float(role.item()),
                    "grad_norm": float(grad_norm.item()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "examples_per_second": (global_step + 1) * args.batch_size / elapsed,
                }, sort_keys=True), flush=True)
            global_step += 1

    metadata["elapsed_seconds"] = time.time() - started
    metadata["final_adapter_sha256"] = adapter_hash(compiler)
    output = {
        "compiler": metadata,
        "adapter_state": adapter_state(compiler),
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
