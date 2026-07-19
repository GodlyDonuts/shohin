#!/usr/bin/env python3
"""Train the tied S3 categorical register from independent atomic updates."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from categorical_permutation_executor import (
    S3CategoricalPermutationExecutor,
    categorical_executor_loss,
    categorical_identity_packet,
    module_state_hash,
    select_categorical_operations,
)
from model import GPTConfig
from referential_gather_delete_executor import execution_targets
from referential_literal_pointer_compiler import load_examples, make_batches, pad_batch, sha256_file
from train_referential_gather_delete_executor import load_frozen_compiler, lr_scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--expected-compiler-sha256", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026071903)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("S3 categorical executor training requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing S3 executor output")
    if sha256_file(args.compiler) != args.expected_compiler_sha256:
        raise SystemExit("S3 compiler hash mismatch")
    report = json.load(open(args.report))
    if not report.get("all_gates_pass"):
        raise SystemExit("factorized corpus report did not pass")
    if report.get("artifacts", {}).get("train", {}).get("sha256") != sha256_file(args.data):
        raise SystemExit("factorized report does not bind S3 training data")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda"
    checkpoint, compiler, compiler_metadata = load_frozen_compiler(
        args.base, args.compiler, args.tokenizer, device,
    )
    tokenizer = Tokenizer.from_file(args.tokenizer)
    cfg = GPTConfig(**checkpoint["cfg"])
    examples = load_examples(
        args.data, tokenizer, "train", cfg.seq_len, keep_evidence=True,
    )
    executor = S3CategoricalPermutationExecutor(
        identity_context_width=int(cfg.d_model),
        context_width=int(compiler_metadata["width"]),
        width=args.width,
    ).to(device)
    base_parameters = sum(parameter.numel() for parameter in compiler.model.parameters())
    compiler_parameters = compiler.adapter_num_params()
    executor_parameters = executor.num_params()
    total_parameters = base_parameters + compiler_parameters + executor_parameters
    if total_parameters >= 150_000_000:
        raise SystemExit("S3 system exceeds strict 150M parameter cap")
    initial_hash = module_state_hash(executor)
    optimizer = torch.optim.AdamW(
        executor.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    epoch_batches = [
        make_batches(examples, args.batch_size, args.seed + epoch)
        for epoch in range(args.epochs)
    ]
    total_steps = sum(len(batches) for batches in epoch_batches)
    metadata = {
        "protocol": "r12_s3_categorical_permutation_executor_v1",
        "training_identity_mode": "gold",
        "training_contract": (
            "independent atomic op0/op1 updates from identity state; no composed or "
            "long-program supervision"
        ),
        "source_deleted": True,
        "base_sha256": sha256_file(args.base),
        "compiler_sha256": sha256_file(args.compiler),
        "compiler_adapter_sha256": compiler_metadata["final_adapter_sha256"],
        "data_sha256": sha256_file(args.data),
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "examples": len(examples),
        "atomic_examples_per_epoch": 2 * len(examples),
        "updates": total_steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "warmup": args.warmup,
        "clip": args.clip,
        "seed": args.seed,
        "identity_context_width": int(cfg.d_model),
        "context_width": int(compiler_metadata["width"]),
        "executor_width": args.width,
        "base_parameters": base_parameters,
        "compiler_parameters": compiler_parameters,
        "executor_parameters": executor_parameters,
        "total_parameters": total_parameters,
        "base_parameters_trainable": 0,
        "compiler_parameters_trainable": 0,
        "initial_executor_sha256": initial_hash,
        "confirmation_access": 0,
        "claim_boundary": (
            "Public-development source-deleted S3 execution component. External schedule/halt; "
            "no confirmation, autonomous reasoning, or novelty claim."
        ),
    }
    print(json.dumps({"s3_training": metadata}, sort_keys=True), flush=True)

    started = time.time()
    global_step = 0
    executor.train()
    for epoch, batches in enumerate(epoch_batches):
        for indices in batches:
            selected, ids, valid = pad_batch(examples, indices, device)
            optimizer.param_groups[0]["lr"] = args.lr * lr_scale(
                global_step, total_steps, args.warmup,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                compiler_outputs = compiler(ids, valid)
                packet = categorical_identity_packet(
                    compiler_outputs, selected, ids, valid, mode="gold",
                )
            atomic_losses = []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for operation_index in (0, 1):
                    outputs = executor(select_categorical_operations(packet, (operation_index,)))
                    targets = [
                        execution_targets(example, (operation_index,)) for example in selected
                    ]
                    atomic_losses.append(categorical_executor_loss(outputs, targets))
                losses = {
                    name: torch.stack([loss[name] for loss in atomic_losses]).mean()
                    for name in atomic_losses[0]
                }
            loss = losses["total"]
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite S3 loss at update {}".format(global_step))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(executor.parameters(), args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite S3 gradient at update {}".format(global_step))
            optimizer.step()
            if global_step % args.log_every == 0:
                print(json.dumps({
                    "update": global_step,
                    "epoch": epoch,
                    "grad_norm": float(grad_norm),
                    "lr": optimizer.param_groups[0]["lr"],
                    **{"{}_loss".format(name): float(value) for name, value in losses.items()},
                }, sort_keys=True), flush=True)
            global_step += 1

    metadata["elapsed_seconds"] = time.time() - started
    metadata["final_executor_sha256"] = module_state_hash(executor)
    output = {
        "executor": metadata,
        "executor_state": {
            name: value.detach().cpu() for name, value in executor.state_dict().items()
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    print(json.dumps({
        "saved": os.path.realpath(args.out),
        "executor_sha256": metadata["final_executor_sha256"],
        "elapsed_seconds": metadata["elapsed_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
