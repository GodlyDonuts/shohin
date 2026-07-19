#!/usr/bin/env python3
"""Train atomic source-deleted list updates for two-step composition evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from referential_gather_delete_executor import (
    GatherDeletePermutationExecutor,
    SourceRetainedAnswerControl,
    execution_targets,
    executor_loss,
    executor_state_hash,
    gather_source_deleted_packet,
    select_packet_operations,
)
from referential_literal_pointer_compiler import (
    OrdinaryTokenTaggerCompiler,
    load_examples,
    make_batches,
    pad_batch,
    sha256_file,
)


EXPECTED_COMPILER_PROTOCOL = (
    "r12_referential_literal_pointer_compiler_ordinary_tagger_factorized_development"
)


def lr_scale(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_frozen_compiler(base_path, compiler_path, tokenizer_path, device):
    checkpoint = torch.load(base_path, map_location="cpu")
    bundle = torch.load(compiler_path, map_location="cpu")
    metadata = bundle.get("compiler", {})
    if metadata.get("protocol") != EXPECTED_COMPILER_PROTOCOL:
        raise SystemExit("Stage B requires the qualified ordinary factorized compiler")
    if metadata.get("confirmation_access") != 0:
        raise SystemExit("compiler metadata records confirmation access")
    if metadata.get("base_sha256") != sha256_file(base_path):
        raise SystemExit("compiler/base identity mismatch")
    if metadata.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise SystemExit("compiler/tokenizer identity mismatch")
    cfg = GPTConfig(**checkpoint["cfg"])
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    compiler = OrdinaryTokenTaggerCompiler(
        model,
        layer=int(metadata["layer"]),
        width=int(metadata["width"]),
        heads=int(metadata["heads"]),
        encoder_layers=int(metadata["encoder_layers"]),
        ff=int(metadata["ff"]),
    ).to(device).eval()
    missing, unexpected = compiler.load_state_dict(bundle["adapter_state"], strict=False)
    missing = [name for name in missing if not name.startswith("model.")]
    unexpected = [name for name in unexpected if not name.startswith("model.")]
    if missing or unexpected:
        raise SystemExit("compiler adapter mismatch missing={} unexpected={}".format(
            missing, unexpected,
        ))
    compiler.requires_grad_(False)
    return checkpoint, compiler, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--expected-compiler-sha256", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--arm", choices=("tied", "untied", "tied_composed", "source_retained"),
        required=True,
    )
    parser.add_argument("--packet-oracle", choices=("none", "full"), default="none")
    parser.add_argument(
        "--packet-mode",
        choices=("contextual_softmax", "lexical_sigmoid_span"),
        default="contextual_softmax",
    )
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026071901)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Stage-B executor training requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing executor output")
    if args.arm == "source_retained" and args.packet_oracle != "none":
        raise SystemExit("source-retained control does not use a packet oracle")
    if args.epochs <= 0 or args.batch_size <= 0 or args.width <= 0:
        raise SystemExit("invalid executor training dimensions")
    if sha256_file(args.compiler) != args.expected_compiler_sha256:
        raise SystemExit("compiler file does not match frozen Stage-B identity")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    report = json.load(open(args.report))
    if not report.get("all_gates_pass"):
        raise SystemExit("factorized corpus report did not pass")
    if report.get("artifacts", {}).get("train", {}).get("sha256") != sha256_file(args.data):
        raise SystemExit("factorized report does not bind training bytes")
    if report.get("tokenizer_sha256") != sha256_file(args.tokenizer):
        raise SystemExit("factorized report does not bind tokenizer")

    device = "cuda"
    checkpoint, compiler, compiler_metadata = load_frozen_compiler(
        args.base, args.compiler, args.tokenizer, device,
    )
    tokenizer = Tokenizer.from_file(args.tokenizer)
    cfg = GPTConfig(**checkpoint["cfg"])
    examples = load_examples(
        args.data,
        tokenizer,
        "train",
        cfg.seq_len,
        keep_evidence=True,
        limit=args.max_examples,
    )
    context_width = int(compiler_metadata["width"])
    identity_width = (
        int(cfg.d_model)
        if args.packet_mode == "lexical_sigmoid_span" else
        context_width
    )
    if args.arm == "source_retained":
        executor = SourceRetainedAnswerControl(
            packet_width=context_width,
            width=args.width,
            heads=8,
            layers=2,
            ff=4 * args.width,
        ).to(device)
    else:
        executor = GatherDeletePermutationExecutor(
            identity_width=identity_width,
            context_width=context_width,
            width=args.width,
            tied=args.arm in {"tied", "tied_composed"},
        ).to(device)
    base_parameters = sum(parameter.numel() for parameter in compiler.model.parameters())
    compiler_parameters = compiler.adapter_num_params()
    executor_parameters = executor.num_params()
    total_parameters = base_parameters + compiler_parameters + executor_parameters
    if total_parameters >= 150_000_000:
        raise SystemExit("Stage-B system exceeds strict 150M parameter cap")
    initial_executor_sha256 = executor_state_hash(executor)
    optimizer = torch.optim.AdamW(
        executor.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    epoch_batches = [
        make_batches(examples, args.batch_size, args.seed + epoch)
        for epoch in range(args.epochs)
    ]
    total_steps = sum(len(batches) for batches in epoch_batches)
    metadata = {
        "protocol": (
            "r12_referential_gather_delete_executor_stage_b_v1_1"
            if args.packet_mode == "lexical_sigmoid_span" else
            "r12_referential_gather_delete_executor_stage_b_v1"
        ),
        "arm": args.arm,
        "source_deleted": args.arm != "source_retained",
        "training_contract": (
            "full two-operation transition/answer supervision; favorable architecture ceiling"
            if args.arm == "tied_composed" else
            "atomic one-operation supervision only; op0 and op1 are independently applied "
            "from the identity initial state; two-step composition is evaluation-only"
            if args.arm != "source_retained" else
            "favorable direct-answer control trained on complete two-operation answers with "
            "unrestricted frozen compiler source-memory access"
        ),
        "packet_oracle": args.packet_oracle,
        "packet_mode": args.packet_mode,
        "base_sha256": sha256_file(args.base),
        "compiler_file_sha256": sha256_file(args.compiler),
        "compiler_adapter_sha256": compiler_metadata["final_adapter_sha256"],
        "data_sha256": sha256_file(args.data),
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "examples": len(examples),
        "atomic_examples_per_epoch": (
            len(examples) * 2 if args.arm in {"tied", "untied"} else 0
        ),
        "full_composition_examples_per_epoch": (
            len(examples) if args.arm == "tied_composed" else 0
        ),
        "full_answer_examples_per_epoch": (
            len(examples) if args.arm == "source_retained" else 0
        ),
        "updates": total_steps,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup": args.warmup,
        "clip": args.clip,
        "seed": args.seed,
        "packet_width": context_width,
        "identity_width": identity_width,
        "context_width": context_width,
        "executor_width": args.width,
        "base_parameters": base_parameters,
        "compiler_parameters": compiler_parameters,
        "executor_parameters": executor_parameters,
        "total_parameters": total_parameters,
        "base_parameters_trainable": 0,
        "compiler_parameters_trainable": 0,
        "initial_executor_sha256": initial_executor_sha256,
        "confirmation_access": 0,
        "development_evaluation_access": 0,
        "claim_boundary": (
            "Development-only source-deleted two-step list-execution component. No sealed "
            "confirmation, natural-language reasoning, halt, rollout, or novelty claim."
        ),
    }
    print(json.dumps({"executor_training": metadata}, sort_keys=True), flush=True)

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
            if args.arm == "source_retained":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = executor(compiler_outputs["memory"], valid)
                    targets = torch.tensor(
                        [execution_targets(example).answer_identity for example in selected],
                        device=device,
                        dtype=torch.long,
                    )
                    loss = F.cross_entropy(outputs["answer_logits"], targets)
                    losses = {"total": loss, "answer": loss}
            else:
                packet = gather_source_deleted_packet(
                    compiler_outputs,
                    selected,
                    valid,
                    oracle=args.packet_oracle,
                    packet_mode=args.packet_mode,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if args.arm == "tied_composed":
                        outputs = executor(packet, cell_indices=(0, 1))
                        losses = executor_loss(
                            outputs, [execution_targets(example) for example in selected],
                        )
                    else:
                        atomic_losses = []
                        for operation_index in (0, 1):
                            atomic_packet = select_packet_operations(packet, (operation_index,))
                            outputs = executor(atomic_packet, cell_indices=(operation_index,))
                            targets = [
                                execution_targets(example, (operation_index,))
                                for example in selected
                            ]
                            atomic_losses.append(executor_loss(outputs, targets))
                        losses = {
                            name: torch.stack([item[name] for item in atomic_losses]).mean()
                            for name in atomic_losses[0]
                        }
                    loss = losses["total"]
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite Stage-B loss at update {}".format(global_step))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(executor.parameters(), args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite Stage-B gradient at update {}".format(global_step))
            optimizer.step()
            if global_step % args.log_every == 0:
                elapsed = max(1e-6, time.time() - started)
                record = {
                    "update": global_step,
                    "epoch": epoch,
                    "grad_norm": float(grad_norm.item()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "examples_per_second": (global_step + 1) * args.batch_size / elapsed,
                }
                record.update({
                    "{}_loss".format(name): float(value.item())
                    for name, value in losses.items()
                })
                print(json.dumps(record, sort_keys=True), flush=True)
            global_step += 1

    metadata["elapsed_seconds"] = time.time() - started
    metadata["final_executor_sha256"] = executor_state_hash(executor)
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
