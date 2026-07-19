#!/usr/bin/env python3
"""Train frozen-trunk S8 treatment and shuffled graph compilers."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from s7_learned_cayley_generator import LearnedCayleyGenerator, PRIMARY_MODULI
from s8_nil_linked_graph_compiler import (
    NilLinkedGraphCompiler,
    adapter_hash,
    adapter_state,
    compiler_loss,
    load_examples,
    load_adapter_state,
    make_batches,
    pad_batch,
    sha256_file,
    shuffle_supervision,
)


def lr_scale(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _load_rows(path: Path, report: dict[str, object], name: str) -> list[dict[str, object]]:
    if sha256_file(path) != report["files"][name]["sha256"]:
        raise SystemExit(f"S8 {name} hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fit_generator(
    model: LearnedCayleyGenerator,
    rows: list[dict[str, object]],
    target_field: str,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    initial_loss = None
    final_loss = None
    for _ in range(1000):
        terms = []
        for modulus in PRIMARY_MODULI:
            selected = [row for row in rows if int(row["modulus"]) == modulus]
            current = torch.tensor(
                [row["current_symbol"] for row in selected],
                dtype=torch.long,
                device=model.successor(modulus).device,
            )
            targets = torch.tensor(
                [row[target_field] for row in selected],
                dtype=torch.long,
                device=model.successor(modulus).device,
            )
            terms.append(
                F.cross_entropy(
                    model.successor(modulus).index_select(0, current), targets
                )
            )
            zero_target = torch.tensor(
                [selected[0]["zero_symbol"]],
                dtype=torch.long,
                device=model.zero(modulus).device,
            )
            terms.append(F.cross_entropy(model.zero(modulus).unsqueeze(0), zero_target))
        loss = torch.stack(terms).mean()
        if initial_loss is None:
            initial_loss = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    correct = 0
    total = 0
    zero_correct = 0
    with torch.no_grad():
        for modulus in PRIMARY_MODULI:
            selected = [row for row in rows if int(row["modulus"]) == modulus]
            expected = tuple(int(row[target_field]) for row in selected)
            predicted = model.discrete_successor(modulus)
            correct += sum(int(a == b) for a, b in zip(predicted, expected, strict=True))
            total += len(expected)
            zero_correct += int(model.discrete_zero(modulus) == int(selected[0]["zero_symbol"]))
    return {
        "target_field": target_field,
        "updates": 1000,
        "learning_rate": 0.05,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "successor_correct": correct,
        "successor_total": total,
        "successor_accuracy": correct / total,
        "zero_correct": zero_correct,
        "zero_total": len(PRIMARY_MODULI),
        "zero_accuracy": zero_correct / len(PRIMARY_MODULI),
    }


def _new_compiler(cfg, base_state, initializer, device, args):
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(base_state)
    compiler = NilLinkedGraphCompiler(
        model,
        layer=args.layer,
        width=args.width,
        heads=args.heads,
        encoder_layers=args.encoder_layers,
        ff=args.ff,
    ).to(device)
    loaded = compiler.initialize_memory_encoder(initializer["adapter_state"])
    return compiler, loaded


def _fit_compiler(compiler, examples, args, seed, label):
    trainable = list(compiler.adapter_parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    batches_by_epoch = [
        make_batches(examples, args.batch_size, seed + epoch)
        for epoch in range(args.epochs)
    ]
    total_steps = sum(map(len, batches_by_epoch))
    started = time.time()
    global_step = 0
    final = {}
    compiler.train()
    compiler.model.eval()
    for epoch, batches in enumerate(batches_by_epoch):
        for indices in batches:
            _, ids, valid, roles, ranks = pad_batch(examples, indices, "cuda")
            optimizer.param_groups[0]["lr"] = args.lr * lr_scale(
                global_step, total_steps, args.warmup
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = compiler(ids, valid)
                loss, components = compiler_loss(
                    outputs,
                    roles,
                    ranks,
                    role_weight=args.role_weight,
                    rank_weight=args.rank_weight,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite S8 {label} loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite S8 {label} gradient")
            optimizer.step()
            final = {
                "loss": float(loss.item()),
                "role_loss": float(components["role"].item()),
                "rank_loss": float(components["rank"].item()),
                "grad_norm": float(grad_norm.item()),
            }
            if global_step % args.log_every == 0:
                print(json.dumps({
                    "arm": label,
                    "update": global_step,
                    "epoch": epoch,
                    **final,
                    "lr": optimizer.param_groups[0]["lr"],
                }, sort_keys=True), flush=True)
            global_step += 1
    return {
        "updates": total_steps,
        "elapsed_seconds": time.time() - started,
        "final": final,
        "adapter_sha256": adapter_hash(compiler),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
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
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("S8 training requires CUDA")
    if args.out.exists():
        raise SystemExit(f"refusing existing S8 checkpoint: {args.out}")
    if args.epochs != 1 or args.batch_size != 64:
        raise SystemExit("S8 v1 freezes one epoch and batch size 64")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    report_path = args.data_dir / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("schema") != "r12_s8_nil_linked_law_graph_board_report_v1":
        raise SystemExit("unexpected S8 board schema")
    if report.get("decision") != "admit_s8_nil_linked_law_graph_board":
        raise SystemExit("S8 board is not admitted")
    if report["audit"]["development_accesses"] != 0 or report["audit"]["confirmation_accesses"] != 0:
        raise SystemExit("S8 board access counter is not zero")
    if sha256_file(args.tokenizer) != report["tokenizer_sha256"]:
        raise SystemExit("S8 tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**base["cfg"])
    train_path = args.data_dir / "train.jsonl"
    if sha256_file(train_path) != report["files"]["train.jsonl"]["sha256"]:
        raise SystemExit("S8 train hash mismatch")
    examples = load_examples(
        train_path, tokenizer, "s8_nil_graph_train", cfg.seq_len
    )
    if len(examples) != 48000:
        raise SystemExit("S8 v1 requires exactly 48,000 training examples")
    initializer = torch.load(args.initializer, map_location="cpu", weights_only=False)
    if initializer.get("parser", {}).get("protocol") != "r12_s4_self_delimiting_event_parser_treatment_v1":
        raise SystemExit("S8 initializer is not the confirmed S4 parser family")
    if initializer["parser"]["base_sha256"] != sha256_file(args.base):
        raise SystemExit("S8 initializer/base mismatch")

    treatment, initialized = _new_compiler(
        cfg, base["model"], initializer, "cuda", args
    )
    base_parameters = sum(parameter.numel() for parameter in treatment.model.parameters())
    adapter_parameters = treatment.adapter_num_params()
    generator_parameters = LearnedCayleyGenerator().num_params()
    total_parameters = base_parameters + adapter_parameters + generator_parameters
    if adapter_parameters > 16_000_000 or total_parameters >= 150_000_000:
        raise SystemExit("S8 compiler violates parameter cap")
    initial_adapter = adapter_state(treatment)
    initial_adapter_sha = adapter_hash(treatment)
    treatment_fit = _fit_compiler(treatment, examples, args, args.seed, "treatment")
    treatment_state = adapter_state(treatment)
    del treatment
    torch.cuda.empty_cache()

    shuffled, initialized_shuffled = _new_compiler(
        cfg, base["model"], initializer, "cuda", args
    )
    load_adapter_state(shuffled, initial_adapter)
    shuffled_examples = shuffle_supervision(examples, args.seed ^ 0x5A8F11ED)
    shuffled_fit = _fit_compiler(
        shuffled, shuffled_examples, args, args.seed, "shuffled"
    )
    shuffled_state = adapter_state(shuffled)

    generator_rows = _load_rows(
        args.data_dir / "generator_train.jsonl", report, "generator_train.jsonl"
    )
    if len(generator_rows) != 23:
        raise SystemExit("S8 generator row count mismatch")
    true_generator = LearnedCayleyGenerator().to("cuda")
    false_generator = LearnedCayleyGenerator().to("cuda")
    true_fit = _fit_generator(true_generator, generator_rows, "next_symbol")
    false_fit = _fit_generator(false_generator, generator_rows, "false_next_symbol")
    if true_fit["successor_accuracy"] != 1.0 or true_fit["zero_accuracy"] != 1.0:
        raise SystemExit("S8 true generator failed exact fit")
    if false_fit["successor_accuracy"] != 1.0 or false_fit["zero_accuracy"] != 1.0:
        raise SystemExit("S8 false generator failed exact fit")

    output = {
        "schema": "r12_s8_nil_linked_law_graph_checkpoint_v1",
        "source_commit": report["source_commit"],
        "board_report_sha256": sha256_file(report_path),
        "base_sha256": sha256_file(args.base),
        "initializer_sha256": sha256_file(args.initializer),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "seed": args.seed,
        "parameters": {
            "base": base_parameters,
            "graph_compiler": adapter_parameters,
            "generator": generator_parameters,
            "complete_system": total_parameters,
        },
        "architecture": {
            "layer": args.layer,
            "width": args.width,
            "heads": args.heads,
            "encoder_layers": args.encoder_layers,
            "ff": args.ff,
            "initialized_memory_keys": len(initialized),
            "shuffled_initialized_memory_keys": len(initialized_shuffled),
        },
        "training_contract": (
            "48,000 whole-source graph-field rows; zero final-state, answer, recurrent, "
            "development-law, or confirmation-law supervision; generator sees only 23 "
            "successor cells plus three zero anchors"
        ),
        "initial_adapter_sha256": initial_adapter_sha,
        "treatment_fit": treatment_fit,
        "shuffled_fit": shuffled_fit,
        "generator_fit": true_fit,
        "false_generator_fit": false_fit,
        "treatment_adapter_state": treatment_state,
        "shuffled_adapter_state": shuffled_state,
        "generator_state": {
            name: value.detach().cpu() for name, value in true_generator.state_dict().items()
        },
        "false_generator_state": {
            name: value.detach().cpu() for name, value in false_generator.state_dict().items()
        },
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    print(json.dumps({
        "saved": os.path.realpath(args.out),
        "parameters": output["parameters"],
        "treatment_adapter_sha256": treatment_fit["adapter_sha256"],
        "shuffled_adapter_sha256": shuffled_fit["adapter_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
