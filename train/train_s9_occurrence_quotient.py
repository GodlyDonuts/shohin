#!/usr/bin/env python3
"""Train frozen-trunk S9 class-aware and matched-control compilers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from s7_learned_cayley_generator import LearnedCayleyGenerator
from s9_occurrence_quotient_compiler import (
    MAX_SPAN_WIDTH,
    OccurrenceQuotientCompiler,
    adapter_hash,
    adapter_state,
    compiler_loss,
    load_adapter_state,
    load_examples,
    make_batches,
    pad_batch,
    sha256_file,
    shuffle_relation_supervision,
)
from train_s8_nil_linked_graph import _fit_generator


def lr_scale(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _rows(path: Path, report, name: str):
    if sha256_file(path) != report["files"][name]["sha256"]:
        raise SystemExit(f"S9 {name} hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _new_compiler(cfg, base_state, initializer, args):
    model = GPT(cfg).to("cuda").eval()
    model.load_state_dict(base_state)
    compiler = OccurrenceQuotientCompiler(
        model,
        layer=args.layer,
        width=args.width,
        heads=args.heads,
        encoder_layers=args.encoder_layers,
        ff=args.ff,
    ).to("cuda")
    loaded = compiler.initialize_memory_encoder(initializer["treatment_adapter_state"])
    return compiler, loaded


def _fit(compiler, examples, args, seed, label, class_messages):
    trainable = list(compiler.adapter_parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    batches = make_batches(examples, args.batch_size, seed)
    started = time.time()
    final = {}
    compiler.train()
    compiler.model.eval()
    for step, indices in enumerate(batches):
        _, _, ids, valid, candidates = pad_batch(
            examples,
            indices,
            "cuda",
            negative_limit=args.negative_candidates,
            seed=seed ^ step,
        )
        optimizer.param_groups[0]["lr"] = args.lr * lr_scale(
            step, len(batches), args.warmup
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = compiler(ids, valid, candidates, class_messages=class_messages)
            loss = compiler_loss(outputs, candidates["target"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite S9 {label} loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite S9 {label} gradient")
        optimizer.step()
        predictions = outputs["role_logits"].argmax(-1)
        positive = candidates["target"] != 0
        final = {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()),
            "candidate_accuracy": float(
                (predictions == candidates["target"]).float().mean().item()
            ),
            "positive_accuracy": float(
                (predictions[positive] == candidates["target"][positive]).float().mean().item()
            ),
        }
        if step % args.log_every == 0:
            print(json.dumps({
                "arm": label,
                "update": step,
                **final,
                "lr": optimizer.param_groups[0]["lr"],
            }, sort_keys=True), flush=True)
    return {
        "updates": len(batches),
        "elapsed_seconds": time.time() - started,
        "class_messages": class_messages,
        "negative_candidates_per_row": args.negative_candidates,
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
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=5)
    parser.add_argument("--ff", type=int, default=1408)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--negative-candidates", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("S9 training requires CUDA")
    if args.out.exists():
        raise SystemExit(f"refusing existing S9 checkpoint: {args.out}")
    if args.batch_size != 64 or args.negative_candidates != 128:
        raise SystemExit("S9 v1 freezes batch 64 and 128 negative spans per row")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    report_path = args.data_dir / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("schema") != "r12_s8_nil_linked_law_graph_board_report_v1":
        raise SystemExit("unexpected S9 board schema")
    if report.get("decision") != "admit_s8_nil_linked_law_graph_board":
        raise SystemExit("S9 board is not admitted")
    if report["source_commit"] != args.source_commit:
        raise SystemExit("S9 board/source commit mismatch")
    if report["audit"]["development_accesses"] or report["audit"]["confirmation_accesses"]:
        raise SystemExit("S9 score board was already accessed")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**base["cfg"])
    initializer = torch.load(args.initializer, map_location="cpu", weights_only=False)
    if initializer.get("schema") != "r12_s8_nil_linked_law_graph_checkpoint_v1":
        raise SystemExit("S9 initializer is not the closed S8.1 checkpoint")
    if initializer.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("S9 initializer/base mismatch")
    train_path = args.data_dir / "train.jsonl"
    examples = load_examples(train_path, tokenizer, "s8_nil_graph_train", cfg.seq_len)
    if len(examples) != 48000:
        raise SystemExit("S9 v1 requires exactly 48,000 training rows")
    shuffled = shuffle_relation_supervision(examples, args.seed ^ 0x59A9)
    generator_rows = _rows(
        args.data_dir / "generator_train.jsonl", report, "generator_train.jsonl"
    )
    generator = LearnedCayleyGenerator().to("cuda")
    generator_fit = _fit_generator(generator, generator_rows, "next_symbol")
    if len(generator_rows) != 23:
        raise SystemExit("S9 generator row count mismatch")
    if generator_fit["successor_accuracy"] != 1.0 or generator_fit["zero_accuracy"] != 1.0:
        raise RuntimeError("S9 generator failed exact fit")

    treatment, loaded = _new_compiler(cfg, base["model"], initializer, args)
    base_parameters = sum(value.numel() for value in treatment.model.parameters())
    compiler_parameters = treatment.adapter_num_params()
    if compiler_parameters > 16_000_000:
        raise RuntimeError("S9 compiler exceeds its 16M cap")
    initial = adapter_state(treatment)
    treatment_fit = _fit(treatment, examples, args, args.seed, "treatment", True)
    treatment_state = adapter_state(treatment)
    del treatment
    torch.cuda.empty_cache()

    no_class, loaded_no_class = _new_compiler(cfg, base["model"], initializer, args)
    load_adapter_state(no_class, initial)
    no_class_fit = _fit(
        no_class, examples, args, args.seed, "no_class_message", False
    )
    no_class_state = adapter_state(no_class)
    del no_class
    torch.cuda.empty_cache()

    shuffled_model, loaded_shuffled = _new_compiler(
        cfg, base["model"], initializer, args
    )
    load_adapter_state(shuffled_model, initial)
    shuffled_fit = _fit(
        shuffled_model, shuffled, args, args.seed, "shuffled_relations", True
    )
    shuffled_state = adapter_state(shuffled_model)
    parameters = {
        "base": base_parameters,
        "compiler": compiler_parameters,
        "generator": sum(value.numel() for value in generator.parameters()),
    }
    parameters["complete_system"] = sum(parameters.values())
    if parameters["complete_system"] >= 150_000_000:
        raise RuntimeError("S9 complete system exceeds 150M")
    checkpoint = {
        "schema": "r12_s9_occurrence_quotient_checkpoint_v1",
        "source_commit": args.source_commit,
        "seed": args.seed,
        "base_sha256": sha256_file(args.base),
        "initializer_sha256": sha256_file(args.initializer),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "board_report_sha256": sha256_file(report_path),
        "architecture": {
            "layer": args.layer,
            "width": args.width,
            "heads": args.heads,
            "encoder_layers": args.encoder_layers,
            "ff": args.ff,
            "max_span_width": MAX_SPAN_WIDTH,
            "negative_candidates_per_row": args.negative_candidates,
        },
        "parameters": parameters,
        "training_contract": (
            "48,000 source-to-span/relation rows; no final-state, answer, recurrent, "
            "development-law, or confirmation-law supervision"
        ),
        "initializer_loaded": loaded,
        "initializer_control_equal": loaded == loaded_no_class == loaded_shuffled,
        "generator_fit": generator_fit,
        "generator_state": generator.state_dict(),
        "treatment_fit": treatment_fit,
        "treatment_adapter_state": treatment_state,
        "no_class_fit": no_class_fit,
        "no_class_adapter_state": no_class_state,
        "shuffled_fit": shuffled_fit,
        "shuffled_adapter_state": shuffled_state,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    print(json.dumps({
        "out": str(args.out),
        "parameters": parameters,
        "treatment": treatment_fit,
        "no_class": no_class_fit,
        "shuffled": shuffled_fit,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
