#!/usr/bin/env python3
"""Train equal-budget S9.1 alpha-closed compiler arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from s7_learned_cayley_generator import LearnedCayleyGenerator
from s8_nil_linked_graph_compiler import compile_row as compile_s8_row, recode_operation_ids
from s9_occurrence_quotient_compiler import (
    MAX_SPAN_WIDTH,
    OccurrenceQuotientCompiler,
    adapter_hash,
    adapter_state,
    compile_row as compile_s9_row,
    compiler_loss,
    load_adapter_state,
    load_examples,
    make_batches,
    pad_batch,
    sha256_file,
    shuffle_relation_supervision,
)
from s9_1_alpha_closed_compiler import aligned_positive_logits, orbit_consistency_loss
from train_s8_nil_linked_graph import _fit_generator


PAIRED_SOURCE_ROWS = 24_000
CHARGED_VIEWS = 48_000
ORBIT_WEIGHT = 0.25


def lr_scale(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _rows(path: Path, report, name: str):
    if sha256_file(path) != report["files"][name]["sha256"]:
        raise SystemExit(f"S9.1 {name} hash mismatch")
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


def _paired_subset(examples, tokenizer, seed):
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    indices = indices[:PAIRED_SOURCE_ROWS]
    original = [examples[index] for index in indices]
    recoded = [
        compile_s9_row(
            recode_operation_ids(compile_s8_row(example.row, tokenizer), tokenizer).row,
            tokenizer,
        )
        for example in original
    ]
    for first, second in zip(original, recoded, strict=True):
        if tuple(target for _, _, target in first.gold) != tuple(
            target for _, _, target in second.gold
        ):
            raise RuntimeError("S9.1 recoding changed relation occurrence order")
        if max(end - start + 1 for start, end, _ in second.gold) > MAX_SPAN_WIDTH:
            raise RuntimeError("S9.1 recoding exceeded the proposal cap")
    digest = hashlib.sha256(
        json.dumps(indices, separators=(",", ":")).encode()
    ).hexdigest()
    return original, recoded, digest


def _fit(compiler, original, recoded, args, seed, label, class_messages):
    trainable = list(compiler.adapter_parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    pair_batch = args.batch_size // 2
    batches = make_batches(original, pair_batch, seed)
    if len(batches) != 750 or any(len(batch) != pair_batch for batch in batches):
        raise RuntimeError("S9.1 frozen pair/update budget changed")
    started = time.time()
    final = {}
    compiler.train()
    compiler.model.eval()
    for step, indices in enumerate(batches):
        first = [original[index] for index in indices]
        second = [recoded[index] for index in indices]
        combined = first + second
        selected, candidate_rows, ids, valid, candidates = pad_batch(
            combined,
            list(range(len(combined))),
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
            supervised = compiler_loss(outputs, candidates["target"])
            positive_logits, positive_targets = aligned_positive_logits(
                selected, candidate_rows, outputs["role_logits"]
            )
            split = sum(len(example.gold) for example in first)
            orbit = orbit_consistency_loss(
                positive_logits[:split],
                positive_logits[split:],
                positive_targets[:split],
                positive_targets[split:],
            )
            loss = supervised + args.orbit_weight * orbit
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite S9.1 {label} loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite S9.1 {label} gradient")
        optimizer.step()
        predictions = outputs["role_logits"].argmax(-1)
        positive = candidates["target"] != 0
        final = {
            "loss": float(loss.item()),
            "supervised_loss": float(supervised.item()),
            "orbit_loss": float(orbit.item()),
            "grad_norm": float(grad_norm.item()),
            "candidate_accuracy": float(
                (predictions == candidates["target"]).float().mean().item()
            ),
            "positive_accuracy": float(
                (predictions[positive] == candidates["target"][positive])
                .float().mean().item()
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
        "unique_sources": len(original),
        "charged_views": 2 * len(original),
        "elapsed_seconds": time.time() - started,
        "class_messages": class_messages,
        "orbit_weight": args.orbit_weight,
        "negative_candidates_per_view": args.negative_candidates,
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
    parser.add_argument("--orbit-weight", type=float, default=ORBIT_WEIGHT)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("S9.1 training requires CUDA")
    if args.out.exists():
        raise SystemExit(f"refusing existing S9.1 checkpoint: {args.out}")
    if (
        args.batch_size != 64
        or args.negative_candidates != 128
        or args.orbit_weight != ORBIT_WEIGHT
    ):
        raise SystemExit("S9.1 frozen batch, negative, or orbit budget changed")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    report_path = args.data_dir / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("schema") != "r12_s8_nil_linked_law_graph_board_report_v1":
        raise SystemExit("unexpected S9.1 board schema")
    if report.get("decision") != "admit_s8_nil_linked_law_graph_board":
        raise SystemExit("S9.1 board is not admitted")
    if report["source_commit"] != args.source_commit:
        raise SystemExit("S9.1 board/source commit mismatch")
    if report["audit"]["development_accesses"] or report["audit"]["confirmation_accesses"]:
        raise SystemExit("S9.1 score board was already accessed")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**base["cfg"])
    initializer = torch.load(args.initializer, map_location="cpu", weights_only=False)
    if initializer.get("schema") != "r12_s8_nil_linked_law_graph_checkpoint_v1":
        raise SystemExit("S9.1 initializer is not the closed S8.1 checkpoint")
    if initializer.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("S9.1 initializer/base mismatch")
    examples = load_examples(
        args.data_dir / "train.jsonl", tokenizer, "s8_nil_graph_train", cfg.seq_len
    )
    if len(examples) != CHARGED_VIEWS:
        raise SystemExit("S9.1 requires the admitted 48,000-row source pool")
    original, recoded, subset_sha256 = _paired_subset(
        examples, tokenizer, args.seed ^ 0xA1FA
    )

    shuffled_seed = args.seed ^ 0x591A
    shuffled_original = shuffle_relation_supervision(original, shuffled_seed)
    shuffled_recoded = shuffle_relation_supervision(recoded, shuffled_seed)
    if any(
        tuple(value[2] for value in first.gold)
        != tuple(value[2] for value in second.gold)
        for first, second in zip(shuffled_original, shuffled_recoded, strict=True)
    ):
        raise RuntimeError("S9.1 shuffled orbit labels diverged")

    generator_rows = _rows(
        args.data_dir / "generator_train.jsonl", report, "generator_train.jsonl"
    )
    generator = LearnedCayleyGenerator().to("cuda")
    generator_fit = _fit_generator(generator, generator_rows, "next_symbol")
    if len(generator_rows) != 23:
        raise SystemExit("S9.1 generator row count mismatch")
    if generator_fit["successor_accuracy"] != 1.0 or generator_fit["zero_accuracy"] != 1.0:
        raise RuntimeError("S9.1 generator failed exact fit")

    treatment, loaded = _new_compiler(cfg, base["model"], initializer, args)
    base_parameters = sum(value.numel() for value in treatment.model.parameters())
    compiler_parameters = treatment.adapter_num_params()
    if compiler_parameters > 16_000_000:
        raise RuntimeError("S9.1 compiler exceeds its 16M cap")
    initial = adapter_state(treatment)
    treatment_fit = _fit(
        treatment, original, recoded, args, args.seed, "treatment", True
    )
    treatment_state = adapter_state(treatment)
    del treatment
    torch.cuda.empty_cache()

    no_class, loaded_no_class = _new_compiler(cfg, base["model"], initializer, args)
    load_adapter_state(no_class, initial)
    no_class_fit = _fit(
        no_class, original, recoded, args, args.seed, "no_class_message", False
    )
    no_class_state = adapter_state(no_class)
    del no_class
    torch.cuda.empty_cache()

    shuffled_model, loaded_shuffled = _new_compiler(cfg, base["model"], initializer, args)
    load_adapter_state(shuffled_model, initial)
    shuffled_fit = _fit(
        shuffled_model,
        shuffled_original,
        shuffled_recoded,
        args,
        args.seed,
        "shuffled_relations",
        True,
    )
    shuffled_state = adapter_state(shuffled_model)
    parameters = {
        "base": base_parameters,
        "compiler": compiler_parameters,
        "generator": sum(value.numel() for value in generator.parameters()),
    }
    parameters["complete_system"] = sum(parameters.values())
    if parameters["complete_system"] >= 150_000_000:
        raise RuntimeError("S9.1 complete system exceeds 150M")
    checkpoint = {
        "schema": "r12_s9_1_alpha_closed_checkpoint_v1",
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
            "negative_candidates_per_view": args.negative_candidates,
        },
        "parameters": parameters,
        "training_contract": (
            "24,000 unique graph-field-only sources paired with one operation recode; "
            "48,000 charged views and 750 updates per arm; no final-state, answer, "
            "recurrent, development-law, or confirmation-law supervision"
        ),
        "paired_source_indices_sha256": subset_sha256,
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
        "paired_source_indices_sha256": subset_sha256,
        "treatment": treatment_fit,
        "no_class": no_class_fit,
        "shuffled": shuffled_fit,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
