#!/usr/bin/env python3
"""Measure cross-occurrence entity identity in candidate Stage-B packets."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPTConfig
from referential_literal_pointer_compiler import (
    load_examples,
    make_batches,
    pad_batch,
    sha256_file,
)
from train_referential_gather_delete_executor import load_frozen_compiler


def normalized_sigmoid_weights(logits, valid_mask):
    weights = torch.sigmoid(logits.float()) * valid_mask.float()
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def gold_weights(examples, label, length, device):
    weights = torch.zeros((len(examples), length), device=device)
    for row, example in enumerate(examples):
        positions = tuple(example.target_positions[label])
        weights[row, list(positions)] = 1.0 / len(positions)
    return weights


def gather(memory, weights):
    return torch.einsum("bl,bld->bd", weights.to(memory.dtype), memory).float()


def identity_predictions(initial, operations):
    initial = F.normalize(initial.float(), dim=-1)
    predictions = []
    for operation in operations:
        operation = F.normalize(operation.float(), dim=-1)
        predictions.append(torch.einsum("bd,bid->bi", operation, initial).argmax(-1))
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--split", default="development_compositional")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("identity-packet probe requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing identity-packet report")
    report = json.load(open(args.report))
    if report.get("artifacts", {}).get(args.split, {}).get("sha256") != sha256_file(args.data):
        raise SystemExit("factorized report does not bind identity-probe data")
    checkpoint, compiler, compiler_metadata = load_frozen_compiler(
        args.base, args.compiler, args.tokenizer, "cuda",
    )
    tokenizer = Tokenizer.from_file(args.tokenizer)
    cfg = GPTConfig(**checkpoint["cfg"])
    examples = load_examples(
        args.data, tokenizer, args.split, cfg.seq_len, keep_evidence=True,
    )
    methods = (
        "contextual_softmax",
        "lexical_softmax",
        "lexical_sigmoid_span",
        "lexical_gold_span",
    )
    correct = collections.Counter()
    totals = collections.Counter()
    by_surface = {
        method: collections.Counter() for method in methods
    }
    batches = make_batches(examples, args.batch_size, seed=0, shuffle=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for indices in batches:
            selected, ids, valid = pad_batch(examples, indices, "cuda")
            outputs = compiler(ids, valid)
            lexical_memory = compiler.model.tok(ids).detach()
            representations = {}
            for method in methods:
                memory = outputs["memory"] if method == "contextual_softmax" else lexical_memory
                fields = {}
                for label in (
                    "intro.entity0", "intro.entity1", "intro.entity2",
                    "op0.entity", "op1.entity",
                ):
                    logits = outputs["pointer_logits"][label]
                    if method == "lexical_sigmoid_span":
                        weights = normalized_sigmoid_weights(logits, valid)
                    elif method == "lexical_gold_span":
                        weights = gold_weights(selected, label, ids.shape[1], ids.device)
                    else:
                        weights = F.softmax(
                            logits.float().masked_fill(~valid, -1e9), dim=-1,
                        )
                    fields[label] = gather(memory, weights)
                representations[method] = fields
            for method, fields in representations.items():
                initial = torch.stack([
                    fields["intro.entity{}".format(index)] for index in range(3)
                ], dim=1)
                predictions = identity_predictions(initial, [
                    fields["op0.entity"], fields["op1.entity"],
                ])
                for row, example in enumerate(selected):
                    surface = example.surface_type
                    for operation_index, prediction in enumerate(predictions):
                        target = example.initial_order.index(
                            example.program[operation_index][1],
                        )
                        hit = int(prediction[row]) == target
                        correct[method] += int(hit)
                        totals[method] += 1
                        by_surface[method]["{}_correct".format(surface)] += int(hit)
                        by_surface[method]["{}_total".format(surface)] += 1
    result = {
        "schema": "r12_referential_identity_packet_probe_v1",
        "base_sha256": sha256_file(args.base),
        "compiler_file_sha256": sha256_file(args.compiler),
        "compiler_adapter_sha256": compiler_metadata["final_adapter_sha256"],
        "data_sha256": sha256_file(args.data),
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "split": args.split,
        "rows": len(examples),
        "entity_references": len(examples) * 2,
        "methods": {
            method: {
                "correct": correct[method],
                "total": totals[method],
                "accuracy": correct[method] / totals[method],
                "by_surface": {
                    surface: (
                        by_surface[method]["{}_correct".format(surface)]
                        / by_surface[method]["{}_total".format(surface)]
                    )
                    for surface in sorted({example.surface_type for example in examples})
                },
            }
            for method in methods
        },
        "confirmation_access": 0,
        "claim_boundary": (
            "No-fit packet identity diagnostic only. No executor, confirmation, reasoning, "
            "or novelty claim."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
