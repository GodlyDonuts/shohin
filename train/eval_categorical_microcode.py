#!/usr/bin/env python3
"""Evaluate semantic compilation and learned categorical execution."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from categorical_microcode import (
    CategoricalMicrocodeCompiler,
    alu_basis_accuracy,
    compile_example,
    execute_program,
    sha256_file,
)
from model import GPT, GPTConfig


def load_examples(path, tokenizer, seq_len):
    examples = []
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            example = compile_example(row, tokenizer)
            if len(example.ids) > seq_len:
                raise ValueError("held-out row {} exceeds sequence length".format(line_number))
            examples.append(example)
    if not examples:
        raise ValueError("no held-out compiler examples")
    return examples


def all_batches(examples, batch_size):
    buckets = collections.defaultdict(list)
    for index, example in enumerate(examples):
        buckets[len(example.ids)].append(index)
    return [
        indices[offset:offset + batch_size]
        for length, indices in sorted(buckets.items())
        for offset in range(0, len(indices), batch_size)
    ]


def flatten_positions(examples, indices, device):
    batch_indices, positions = [], []
    for local, index in enumerate(indices):
        for position in examples[index].operation_positions:
            batch_indices.append(local)
            positions.append(position)
    return (
        torch.tensor(batch_indices, dtype=torch.long, device=device),
        torch.tensor(positions, dtype=torch.long, device=device),
    )


def safe_execute(example, opcodes, query, table):
    try:
        return execute_program(
            example.initial_values, list(map(int, opcodes)), example.operation_values,
            int(query), table,
        )
    except (IndexError, ValueError):
        return None


def summarize(records):
    output = {}
    for regime in ["all"] + sorted({record["regime"] for record in records}):
        selected = records if regime == "all" else [record for record in records if record["regime"] == regime]
        op_total = sum(record["operation_total"] for record in selected)
        output[regime] = {
            "cases": len(selected),
            "operation_accuracy": sum(record["operation_correct"] for record in selected) / op_total,
            "query_accuracy": sum(record["query_correct"] for record in selected) / len(selected),
            "program_exact": sum(record["program_exact"] for record in selected),
            "program_exact_accuracy": sum(record["program_exact"] for record in selected) / len(selected),
            "answer_correct": sum(record["answer_correct"] for record in selected),
            "answer_accuracy": sum(record["answer_correct"] for record in selected) / len(selected),
            "oracle_answer_correct": sum(record["oracle_answer_correct"] for record in selected),
            "oracle_answer_accuracy": sum(record["oracle_answer_correct"] for record in selected) / len(selected),
            "shuffled_answer_correct": sum(record.get("shuffled_answer_correct", False) for record in selected),
            "shuffled_answer_accuracy": sum(record.get("shuffled_answer_correct", False) for record in selected) / len(selected),
        }
    return output


def locked_gates(summary, basis_correct, basis_total):
    return {
        "learned_alu_basis_exact": basis_correct == basis_total == 400,
        "fit_iid_answer_at_least_0_70": summary["fit_iid"]["answer_accuracy"] >= 0.70,
        "depth_ood_answer_at_least_0_60": summary["depth_ood"]["answer_accuracy"] >= 0.60,
        "language_ood_answer_at_least_0_50": summary["language_ood"]["answer_accuracy"] >= 0.50,
        "full_ood_answer_at_least_0_40": summary["full_ood"]["answer_accuracy"] >= 0.40,
        "all_program_exact_at_least_0_50": summary["all"]["program_exact_accuracy"] >= 0.50,
        "answer_over_shuffled_margin_at_least_0_20": (
            summary["all"]["answer_accuracy"] - summary["all"]["shuffled_answer_accuracy"] >= 0.20
        ),
        "oracle_executor_exact": summary["all"]["oracle_answer_accuracy"] == 1.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("categorical microcode evaluation requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing output: {}".format(args.out))

    adapter_checkpoint = torch.load(args.adapter, map_location="cpu")
    metadata = adapter_checkpoint.get("categorical_microcode", {})
    if metadata.get("protocol") != "causal_microcode_bottleneck_v1":
        raise SystemExit("invalid microcode adapter metadata")
    if metadata.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("adapter does not bind supplied base")
    with open(args.admission) as source:
        admission = json.load(source)
    data_sha256 = sha256_file(args.data)
    tokenizer_sha256 = sha256_file(args.tokenizer)
    if not admission.get("all_checks_pass"):
        raise SystemExit("categorical microcode admission did not pass")
    if admission.get("eval_sha256") != data_sha256:
        raise SystemExit("categorical microcode admission does not bind evaluation data")
    if admission.get("train_sha256") != metadata.get("data_sha256"):
        raise SystemExit("adapter training data is not bound by admission")
    if admission.get("tokenizer_sha256") != tokenizer_sha256:
        raise SystemExit("categorical microcode admission does not bind tokenizer")
    if metadata.get("admission_sha256") != sha256_file(args.admission):
        raise SystemExit("adapter does not bind supplied admission report")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    base_checkpoint = torch.load(args.base, map_location="cpu")
    cfg = GPTConfig(**base_checkpoint["cfg"])
    examples = load_examples(args.data, tokenizer, cfg.seq_len)
    batches = all_batches(examples, args.batch_size)

    model = GPT(cfg).to("cuda").eval()
    model.load_state_dict(base_checkpoint["model"])
    compiler = CategoricalMicrocodeCompiler(
        model, layer=int(metadata["layer"]), hidden=int(metadata["hidden"]),
    ).to("cuda").eval()
    missing, unexpected = compiler.load_state_dict(adapter_checkpoint["adapter_state"], strict=False)
    missing = [name for name in missing if not name.startswith("model.")]
    unexpected = [name for name in unexpected if not name.startswith("model.")]
    if missing or unexpected:
        raise SystemExit("adapter mismatch missing={} unexpected={}".format(missing, unexpected))

    predictions = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_number, indices in enumerate(batches, 1):
            ids = torch.tensor([examples[index].ids for index in indices], dtype=torch.long, device="cuda")
            hidden = compiler.encode(ids)
            op_batch, op_positions = flatten_positions(examples, indices, "cuda")
            op_logits = compiler.classify_positions(hidden, op_batch, op_positions, "operation")
            query_positions = torch.tensor(
                [examples[index].query_position for index in indices], dtype=torch.long, device="cuda",
            )
            query_logits = compiler.classify_positions(
                hidden, torch.arange(len(indices), device="cuda"), query_positions, "query",
            )
            op_predictions = op_logits.argmax(dim=-1).cpu().tolist()
            query_predictions = query_logits.argmax(dim=-1).cpu().tolist()
            cursor = 0
            for local, index in enumerate(indices):
                count = len(examples[index].operation_targets)
                predictions[index] = {
                    "operations": op_predictions[cursor:cursor + count],
                    "query": query_predictions[local],
                }
                cursor += count
            if batch_number % 20 == 0 or batch_number == len(batches):
                print("[microcode-eval] {}/{} batches".format(batch_number, len(batches)), flush=True)

    table = compiler.transition_logits.detach().cpu()
    basis_correct, basis_total = alu_basis_accuracy(table)
    records = []
    for index, example in enumerate(examples):
        predicted_ops = predictions[index]["operations"]
        predicted_query = predictions[index]["query"]
        predicted_answer = safe_execute(example, predicted_ops, predicted_query, table)
        oracle_answer = safe_execute(example, example.operation_targets, example.query_target, table)
        operation_correct = sum(
            int(predicted == target) for predicted, target in zip(predicted_ops, example.operation_targets)
        )
        query_correct = predicted_query == example.query_target
        records.append({
            "index": index,
            "reference": example.reference,
            "regime": example.regime,
            "depth": len(example.operation_targets),
            "operation_targets": list(example.operation_targets),
            "operation_predictions": predicted_ops,
            "operation_values": list(example.operation_values),
            "operation_correct": operation_correct,
            "operation_total": len(example.operation_targets),
            "query_target": example.query_target,
            "query_prediction": predicted_query,
            "query_correct": query_correct,
            "program_exact": operation_correct == len(example.operation_targets) and query_correct,
            "expected_answer": example.answer,
            "predicted_answer": predicted_answer,
            "answer_correct": predicted_answer == example.answer,
            "oracle_answer": oracle_answer,
            "oracle_answer_correct": oracle_answer == example.answer,
        })

    by_depth = collections.defaultdict(list)
    for index, record in enumerate(records):
        by_depth[record["depth"]].append(index)
    for indices in by_depth.values():
        for offset, index in enumerate(indices):
            donor_index = indices[(offset + 1) % len(indices)]
            donor = predictions[donor_index]
            shuffled_answer = safe_execute(
                examples[index], donor["operations"], donor["query"], table,
            )
            records[index]["shuffled_answer"] = shuffled_answer
            records[index]["shuffled_answer_correct"] = shuffled_answer == examples[index].answer

    summary = summarize(records)
    gates = locked_gates(summary, basis_correct, basis_total)
    result = {
        "audit": "causal_microcode_bottleneck_eval_v1",
        "base": os.path.realpath(args.base),
        "base_sha256": sha256_file(args.base),
        "adapter": os.path.realpath(args.adapter),
        "adapter_sha256": sha256_file(args.adapter),
        "adapter_metadata": metadata,
        "data": os.path.realpath(args.data),
        "data_sha256": data_sha256,
        "admission": os.path.realpath(args.admission),
        "admission_sha256": sha256_file(args.admission),
        "cases": len(examples),
        "batches": len(batches),
        "alu_basis": {"correct": basis_correct, "total": basis_total},
        "summary": summary,
        "gates": gates,
        "advance_to_decoder_bridge": all(gates.values()),
        "records": records,
        "claim_boundary": (
            "A pass establishes narrow text-to-microcode compilation plus learned categorical execution only; "
            "numeric literals and line boundaries are deterministic lexical inputs."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[microcode-eval] " + json.dumps({
        "summary": summary, "gates": gates,
        "advance_to_decoder_bridge": result["advance_to_decoder_bridge"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
