#!/usr/bin/env python3
"""Render direct text-to-program interaction transcripts for a microcode adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from categorical_microcode import (
    CategoricalMicrocodeCompiler, OPCODES, QUERIES, compile_example, execute_program, sha256_file,
)
from role_equivariant_microcode import RoleEquivariantMicrocodeCompiler
from model import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("manual microcode inspection requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing transcript: {}".format(args.out))
    adapter_checkpoint = torch.load(args.adapter, map_location="cpu")
    metadata = adapter_checkpoint.get("categorical_microcode", {})
    if metadata.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("adapter does not bind supplied base")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    examples = [compile_example(row, tokenizer) for row in rows]
    base_checkpoint = torch.load(args.base, map_location="cpu")
    cfg = GPTConfig(**base_checkpoint["cfg"])
    model = GPT(cfg).to("cuda").eval()
    model.load_state_dict(base_checkpoint["model"])
    compiler_class = (
        RoleEquivariantMicrocodeCompiler
        if metadata.get("protocol") == "causal_microcode_role_equivariance_v3"
        else CategoricalMicrocodeCompiler
    )
    compiler = compiler_class(
        model, layer=int(metadata["layer"]), hidden=int(metadata["hidden"]),
    ).to("cuda").eval()
    missing, unexpected = compiler.load_state_dict(adapter_checkpoint["adapter_state"], strict=False)
    if [name for name in missing if not name.startswith("model.")] or unexpected:
        raise SystemExit("adapter state mismatch")

    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row, example in zip(rows, examples):
            ids = torch.tensor([example.ids], dtype=torch.long, device="cuda")
            hidden = compiler.encode(ids)
            op_predictions = []
            for position in example.operation_positions:
                logits = compiler.classify_positions(
                    hidden, torch.tensor([0], device="cuda"),
                    torch.tensor([position], device="cuda"), "operation",
                )
                op_predictions.append(int(logits.argmax(dim=-1).item()))
            query_logits = compiler.classify_positions(
                hidden, torch.tensor([0], device="cuda"),
                torch.tensor([example.query_position], device="cuda"), "query",
            )
            query_prediction = int(query_logits.argmax(dim=-1).item())
            try:
                answer = execute_program(
                    example.initial_values, op_predictions, example.operation_values,
                    query_prediction, compiler.transition_logits.detach().cpu(),
                )
            except (IndexError, ValueError):
                answer = None
            record = {
                "reference": example.reference,
                "question": row["question"],
                "expected_operations": [OPCODES[index] for index in example.operation_targets],
                "predicted_operations": [OPCODES[index] for index in op_predictions],
                "expected_query": QUERIES[example.query_target],
                "predicted_query": QUERIES[query_prediction],
                "expected_answer": example.answer,
                "predicted_answer": answer,
                "program_exact": (
                    tuple(op_predictions) == example.operation_targets
                    and query_prediction == example.query_target
                ),
                "answer_correct": answer == example.answer,
            }
            records.append(record)
            print("\n[manual] {}\n{}\nprogram={} query={} answer={} expected={} correct={}".format(
                example.reference, row["question"], record["predicted_operations"],
                record["predicted_query"], answer, example.answer, record["answer_correct"],
            ), flush=True)
    result = {
        "audit": "categorical_microcode_manual_interaction_v1",
        "base": os.path.realpath(args.base),
        "base_sha256": sha256_file(args.base),
        "adapter": os.path.realpath(args.adapter),
        "adapter_sha256": sha256_file(args.adapter),
        "adapter_metadata": metadata,
        "data": os.path.realpath(args.data),
        "data_sha256": sha256_file(args.data),
        "cases": len(records),
        "program_exact": sum(record["program_exact"] for record in records),
        "answer_correct": sum(record["answer_correct"] for record in records),
        "records": records,
        "claim_boundary": (
            "These are hand-authored direct compiler interactions. Structured fields are used for "
            "scoring and lexical positions; the neural compiler sees only question text."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
