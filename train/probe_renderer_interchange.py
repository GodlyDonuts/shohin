#!/usr/bin/env python3
"""Causal interchange audit for arithmetic inside a familiar worked renderer.

The source problem and displayed equation state are crossed. Exact equal-length
candidate continuations test whether the frozen model follows the source or the
intervened visible state. This is a diagnostic only; the full source remains in
context and externally computed candidates are used only for scoring.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_cells():
    operations = [
        {
            "operation": "add",
            "sources": (
                {"name": "A", "question": "Start at 12, add 7, multiply by 3, then subtract 5.", "source_state": 12},
                {"name": "B", "question": "Start at 14, add 7, multiply by 3, then subtract 5.", "source_state": 14},
            ),
            "displayed_states": (12, 14),
            "operand": 7,
            "symbol": "+",
        },
        {
            "operation": "multiply",
            "sources": (
                {"name": "A", "question": "Start at 12, add 7, multiply by 3, then subtract 5.", "source_state": 19},
                {"name": "B", "question": "Start at 13, add 8, multiply by 3, then subtract 5.", "source_state": 21},
            ),
            "displayed_states": (19, 21),
            "operand": 3,
            "symbol": "*",
        },
        {
            "operation": "subtract",
            "sources": (
                {"name": "A", "question": "Start at 12, add 7, multiply by 3, then subtract 5.", "source_state": 57},
                {"name": "B", "question": "Start at 13, add 8, multiply by 3, then subtract 5.", "source_state": 63},
            ),
            "displayed_states": (57, 63),
            "operand": 5,
            "symbol": "-",
        },
    ]
    cells = []
    for spec in operations:
        for source in spec["sources"]:
            for displayed in spec["displayed_states"]:
                source_result = apply_named(source["source_state"], spec["operation"], spec["operand"])
                local_result = apply_named(displayed, spec["operation"], spec["operand"])
                prompt = (
                    f"Problem: {source['question']}\n"
                    f"Work: {displayed}{spec['symbol']}{spec['operand']}="
                )
                cells.append({
                    "operation": spec["operation"],
                    "source": source["name"],
                    "source_question": source["question"],
                    "source_state": source["source_state"],
                    "displayed_state": displayed,
                    "operand": spec["operand"],
                    "prompt": prompt,
                    "source_result": source_result,
                    "local_result": local_result,
                    "crossed": source["source_state"] != displayed,
                })
    return cells


def apply_named(value, operation, operand):
    if operation == "add":
        return value + operand
    if operation == "multiply":
        return value * operand
    if operation == "subtract":
        return value - operand
    raise ValueError(operation)


def sequence_logprob(model, tokenizer, prompt, continuation, device):
    prompt_ids = tokenizer.encode(prompt).ids
    continuation_ids = tokenizer.encode(continuation).ids
    if not prompt_ids or not continuation_ids:
        raise ValueError("prompt and continuation must tokenize nonempty")
    full = prompt_ids + continuation_ids
    x = torch.tensor(full[:-1], dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        logits, _ = model(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
    start = len(prompt_ids) - 1
    values = []
    for offset, target in enumerate(continuation_ids):
        values.append(float(log_probs[0, start + offset, target].item()))
    return {
        "continuation": continuation,
        "token_ids": continuation_ids,
        "token_logprobs": values,
        "sum_logprob": sum(values),
        "mean_logprob": sum(values) / len(values),
    }


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint, model = load_model(args.ckpt, device)
    rows = []
    for cell in frozen_cells():
        source_text = f" {cell['source_result']}"
        local_text = f" {cell['local_result']}"
        source_score = sequence_logprob(model, tokenizer, cell["prompt"], source_text, device)
        if source_text == local_text:
            local_score = dict(source_score)
        else:
            local_score = sequence_logprob(model, tokenizer, cell["prompt"], local_text, device)
        if len(source_score["token_ids"]) != len(local_score["token_ids"]):
            raise RuntimeError("interchange candidates must have equal token counts")
        margin = local_score["sum_logprob"] - source_score["sum_logprob"]
        row = dict(cell)
        row.update({
            "source_score": source_score,
            "local_score": local_score,
            "local_minus_source_logprob": margin,
            "winner": "tie" if source_text == local_text else "local" if margin > 0 else "source",
        })
        rows.append(row)
        print(
            f"[interchange] operation={cell['operation']} source={cell['source']} "
            f"displayed={cell['displayed_state']} crossed={cell['crossed']} margin={margin:.4f}",
            flush=True,
        )

    crossed = [row for row in rows if row["crossed"]]
    result = {
        "audit": "renderer_interchange_causal_audit_v1",
        "interpretation": "posthoc_visible_state_causality_diagnostic_only",
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_sha256": sha256_file(args.ckpt),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "device": device,
        "resource_ledger": {
            "cells": len(rows),
            "crossed_cells": len(crossed),
            "candidate_sequence_evaluations": sum(1 if row["source_result"] == row["local_result"] else 2 for row in rows),
            "generated_tokens": 0,
            "training_tokens": 0,
            "retries": 0,
            "verifier_feedback": 0,
            "source_retained": True,
            "external_candidate_construction": True,
        },
        "summary": {
            "crossed_local_wins": sum(row["winner"] == "local" for row in crossed),
            "crossed_source_wins": sum(row["winner"] == "source" for row in crossed),
            "crossed_ties": sum(row["winner"] == "tie" for row in crossed),
            "minimum_crossed_absolute_margin": min(abs(row["local_minus_source_logprob"]) for row in crossed),
            "by_operation": {
                operation: {
                    "local_wins": sum(row["winner"] == "local" for row in crossed if row["operation"] == operation),
                    "source_wins": sum(row["winner"] == "source" for row in crossed if row["operation"] == operation),
                }
                for operation in ("add", "multiply", "subtract")
            },
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
