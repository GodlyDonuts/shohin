#!/usr/bin/env python3
"""Render direct text-only interactions for a referential slot compiler."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from categorical_microcode import OPCODES, QUERIES, execute_program, sha256_file
from model import GPT, GPTConfig
from referential_slot_microcode import ReferentialSlotMicrocodeCompiler, compile_referential_example


def attention_hit(weights, positions, targets):
    if not targets:
        return None
    return int(positions[int(weights.argmax().item())]) in set(map(int, targets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--label-admission", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("referential manual inspection requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing transcript")
    adapter_checkpoint = torch.load(args.adapter, map_location="cpu")
    metadata = adapter_checkpoint.get("categorical_microcode", {})
    if metadata.get("protocol") != "causal_microcode_referential_slots_v4":
        raise SystemExit("invalid referential adapter")
    if metadata.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("adapter does not bind supplied base")
    label_admission = json.load(open(args.label_admission))
    if not label_admission.get("all_checks_pass"):
        raise SystemExit("mention-label admission failed")
    if label_admission["datasets"]["manual"].get("sha256") != sha256_file(args.data):
        raise SystemExit("mention-label admission does not bind manual board")
    if metadata.get("label_admission_sha256") != sha256_file(args.label_admission):
        raise SystemExit("adapter does not bind mention-label admission")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    examples = [compile_referential_example(row, tokenizer) for row in rows]
    base_checkpoint = torch.load(args.base, map_location="cpu")
    cfg = GPTConfig(**base_checkpoint["cfg"])
    model = GPT(cfg).to("cuda").eval()
    model.load_state_dict(base_checkpoint["model"])
    compiler = ReferentialSlotMicrocodeCompiler(
        model, layer=int(metadata["layer"]), hidden=int(metadata["hidden"]),
        role_mode=metadata["role_mode"],
    ).to("cuda").eval()
    missing, unexpected = compiler.load_state_dict(adapter_checkpoint["adapter_state"], strict=False)
    missing = [name for name in missing if not name.startswith("model.")]
    unexpected = [name for name in unexpected if not name.startswith("model.")]
    if missing or unexpected:
        raise SystemExit("adapter mismatch missing={} unexpected={}".format(missing, unexpected))

    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row, wrapped in zip(rows, examples):
            example = wrapped.compiled
            ids = torch.tensor([example.ids], dtype=torch.long, device="cuda")
            hidden, identity = compiler.encode(ids)
            output = compiler.classify_text(
                hidden[0], identity[0], wrapped.intro_positions,
                wrapped.operation_spans, wrapped.query_span,
            )
            op_predictions = [
                int(compiler.compose_operation_logits(
                    item["kind_logits"].unsqueeze(0), item["role_logits"].unsqueeze(0),
                ).argmax().item()) for item in output["operations"]
            ]
            query_prediction = int(compiler.compose_query_logits(
                output["query"]["kind_logits"].unsqueeze(0),
                output["query"]["role_logits"].unsqueeze(0),
            ).argmax().item())
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
                "program_exact": tuple(op_predictions) == example.operation_targets and query_prediction == example.query_target,
                "answer_correct": answer == example.answer,
                "intro_slot_hits": [
                    attention_hit(
                        output["intro_weights"][:, slot], output["intro_positions"],
                        wrapped.intro_slot_targets[slot],
                    ) for slot in range(2)
                ],
                "operation_mention_hits": [
                    attention_hit(item["target_weights"], item["positions"], targets)
                    for item, targets in zip(output["operations"], wrapped.operation_mention_targets)
                ],
                "query_mention_hit": attention_hit(
                    output["query"]["target_weights"], output["query"]["positions"],
                    wrapped.query_mention_target,
                ),
            }
            records.append(record)
            print("\n[referential-manual] {}\n{}\nprogram={} query={} answer={} expected={} correct={}".format(
                example.reference, row["question"], record["predicted_operations"],
                record["predicted_query"], answer, example.answer, record["answer_correct"],
            ), flush=True)

    role_operation_hits = [
        hit for record in records for hit in record["operation_mention_hits"] if hit is not None
    ]
    query_hits = [record["query_mention_hit"] for record in records if record["query_mention_hit"] is not None]
    result = {
        "audit": "referential_slot_manual_interaction_v4",
        "base": os.path.realpath(args.base),
        "base_sha256": sha256_file(args.base),
        "adapter": os.path.realpath(args.adapter),
        "adapter_sha256": sha256_file(args.adapter),
        "adapter_metadata": metadata,
        "data": os.path.realpath(args.data),
        "data_sha256": sha256_file(args.data),
        "label_admission": os.path.realpath(args.label_admission),
        "label_admission_sha256": sha256_file(args.label_admission),
        "cases": len(records),
        "program_exact": sum(record["program_exact"] for record in records),
        "answer_correct": sum(record["answer_correct"] for record in records),
        "intro_slot_attention": sum(sum(record["intro_slot_hits"]) for record in records),
        "intro_slot_attention_total": 2 * len(records),
        "operation_mention_attention": sum(role_operation_hits),
        "operation_mention_attention_total": len(role_operation_hits),
        "query_mention_attention": sum(query_hits),
        "query_mention_attention_total": len(query_hits),
        "records": records,
        "claim_boundary": (
            "Hand-authored text-only entity-binding interactions. Structured keys score predicted "
            "mentions but are not passed to the compiler."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
