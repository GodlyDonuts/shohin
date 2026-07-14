#!/usr/bin/env python3
"""Evaluate binding-first referential slot compilation and exact execution."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from categorical_microcode import alu_basis_accuracy, execute_program, sha256_file
from eval_categorical_microcode import locked_gates, summarize
from model import GPT, GPTConfig
from referential_slot_microcode import ReferentialSlotMicrocodeCompiler, compile_referential_example
from role_equivariant_microcode import IGNORE_ROLE, factor_operation, factor_query


def load_examples(path, tokenizer, seq_len):
    examples = []
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            example = compile_referential_example(json.loads(line), tokenizer)
            if len(example.compiled.ids) > seq_len:
                raise ValueError("held-out row {} exceeds sequence length".format(line_number))
            examples.append(example)
    if not examples:
        raise ValueError("no referential examples")
    return examples


def all_batches(examples, batch_size):
    buckets = collections.defaultdict(list)
    for index, example in enumerate(examples):
        buckets[len(example.compiled.ids)].append(index)
    return [
        indices[offset:offset + batch_size]
        for _, indices in sorted(buckets.items())
        for offset in range(0, len(indices), batch_size)
    ]


def safe_execute(example, opcodes, query, table):
    try:
        return execute_program(
            example.initial_values, list(map(int, opcodes)), example.operation_values,
            int(query), table,
        )
    except (IndexError, ValueError):
        return None


def attention_hit(weights, positions, targets):
    if not targets:
        return None
    predicted = int(positions[int(weights.argmax().item())])
    return predicted in set(map(int, targets))


def factor_diagnostics(records):
    output = {}
    regimes = ["all"] + sorted({record["regime"] for record in records})
    for regime in regimes:
        selected = records if regime == "all" else [record for record in records if record["regime"] == regime]
        operation_total = sum(len(record["operation_targets"]) for record in selected)
        role_operations = sum(sum(role != IGNORE_ROLE for role in record["operation_role_targets"]) for record in selected)
        kind_correct_role_ops = sum(sum(
            target_role != IGNORE_ROLE and predicted_kind == target_kind
            for target_kind, predicted_kind, target_role in zip(
                record["operation_kind_targets"], record["operation_kind_predictions"],
                record["operation_role_targets"],
            )
        ) for record in selected)
        operation_role_correct_given_kind = sum(sum(
            target_role != IGNORE_ROLE and predicted_kind == target_kind and predicted_role == target_role
            for target_kind, predicted_kind, target_role, predicted_role in zip(
                record["operation_kind_targets"], record["operation_kind_predictions"],
                record["operation_role_targets"], record["operation_role_predictions"],
            )
        ) for record in selected)
        role_queries = [record for record in selected if record["query_role_target"] != IGNORE_ROLE]
        operation_mentions = [
            hit for record in selected for hit in record["operation_mention_hits"] if hit is not None
        ]
        query_mentions = [record["query_mention_hit"] for record in selected if record["query_mention_hit"] is not None]
        intro_mentions = [hit for record in selected for hit in record["intro_slot_hits"]]
        by_kind = {}
        for kind in range(5):
            cells = [
                (target_kind, predicted_kind, target_role, predicted_role)
                for record in selected
                for target_kind, predicted_kind, target_role, predicted_role in zip(
                    record["operation_kind_targets"], record["operation_kind_predictions"],
                    record["operation_role_targets"], record["operation_role_predictions"],
                ) if target_kind == kind
            ]
            by_kind[str(kind)] = {
                "cases": len(cells),
                "kind_correct": sum(predicted_kind == target_kind for target_kind, predicted_kind, _, _ in cells),
                "exact_factor": sum(
                    predicted_kind == target_kind and (
                        target_role == IGNORE_ROLE or predicted_role == target_role
                    ) for target_kind, predicted_kind, target_role, predicted_role in cells
                ),
            }
        output[regime] = {
            "cases": len(selected),
            "operation_kind_accuracy": sum(sum(record["operation_kind_correct"]) for record in selected) / operation_total,
            "operation_role_joint_accuracy": sum(sum(record["operation_role_joint_correct"]) for record in selected) / role_operations,
            "operation_role_given_kind_accuracy": (
                operation_role_correct_given_kind / kind_correct_role_ops if kind_correct_role_ops else 0.0
            ),
            "query_kind_accuracy": sum(record["query_kind_correct"] for record in selected) / len(selected),
            "query_role_joint_accuracy": (
                sum(record["query_role_joint_correct"] for record in role_queries) / len(role_queries)
                if role_queries else 0.0
            ),
            "intro_slot_attention_accuracy": sum(intro_mentions) / len(intro_mentions),
            "operation_mention_attention_accuracy": sum(operation_mentions) / len(operation_mentions),
            "query_mention_attention_accuracy": sum(query_mentions) / len(query_mentions) if query_mentions else 0.0,
            "by_operation_kind": by_kind,
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--label-admission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("referential slot evaluation requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing output")

    adapter_checkpoint = torch.load(args.adapter, map_location="cpu")
    metadata = adapter_checkpoint.get("categorical_microcode", {})
    if metadata.get("protocol") != "causal_microcode_referential_slots_v4":
        raise SystemExit("invalid referential adapter metadata")
    if metadata.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("adapter does not bind supplied base")
    admission = json.load(open(args.admission))
    label_admission = json.load(open(args.label_admission))
    data_sha256 = sha256_file(args.data)
    if not admission.get("all_checks_pass") or admission.get("eval_sha256") != data_sha256:
        raise SystemExit("structural admission does not bind evaluation data")
    if admission.get("train_sha256") != metadata.get("data_sha256"):
        raise SystemExit("structural admission does not bind adapter data")
    if not label_admission.get("all_checks_pass"):
        raise SystemExit("mention-label admission failed")
    if label_admission["datasets"]["eval"].get("sha256") != data_sha256:
        raise SystemExit("mention-label admission does not bind evaluation data")
    if metadata.get("label_admission_sha256") != sha256_file(args.label_admission):
        raise SystemExit("adapter does not bind mention-label admission")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    base_checkpoint = torch.load(args.base, map_location="cpu")
    cfg = GPTConfig(**base_checkpoint["cfg"])
    examples = load_examples(args.data, tokenizer, cfg.seq_len)
    batches = all_batches(examples, args.batch_size)

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

    predictions = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_number, indices in enumerate(batches, 1):
            ids = torch.tensor(
                [examples[index].compiled.ids for index in indices], dtype=torch.long, device="cuda",
            )
            hidden, identity = compiler.encode(ids)
            for local, index in enumerate(indices):
                example = examples[index]
                output = compiler.classify_text(
                    hidden[local], identity[local], example.intro_positions,
                    example.operation_spans, example.query_span,
                )
                op_kind = [int(item["kind_logits"].argmax().item()) for item in output["operations"]]
                op_role = [int(item["role_logits"].argmax().item()) for item in output["operations"]]
                opcodes = [
                    int(compiler.compose_operation_logits(
                        item["kind_logits"].unsqueeze(0), item["role_logits"].unsqueeze(0),
                    ).argmax().item()) for item in output["operations"]
                ]
                query_kind = int(output["query"]["kind_logits"].argmax().item())
                query_role = int(output["query"]["role_logits"].argmax().item())
                query = int(compiler.compose_query_logits(
                    output["query"]["kind_logits"].unsqueeze(0),
                    output["query"]["role_logits"].unsqueeze(0),
                ).argmax().item())
                predictions[index] = {
                    "operations": opcodes,
                    "operation_kind": op_kind,
                    "operation_role": op_role,
                    "query": query,
                    "query_kind": query_kind,
                    "query_role": query_role,
                    "intro_slot_hits": [
                        attention_hit(
                            output["intro_weights"][:, slot], output["intro_positions"],
                            example.intro_slot_targets[slot],
                        ) for slot in range(2)
                    ],
                    "operation_mention_hits": [
                        attention_hit(item["target_weights"], item["positions"], targets)
                        for item, targets in zip(output["operations"], example.operation_mention_targets)
                    ],
                    "query_mention_hit": attention_hit(
                        output["query"]["target_weights"], output["query"]["positions"],
                        example.query_mention_target,
                    ),
                }
            if batch_number % 20 == 0 or batch_number == len(batches):
                print("[referential-eval] {}/{} batches".format(batch_number, len(batches)), flush=True)

    table = compiler.transition_logits.detach().cpu()
    basis_correct, basis_total = alu_basis_accuracy(table)
    records = []
    for index, wrapped in enumerate(examples):
        example = wrapped.compiled
        prediction = predictions[index]
        predicted_answer = safe_execute(example, prediction["operations"], prediction["query"], table)
        oracle_answer = safe_execute(example, example.operation_targets, example.query_target, table)
        op_kind_targets = [factor_operation(target)[0] for target in example.operation_targets]
        op_role_targets = [factor_operation(target)[1] for target in example.operation_targets]
        query_kind_target, query_role_target = factor_query(example.query_target)
        operation_correct = sum(
            predicted == target for predicted, target in zip(prediction["operations"], example.operation_targets)
        )
        query_correct = prediction["query"] == example.query_target
        records.append({
            "index": index,
            "reference": example.reference,
            "regime": example.regime,
            "depth": len(example.operation_targets),
            "operation_targets": list(example.operation_targets),
            "operation_predictions": prediction["operations"],
            "operation_values": list(example.operation_values),
            "operation_correct": operation_correct,
            "operation_total": len(example.operation_targets),
            "operation_kind_targets": op_kind_targets,
            "operation_kind_predictions": prediction["operation_kind"],
            "operation_kind_correct": [
                predicted == target for predicted, target in zip(prediction["operation_kind"], op_kind_targets)
            ],
            "operation_role_targets": op_role_targets,
            "operation_role_predictions": prediction["operation_role"],
            "operation_role_joint_correct": [
                target_role != IGNORE_ROLE and predicted_kind == target_kind and predicted_role == target_role
                for target_kind, predicted_kind, target_role, predicted_role in zip(
                    op_kind_targets, prediction["operation_kind"], op_role_targets, prediction["operation_role"],
                )
            ],
            "query_target": example.query_target,
            "query_prediction": prediction["query"],
            "query_correct": query_correct,
            "query_kind_target": query_kind_target,
            "query_kind_prediction": prediction["query_kind"],
            "query_kind_correct": prediction["query_kind"] == query_kind_target,
            "query_role_target": query_role_target,
            "query_role_prediction": prediction["query_role"],
            "query_role_joint_correct": (
                query_role_target != IGNORE_ROLE and prediction["query_kind"] == query_kind_target
                and prediction["query_role"] == query_role_target
            ),
            "intro_slot_hits": prediction["intro_slot_hits"],
            "operation_mention_hits": prediction["operation_mention_hits"],
            "query_mention_hit": prediction["query_mention_hit"],
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
            donor = predictions[indices[(offset + 1) % len(indices)]]
            shuffled_answer = safe_execute(
                examples[index].compiled, donor["operations"], donor["query"], table,
            )
            records[index]["shuffled_answer"] = shuffled_answer
            records[index]["shuffled_answer_correct"] = (
                shuffled_answer == examples[index].compiled.answer
            )

    summary = summarize(records)
    gates = locked_gates(summary, basis_correct, basis_total)
    result = {
        "audit": "referential_slot_microcode_eval_v4",
        "base": os.path.realpath(args.base),
        "base_sha256": sha256_file(args.base),
        "adapter": os.path.realpath(args.adapter),
        "adapter_sha256": sha256_file(args.adapter),
        "adapter_metadata": metadata,
        "data": os.path.realpath(args.data),
        "data_sha256": data_sha256,
        "admission": os.path.realpath(args.admission),
        "admission_sha256": sha256_file(args.admission),
        "label_admission": os.path.realpath(args.label_admission),
        "label_admission_sha256": sha256_file(args.label_admission),
        "cases": len(examples),
        "batches": len(batches),
        "alu_basis": {"correct": basis_correct, "total": basis_total},
        "summary": summary,
        "factor_diagnostics": factor_diagnostics(records),
        "gates": gates,
        "advance_to_decoder_bridge": all(gates.values()),
        "records": records,
        "claim_boundary": (
            "A pass establishes narrow text-only dynamic entity binding, categorical compilation, "
            "and exact supplied execution; it is not broad autonomous reasoning."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[referential-eval] " + json.dumps({
        "summary": summary, "gates": gates, "advance_to_decoder_bridge": result["advance_to_decoder_bridge"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
