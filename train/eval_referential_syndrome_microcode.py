#!/usr/bin/env python3
"""Evaluate R9c directional semantics, replay, and common-mode failures."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from categorical_microcode import alu_basis_accuracy, execute_program, sha256_file
from eval_referential_slot_microcode import load_examples
from referential_syndrome_microcode import ReferentialSyndromeBridge
from train_referential_slot_microcode import pad_ids
from train_referential_syndrome_microcode import ARM_CONFIGS, load_pointer


def matched_batches(examples, batch_size):
    buckets = collections.defaultdict(list)
    for index, example in enumerate(examples):
        key = (len(example.compiled.ids), len(example.compiled.operation_targets))
        buckets[key].append(index)
    return [
        indices[offset:offset + int(batch_size)]
        for key, indices in sorted(buckets.items())
        for offset in range(0, len(indices), int(batch_size))
    ]


def safe_execute(example, opcodes, query, table):
    try:
        return execute_program(
            example.initial_values, list(map(int, opcodes)), example.operation_values,
            int(query), table,
        )
    except (IndexError, ValueError):
        return None


def summarize(records, mode):
    selected = [record[mode] for record in records]
    operation_total = sum(len(record["operation_targets"]) for record in records)
    joint_correct = sum(sum(item["joint_operation_correct"]) for item in selected)
    forward_correct = sum(sum(item["forward_operation_correct"]) for item in selected)
    backward_correct = sum(sum(item["backward_operation_correct"]) for item in selected)
    agreements = sum(sum(item["directional_agreement"]) for item in selected)
    agreed_correct = sum(sum(
        agree and correct for agree, correct in zip(
            item["directional_agreement"], item["joint_operation_correct"],
        )
    ) for item in selected)
    agreed_wrong = agreements - agreed_correct
    wrong = operation_total - joint_correct
    return {
        "cases": len(records),
        "operations": operation_total,
        "joint_operation_correct": joint_correct,
        "joint_operation_accuracy": joint_correct / operation_total,
        "forward_operation_accuracy": forward_correct / operation_total,
        "backward_operation_accuracy": backward_correct / operation_total,
        "directional_agreements": agreements,
        "directional_agreement_rate": agreements / operation_total,
        "agreed_correct": agreed_correct,
        "agreed_wrong_common_mode": agreed_wrong,
        "certified_precision": agreed_correct / agreements if agreements else 0.0,
        "common_mode_fraction_of_wrong": agreed_wrong / wrong if wrong else 0.0,
        "program_exact": sum(item["program_exact"] for item in selected),
        "program_exact_accuracy": sum(item["program_exact"] for item in selected) / len(records),
        "answer_correct": sum(item["answer_correct"] for item in selected),
        "answer_accuracy": sum(item["answer_correct"] for item in selected) / len(records),
        "query_correct": sum(item["query_correct"] for item in selected),
        "query_accuracy": sum(item["query_correct"] for item in selected) / len(records),
        "event_updates": sum(item["event_updates"] for item in selected),
        "mean_event_updates": sum(item["event_updates"] for item in selected) / operation_total,
        "mean_final_syndrome": sum(item["syndrome_sum"] for item in selected) / operation_total,
    }


def summaries(records, mode):
    output = {"all": summarize(records, mode)}
    for regime in sorted({record["regime"] for record in records}):
        output[regime] = summarize([record for record in records if record["regime"] == regime], mode)
    return output


def predictions_for_run(encoded, run, examples, table):
    forward = run.forward_logits.argmax(dim=-1)
    backward = run.backward_logits.argmax(dim=-1)
    joint = (0.5 * (run.forward_logits + run.backward_logits)).argmax(dim=-1)
    query = encoded.query_logits.argmax(dim=-1)
    output = []
    for local, wrapped in enumerate(examples):
        example = wrapped.compiled
        target = encoded.operation_targets[local]
        forward_prediction = forward[local].tolist()
        backward_prediction = backward[local].tolist()
        joint_prediction = joint[local].tolist()
        query_prediction = int(query[local].item())
        operation_correct = [
            predicted == int(expected)
            for predicted, expected in zip(joint_prediction, target.tolist())
        ]
        agreement = [left == right for left, right in zip(forward_prediction, backward_prediction)]
        answer = safe_execute(example, joint_prediction, query_prediction, table)
        output.append({
            "forward_operations": forward_prediction,
            "backward_operations": backward_prediction,
            "joint_operations": joint_prediction,
            "forward_operation_correct": [
                predicted == int(expected)
                for predicted, expected in zip(forward_prediction, target.tolist())
            ],
            "backward_operation_correct": [
                predicted == int(expected)
                for predicted, expected in zip(backward_prediction, target.tolist())
            ],
            "joint_operation_correct": operation_correct,
            "directional_agreement": agreement,
            "query_prediction": query_prediction,
            "query_correct": query_prediction == example.query_target,
            "program_exact": all(operation_correct) and query_prediction == example.query_target,
            "predicted_answer": answer,
            "answer_correct": answer == example.answer,
            "event_updates": sum(int(mask[local].sum().item()) for mask in run.active_masks),
            "syndrome_sum": float(run.syndrome_norm[local].sum().item()),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--pointer-adapter", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--syndrome-threshold", type=float, default=0.05)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("referential syndrome evaluation requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing output")

    checkpoint = torch.load(args.adapter, map_location="cpu")
    metadata = checkpoint.get("referential_syndrome_microcode", {})
    if metadata.get("protocol") != "referential_bidirectional_syndrome_microcode_r9c":
        raise SystemExit("invalid R9c adapter protocol")
    if metadata.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("R9c adapter does not bind supplied base")
    if metadata.get("pointer_adapter_sha256") != sha256_file(args.pointer_adapter):
        raise SystemExit("R9c adapter does not bind supplied pointer")
    arm_name = metadata.get("arm")
    if arm_name not in ARM_CONFIGS or metadata.get("arm_config") != ARM_CONFIGS[arm_name]:
        raise SystemExit("R9c arm metadata differs from frozen runtime contract")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    base, pointer_metadata, pointer = load_pointer(args.base, args.pointer_adapter, "cuda")
    examples = load_examples(args.data, tokenizer, pointer.model.cfg.seq_len)
    batches = matched_batches(examples, args.batch_size)
    bridge = ReferentialSyndromeBridge(
        pointer, pointer_hidden=int(pointer_metadata["hidden"]),
        memory_dim=int(metadata["memory_dim"]),
    ).to("cuda").eval()
    missing, unexpected = bridge.microcode.load_state_dict(checkpoint["microcode_state"], strict=True)
    if missing or unexpected:
        raise SystemExit("R9c microcode mismatch missing={} unexpected={}".format(missing, unexpected))
    table = pointer.transition_logits.detach().cpu()
    basis_correct, basis_total = alu_basis_accuracy(table)
    records = [None] * len(examples)
    arm = ARM_CONFIGS[arm_name]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_number, indices in enumerate(batches, 1):
            selected = [examples[index] for index in indices]
            ids = pad_ids(selected, "cuda")
            encoded = bridge.encode_examples(ids, selected)
            input_goals = encoded.query_goals
            if arm["shuffle_goal"] and len(selected) > 1:
                input_goals = input_goals.roll(1, dims=0)
            common = dict(
                event_features=encoded.event_features,
                values=encoded.values,
                initial_values=encoded.initial_values,
                query_goals=input_goals,
                rounds=int(metadata["rounds"]),
                conditioning=arm["conditioning"],
                use_syndrome=arm["use_syndrome"],
            )
            fixed = bridge.microcode(**common, adaptive=False)
            adaptive = bridge.microcode(
                **common, adaptive=True, syndrome_threshold=args.syndrome_threshold,
            )
            fixed_predictions = predictions_for_run(encoded, fixed, selected, table)
            adaptive_predictions = predictions_for_run(encoded, adaptive, selected, table)
            for local, index in enumerate(indices):
                example = examples[index].compiled
                records[index] = {
                    "index": index,
                    "reference": example.reference,
                    "regime": example.regime,
                    "depth": len(example.operation_targets),
                    "operation_targets": list(example.operation_targets),
                    "operation_values": list(example.operation_values),
                    "query_target": example.query_target,
                    "expected_answer": example.answer,
                    "fixed": fixed_predictions[local],
                    "adaptive": adaptive_predictions[local],
                }
            if batch_number % 20 == 0 or batch_number == len(batches):
                print("[r9c-eval] {}/{} batches".format(batch_number, len(batches)), flush=True)

    result = {
        "audit": "referential_bidirectional_syndrome_microcode_eval_r9c",
        "base": os.path.realpath(args.base),
        "base_sha256": sha256_file(args.base),
        "pointer_adapter": os.path.realpath(args.pointer_adapter),
        "pointer_adapter_sha256": sha256_file(args.pointer_adapter),
        "adapter": os.path.realpath(args.adapter),
        "adapter_sha256": sha256_file(args.adapter),
        "adapter_metadata": metadata,
        "data": os.path.realpath(args.data),
        "data_sha256": sha256_file(args.data),
        "cases": len(examples),
        "batches": len(batches),
        "syndrome_threshold": args.syndrome_threshold,
        "alu_basis": {"correct": basis_correct, "total": basis_total},
        "fixed": summaries(records, "fixed"),
        "adaptive": summaries(records, "adaptive"),
        "records": records,
        "claim_boundary": (
            "Used-board matched-arm mechanism evaluation. Scores cannot establish fresh reasoning or "
            "context scaling; agreed-wrong operations are explicit common-mode certification failures."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(args.out).with_suffix(Path(args.out).suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.out)
    print("[r9c-eval] " + json.dumps({
        "arm": arm_name,
        "fixed": result["fixed"]["all"],
        "adaptive": result["adaptive"]["all"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
