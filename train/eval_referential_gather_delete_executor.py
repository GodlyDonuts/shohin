#!/usr/bin/env python3
"""Evaluate atomic-trained Stage-B executors on two-step source-deleted composition."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPTConfig
from referential_gather_delete_executor import (
    GatherDeletePermutationExecutor,
    SourceRetainedAnswerControl,
    execution_targets,
    executor_state_hash,
    gather_source_deleted_packet,
    shuffle_operation_packet,
    shuffle_query_packet,
)
from referential_literal_pointer_compiler import (
    load_examples,
    make_batches,
    pad_batch,
    sha256_file,
)
from train_referential_gather_delete_executor import load_frozen_compiler


def summarize(records):
    if not records:
        return {}
    total = len(records)
    summary = {
        "rows": total,
        "answer_accuracy": sum(row["answer_correct"] for row in records) / total,
    }
    if records[0]["final_assignment_exact"] is not None:
        summary.update({
            "query_accuracy": sum(row["query_correct"] for row in records) / total,
            "final_assignment_exact": sum(
                row["final_assignment_exact"] for row in records
            ) / total,
            "all_transitions_exact": sum(
                row["all_transitions_exact"] for row in records
            ) / total,
            "transition0_exact": sum(row["transition_exact"][0] for row in records) / total,
            "transition1_exact": sum(row["transition_exact"][1] for row in records) / total,
            "entity_match_accuracy": sum(
                sum(row["entity_match_correct"]) for row in records
            ) / (2 * total),
            "amount_accuracy": sum(
                sum(row["amount_correct"]) for row in records
            ) / (2 * total),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--executor", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--split", choices=(
        "development_compositional", "development_lexical_ood",
    ), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--packet-oracle", choices=("none", "full"), default="none")
    parser.add_argument("--shuffle-operations", action="store_true")
    parser.add_argument("--shuffle-query", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Stage-B executor evaluation requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing Stage-B evaluation output")
    report = json.load(open(args.report))
    if not report.get("all_gates_pass"):
        raise SystemExit("factorized corpus report did not pass")
    if report.get("artifacts", {}).get(args.split, {}).get("sha256") != sha256_file(args.data):
        raise SystemExit("factorized report does not bind evaluation bytes")
    bundle = torch.load(args.executor, map_location="cpu")
    metadata = bundle.get("executor", {})
    if metadata.get("protocol") != "r12_referential_gather_delete_executor_stage_b_v1":
        raise SystemExit("invalid Stage-B executor protocol")
    if metadata.get("confirmation_access") != 0:
        raise SystemExit("executor metadata records confirmation access")
    for field, path in (
        ("base_sha256", args.base),
        ("compiler_file_sha256", args.compiler),
        ("tokenizer_sha256", args.tokenizer),
    ):
        if metadata.get(field) != sha256_file(path):
            raise SystemExit("Stage-B {} identity mismatch".format(field))
    if metadata["arm"] == "source_retained" and (
        args.packet_oracle != "none" or args.shuffle_operations or args.shuffle_query
    ):
        raise SystemExit("source-retained control cannot use packet interventions")
    if args.shuffle_operations and args.shuffle_query:
        raise SystemExit("apply one Stage-B packet intervention per evaluation")

    device = "cuda"
    checkpoint, compiler, compiler_metadata = load_frozen_compiler(
        args.base, args.compiler, args.tokenizer, device,
    )
    tokenizer = Tokenizer.from_file(args.tokenizer)
    cfg = GPTConfig(**checkpoint["cfg"])
    examples = load_examples(
        args.data,
        tokenizer,
        args.split,
        cfg.seq_len,
        keep_evidence=True,
        limit=args.limit,
    )
    if metadata["arm"] == "source_retained":
        executor = SourceRetainedAnswerControl(
            packet_width=int(metadata["packet_width"]),
            width=int(metadata["executor_width"]),
            heads=8,
            layers=2,
            ff=4 * int(metadata["executor_width"]),
        ).to(device).eval()
    else:
        executor = GatherDeletePermutationExecutor(
            packet_width=int(metadata["packet_width"]),
            width=int(metadata["executor_width"]),
            tied=metadata["arm"] == "tied",
        ).to(device).eval()
    executor.load_state_dict(bundle["executor_state"], strict=True)
    if executor_state_hash(executor) != metadata["final_executor_sha256"]:
        raise SystemExit("executor state hash mismatch")

    records = [None] * len(examples)
    batches = make_batches(examples, args.batch_size, seed=0, shuffle=False)
    intervention_rows = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_number, indices in enumerate(batches, 1):
            selected, ids, valid = pad_batch(examples, indices, device)
            compiler_outputs = compiler(ids, valid)
            targets = [execution_targets(example) for example in selected]
            if metadata["arm"] == "source_retained":
                outputs = executor(compiler_outputs["memory"], valid)
                answers = outputs["answer_logits"].argmax(-1).tolist()
                transition_predictions = None
                final_predictions = None
                query_predictions = None
                entity_predictions = None
                amount_predictions = None
            else:
                packet = gather_source_deleted_packet(
                    compiler_outputs,
                    selected,
                    valid,
                    oracle=args.packet_oracle,
                )
                if args.shuffle_operations and len(selected) > 1:
                    permutation = torch.roll(
                        torch.arange(len(selected), device=device), shifts=1,
                    )
                    packet = shuffle_operation_packet(packet, permutation)
                    intervention_rows += len(selected)
                if args.shuffle_query and len(selected) > 1:
                    permutation = torch.roll(
                        torch.arange(len(selected), device=device), shifts=1,
                    )
                    packet = shuffle_query_packet(packet, permutation)
                    intervention_rows += len(selected)
                outputs = executor(packet, cell_indices=(0, 1))
                answers = outputs["answer_probabilities"].argmax(-1).tolist()
                transition_predictions = [
                    logits.argmax(-1).tolist() for logits in outputs["transition_logits"]
                ]
                final_predictions = outputs["assignment"].argmax(-1).tolist()
                query_predictions = outputs["query_logits"].argmax(-1).tolist()
                entity_predictions = [
                    logits.argmax(-1).tolist() for logits in outputs["entity_match_logits"]
                ]
                amount_predictions = [
                    logits.argmax(-1).tolist() for logits in outputs["amount_logits"]
                ]
            for local, global_index in enumerate(indices):
                target = targets[local]
                if transition_predictions is None:
                    transition_exact = None
                    final_exact = None
                    query_correct = None
                    entity_correct = None
                    amount_correct = None
                    all_transitions = None
                else:
                    transition_exact = [
                        tuple(predictions[local]) == target.transition_sources[step]
                        for step, predictions in enumerate(transition_predictions)
                    ]
                    final_exact = tuple(final_predictions[local]) == target.final_identities
                    query_correct = int(query_predictions[local]) == target.query_position
                    entity_correct = [
                        int(predictions[local]) == target.entity_locations[step]
                        for step, predictions in enumerate(entity_predictions)
                    ]
                    amount_correct = [
                        int(predictions[local]) == target.amounts[step]
                        for step, predictions in enumerate(amount_predictions)
                    ]
                    all_transitions = all(transition_exact)
                example = selected[local]
                records[global_index] = {
                    "id": example.row_id,
                    "group": example.group,
                    "surface_type": example.surface_type,
                    "factors": dict(example.factors),
                    "answer_identity_prediction": int(answers[local]),
                    "answer_identity_target": target.answer_identity,
                    "answer_correct": int(answers[local]) == target.answer_identity,
                    "transition_exact": transition_exact,
                    "all_transitions_exact": all_transitions,
                    "final_assignment_exact": final_exact,
                    "query_correct": query_correct,
                    "entity_match_correct": entity_correct,
                    "amount_correct": amount_correct,
                }
            if batch_number % 25 == 0:
                print("[rgde-eval] {}/{} batches".format(
                    batch_number, len(batches),
                ), flush=True)

    by_surface = {
        surface: summarize([row for row in records if row["surface_type"] == surface])
        for surface in sorted({row["surface_type"] for row in records})
    }
    groups = collections.defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)
    group_summary = {
        "groups": len(groups),
        "all_four_answers_correct": sum(
            len(rows) == 4 and all(row["answer_correct"] for row in rows)
            for rows in groups.values()
        ),
        "all_four_final_assignments_exact": (
            None if metadata["arm"] == "source_retained" else sum(
                len(rows) == 4 and all(row["final_assignment_exact"] for row in rows)
                for rows in groups.values()
            )
        ),
        "all_four_transitions_exact": (
            None if metadata["arm"] == "source_retained" else sum(
                len(rows) == 4 and all(row["all_transitions_exact"] for row in rows)
                for rows in groups.values()
            )
        ),
    }
    result = {
        "schema": "r12_referential_gather_delete_executor_eval_v1",
        "split": args.split,
        "arm": metadata["arm"],
        "source_deleted": metadata["source_deleted"],
        "training_contract": metadata["training_contract"],
        "base_sha256": sha256_file(args.base),
        "compiler_file_sha256": sha256_file(args.compiler),
        "compiler_adapter_sha256": compiler_metadata["final_adapter_sha256"],
        "executor_file_sha256": sha256_file(args.executor),
        "executor_state_sha256": metadata["final_executor_sha256"],
        "data_sha256": sha256_file(args.data),
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "packet_oracle": args.packet_oracle,
        "shuffle_operations": args.shuffle_operations,
        "shuffle_query": args.shuffle_query,
        "intervention_rows": intervention_rows,
        "host_supplied_inference_fields": {
            "source_role_positions": args.packet_oracle == "full",
            "operation_classes": args.packet_oracle == "full",
            "state_update": False,
            "query_answer": False,
        },
        "confirmation_access": 0,
        "overall": summarize(records),
        "by_surface": by_surface,
        "group_summary": group_summary,
        "records": records,
        "claim_boundary": (
            "Development-only atomic-trained two-step list execution. Source-deleted treatment "
            "is a component test, not natural-language reasoning, halt, rollout, or novelty."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(Path(args.out).resolve()),
        "overall": result["overall"],
        "group_summary": group_summary,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
