#!/usr/bin/env python3
"""Evaluate the lexical closed-S3 system on its one-shot confirmation board."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from categorical_permutation_executor import (
    S3ClosedActionPermutationExecutor,
    module_state_hash,
)
from eval_rgde_depth_confirmation import long_targets
from eval_s3_categorical_depth import compile_packets, stack_packets
from model import GPTConfig
from referential_gather_delete_executor import semantic_derangement_permutation
from referential_literal_pointer_compiler import KIND_TO_ID, sha256_file
from train_referential_gather_delete_executor import load_frozen_compiler


def summarize(records):
    total = len(records)
    steps = sum(len(row["entity_correct"]) for row in records)
    result = {
        "rows": total,
        "answer_accuracy": sum(row["answer_correct"] for row in records) / total,
        "final_assignment_exact": sum(row["final_exact"] for row in records) / total,
        "all_transitions_exact": sum(row["all_transitions_exact"] for row in records) / total,
        "query_accuracy": sum(row["query_correct"] for row in records) / total,
        "entity_match_accuracy": sum(
            sum(row["entity_correct"]) for row in records
        ) / steps,
        "amount_accuracy": sum(sum(row["amount_correct"]) for row in records) / steps,
        "kind_accuracy": sum(sum(row["kind_correct"]) for row in records) / steps,
        "kind_lexical_coverage": sum(
            sum(row["kind_lexical_matched"]) for row in records
        ) / steps,
    }
    return result


def operation_key(row):
    initial = tuple(row["initial_order"])
    return tuple(
        (operation["kind"], initial.index(operation["entity"]), int(operation["amount"]))
        for operation in row["program"]
    )


def apply_intervention(rows, packets, intervention):
    if intervention == "none":
        return packets, 0
    output = [dict(packet) for packet in packets]
    by_depth = collections.defaultdict(list)
    for index, row in enumerate(rows):
        by_depth[int(row["depth"])].append(index)
    changed = 0
    for indices in by_depth.values():
        keys = (
            [operation_key(rows[index]) for index in indices]
            if intervention == "operations" else
            [int(rows[index]["query"]["position"]) for index in indices]
        )
        permutation = semantic_derangement_permutation(keys).tolist()
        for local, source_local in enumerate(permutation):
            destination = indices[local]
            source = indices[source_local]
            if keys[local] == keys[source_local]:
                raise AssertionError("semantic intervention did not change the field")
            if intervention == "operations":
                output[destination]["operations"] = packets[source]["operations"]
            else:
                output[destination]["query"] = packets[source]["query"]
            changed += 1
    return output, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--executor", required=True)
    parser.add_argument("--lexicon", required=True)
    parser.add_argument("--board", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--identity-mode", choices=("mean", "ordered", "gold"), required=True)
    parser.add_argument("--intervention", choices=("none", "operations", "query"), default="none")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("S3 lexical confirmation requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing S3 lexical confirmation output")
    report = json.load(open(args.report))
    if (
        report.get("schema") != "r12_s3_lexical_confirmation_report_v1"
        or not report.get("all_gates_pass")
        or report.get("old_confirmation_access") != 0
    ):
        raise SystemExit("S3 lexical confirmation report is not admitted")
    if report["artifact"]["sha256"] != sha256_file(args.board):
        raise SystemExit("confirmation report does not bind board")
    lexicon = json.load(open(args.lexicon))
    if not lexicon.get("all_gates_pass") or sha256_file(args.lexicon) != report["kind_lexicon_sha256"]:
        raise SystemExit("confirmation lexicon mismatch")
    rows = [json.loads(line) for line in open(args.board) if line.strip()]
    if not rows or any(row.get("split") != "confirmation_depth" for row in rows):
        raise SystemExit("invalid S3 lexical confirmation board")
    bundle = torch.load(args.executor, map_location="cpu")
    metadata = bundle.get("executor", {})
    if metadata.get("protocol") != "r12_s3_equivariant_permutation_executor_v1_1":
        raise SystemExit("confirmation requires frozen equivariant v1.1 state")
    checkpoint, compiler, compiler_metadata = load_frozen_compiler(
        args.base, args.compiler, args.tokenizer, "cuda",
    )
    cfg = GPTConfig(**checkpoint["cfg"])
    tokenizer = Tokenizer.from_file(args.tokenizer)
    packets, chunks = compile_packets(
        rows, tokenizer, compiler, cfg, args.identity_mode, "cuda", args.batch_size,
        lexicon=lexicon,
    )
    packets, intervention_rows = apply_intervention(rows, packets, args.intervention)
    executor = S3ClosedActionPermutationExecutor(
        identity_context_width=int(metadata["identity_context_width"]),
        context_width=int(metadata["context_width"]),
        width=int(metadata["executor_width"]),
    ).to("cuda").eval()
    executor.load_state_dict(bundle["executor_state"], strict=True)
    if module_state_hash(executor) != metadata["final_executor_sha256"]:
        raise SystemExit("confirmation executor state mismatch")

    records = [None] * len(rows)
    by_depth = collections.defaultdict(list)
    for index, row in enumerate(rows):
        by_depth[int(row["depth"])].append(index)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for depth, indices in sorted(by_depth.items()):
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start:start + args.batch_size]
                outputs = executor(stack_packets([packets[index] for index in batch_indices], "cuda"))
                transitions = [matrix.argmax(-1).tolist() for matrix in outputs["transition_matrices"]]
                final = outputs["assignment"].argmax(-1).tolist()
                query = outputs["query_logits"].argmax(-1).tolist()
                answers = outputs["answer_probabilities"].argmax(-1).tolist()
                entities = [value.argmax(-1).tolist() for value in outputs["entity_match_logits"]]
                amounts = [value.argmax(-1).tolist() for value in outputs["amount_logits"]]
                kinds = [value.tolist() for value in outputs["kind_predictions"]]
                for local, index in enumerate(batch_indices):
                    target = long_targets(rows[index])
                    records[index] = {
                        "id": rows[index]["id"],
                        "group": int(rows[index]["group"]),
                        "surface_type": rows[index]["surface_type"],
                        "depth": depth,
                        "answer_correct": int(answers[local]) == target["answer"],
                        "final_exact": tuple(final[local]) == target["final"],
                        "all_transitions_exact": all(
                            tuple(transitions[step][local]) == target["transitions"][step]
                            for step in range(depth)
                        ),
                        "query_correct": int(query[local]) == target["query"],
                        "entity_correct": [
                            int(entities[step][local]) == target["entity_locations"][step]
                            for step in range(depth)
                        ],
                        "amount_correct": [
                            int(amounts[step][local]) == target["amounts"][step]
                            for step in range(depth)
                        ],
                        "kind_correct": [
                            int(kinds[step][local]) == KIND_TO_ID[rows[index]["program"][step]["kind"]]
                            for step in range(depth)
                        ],
                        "kind_lexical_matched": [
                            bool(packets[index]["operations"][step]["kind_lexical_matched"])
                            for step in range(depth)
                        ],
                    }
    groups = collections.defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)
    result = {
        "schema": "r12_s3_lexical_confirmation_eval_v1",
        "identity_mode": args.identity_mode,
        "intervention": args.intervention,
        "base_sha256": sha256_file(args.base),
        "compiler_sha256": sha256_file(args.compiler),
        "compiler_adapter_sha256": compiler_metadata["final_adapter_sha256"],
        "executor_sha256": sha256_file(args.executor),
        "executor_state_sha256": metadata["final_executor_sha256"],
        "kind_lexicon_sha256": sha256_file(args.lexicon),
        "board_sha256": sha256_file(args.board),
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "evaluator_sha256": sha256_file(__file__),
        "rows": len(rows),
        "chunks": chunks,
        "overall": summarize(records),
        "by_depth": {
            str(depth): summarize([row for row in records if row["depth"] == depth])
            for depth in range(3, 9)
        },
        "by_surface": {
            surface: summarize([row for row in records if row["surface_type"] == surface])
            for surface in sorted({row["surface_type"] for row in records})
        },
        "group_summary": {
            "groups": len(groups),
            "all_four_answers_correct": sum(
                len(group) == 4 and all(row["answer_correct"] for row in group)
                for group in groups.values()
            ),
            "all_four_state_exact": sum(
                len(group) == 4 and all(row["final_exact"] for row in group)
                for group in groups.values()
            ),
        },
        "intervention_rows": intervention_rows,
        "fit_updates": 0,
        "confirmation_evaluation": 1,
        "old_confirmation_access": 0,
        "claim_boundary": (
            "One-shot known-atom source-deleted S3 confirmation with external schedule/halt; "
            "not unseen-phrase generalization or autonomous reasoning."
        ),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "identity_mode": args.identity_mode, "intervention": args.intervention,
        "overall": result["overall"], "out": str(Path(args.out).resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
