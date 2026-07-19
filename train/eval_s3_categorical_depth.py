#!/usr/bin/env python3
"""Evaluate the S3 categorical register on public depths three through eight."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from categorical_permutation_executor import (
    S3CategoricalPermutationExecutor,
    S3ClosedActionPermutationExecutor,
    S3EquivariantPermutationExecutor,
    categorical_identity_packet,
    apply_lexical_kind_override,
    module_state_hash,
)
from eval_rgde_depth_confirmation import long_targets, summarize
from model import GPTConfig
from referential_literal_pointer_compiler import compile_row, make_batches, pad_batch, sha256_file
from train_referential_gather_delete_executor import load_frozen_compiler


def load_board(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows or any(row.get("split") != "development_relational" for row in rows):
        raise ValueError("invalid S3 depth-development board")
    return rows


def extract_operation(operation, index):
    return {name: value[index].float().cpu() for name, value in operation.items()}


def compile_packets(rows, tokenizer, compiler, cfg, mode, device, batch_size, lexicon=None):
    examples = []
    mapping = []
    active_counts = []
    for row in rows:
        indices = []
        counts = []
        for chunk in row["chunks"]:
            indices.append(len(examples))
            counts.append(int(chunk["active_operations"]))
            examples.append(compile_row(chunk, tokenizer, keep_evidence=True))
        mapping.append(indices)
        active_counts.append(counts)
    flat = [None] * len(examples)
    batches = make_batches(examples, batch_size, seed=0, shuffle=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for indices in batches:
            selected, ids, valid = pad_batch(examples, indices, device)
            if ids.shape[1] > cfg.seq_len:
                raise ValueError("S3 depth chunk exceeds sequence length")
            outputs = compiler(ids, valid)
            packet = categorical_identity_packet(outputs, selected, ids, valid, mode=mode)
            if lexicon is not None:
                packet = apply_lexical_kind_override(
                    packet, outputs, ids, valid, lexicon,
                )
            for local, global_index in enumerate(indices):
                flat[global_index] = {
                    "operations": tuple(
                        extract_operation(operation, local) for operation in packet["operations"]
                    ),
                    "query": packet["query"][local].float().cpu(),
                }
    packets = []
    for indices, counts in zip(mapping, active_counts):
        chunks = [flat[index] for index in indices]
        operations = []
        for chunk, active in zip(chunks, counts):
            operations.extend(chunk["operations"][:active])
        packets.append({"operations": tuple(operations), "query": chunks[-1]["query"]})
    return packets, len(examples)


def stack_packets(packets, device):
    depth = len(packets[0]["operations"])
    return {
        "operations": tuple({
            name: torch.stack([
                packet["operations"][step][name] for packet in packets
            ]).to(device)
            for name in packets[0]["operations"][step]
        } for step in range(depth)),
        "query": torch.stack([packet["query"] for packet in packets]).to(device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--executor", required=True)
    parser.add_argument("--board", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--identity-mode", choices=("mean", "ordered", "gold"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--closed-action", action="store_true")
    parser.add_argument("--kind-lexicon")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("S3 depth evaluation requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing S3 depth evaluation")
    report = json.load(open(args.report))
    if not report.get("all_gates_pass") or report.get("confirmation_access") != 0:
        raise SystemExit("S3 depth-development report is not admitted")
    if report["artifact"]["sha256"] != sha256_file(args.board):
        raise SystemExit("S3 report does not bind depth board")
    bundle = torch.load(args.executor, map_location="cpu")
    metadata = bundle.get("executor", {})
    if metadata.get("protocol") not in {
        "r12_s3_categorical_permutation_executor_v1",
        "r12_s3_equivariant_permutation_executor_v1_1",
    }:
        raise SystemExit("invalid S3 depth executor protocol")
    checkpoint, compiler, compiler_metadata = load_frozen_compiler(
        args.base, args.compiler, args.tokenizer, "cuda",
    )
    cfg = GPTConfig(**checkpoint["cfg"])
    tokenizer = Tokenizer.from_file(args.tokenizer)
    lexicon = None
    if args.kind_lexicon:
        lexicon = json.load(open(args.kind_lexicon))
        if not lexicon.get("all_gates_pass") or lexicon.get("development_access") != 0:
            raise SystemExit("kind lexicon is not admitted")
    rows = load_board(args.board)
    packets, chunks = compile_packets(
        rows, tokenizer, compiler, cfg, args.identity_mode, "cuda", args.batch_size,
        lexicon=lexicon,
    )
    if args.closed_action:
        if metadata["protocol"] != "r12_s3_equivariant_permutation_executor_v1_1":
            raise SystemExit("closed action requires the frozen equivariant v1.1 state")
        executor_class = S3ClosedActionPermutationExecutor
    else:
        executor_class = (
            S3EquivariantPermutationExecutor
            if metadata["protocol"] == "r12_s3_equivariant_permutation_executor_v1_1"
            else S3CategoricalPermutationExecutor
        )
    executor = executor_class(
        identity_context_width=int(metadata["identity_context_width"]),
        context_width=int(metadata["context_width"]),
        width=int(metadata["executor_width"]),
    ).to("cuda").eval()
    executor.load_state_dict(bundle["executor_state"], strict=True)
    if module_state_hash(executor) != metadata["final_executor_sha256"]:
        raise SystemExit("S3 depth executor state mismatch")

    records = [None] * len(rows)
    by_depth = collections.defaultdict(list)
    for index, row in enumerate(rows):
        by_depth[int(row["depth"])].append(index)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for depth, indices in sorted(by_depth.items()):
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start:start + args.batch_size]
                outputs = executor(stack_packets([packets[index] for index in batch_indices], "cuda"))
                transitions = [
                    matrix.argmax(-1).tolist() for matrix in outputs["transition_matrices"]
                ]
                final = outputs["assignment"].argmax(-1).tolist()
                query = outputs["query_logits"].argmax(-1).tolist()
                answers = outputs["answer_probabilities"].argmax(-1).tolist()
                entities = [
                    logits.argmax(-1).tolist() for logits in outputs["entity_match_logits"]
                ]
                amounts = [logits.argmax(-1).tolist() for logits in outputs["amount_logits"]]
                kinds = [prediction.tolist() for prediction in outputs.get("kind_predictions", ())]
                for local, index in enumerate(batch_indices):
                    target = long_targets(rows[index])
                    transition_exact = [
                        tuple(transitions[step][local]) == target["transitions"][step]
                        for step in range(depth)
                    ]
                    records[index] = {
                        "id": rows[index]["id"],
                        "surface_type": rows[index]["surface_type"],
                        "depth": depth,
                        "answer_correct": int(answers[local]) == target["answer"],
                        "final_exact": tuple(final[local]) == target["final"],
                        "all_transitions_exact": all(transition_exact),
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
                            int(kinds[step][local]) == (0 if rows[index]["program"][step]["kind"] == "left" else 1)
                            for step in range(depth)
                        ] if kinds else [],
                        "kind_lexical_matched": [
                            bool(packets[index]["operations"][step]["kind_lexical_matched"])
                            for step in range(depth)
                        ] if lexicon is not None else [],
                    }
    result = {
        "schema": "r12_s3_categorical_depth_eval_v1",
        "identity_mode": args.identity_mode,
        "action_protocol": "closed_s3_v1_2" if args.closed_action else "learned",
        "kind_protocol": "training_lexicon_v1" if lexicon is not None else "neural",
        "base_sha256": sha256_file(args.base),
        "compiler_sha256": sha256_file(args.compiler),
        "compiler_adapter_sha256": compiler_metadata["final_adapter_sha256"],
        "executor_sha256": sha256_file(args.executor),
        "executor_state_sha256": metadata["final_executor_sha256"],
        "board_sha256": sha256_file(args.board),
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "kind_lexicon_sha256": sha256_file(args.kind_lexicon) if lexicon is not None else None,
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
        "fit_updates": 0,
        "confirmation_access": 0,
        "records": records,
        "claim_boundary": (
            "Public depth-3--8 S3 component development with external schedule/halt. "
            "No confirmation, autonomous reasoning, or novelty claim."
        ),
    }
    if args.closed_action:
        result["overall"]["kind_accuracy"] = (
            sum(sum(row["kind_correct"]) for row in records)
            / sum(len(row["kind_correct"]) for row in records)
        )
        for depth in range(3, 9):
            selected = [row for row in records if row["depth"] == depth]
            result["by_depth"][str(depth)]["kind_accuracy"] = (
                sum(sum(row["kind_correct"]) for row in selected)
                / sum(len(row["kind_correct"]) for row in selected)
            )
    if lexicon is not None:
        result["overall"]["kind_lexical_coverage"] = (
            sum(sum(row["kind_lexical_matched"]) for row in records)
            / sum(len(row["kind_lexical_matched"]) for row in records)
        )
        for depth in range(3, 9):
            selected = [row for row in records if row["depth"] == depth]
            result["by_depth"][str(depth)]["kind_lexical_coverage"] = (
                sum(sum(row["kind_lexical_matched"]) for row in selected)
                / sum(len(row["kind_lexical_matched"]) for row in selected)
            )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "identity_mode": args.identity_mode,
        "out": str(Path(args.out).resolve()),
        "overall": result["overall"],
        "by_depth": result["by_depth"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
