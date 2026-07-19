#!/usr/bin/env python3
"""Evaluate a frozen RGDE executor on fresh three-to-eight-step packet streams."""

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
    executor_state_hash,
    gather_source_deleted_packet,
    semantic_derangement_permutation,
)
from referential_literal_pointer_compiler import (
    compile_row,
    make_batches,
    pad_batch,
    sha256_file,
)
from train_referential_gather_delete_executor import load_frozen_compiler


def load_board(path):
    rows = []
    with open(path) as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    if not rows or any(row.get("split") != "confirmation_depth" for row in rows):
        raise ValueError("invalid depth-confirmation board")
    return rows


def long_targets(row):
    initial = tuple(row["initial_order"])
    state = [0, 1, 2]
    transitions = []
    entity_locations = []
    amounts = []
    for operation in row["program"]:
        identity = initial.index(operation["entity"])
        source = state.index(identity)
        amount = int(operation["amount"])
        destination = (
            max(0, source - amount)
            if operation["kind"] == "left" else
            min(2, source + amount)
        )
        next_state = list(state)
        next_state.insert(destination, next_state.pop(source))
        transitions.append(tuple(state.index(identity) for identity in next_state))
        entity_locations.append(source)
        amounts.append(amount - 1)
        state = next_state
    query = int(row["query"]["position"])
    return {
        "transitions": tuple(transitions),
        "entity_locations": tuple(entity_locations),
        "amounts": tuple(amounts),
        "query": query,
        "answer": state[query],
        "final": tuple(state),
    }


def extract_packet(packet, index):
    return {
        "initial_entities": packet["initial_entities"][index].float().cpu(),
        "operations": tuple({
            name: value[index].float().cpu()
            for name, value in operation.items()
        } for operation in packet["operations"]),
        "query": packet["query"][index].float().cpu(),
    }


def compile_packets(rows, tokenizer, compiler, cfg, oracle, device, batch_size):
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
    flat_packets = [None] * len(examples)
    batches = make_batches(examples, batch_size, seed=0, shuffle=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_number, indices in enumerate(batches, 1):
            selected, ids, valid = pad_batch(examples, indices, device)
            if ids.shape[1] > cfg.seq_len:
                raise ValueError("depth-confirmation chunk exceeds model sequence length")
            outputs = compiler(ids, valid)
            packet = gather_source_deleted_packet(
                outputs,
                selected,
                valid,
                oracle=oracle,
                packet_mode="lexical_sigmoid_span",
            )
            for local, global_index in enumerate(indices):
                flat_packets[global_index] = extract_packet(packet, local)
            if batch_number % 50 == 0:
                print("[rgde-depth] compiled {}/{} batches".format(
                    batch_number, len(batches),
                ), flush=True)
    row_packets = []
    for indices, counts in zip(mapping, active_counts):
        chunks = [flat_packets[index] for index in indices]
        operations = []
        for chunk, active in zip(chunks, counts):
            operations.extend(chunk["operations"][:active])
        row_packets.append({
            "initial_entities": chunks[0]["initial_entities"],
            "operations": tuple(operations),
            "query": chunks[-1]["query"],
        })
    if any(len(packet["operations"]) != int(row["depth"])
           for packet, row in zip(row_packets, rows)):
        raise AssertionError("packet stream depth mismatch")
    return row_packets, len(examples)


def stack_packets(packets, device):
    depth = len(packets[0]["operations"])
    if any(len(packet["operations"]) != depth for packet in packets):
        raise ValueError("stacked packet depths differ")
    return {
        "initial_entities": torch.stack([
            packet["initial_entities"] for packet in packets
        ]).to(device),
        "operations": tuple({
            name: torch.stack([
                packet["operations"][step][name] for packet in packets
            ]).to(device)
            for name in packets[0]["operations"][step]
        } for step in range(depth)),
        "query": torch.stack([packet["query"] for packet in packets]).to(device),
        "oracle": "stream",
        "packet_mode": "lexical_sigmoid_span",
    }


def summarize(records):
    total = len(records)
    return {
        "rows": total,
        "answer_accuracy": sum(row["answer_correct"] for row in records) / total,
        "final_assignment_exact": sum(row["final_exact"] for row in records) / total,
        "all_transitions_exact": sum(row["all_transitions_exact"] for row in records) / total,
        "query_accuracy": sum(row["query_correct"] for row in records) / total,
        "entity_match_accuracy": (
            sum(sum(row["entity_correct"]) for row in records)
            / sum(len(row["entity_correct"]) for row in records)
        ),
        "amount_accuracy": (
            sum(sum(row["amount_correct"]) for row in records)
            / sum(len(row["amount_correct"]) for row in records)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--executor", required=True)
    parser.add_argument("--board", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--packet-oracle", choices=("none", "full"), default="none")
    parser.add_argument("--intervention", choices=("none", "operations", "query"), default="none")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("RGDE depth confirmation requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing depth-confirmation result")
    report = json.load(open(args.report))
    if not report.get("all_gates_pass") or report.get("old_confirmation_access") != 0:
        raise SystemExit("depth-confirmation report is not admitted")
    if report["artifact"]["sha256"] != sha256_file(args.board):
        raise SystemExit("depth-confirmation report does not bind board")
    bundle = torch.load(args.executor, map_location="cpu")
    metadata = bundle.get("executor", {})
    if metadata.get("protocol") != "r12_referential_gather_delete_executor_stage_b_v1_1":
        raise SystemExit("depth confirmation requires RGDE v1.1")
    if metadata.get("arm") != "tied" or metadata.get("packet_oracle") != "none":
        raise SystemExit("depth confirmation requires tied predicted atomic executor")
    if metadata.get("confirmation_access") != 0:
        raise SystemExit("executor records prior confirmation access")

    device = "cuda"
    checkpoint, compiler, compiler_metadata = load_frozen_compiler(
        args.base, args.compiler, args.tokenizer, device,
    )
    cfg = GPTConfig(**checkpoint["cfg"])
    tokenizer = Tokenizer.from_file(args.tokenizer)
    rows = load_board(args.board)
    packets, chunks = compile_packets(
        rows, tokenizer, compiler, cfg, args.packet_oracle, device, args.batch_size,
    )
    executor = GatherDeletePermutationExecutor(
        identity_width=int(metadata["identity_width"]),
        context_width=int(metadata["context_width"]),
        width=int(metadata["executor_width"]),
        tied=True,
    ).to(device).eval()
    executor.load_state_dict(bundle["executor_state"], strict=True)
    if executor_state_hash(executor) != metadata["final_executor_sha256"]:
        raise SystemExit("depth executor state hash mismatch")

    source_for_row = list(range(len(rows)))
    if args.intervention != "none":
        by_depth = collections.defaultdict(list)
        for index, row in enumerate(rows):
            by_depth[int(row["depth"])].append(index)
        for depth, indices in by_depth.items():
            keys = (
                [canonical_program(rows[index]) for index in indices]
                if args.intervention == "operations" else
                [int(rows[index]["query"]["position"]) for index in indices]
            )
            local = semantic_derangement_permutation(keys).tolist()
            for destination, source in enumerate(local):
                source_for_row[indices[destination]] = indices[source]
            if any(source_for_row[index] == index for index in indices):
                raise AssertionError("depth {} intervention retained a row".format(depth))

    records = [None] * len(rows)
    by_depth = collections.defaultdict(list)
    for index, row in enumerate(rows):
        by_depth[int(row["depth"])].append(index)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for depth, depth_indices in sorted(by_depth.items()):
            for start in range(0, len(depth_indices), args.batch_size):
                indices = depth_indices[start:start + args.batch_size]
                selected_packets = []
                for index in indices:
                    packet = dict(packets[index])
                    source = packets[source_for_row[index]]
                    if args.intervention == "operations":
                        packet["operations"] = source["operations"]
                    elif args.intervention == "query":
                        packet["query"] = source["query"]
                    selected_packets.append(packet)
                outputs = executor(
                    stack_packets(selected_packets, device),
                    cell_indices=tuple(range(depth)),
                )
                transitions = [
                    logits.argmax(-1).tolist() for logits in outputs["transition_logits"]
                ]
                final = outputs["assignment"].argmax(-1).tolist()
                query_predictions = outputs["query_logits"].argmax(-1).tolist()
                answers = outputs["answer_probabilities"].argmax(-1).tolist()
                entities = [
                    logits.argmax(-1).tolist() for logits in outputs["entity_match_logits"]
                ]
                amounts = [
                    logits.argmax(-1).tolist() for logits in outputs["amount_logits"]
                ]
                for local, index in enumerate(indices):
                    target = long_targets(rows[index])
                    transition_exact = [
                        tuple(transitions[step][local]) == target["transitions"][step]
                        for step in range(depth)
                    ]
                    records[index] = {
                        "id": rows[index]["id"],
                        "group": rows[index]["group"],
                        "surface_type": rows[index]["surface_type"],
                        "depth": depth,
                        "answer_correct": int(answers[local]) == target["answer"],
                        "final_exact": tuple(final[local]) == target["final"],
                        "transition_exact": transition_exact,
                        "all_transitions_exact": all(transition_exact),
                        "query_correct": int(query_predictions[local]) == target["query"],
                        "entity_correct": [
                            int(entities[step][local]) == target["entity_locations"][step]
                            for step in range(depth)
                        ],
                        "amount_correct": [
                            int(amounts[step][local]) == target["amounts"][step]
                            for step in range(depth)
                        ],
                    }
    groups = collections.defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)
    result = {
        "schema": "r12_rgde_depth_confirmation_eval_v1",
        "base_sha256": sha256_file(args.base),
        "compiler_sha256": sha256_file(args.compiler),
        "executor_sha256": sha256_file(args.executor),
        "executor_state_sha256": metadata["final_executor_sha256"],
        "board_sha256": sha256_file(args.board),
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "evaluator_sha256": sha256_file(__file__),
        "packet_oracle": args.packet_oracle,
        "intervention": args.intervention,
        "intervention_rows": 0 if args.intervention == "none" else len(rows),
        "rows": len(rows),
        "chunks_compiled": chunks,
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
            "all_four_final_exact": sum(
                len(group) == 4 and all(row["final_exact"] for row in group)
                for group in groups.values()
            ),
        },
        "host_supplied_inference_fields": {
            "operation_count": True,
            "halt_after": True,
            "operation_semantics": False,
            "entity_bindings": False,
            "state_update": False,
            "query_answer": False,
        },
        "old_confirmation_access": 0,
        "records": records,
        "claim_boundary": (
            "Fresh packet-stream recurrent-depth confirmation with external schedule/halt. "
            "Not autonomous language reasoning or learned halting."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(Path(args.out).resolve()),
        "overall": result["overall"],
        "by_depth": result["by_depth"],
        "groups": result["group_summary"],
    }, sort_keys=True))


def canonical_program(row):
    return json.dumps(row["program"], sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()
