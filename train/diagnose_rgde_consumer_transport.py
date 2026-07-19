#!/usr/bin/env python3
"""No-fit matched diagnosis of RGDE identity transport into the frozen executor."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_rgde_depth_confirmation import extract_packet, long_targets, stack_packets, summarize
from model import GPTConfig
from probe_rgde_relational_identity import lexical_mean_scores, ordered_sequence_scores, role_weights
from referential_gather_delete_executor import (
    GatherDeletePermutationExecutor,
    executor_state_hash,
    gather_source_deleted_packet,
)
from referential_literal_pointer_compiler import compile_row, make_batches, pad_batch, sha256_file
from train_referential_gather_delete_executor import load_frozen_compiler


ARMS = ("current", "mean_rebound", "ordered_rebound", "gold_rebound")


def load_board(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows or any(row.get("split") != "development_relational" for row in rows):
        raise ValueError("invalid relational-development board")
    return rows


def rebind_packet(packet, identities):
    if len(identities) != len(packet["operations"]):
        raise ValueError("one identity is required per operation")
    initial = packet["initial_entities"]
    operations = []
    for operation, identity in zip(packet["operations"], identities):
        if int(identity) not in (0, 1, 2):
            raise ValueError("rebound identity is outside the three-slot state")
        rebound = dict(operation)
        rebound["entity"] = initial[int(identity)].clone()
        operations.append(rebound)
    return {
        "initial_entities": initial,
        "operations": tuple(operations),
        "query": packet["query"],
    }


def compile_variants(rows, tokenizer, compiler, cfg, device, batch_size):
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
    flat_identities = [None] * len(examples)
    batches = make_batches(examples, batch_size, seed=0, shuffle=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_number, indices in enumerate(batches, 1):
            selected, ids, valid = pad_batch(examples, indices, device)
            if ids.shape[1] > cfg.seq_len:
                raise ValueError("consumer-diagnostic chunk exceeds sequence length")
            outputs = compiler(ids, valid)
            packet = gather_source_deleted_packet(
                outputs, selected, valid, oracle="none", packet_mode="lexical_sigmoid_span",
            )
            lexical = compiler.model.tok(ids).detach()
            pointer_logits = outputs["pointer_logits"]
            weights = {
                label: role_weights(pointer_logits[label], valid, 0.5)
                for label in (
                    "intro.entity0", "intro.entity1", "intro.entity2",
                    "op0.entity", "op1.entity",
                )
            }
            mean_weights = {
                label: role_weights(pointer_logits[label], valid, 1.0)
                for label in (
                    "intro.entity0", "intro.entity1", "intro.entity2",
                    "op0.entity", "op1.entity",
                )
            }
            score_maps = []
            for operation_index in range(2):
                label = "op{}.entity".format(operation_index)
                score_maps.append({
                    "mean_rebound": lexical_mean_scores(
                        lexical,
                        [mean_weights["intro.entity{}".format(i)] for i in range(3)],
                        mean_weights[label],
                    ),
                    "ordered_rebound": ordered_sequence_scores(
                        ids,
                        [weights["intro.entity{}".format(i)] for i in range(3)],
                        weights[label],
                    ),
                })
            for local, global_index in enumerate(indices):
                flat_packets[global_index] = extract_packet(packet, local)
                identity_sets = {"mean_rebound": [], "ordered_rebound": [], "gold_rebound": []}
                for operation_index in range(2):
                    identity_sets["mean_rebound"].append(
                        int(score_maps[operation_index]["mean_rebound"][local].argmax())
                    )
                    identity_sets["ordered_rebound"].append(
                        int(score_maps[operation_index]["ordered_rebound"][local].argmax())
                    )
                    identity_sets["gold_rebound"].append(
                        selected[local].initial_order.index(
                            selected[local].program[operation_index][1],
                        )
                    )
                flat_identities[global_index] = identity_sets
            if batch_number % 25 == 0:
                print("[rgde-consumer] compiled {}/{} batches".format(
                    batch_number, len(batches),
                ), flush=True)

    packets = {arm: [] for arm in ARMS}
    identity_hits = {arm: [0, 0] for arm in ARMS if arm != "current"}
    for row, indices, counts in zip(rows, mapping, active_counts):
        chunks = [flat_packets[index] for index in indices]
        current_operations = []
        identities = {arm: [] for arm in ARMS if arm != "current"}
        operation_cursor = 0
        for chunk_index, (chunk, active) in enumerate(zip(chunks, counts)):
            current_operations.extend(chunk["operations"][:active])
            for local_operation in range(active):
                operation = row["program"][operation_cursor]
                target = tuple(row["initial_order"]).index(operation["entity"])
                for arm in identities:
                    prediction = flat_identities[indices[chunk_index]][arm][local_operation]
                    identities[arm].append(prediction)
                    identity_hits[arm][0] += int(prediction == target)
                    identity_hits[arm][1] += 1
                operation_cursor += 1
        current = {
            "initial_entities": chunks[0]["initial_entities"],
            "operations": tuple(current_operations),
            "query": chunks[-1]["query"],
        }
        packets["current"].append(current)
        for arm in identities:
            packets[arm].append(rebind_packet(current, identities[arm]))
    return packets, len(examples), identity_hits


def evaluate_arm(rows, packets, executor, device, batch_size):
    records = [None] * len(rows)
    by_depth = collections.defaultdict(list)
    for index, row in enumerate(rows):
        by_depth[int(row["depth"])].append(index)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for depth, indices in sorted(by_depth.items()):
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                outputs = executor(
                    stack_packets([packets[index] for index in batch_indices], device),
                    cell_indices=tuple(range(depth)),
                )
                transitions = [logits.argmax(-1).tolist() for logits in outputs["transition_logits"]]
                final = outputs["assignment"].argmax(-1).tolist()
                query = outputs["query_logits"].argmax(-1).tolist()
                answers = outputs["answer_probabilities"].argmax(-1).tolist()
                entities = [logits.argmax(-1).tolist() for logits in outputs["entity_match_logits"]]
                amounts = [logits.argmax(-1).tolist() for logits in outputs["amount_logits"]]
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
                    }
    return {
        "overall": summarize(records),
        "by_depth": {
            str(depth): summarize([record for record in records if record["depth"] == depth])
            for depth in range(3, 9)
        },
        "by_surface": {
            surface: summarize([
                record for record in records if record["surface_type"] == surface
            ]) for surface in sorted({row["surface_type"] for row in rows})
        },
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
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("RGDE consumer transport diagnostic requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing consumer diagnostic output")
    report = json.load(open(args.report))
    if not report.get("all_gates_pass") or report.get("confirmation_access") != 0:
        raise SystemExit("relational-development report is not admitted")
    if report["artifact"]["sha256"] != sha256_file(args.board):
        raise SystemExit("report does not bind consumer diagnostic board")

    bundle = torch.load(args.executor, map_location="cpu")
    executor_metadata = bundle.get("executor", {})
    if executor_metadata.get("protocol") != "r12_referential_gather_delete_executor_stage_b_v1_1":
        raise SystemExit("consumer diagnostic requires RGDE v1.1")
    if executor_metadata.get("arm") != "tied" or executor_metadata.get("confirmation_access") != 0:
        raise SystemExit("consumer diagnostic requires the clean tied executor")
    checkpoint, compiler, compiler_metadata = load_frozen_compiler(
        args.base, args.compiler, args.tokenizer, "cuda",
    )
    cfg = GPTConfig(**checkpoint["cfg"])
    tokenizer = Tokenizer.from_file(args.tokenizer)
    rows = load_board(args.board)
    packets, chunks, identity_hits = compile_variants(
        rows, tokenizer, compiler, cfg, "cuda", args.batch_size,
    )
    executor = GatherDeletePermutationExecutor(
        identity_width=int(executor_metadata["identity_width"]),
        context_width=int(executor_metadata["context_width"]),
        width=int(executor_metadata["executor_width"]),
        tied=True,
    ).to("cuda").eval()
    executor.load_state_dict(bundle["executor_state"], strict=True)
    if executor_state_hash(executor) != executor_metadata["final_executor_sha256"]:
        raise SystemExit("consumer diagnostic executor state mismatch")

    arms = {
        arm: evaluate_arm(rows, packets[arm], executor, "cuda", args.batch_size)
        for arm in ARMS
    }
    current = arms["current"]["overall"]
    mean = arms["mean_rebound"]["overall"]
    ordered = arms["ordered_rebound"]["overall"]
    gold = arms["gold_rebound"]["overall"]
    observations = {
        "public_board_reproduces_match_loss": current["entity_match_accuracy"] < 0.90,
        "mean_rebinding_recovers_10pp_answers": (
            mean["answer_accuracy"] - current["answer_accuracy"] >= 0.10
        ),
        "mean_rebinding_recovers_10pp_state": (
            mean["final_assignment_exact"] - current["final_assignment_exact"] >= 0.10
        ),
        "ordered_adds_1pp_answers_over_mean": (
            ordered["answer_accuracy"] - mean["answer_accuracy"] >= 0.01
        ),
        "gold_answer_ceiling_at_least_99pct": gold["answer_accuracy"] >= 0.99,
        "gold_state_ceiling_at_least_99pct": gold["final_assignment_exact"] >= 0.99,
    }
    if (
        observations["public_board_reproduces_match_loss"]
        and observations["mean_rebinding_recovers_10pp_answers"]
        and observations["mean_rebinding_recovers_10pp_state"]
    ):
        diagnosis = "localize_transport_loss_to_learned_consumer_matcher"
    elif current["answer_accuracy"] >= 0.95 and current["final_assignment_exact"] >= 0.95:
        diagnosis = "public_board_does_not_reproduce_end_to_end_failure"
    else:
        diagnosis = "transport_failure_not_localized"
    result = {
        "schema": "r12_rgde_consumer_transport_diagnostic_v1",
        "diagnosis": diagnosis,
        "base_sha256": sha256_file(args.base),
        "compiler_sha256": sha256_file(args.compiler),
        "compiler_adapter_sha256": compiler_metadata["final_adapter_sha256"],
        "executor_sha256": sha256_file(args.executor),
        "executor_state_sha256": executor_metadata["final_executor_sha256"],
        "board_sha256": sha256_file(args.board),
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "diagnostic_sha256": sha256_file(__file__),
        "rows": len(rows),
        "chunks": chunks,
        "arms": arms,
        "identity_selection": {
            arm: {
                "correct": hits[0],
                "total": hits[1],
                "accuracy": hits[0] / hits[1],
            } for arm, hits in identity_hits.items()
        },
        "observations": observations,
        "fit_updates": 0,
        "confirmation_access": 0,
        "claim_boundary": (
            "No-fit public matched-consumer diagnostic only. Failed carriers are not "
            "promoted; no confirmation, autonomous reasoning, halt, or novelty claim."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "diagnosis": diagnosis,
        "observations": observations,
        "out": str(Path(args.out).resolve()),
        "scores": {arm: values["overall"] for arm, values in arms.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
