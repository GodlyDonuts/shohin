#!/usr/bin/env python3
"""Evaluate an atomic-trained S3 executor on public two-step composition."""

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
from model import GPTConfig
from referential_gather_delete_executor import execution_targets
from referential_literal_pointer_compiler import load_examples, make_batches, pad_batch, sha256_file
from train_referential_gather_delete_executor import load_frozen_compiler


def summarize(records):
    total = len(records)
    return {
        "rows": total,
        "answer_accuracy": sum(row["answer_correct"] for row in records) / total,
        "final_assignment_exact": sum(row["final_exact"] for row in records) / total,
        "all_transitions_exact": sum(row["all_transitions_exact"] for row in records) / total,
        "query_accuracy": sum(row["query_correct"] for row in records) / total,
        "entity_match_accuracy": sum(
            sum(row["entity_correct"]) for row in records
        ) / (2 * total),
        "amount_accuracy": sum(sum(row["amount_correct"]) for row in records) / (2 * total),
    }


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
    parser.add_argument("--identity-mode", choices=("mean", "ordered", "gold"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--closed-action", action="store_true")
    parser.add_argument("--kind-lexicon")
    parser.add_argument("--kind-decoder", choices=("mass", "pointer_anchor"), default="mass")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("S3 evaluation requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing S3 evaluation")
    report = json.load(open(args.report))
    if not report.get("all_gates_pass"):
        raise SystemExit("factorized report is not admitted")
    if report["artifacts"][args.split]["sha256"] != sha256_file(args.data):
        raise SystemExit("factorized report does not bind S3 evaluation data")
    bundle = torch.load(args.executor, map_location="cpu")
    metadata = bundle.get("executor", {})
    if metadata.get("protocol") not in {
        "r12_s3_categorical_permutation_executor_v1",
        "r12_s3_equivariant_permutation_executor_v1_1",
    }:
        raise SystemExit("invalid S3 executor protocol")
    if metadata.get("confirmation_access") != 0:
        raise SystemExit("S3 executor records confirmation access")
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
    examples = load_examples(
        args.data, tokenizer, args.split, cfg.seq_len, keep_evidence=True,
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
        raise SystemExit("S3 executor state mismatch")

    records = [None] * len(examples)
    batches = make_batches(examples, args.batch_size, seed=0, shuffle=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for indices in batches:
            selected, ids, valid = pad_batch(examples, indices, "cuda")
            compiler_outputs = compiler(ids, valid)
            packet = categorical_identity_packet(
                compiler_outputs, selected, ids, valid, mode=args.identity_mode,
            )
            if lexicon is not None:
                packet = apply_lexical_kind_override(
                    packet, compiler_outputs, ids, valid, lexicon,
                    decoder=args.kind_decoder,
                )
            outputs = executor(packet)
            transitions = [matrix.argmax(-1).tolist() for matrix in outputs["transition_matrices"]]
            final = outputs["assignment"].argmax(-1).tolist()
            query = outputs["query_logits"].argmax(-1).tolist()
            answers = outputs["answer_probabilities"].argmax(-1).tolist()
            entities = [logits.argmax(-1).tolist() for logits in outputs["entity_match_logits"]]
            amounts = [logits.argmax(-1).tolist() for logits in outputs["amount_logits"]]
            kinds = [prediction.tolist() for prediction in outputs.get("kind_predictions", ())]
            targets = [execution_targets(example) for example in selected]
            for local, global_index in enumerate(indices):
                target = targets[local]
                transition_exact = [
                    tuple(transitions[step][local]) == target.transition_sources[step]
                    for step in range(2)
                ]
                records[global_index] = {
                    "id": selected[local].row_id,
                    "group": selected[local].group,
                    "surface_type": selected[local].surface_type,
                    "answer_correct": int(answers[local]) == target.answer_identity,
                    "final_exact": tuple(final[local]) == target.final_identities,
                    "all_transitions_exact": all(transition_exact),
                    "query_correct": int(query[local]) == target.query_position,
                    "entity_correct": [
                        int(entities[step][local]) == target.entity_locations[step]
                        for step in range(2)
                    ],
                    "amount_correct": [
                        int(amounts[step][local]) == target.amounts[step]
                        for step in range(2)
                    ],
                    "kind_correct": [
                        int(kinds[step][local]) == selected[local].kind_targets[step]
                        for step in range(2)
                    ] if kinds else [],
                    "kind_lexical_matched": [
                        bool(packet["operations"][step]["kind_lexical_matched"][local])
                        for step in range(2)
                    ] if lexicon is not None else [],
                }
    groups = collections.defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)
    result = {
        "schema": "r12_s3_categorical_permutation_eval_v1",
        "split": args.split,
        "identity_mode": args.identity_mode,
        "action_protocol": "closed_s3_v1_2" if args.closed_action else "learned",
        "kind_protocol": (
            "training_lexicon_pointer_anchor_v1"
            if lexicon is not None and args.kind_decoder == "pointer_anchor"
            else "training_lexicon_v1" if lexicon is not None else "neural"
        ),
        "base_sha256": sha256_file(args.base),
        "compiler_sha256": sha256_file(args.compiler),
        "compiler_adapter_sha256": compiler_metadata["final_adapter_sha256"],
        "executor_sha256": sha256_file(args.executor),
        "executor_state_sha256": metadata["final_executor_sha256"],
        "data_sha256": sha256_file(args.data),
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "kind_lexicon_sha256": sha256_file(args.kind_lexicon) if lexicon is not None else None,
        "evaluator_sha256": sha256_file(__file__),
        "overall": summarize(records),
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
        "fit_updates": 0,
        "confirmation_access": 0,
        "records": records,
        "claim_boundary": (
            "Public two-step S3 component development. External schedule/halt; no "
            "confirmation, autonomous reasoning, or novelty claim."
        ),
    }
    if args.closed_action:
        result["overall"]["kind_accuracy"] = (
            sum(sum(row["kind_correct"]) for row in records) / (2 * len(records))
        )
    if lexicon is not None:
        result["overall"]["kind_lexical_coverage"] = (
            sum(sum(row["kind_lexical_matched"]) for row in records) / (2 * len(records))
        )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "identity_mode": args.identity_mode,
        "out": str(Path(args.out).resolve()),
        "overall": result["overall"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
