#!/usr/bin/env python3
"""Audit whether the RGDE depth board encodes its active-operation schedule.

The depth board pads an odd final chunk with a normal operation and stores the
true active count only in metadata.  This audit asks the narrower, invariant
question needed for a semantic halt claim: is the active/padding label a
function of the second operation's semantics relative to the initial roster?
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from semantic_compiler_falsifier import (
    Operation,
    apply_program_pop_insert,
    canonical_json,
    sha256_file,
)


def operation_from_dict(value):
    return Operation(str(value["kind"]), str(value["entity"]), int(value["amount"]))


def relative_signature(operation, initial):
    return (
        operation.direction,
        tuple(initial).index(operation.entity),
        int(operation.amount),
    )


def execute(initial, program, query_position):
    terminal = apply_program_pop_insert(tuple(initial), tuple(program))
    return terminal, terminal[int(query_position)]


def schedule_program(row, policy):
    operations = []
    filler_signature = ("left", 0, 1)
    for chunk in row["chunks"]:
        rendered = [operation_from_dict(value) for value in chunk["program"]]
        if len(rendered) != 2:
            raise ValueError("every depth-board chunk must render two operations")
        operations.append(rendered[0])
        if policy == "oracle":
            if int(chunk["active_operations"]) == 2:
                operations.append(rendered[1])
        elif policy == "keep_all":
            operations.append(rendered[1])
        elif policy == "drop_filler_signature":
            if relative_signature(rendered[1], row["initial_order"]) != filler_signature:
                operations.append(rendered[1])
        else:
            raise ValueError("unknown schedule policy {}".format(policy))
    return tuple(operations)


def evaluate_policy(rows, policy):
    exact_programs = 0
    exact_states = 0
    correct_answers = 0
    for row in rows:
        predicted = schedule_program(row, policy)
        gold = tuple(operation_from_dict(value) for value in row["program"])
        exact_programs += int(predicted == gold)
        predicted_state, predicted_answer = execute(
            row["initial_order"], predicted, row["query"]["position"],
        )
        gold_state, gold_answer = execute(
            row["initial_order"], gold, row["query"]["position"],
        )
        exact_states += int(predicted_state == gold_state)
        correct_answers += int(predicted_answer == gold_answer)
    total = len(rows)
    return {
        "rows": total,
        "exact_programs": exact_programs,
        "exact_program_accuracy": exact_programs / total,
        "exact_states": exact_states,
        "exact_state_accuracy": exact_states / total,
        "correct_answers": correct_answers,
        "answer_accuracy": correct_answers / total,
    }


def audit(rows):
    labels_by_signature = collections.defaultdict(collections.Counter)
    active_counts = collections.Counter()
    filler_signature = ("left", 0, 1)
    for row in rows:
        for chunk in row["chunks"]:
            active = int(chunk["active_operations"])
            if active not in {1, 2}:
                raise ValueError("invalid active-operation count")
            if len(chunk["program"]) != 2:
                raise ValueError("chunk does not contain exactly two rendered operations")
            signature = relative_signature(
                operation_from_dict(chunk["program"][1]), row["initial_order"],
            )
            labels_by_signature[signature][active] += 1
            active_counts[active] += 1

    conflicting = {
        canonical_json(signature): dict(sorted(labels.items()))
        for signature, labels in sorted(labels_by_signature.items())
        if len(labels) > 1
    }
    minimum_errors = sum(
        sum(labels.values()) - max(labels.values())
        for labels in labels_by_signature.values()
    )
    filler_labels = labels_by_signature[filler_signature]
    return {
        "rows": len(rows),
        "chunks": sum(active_counts.values()),
        "active_count_histogram": {str(key): value for key, value in sorted(active_counts.items())},
        "semantic_signature_count": len(labels_by_signature),
        "filler_signature": list(filler_signature),
        "filler_signature_label_histogram": {
            str(key): value for key, value in sorted(filler_labels.items())
        },
        "conflicting_signatures": conflicting,
        "conflicting_signature_count": len(conflicting),
        "minimum_equivariant_signature_classifier_errors": minimum_errors,
        "minimum_equivariant_signature_classifier_error_rate": (
            minimum_errors / max(1, sum(active_counts.values()))
        ),
        "policies": {
            policy: evaluate_policy(rows, policy)
            for policy in ("oracle", "keep_all", "drop_filler_signature")
        },
        "all_gates_pass": bool(
            filler_labels[1] > 0
            and filler_labels[2] > 0
            and conflicting
            and minimum_errors > 0
        ),
        "claim_boundary": (
            "The active/padding label is not a function of the second operation's "
            "equivariant semantic signature. This does not prove arbitrary text "
            "memorization impossible; it rejects a semantic learned-halt claim on "
            "this corpus and requires a self-delimiting replacement."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing report {}".format(output))
    rows = [json.loads(line) for line in open(args.board) if line.strip()]
    if not rows:
        raise SystemExit("empty board")
    if any(row.get("split") != "development_relational" for row in rows):
        raise SystemExit("unexpected board split")
    report = {
        "schema": "r12_s3_schedule_identifiability_audit_v1",
        "board": {"path": str(Path(args.board)), "sha256": sha256_file(args.board)},
        **audit(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit("schedule identifiability audit did not reproduce the conflict")


if __name__ == "__main__":
    main()
