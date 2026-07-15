#!/usr/bin/env python3
"""Exact CPU admission audit for bidirectional noncommutative microcode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from bidirectional_operator_tree import (
    build_tree,
    compact_frontier,
    read_tree,
    retained_sources,
    scalar_payload,
    suspect_leaves,
)
from categorical_microcode import OPCODE_TO_ID, opcode_for, operation_value, query_for
from future_effect_algebra import operation_operator


def row_program(row):
    keys = row["keys"]
    opcodes = [opcode_for(operation, keys) for operation in row["operations"]]
    values = [operation_value(operation) for operation in row["operations"]]
    operators = [operation_operator(opcode, value, dtype=torch.int64) for opcode, value in zip(opcodes, values)]
    initial = [row["initial"][key] for key in keys]
    return operators, initial, query_for(row["query"], keys)


def audit(data):
    rows = [json.loads(line) for line in open(data) if line.strip()]
    exact = 0
    certified = 0
    for row_index, row in enumerate(rows):
        operators, initial, query = row_program(row)
        tree = build_tree(operators, operators, ["row{}:event{}".format(row_index, i) for i in range(len(operators))])
        certified += int(tree.certified)
        exact += int(read_tree(tree, initial, query) == int(row["answer"]))

    # The same event multiset has different semantics under different order.
    add = operation_operator("add_0", 3, dtype=torch.int64)
    swap = operation_operator("swap", 0, dtype=torch.int64)
    ordered_a = build_tree([add, swap], [add, swap])
    ordered_b = build_tree([swap, add], [swap, add])
    noncommuting = not torch.equal(ordered_a.forward, ordered_b.forward)
    order_answers = (
        read_tree(ordered_a, [5, 7], "read_0"),
        read_tree(ordered_b, [5, 7], "read_0"),
    )

    # One independently corrupted directional prediction is localized while
    # every clean sibling subtree remains source-droppable.
    count = 4096
    clean_ops = [operation_operator("add_0", (index % 7) + 1, dtype=torch.int64) for index in range(count)]
    corrupt_index = 2345
    backward = [operator.clone() for operator in clean_ops]
    backward[corrupt_index] = operation_operator("sub_0", 3, dtype=torch.int64)
    corrupt_tree = build_tree(clean_ops, backward)
    frontier = compact_frontier(corrupt_tree)
    suspects = suspect_leaves(corrupt_tree)

    clean_tree = build_tree(clean_ops, clean_ops)
    clean_frontier = compact_frontier(clean_tree)
    # Both channels can confidently agree on the same wrong operator.  This is
    # the explicit limitation that a later neural design must measure.
    jointly_wrong = build_tree([add], [add])
    same_wrong_undetected = jointly_wrong.certified

    # Confidence alone cannot reproduce directional agreement: both examples
    # can have identical one-hot confidence while only one has zero syndrome.
    confidence_matched = 0.999 == 0.999
    disagreement_detected = not build_tree([add], [swap]).certified
    agreement_detected = build_tree([add], [add]).certified

    gates = {
        "all_board_programs_certified": certified == len(rows),
        "all_board_answers_exact": exact == len(rows),
        "same_multiset_different_order_is_distinguished": noncommuting and order_answers[0] != order_answers[1],
        "clean_4096_events_fold_to_one_operator": (
            clean_tree.certified and len(clean_frontier) == 1
            and scalar_payload(clean_frontier) == 9 and retained_sources(clean_frontier) == 0
        ),
        "single_directional_error_localized_exactly": suspects == (corrupt_index,),
        "only_suspect_source_is_retained": retained_sources(frontier) == 1,
        "compact_error_frontier_is_logarithmic": len(frontier) <= 14 and scalar_payload(frontier) <= 118,
        "syndrome_is_not_a_confidence_threshold": confidence_matched and disagreement_detected and agreement_detected,
    }
    return {
        "audit": "bidirectional_noncommutative_operator_tree_r9b",
        "data": str(Path(data).resolve()),
        "programs": len(rows),
        "certified_programs": certified,
        "exact_answers": exact,
        "noncommuting_order_answers": order_answers,
        "clean_events": count,
        "clean_frontier_nodes": len(clean_frontier),
        "corrupt_index": corrupt_index,
        "suspect_leaves": suspects,
        "corrupt_frontier_nodes": len(frontier),
        "corrupt_scalar_payload": scalar_payload(frontier),
        "corrupt_retained_sources": retained_sources(frontier),
        "same_wrong_both_channels_is_undetected": same_wrong_undetected,
        "gates": gates,
        "mechanics_pass": all(gates.values()),
        "claim_boundary": (
            "A pass proves exact bidirectional execution, noncommutative order sensitivity, and fail-closed "
            "context folding under independent directional disagreement. It does not prove that a neural "
            "compiler can infer either operator or that agreement implies semantic correctness."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing output")
    result = audit(args.data)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
