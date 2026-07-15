#!/usr/bin/env python3
"""Exact mechanics gates for the R10 Version-Space Product Tree."""

import torch

from categorical_microcode import OPCODES, QUERIES
from future_effect_algebra import operation_operator, query_operator
from version_space_product_tree import (
    ExactAffineTransform,
    build_tree,
    chronological_compose,
    compact_frontier,
    compose_chunks,
    leaf_node,
    merge_nodes,
    opcode_candidate,
    operation_transform,
    query_agreement,
    query_set_agreement,
    query_row,
    read_tree,
    refine_leaf,
    retained_sources,
    transform_candidate,
    version_space_size,
)


def candidate_signature(node):
    return tuple(
        (candidate.transform, candidate.support_commitment)
        for candidate in node.candidates
    )


def transform_signature(node):
    return tuple(candidate.transform for candidate in node.candidates)


def test_exact_operator_parity_with_established_algebra():
    for opcode in OPCODES:
        values = (0,) if opcode.startswith(("merge_", "swap")) else (0, 1, 17)
        for value in values:
            exact = torch.tensor(operation_transform(opcode, value).rows, dtype=torch.float64)
            assert torch.equal(exact, operation_operator(opcode, value))
    for query in QUERIES:
        exact = torch.tensor((*query_row(query), 0), dtype=torch.float64)
        assert torch.equal(exact, query_operator(query))


def test_noncommutative_chronological_order():
    add = opcode_candidate("add_0", 4)
    swap = opcode_candidate("swap")
    add_then_swap = build_tree(((add,), (swap,)), ("add", "swap"), cap=8)
    swap_then_add = build_tree(((swap,), (add,)), ("swap", "add"), cap=8)

    assert add_then_swap.unique_transform == operation_transform("add_0", 4).followed_by(
        operation_transform("swap")
    )
    assert add_then_swap.unique_transform != swap_then_add.unique_transform
    assert read_tree(add_then_swap, (3, 8), "read_1") == 7
    assert read_tree(swap_then_add, (3, 8), "read_0") == 12
    assert retained_sources(add_then_swap) == 0
    assert version_space_size(add_then_swap) == 1


def test_distinct_opcode_paths_collapse_to_one_complete_transform():
    earlier = leaf_node(
        0,
        (opcode_candidate("add_0", 1), opcode_candidate("swap")),
        "earlier",
        cap=8,
    )
    later = leaf_node(
        1,
        (opcode_candidate("sub_0", 1), opcode_candidate("swap")),
        "later",
        cap=8,
    )
    product = merge_nodes(earlier, later)

    # Four opcode paths produce only three complete effects: add/sub and
    # swap/swap both denote identity. The commitment binds both derivations.
    assert version_space_size(product) == 3
    identities = [
        candidate for candidate in product.candidates
        if candidate.transform == ExactAffineTransform.identity()
    ]
    assert len(identities) == 1
    assert len(identities[0].support_commitment) == 64
    swap_only = build_tree(
        ((opcode_candidate("swap"),), (opcode_candidate("swap"),)),
        cap=8,
    )
    assert identities[0].support_commitment != swap_only.candidates[0].support_commitment
    assert retained_sources(product) == 2


def test_query_agreement_is_weaker_than_source_drop_certification():
    ambiguous = leaf_node(
        0,
        (opcode_candidate("add_0", 5), opcode_candidate("add_1", 5)),
        "ambiguous event",
        cap=8,
    )
    agreed = query_agreement(ambiguous, (2, 7), "sum")

    assert agreed.complete and agreed.query_agrees
    assert agreed.answer == 14 and agreed.answers == (14,)
    assert not agreed.source_droppable
    assert not ambiguous.source_droppable
    assert version_space_size(ambiguous) == 2
    assert retained_sources(ambiguous) == 1
    assert read_tree(ambiguous, (2, 7), "sum") == 14

    disagreed = query_agreement(ambiguous, (2, 7), "read_0")
    assert disagreed.complete and not disagreed.query_agrees
    assert disagreed.answers == (2, 7) and disagreed.answer is None
    query_set = query_set_agreement(ambiguous, (2, 7), ("sum", "read_0"))
    assert query_set.complete and not query_set.query_agrees
    assert query_set.answers == (2, 7, 14)


def test_overflow_is_sticky_but_preserves_exact_factorized_siblings():
    overflowed = leaf_node(
        0,
        tuple(opcode_candidate("add_0", value) for value in (1, 2, 3)),
        "overflow source",
        cap=2,
    )
    assert overflowed.overflow
    assert version_space_size(overflowed) is None
    assert overflowed.version_space_lower_bound == 3
    assert not overflowed.source_droppable
    assert retained_sources(overflowed) == 1

    query = query_agreement(overflowed, (0, 0), "read_1")
    assert not query.complete and not query.query_agrees and query.answer is None

    unique = leaf_node(1, (opcode_candidate("swap"),), "otherwise unique", cap=2)
    parent = merge_nodes(overflowed, unique)
    assert parent.overflow and not parent.source_droppable
    assert version_space_size(parent) is None
    assert retained_sources(parent) == 1
    frontier = compact_frontier(parent)
    assert [item["kind"] for item in frontier] == ["source", "transform"]
    assert frontier[0]["source"] == "overflow source"
    assert frontier[1]["retrieval_reference"]["start"] == 1
    assert frontier[1]["retrieval_reference"]["end"] == 2
    assert len(frontier[1]["retrieval_reference"]["node_commitment"]) == 64


def test_chunk_composition_is_associative():
    first = leaf_node(
        0,
        (opcode_candidate("add_0", 1), opcode_candidate("swap")),
        "first",
        cap=64,
    )
    second = leaf_node(
        1,
        (opcode_candidate("sub_0", 1), opcode_candidate("merge_0_1")),
        "second",
        cap=64,
    )
    third = leaf_node(
        2,
        (opcode_candidate("add_1", 2), opcode_candidate("swap")),
        "third",
        cap=64,
    )

    left_grouped = merge_nodes(merge_nodes(first, second), third)
    right_grouped = merge_nodes(first, merge_nodes(second, third))
    balanced = compose_chunks((first, second, third))
    assert not left_grouped.overflow and not right_grouped.overflow
    assert transform_signature(left_grouped) == transform_signature(right_grouped)
    assert transform_signature(left_grouped) == transform_signature(balanced)
    assert left_grouped.retained_source_indices == right_grouped.retained_source_indices
    assert left_grouped.version_space_size == right_grouped.version_space_size


def test_representative_long_exact_histories():
    events = 4096
    specifications = [
        ("merge_0_1", 0) if index % 2 == 0 else ("merge_1_0", 0)
        for index in range(events)
    ]
    candidate_sets = tuple((opcode_candidate(*specification),) for specification in specifications)
    full = build_tree(candidate_sets, cap=8)
    expected = chronological_compose(
        operation_transform(*specification) for specification in specifications
    )

    assert full.events == events
    assert full.source_droppable and not full.overflow
    assert full.unique_transform == expected
    assert version_space_size(full) == 1
    assert retained_sources(full) == 0
    assert max(abs(value) for value in expected.flat) > 2 ** 63
    assert read_tree(full, (1, 1), "sum") == expected.answer((1, 1), "sum")
    frontier = compact_frontier(full)
    assert len(frontier) == 1
    assert "witness" not in frontier[0]
    assert frontier[0]["retrieval_reference"]["start"] == 0
    assert frontier[0]["retrieval_reference"]["end"] == events
    assert len(frontier[0]["support_commitment"]) == 64
    assert len(frontier[0]["node_commitment"]) == 64
    assert "source_leaves" not in full.__dict__
    assert "retained_source_indices" not in full.__dict__

    chunks = []
    width = 257
    for start in range(0, events, width):
        stop = min(start + width, events)
        chunks.append(build_tree(candidate_sets[start:stop], start=start, cap=8))
    chunked = compose_chunks(chunks)
    assert chunked.unique_transform == full.unique_transform
    assert len(chunked.support_commitment) == 64
    assert len(full.support_commitment) == 64
    assert retained_sources(chunked) == 0


def test_monotone_refinement_retains_alternate_derivations():
    identity = transform_candidate(ExactAffineTransform.identity(), "identity")
    swap = opcode_candidate("swap")
    tree_a = build_tree(((identity, swap), (identity, swap)), cap=8)
    tree_b = build_tree(((identity,), (identity, swap)), cap=8)
    assert {candidate.transform for candidate in tree_a.candidates} == {
        candidate.transform for candidate in tree_b.candidates
    }

    refined_a = refine_leaf(tree_a, 1, (identity,))
    refined_b = refine_leaf(tree_b, 1, (identity,))
    assert {candidate.transform for candidate in refined_a.candidates} == {
        ExactAffineTransform.identity(), operation_transform("swap"),
    }
    assert refined_b.unique_transform == ExactAffineTransform.identity()
    rebuilt_a = build_tree(((identity, swap), (identity,)), cap=8)
    rebuilt_b = build_tree(((identity,), (identity,)), cap=8)
    assert candidate_signature(refined_a) == candidate_signature(rebuilt_a)
    assert candidate_signature(refined_b) == candidate_signature(rebuilt_b)

    try:
        refine_leaf(refined_b, 0, (identity, swap))
    except ValueError:
        pass
    else:
        raise AssertionError("candidate-expanding evidence was accepted")


def test_refinement_can_reconstruct_an_overflowed_internal_node():
    identity = transform_candidate(ExactAffineTransform.identity(), "identity")
    left = (identity, opcode_candidate("add_0", 1))
    right = (identity, opcode_candidate("add_1", 1))
    tree = build_tree((left, right), cap=2)
    assert tree.overflow and retained_sources(tree) == 2
    refined = refine_leaf(tree, 0, (identity,))
    rebuilt = build_tree(((identity,), right), cap=2)
    assert not refined.overflow
    assert candidate_signature(refined) == candidate_signature(rebuilt)


def main():
    test_exact_operator_parity_with_established_algebra()
    test_noncommutative_chronological_order()
    test_distinct_opcode_paths_collapse_to_one_complete_transform()
    test_query_agreement_is_weaker_than_source_drop_certification()
    test_overflow_is_sticky_but_preserves_exact_factorized_siblings()
    test_chunk_composition_is_associative()
    test_representative_long_exact_histories()
    test_monotone_refinement_retains_alternate_derivations()
    test_refinement_can_reconstruct_an_overflowed_internal_node()
    print(
        "version-space product tree mechanics: passed "
        "(noncommutative, fail-closed, associative, 4096-event exact history)"
    )


if __name__ == "__main__":
    main()
