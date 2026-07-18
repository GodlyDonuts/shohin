#!/usr/bin/env python3
"""Focused deterministic accounting gates for the exact R10 product tree."""

from version_space_accounting import (
    AccountingContractError,
    account_version_space_tree,
    active_hot_frontier_records,
    canonical_json_bytes,
    external_source_records,
    factorized_provenance_records,
    integer_bit_growth,
    retrieval_reference_records,
)
from version_space_product_tree import (
    build_tree,
    leaf_node,
    merge_nodes,
    opcode_candidate,
)


def test_canonical_payload_and_exact_category_sums_are_deterministic():
    first_payload = {"z": 2, "a": [1, -8]}
    same_payload = {"a": [1, -8], "z": 2}
    assert canonical_json_bytes(first_payload) == b'{"a":[1,-8],"z":2}'
    assert canonical_json_bytes(first_payload) == canonical_json_bytes(same_payload)
    growth = integer_bit_growth(first_payload)
    assert growth.values == 3
    assert growth.magnitude_bits == 7
    assert growth.max_magnitude_bits == 4
    try:
        canonical_json_bytes({"valid": 1, 2: "invalid"})
    except AccountingContractError:
        pass
    else:
        raise AssertionError("non-string canonical mapping key was accepted")

    tree = build_tree(
        ((opcode_candidate("add_0", 3),), (opcode_candidate("swap"),)),
        sources=("source-0", "source-1"),
        start=9,
        cap=8,
    )
    external_a = (
        {"event": 0, "text": "add three"},
        {"event": 1, "text": "swap"},
    )
    external_b = (
        {"text": "add three", "event": 0},
        {"text": "swap", "event": 1},
    )
    accounting_a = account_version_space_tree(tree, external_a)
    accounting_b = account_version_space_tree(tree, external_b)
    assert accounting_a == accounting_b

    hot = active_hot_frontier_records(tree, external_a)
    provenance = factorized_provenance_records(tree)
    external = external_source_records(tree, external_a)
    retrieval = retrieval_reference_records(tree)
    assert accounting_a.active_hot_frontier_bytes == sum(
        len(canonical_json_bytes(record)) for record in hot
    )
    assert accounting_a.factorized_provenance_bytes == sum(
        len(canonical_json_bytes(record)) for record in provenance
    )
    assert accounting_a.external_source_bytes == sum(
        len(canonical_json_bytes(record)) for record in external
    )
    assert accounting_a.retrieval_reference_bytes == sum(
        len(canonical_json_bytes(record)) for record in retrieval
    )
    assert len(hot) == 1 and hot[0]["kind"] == "transform"
    assert len(hot[0]["support_commitment"]) == 64
    assert len(hot[0]["node_commitment"]) == 64
    assert retrieval == ({
        "end": 11,
        "node_commitment": hot[0]["node_commitment"],
        "start": 9,
    },)


def test_ambiguous_tree_keeps_source_hot_and_has_no_retrieval_bytes():
    tree = leaf_node(
        7,
        (opcode_candidate("add_0", 5), opcode_candidate("add_1", 5)),
        "ambiguous-source-reference",
        cap=8,
    )
    external = {7: {"event": "ambiguous", "value": 5}}
    accounting = account_version_space_tree(tree, external)
    hot = active_hot_frontier_records(tree, external)
    provenance = factorized_provenance_records(tree)

    assert len(hot) == 1 and hot[0]["kind"] == "source"
    assert hot[0]["source"] == external[7]
    assert len(hot[0]["source_commitment"]) == 64
    assert retrieval_reference_records(tree) == ()
    assert accounting.retrieval_reference_count == 0
    assert accounting.retrieval_reference_bytes == 0
    assert accounting.retained_source_events == 1
    assert accounting.evicted_source_events == 0
    assert len(provenance) == 1
    assert len(provenance[0]["candidates"]) == 2
    assert all(
        len(candidate["support_commitment"]) == 64
        for candidate in provenance[0]["candidates"]
    )
    assert "source" not in provenance[0]["source_reference"]


def test_overflow_preserves_factorized_children_and_singleton_range_reference():
    overflowed = leaf_node(
        20,
        tuple(opcode_candidate("add_0", value) for value in (1, 2, 3)),
        "overflow-source-reference",
        cap=2,
    )
    singleton = leaf_node(
        21,
        (opcode_candidate("swap"),),
        "singleton-source-reference",
        cap=2,
    )
    tree = merge_nodes(overflowed, singleton)
    external = {
        20: {"event": "overflow"},
        21: {"event": "singleton"},
    }
    accounting = account_version_space_tree(tree, external)
    hot = active_hot_frontier_records(tree, external)
    provenance = factorized_provenance_records(tree)
    retrieval = retrieval_reference_records(tree)

    assert tree.overflow
    assert [record["kind"] for record in hot] == ["source", "transform"]
    assert hot[0]["overflow"] is True
    assert hot[0]["version_space_lower_bound"] == 3
    assert retrieval == ({
        "end": 22,
        "node_commitment": hot[1]["node_commitment"],
        "start": 21,
    },)
    assert accounting.retained_source_events == 1
    assert accounting.evicted_source_events == 1
    assert accounting.retrieval_reference_count == 1
    assert accounting.factorized_node_count == 3
    root_record = provenance[0]
    assert root_record["kind"] == "internal"
    assert root_record["overflow"] is True
    assert root_record["candidates"] == []
    assert len(root_record["children"]) == 2
    assert all(len(child["node_commitment"]) == 64 for child in root_record["children"])


def test_external_source_coverage_fails_closed():
    tree = build_tree(
        ((opcode_candidate("swap"),), (opcode_candidate("swap"),)),
        start=30,
        cap=8,
    )
    for external in (({"event": 0},), {30: {"event": 0}}):
        try:
            account_version_space_tree(tree, external)
        except AccountingContractError:
            pass
        else:
            raise AssertionError("partial external-source accounting was accepted")


def test_4096_event_integer_growth_and_factorized_node_accounting():
    events = 4096
    candidate_sets = tuple(
        (opcode_candidate("merge_0_1" if index % 2 == 0 else "merge_1_0"),)
        for index in range(events)
    )
    tree = build_tree(candidate_sets, cap=8)
    accounting = account_version_space_tree(tree, ("x",) * events)
    hot = active_hot_frontier_records(tree, ("x",) * events)
    retrieval = retrieval_reference_records(tree)

    assert tree.source_droppable and tree.unique_transform is not None
    assert accounting.events == events
    assert accounting.retained_source_events == 0
    assert accounting.evicted_source_events == events
    assert accounting.active_hot_frontier.records == 1
    assert accounting.external_source.records == events
    assert accounting.factorized_node_count == 2 * events - 1
    assert accounting.retrieval_reference_count == 1
    assert hot[0]["start"] == 0 and hot[0]["end"] == events
    assert retrieval[0]["start"] == 0 and retrieval[0]["end"] == events
    assert retrieval[0]["node_commitment"] == hot[0]["node_commitment"]

    expected_max_bits = max(
        max(1, abs(value).bit_length())
        for value in tree.unique_transform.flat
    )
    assert expected_max_bits > 64
    assert accounting.transform_integer_growth.max_magnitude_bits == expected_max_bits
    assert accounting.active_hot_frontier.integer_growth.max_magnitude_bits >= expected_max_bits
    assert accounting.factorized_provenance.integer_growth.max_magnitude_bits >= expected_max_bits
    assert accounting.total_canonical_bytes == (
        accounting.active_hot_frontier_bytes
        + accounting.factorized_provenance_bytes
        + accounting.external_source_bytes
        + accounting.retrieval_reference_bytes
    )


def main():
    test_canonical_payload_and_exact_category_sums_are_deterministic()
    test_ambiguous_tree_keeps_source_hot_and_has_no_retrieval_bytes()
    test_overflow_preserves_factorized_children_and_singleton_range_reference()
    test_external_source_coverage_fails_closed()
    test_4096_event_integer_growth_and_factorized_node_accounting()
    print(
        "version-space accounting: passed "
        "(canonical stores, commitments, ambiguity, overflow, 4096-event growth)"
    )


if __name__ == "__main__":
    main()
