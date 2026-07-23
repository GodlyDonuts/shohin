from __future__ import annotations

from contrastive_fixed_point_board import (
    MAX_OBJECTS,
    closure,
    compose,
    generate_rows,
    identity_relation,
    relation_difference,
    relation_union,
    split_contract,
    validate_row,
)


def test_independent_closure_counts_one_edge_expansions() -> None:
    relation = (
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
    )
    observed, depth = closure(relation)
    assert depth == 3
    assert observed == (
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (1, 1, 1, 0),
        (1, 1, 1, 1),
    )
    one_step = relation_union(
        identity_relation(4),
        compose(relation, identity_relation(4)),
    )
    assert one_step[3][0] == 0


def test_rows_obey_disjoint_scale_and_depth_contracts() -> None:
    for split in ("train", "development", "confirmation"):
        rows = generate_rows(split=split, count=64, seed=557)
        contract = split_contract(split)
        assert len({str(row["semantic_sha256"]) for row in rows}) == len(rows)
        for row in rows:
            validate_row(row)
            assert row["cardinality"] in contract["cardinalities"]
            assert row["a_depth"] in contract["a_depths"]
            assert row["b_depth"] < row["a_depth"]
            assert len(row["input_registers"]) == 6
            assert len(row["target_registers"]) == 6
            assert len(row["answer_bits"]) == MAX_OBJECTS


def test_target_is_antitone_under_a_larger_second_closure() -> None:
    relation_a = (
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
    )
    relation_b_small = (
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    relation_b_large = (
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 0),
    )
    closure_a, _ = closure(relation_a)
    closure_small, _ = closure(relation_b_small)
    closure_large, _ = closure(relation_b_large)
    first = relation_difference(closure_a, closure_small)
    second = relation_difference(closure_a, closure_large)
    assert all(
        second[row][column] <= first[row][column]
        for row in range(4)
        for column in range(4)
    )
    assert first != second


def test_input_packet_does_not_contain_derived_closures_or_answer() -> None:
    for row in generate_rows(split="train", count=32, seed=830):
        registers = row["input_registers"]
        assert registers[3:] == [
            [[0] * MAX_OBJECTS for _ in range(MAX_OBJECTS)]
        ] * 3
        assert row["target_registers"][5] not in registers
