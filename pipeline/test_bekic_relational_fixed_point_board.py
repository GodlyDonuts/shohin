from __future__ import annotations

import copy

import pytest

from bekic_relational_fixed_point_board import (
    ALLOWED_OPERATIONS,
    BekicBoardError,
    DEPENDENCY_TOPOLOGIES,
    canonical_program_semantics,
    empty_relation,
    evaluate_bekic,
    evaluate_program_equations,
    evaluate_simultaneous,
    generate_row,
    generate_rows,
    identity_relation,
    reindex_input_objects,
    reindex_program_nodes,
    reindex_program_variables,
    reindex_relation,
    relation_compose,
    relation_converse,
    relation_intersection,
    relation_subset,
    relation_union,
    select_machine_input,
    split_contract,
    universal_relation,
    validate_row,
)


def _relation(value: object) -> tuple[tuple[int, ...], ...]:
    assert isinstance(value, list)
    return tuple(tuple(int(bit) for bit in row) for row in value)


def _constants(row: dict[str, object]) -> dict[str, tuple[tuple[int, ...], ...]]:
    return {
        str(item["id"]): _relation(item["relation"])
        for item in row["input"]["constants"]
    }


def _targets(row: dict[str, object]) -> dict[str, tuple[tuple[int, ...], ...]]:
    return {
        str(item["id"]): _relation(item["relation"])
        for item in row["targets"]["variables"]
    }


def _program(row: dict[str, object], form: str) -> dict[str, object]:
    return row["input"]["representations"][form]


def test_independent_oracles_and_stored_targets_agree() -> None:
    rows = [
        *generate_rows(split="train", count=6, seed=1201),
        *generate_rows(split="development", count=6, seed=1202),
        *generate_rows(
            split="confirmation",
            count=4,
            seed=1203,
            sealed_confirmation=True,
        ),
    ]
    for row in rows:
        validate_row(
            row,
            sealed_confirmation=row["split"] == "confirmation",
        )
        simultaneous = evaluate_simultaneous(row["input"])
        nested = evaluate_bekic(row["input"])
        assert simultaneous == nested == _targets(row)


def test_paired_graphs_are_bekic_equivalent_after_node_id_erasure() -> None:
    for seed in range(8):
        row = generate_row(split="train", seed=seed)
        simultaneous = _program(row, "simultaneous")
        nested = _program(row, "nested")
        assert simultaneous["form"] == "simultaneous"
        assert nested["form"] == "bekic_nested"
        assert set(node["id"] for node in simultaneous["nodes"]).isdisjoint(
            node["id"] for node in nested["nodes"]
        )
        assert canonical_program_semantics(
            simultaneous
        ) == canonical_program_semantics(nested)
        assert evaluate_simultaneous(row["input"]) == evaluate_bekic(row["input"])


def test_relation_algebra_and_generated_equations_are_monotone() -> None:
    row = generate_row(split="train", seed=411)
    cardinality = row["input"]["cardinality"]
    constants = _constants(row)
    program = _program(row, "simultaneous")
    variables = tuple(program["variables"])
    lower = {
        variables[0]: empty_relation(cardinality),
        variables[1]: empty_relation(cardinality),
    }
    upper = _targets(row)
    lower_outputs = evaluate_program_equations(program, constants, lower)
    upper_outputs = evaluate_program_equations(program, constants, upper)
    assert all(
        relation_subset(lower_outputs[variable], upper_outputs[variable])
        for variable in variables
    )

    first = next(iter(constants.values()))
    bottom = empty_relation(cardinality)
    top = universal_relation(cardinality)
    assert relation_subset(
        relation_union(bottom, first),
        relation_union(first, top),
    )
    assert relation_subset(
        relation_intersection(bottom, first),
        relation_intersection(first, top),
    )
    assert relation_subset(
        relation_compose(bottom, first),
        relation_compose(first, top),
    )
    assert relation_subset(relation_converse(bottom), relation_converse(first))


def test_object_variable_and_program_node_reindexing_preserve_results() -> None:
    row = generate_row(split="development", seed=877)
    original_input = row["input"]
    original = evaluate_simultaneous(original_input)
    cardinality = original_input["cardinality"]

    object_permutation = tuple(reversed(range(cardinality)))
    object_input = reindex_input_objects(original_input, object_permutation)
    object_result = evaluate_simultaneous(object_input)
    assert object_result == {
        variable: reindex_relation(relation, object_permutation)
        for variable, relation in original.items()
    }
    assert evaluate_bekic(object_input) == object_result

    node_input = copy.deepcopy(original_input)
    for form in ("simultaneous", "nested"):
        program = node_input["representations"][form]
        permutation = tuple(reversed(range(len(program["nodes"]))))
        node_input["representations"][form] = reindex_program_nodes(
            program,
            permutation,
        )
    assert evaluate_simultaneous(node_input) == original
    assert evaluate_bekic(node_input) == original

    variable_input = copy.deepcopy(original_input)
    simultaneous, mapping = reindex_program_variables(
        variable_input["representations"]["simultaneous"],
        (1, 0),
    )
    nested, nested_mapping = reindex_program_variables(
        variable_input["representations"]["nested"],
        (1, 0),
    )
    assert mapping == nested_mapping
    variable_input["representations"] = {
        "simultaneous": simultaneous,
        "nested": nested,
    }
    expected = {mapping[variable]: relation for variable, relation in original.items()}
    assert evaluate_simultaneous(variable_input) == expected
    assert evaluate_bekic(variable_input) == expected


def test_program_topology_depth_and_scale_are_factorial_axes() -> None:
    train_contract = split_contract("train")["axis_cells"]["factorial"]
    expected = {
        (cardinality, depth, topology)
        for cardinality in train_contract["cardinalities"]
        for depth in train_contract["expression_depths"]
        for topology in train_contract["dependency_topologies"]
    }
    rows = [generate_row(split="train", seed=seed) for seed in range(len(expected))]
    observed = {
        (
            row["axes"]["cardinality"],
            row["axes"]["expression_depth"],
            row["axes"]["dependency_topology"],
        )
        for row in rows
    }
    assert observed == expected
    assert {row["axes"]["dependency_topology"] for row in rows} == set(
        DEPENDENCY_TOPOLOGIES
    )
    assert all(
        {
            node.get("operation")
            for node in _program(row, "simultaneous")["nodes"]
            if node["kind"] == "operation"
        }
        == set(ALLOWED_OPERATIONS)
        for row in rows
    )


def test_development_cells_isolate_scale_depth_and_joint_shift() -> None:
    rows = [generate_row(split="development", seed=seed) for seed in range(36)]
    by_cell: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_cell.setdefault(str(row["axes"]["axis_cell"]), []).append(row)
    assert set(by_cell) == {"scale_only", "depth_only", "joint"}
    assert {
        row["axes"]["expression_depth"] for row in by_cell["scale_only"]
    } <= {5, 6}
    assert {
        row["axes"]["cardinality"] for row in by_cell["scale_only"]
    } <= {6, 7}
    assert {
        row["axes"]["expression_depth"] for row in by_cell["depth_only"]
    } == {7}
    assert {
        row["axes"]["cardinality"] for row in by_cell["depth_only"]
    } <= {3, 4, 5}
    assert {
        (
            row["axes"]["cardinality"],
            row["axes"]["expression_depth"],
        )
        for row in by_cell["joint"]
    } <= {(6, 7), (7, 7)}


def test_confirmation_generation_and_validation_fail_closed() -> None:
    with pytest.raises(BekicBoardError, match="sealed_confirmation"):
        generate_row(split="confirmation", seed=17)
    with pytest.raises(BekicBoardError, match="sealed_confirmation"):
        generate_rows(split="confirmation", count=2, seed=17)
    row = generate_row(
        split="confirmation",
        seed=17,
        sealed_confirmation=True,
    )
    with pytest.raises(BekicBoardError, match="sealed_confirmation"):
        validate_row(row)
    validate_row(row, sealed_confirmation=True)


def test_split_rows_and_receipts_are_disjoint_and_deterministic() -> None:
    train = generate_rows(split="train", count=10, seed=901)
    development = generate_rows(split="development", count=10, seed=901)
    confirmation = generate_rows(
        split="confirmation",
        count=10,
        seed=901,
        sealed_confirmation=True,
    )
    hash_sets = [
        {row["hashes"]["input_sha256"] for row in rows}
        for rows in (train, development, confirmation)
    ]
    assert hash_sets[0].isdisjoint(hash_sets[1])
    assert hash_sets[0].isdisjoint(hash_sets[2])
    assert hash_sets[1].isdisjoint(hash_sets[2])
    assert generate_row(split="train", seed=771) == generate_row(
        split="train",
        seed=771,
    )
    first = generate_row(split="train", seed=771)
    second = generate_row(split="train", seed=772)
    assert set(_program(first, "simultaneous")["variables"]).isdisjoint(
        _program(second, "simultaneous")["variables"]
    )
    assert {
        node["id"] for node in _program(first, "simultaneous")["nodes"]
    }.isdisjoint(
        node["id"] for node in _program(second, "simultaneous")["nodes"]
    )


def test_inputs_have_no_schedule_or_direct_target_leakage() -> None:
    forbidden = {
        "answer",
        "fixed_point",
        "halt",
        "iteration_count",
        "query",
        "schedule",
        "source",
        "target",
        "trajectory",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return {
                str(key).lower()
                for key in value
            } | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    for row in generate_rows(split="train", count=12, seed=1107):
        input_packet = row["input"]
        input_keys = keys(input_packet)
        assert all(
            not any(fragment in key for fragment in forbidden)
            for key in input_keys
        )
        raw_constants = set(_constants(row).values())
        targets = set(_targets(row).values())
        assert raw_constants.isdisjoint(targets)
        assert "hashes" not in input_packet
        assert "axes" not in input_packet


def test_neural_arm_receives_only_one_program_representation() -> None:
    row = generate_row(split="train", seed=1441)
    simultaneous = select_machine_input(row["input"], "simultaneous")
    nested = select_machine_input(row["input"], "nested")
    assert set(simultaneous) == {
        "schema",
        "cardinality",
        "constants",
        "program",
    }
    assert simultaneous["program"]["form"] == "simultaneous"
    assert nested["program"]["form"] == "bekic_nested"
    assert "representations" not in simultaneous
    assert "representations" not in nested
    assert simultaneous["program"] != nested["program"]
    with pytest.raises(BekicBoardError, match="representation"):
        select_machine_input(row["input"], "both")


def test_hash_and_graph_tampering_is_rejected() -> None:
    row = generate_row(split="train", seed=712)
    tampered_hash = copy.deepcopy(row)
    tampered_hash["hashes"]["input_sha256"] = "0" * 64
    with pytest.raises(BekicBoardError, match="hashes"):
        validate_row(tampered_hash)

    tampered_graph = copy.deepcopy(row)
    program = tampered_graph["input"]["representations"]["simultaneous"]
    operation = next(
        node
        for node in program["nodes"]
        if node.get("operation") == "INTERSECTION"
    )
    operation["operation"] = "UNION"
    tampered_graph["hashes"] = row["hashes"]
    with pytest.raises(BekicBoardError):
        validate_row(tampered_graph)


def test_identity_is_typed_and_not_materialized_as_an_input_relation() -> None:
    row = generate_row(split="train", seed=99)
    cardinality = row["input"]["cardinality"]
    constants = set(_constants(row).values())
    identity = identity_relation(cardinality)
    program = _program(row, "simultaneous")
    assert any(node.get("operation") == "IDENTITY" for node in program["nodes"])
    assert identity not in constants
