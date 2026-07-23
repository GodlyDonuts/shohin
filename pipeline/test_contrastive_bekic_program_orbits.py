from __future__ import annotations

import copy

import pytest
from bekic_relational_fixed_point_board import reindex_relation

from contrastive_bekic_program_orbits import (
    COUNTERFACTUAL_MIN_HAMMING,
    HELD_OUT_MOTIF,
    ISOLATED_COUNTERFACTUAL_ARMS,
    MIN_CONVERGENCE_UPDATES,
    MIN_TOTAL_VARIABLE_CHANGE_STEPS,
    MIN_VARIABLE_CHANGE_STEPS,
    ContrastiveOrbitError,
    _receipt_values,
    assert_split_disjoint,
    canonical_skeleton,
    contains_held_out_motif,
    deserialize_targets,
    evaluate_nested_mu,
    evaluate_simultaneous,
    fixed_point_pressure,
    generate_orbit,
    generate_orbits,
    generate_train_development,
    individual_motif_receipts,
    joint_target_hamming,
    program_receipts,
    program_statistics,
    select_isolated_counterfactual_input,
    select_machine_input,
    split_contract,
    transplant_constants,
    transplant_program,
    validate_orbit,
)


def test_counterfactual_orbits_are_balanced_and_causally_separated() -> None:
    rows = generate_orbits(split="train", count=8, seed=9201)
    observed_skeletons: set[str] = set()
    for row in rows:
        validate_orbit(row)
        p = select_machine_input(row, arm="p", form="simultaneous")
        p_prime = select_machine_input(
            row,
            arm="p_prime",
            form="simultaneous",
        )
        assert program_statistics(p["program"]) == program_statistics(
            p_prime["program"]
        )
        assert (
            joint_target_hamming(
                deserialize_targets(row["targets"]["p"]),
                deserialize_targets(row["targets"]["p_prime"]),
            )
            >= COUNTERFACTUAL_MIN_HAMMING
        )
        assert p["program"] != p_prime["program"]
        observed_skeletons.add(row["receipts"]["p"]["skeleton_sha256"])
    assert len(observed_skeletons) == len(rows)


def test_isolated_counterfactuals_change_one_factor_at_a_time() -> None:
    row = generate_orbit(split="train", seed=9214)
    original = select_machine_input(row, arm="p", form="simultaneous")
    original_target = deserialize_targets(row["targets"]["p"])
    original_statistics = program_statistics(original["program"])
    original_skeleton = canonical_skeleton(original["program"])
    for arm in ISOLATED_COUNTERFACTUAL_ARMS:
        packet = select_isolated_counterfactual_input(
            row,
            arm=arm,
            form="simultaneous",
        )
        target = deserialize_targets(
            row["isolated_counterfactuals"][arm]["targets"]
        )
        assert program_statistics(packet["program"]) == original_statistics
        assert target != original_target
        assert (
            joint_target_hamming(original_target, target)
            == row["counterfactual"]["isolated_joint_target_hamming"][arm]
        )
        if arm == "constant_rewire":
            assert canonical_skeleton(packet["program"]) == original_skeleton
            assert packet["constants"] == original["constants"]
        else:
            assert packet["constants"] == original["constants"]
            assert packet["program"] != original["program"]


def test_recursive_pressure_is_per_variable_and_recomputed() -> None:
    row = generate_orbit(split="development", seed=9215, index=2)
    packets = {
        arm: select_machine_input(
            row,
            arm=arm,
            form="simultaneous",
        )
        for arm in ("p", "p_prime", "p_eq")
    }
    packets.update(
        {
            arm: select_isolated_counterfactual_input(
                row,
                arm=arm,
                form="simultaneous",
            )
            for arm in ISOLATED_COUNTERFACTUAL_ARMS
        }
    )
    receipts = row["receipts"]["recursive_pressure"]
    assert set(receipts) == set(packets)
    for arm, packet in packets.items():
        observed = fixed_point_pressure(packet)
        assert receipts[arm] == observed
        assert len(observed["variables"]) == 2
        assert all(
            variable["changed_bits_total"] >= variable["change_steps"]
            and variable["last_change_update"] <= observed["convergence_updates"]
            for variable in observed["variables"]
        )
        if arm in {"p", "p_eq"}:
            assert (
                observed["minimum_variable_change_steps"]
                >= MIN_VARIABLE_CHANGE_STEPS
            )
            assert (
                observed["convergence_updates"]
                >= MIN_CONVERGENCE_UPDATES
            )
            assert (
                observed["total_variable_change_steps"]
                >= MIN_TOTAL_VARIABLE_CHANGE_STEPS
            )
            assert all(
                variable["change_steps"] >= MIN_VARIABLE_CHANGE_STEPS
                for variable in observed["variables"]
            )


def test_programs_cover_the_declared_monotone_grammar_and_topologies() -> None:
    rows = generate_orbits(split="train", count=16, seed=9202)
    kinds: set[str] = set()
    topologies: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            kind = value.get("kind")
            if isinstance(kind, str):
                kinds.add(kind)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    for row in rows:
        packet = select_machine_input(row, arm="p", form="simultaneous")
        collect(packet["program"])
        topologies.add(program_statistics(packet["program"])["dependency_topology"])
    assert {
        "VARIABLE",
        "CONSTANT",
        "IDENTITY",
        "UNION",
        "INTERSECTION",
        "COMPOSE",
        "CONVERSE",
    } <= kinds
    assert topologies == {
        "independent_dag",
        "dag_x_to_y",
        "dag_y_to_x",
        "mutual_cycle",
    }


def test_explicit_nested_lfp_oracle_agrees_independently() -> None:
    for row in generate_orbits(
        split="development",
        count=10,
        seed=9203,
    ):
        for arm in ("p", "p_prime", "p_eq"):
            simultaneous_packet = select_machine_input(
                row,
                arm=arm,
                form="simultaneous",
            )
            nested_packet = select_machine_input(
                row,
                arm=arm,
                form="nested",
            )
            expression = nested_packet["program"]["expression"]
            assert expression["kind"] == "LET"
            assert expression["value"]["kind"] == "LFP"
            assert expression["body"]["kind"] == "PAIR"
            assert any(
                entry["expression"]["kind"] == "LFP"
                for entry in expression["body"]["entries"]
            )
            assert evaluate_simultaneous(simultaneous_packet) == evaluate_nested_mu(
                nested_packet
            )


def test_peq_alpha_node_object_and_constant_order_rewrite_is_invariant() -> None:
    row = generate_orbit(split="train", seed=9204)
    p_sim = select_machine_input(row, arm="p", form="simultaneous")
    peq_sim = select_machine_input(
        row,
        arm="p_eq",
        form="simultaneous",
    )
    p_nested = select_machine_input(row, arm="p", form="nested")
    peq_nested = select_machine_input(row, arm="p_eq", form="nested")
    assert p_sim["program"]["variables"] != peq_sim["program"]["variables"]
    assert [item["id"] for item in p_sim["constants"]] != [
        item["id"] for item in peq_sim["constants"]
    ]
    assert row["equivalence"]["object_permutation"] != list(range(p_sim["cardinality"]))
    p_node_ids = {
        node["id"]
        for equation in p_sim["program"]["equations"]
        for node in _nodes(equation["expression"])
    }
    peq_node_ids = {
        node["id"]
        for equation in peq_sim["program"]["equations"]
        for node in _nodes(equation["expression"])
    }
    assert p_node_ids.isdisjoint(peq_node_ids)
    assert canonical_skeleton(p_sim["program"]) == canonical_skeleton(
        peq_sim["program"]
    )
    assert program_receipts(p_sim["program"]) == program_receipts(peq_sim["program"])
    assert evaluate_simultaneous(peq_sim) == deserialize_targets(row["targets"]["p_eq"])
    assert evaluate_nested_mu(peq_nested) == deserialize_targets(row["targets"]["p_eq"])
    variable_mapping = row["equivalence"]["variable_mapping"]
    object_permutation = row["equivalence"]["object_permutation"]
    p_target = deserialize_targets(row["targets"]["p"])
    peq_target = deserialize_targets(row["targets"]["p_eq"])
    assert peq_target == {
        variable_mapping[variable]: reindex_relation(
            relation,
            object_permutation,
        )
        for variable, relation in p_target.items()
    }
    assert p_nested["program"] != peq_nested["program"]


def test_constant_sampling_is_role_independent_and_order_is_shuffled() -> None:
    rows = generate_orbits(split="train", count=12, seed=9205)
    at_least_one_nonlexical_order = False
    for row in rows:
        sampling = row["receipts"]["constant_sampling"]
        assert sampling["role_specific_probabilities"] is False
        assert 0.08 <= sampling["shared_bernoulli_probability"] <= 0.30
        packet = select_machine_input(row, arm="p", form="simultaneous")
        serialized_ids = [item["id"] for item in packet["constants"]]
        occurrence_ids = []

        def visit(expression: dict[str, object]) -> None:
            if expression["kind"] == "CONSTANT":
                name = expression["constant"]
                if name not in occurrence_ids:
                    occurrence_ids.append(name)
            elif expression["kind"] == "CONVERSE":
                visit(expression["child"])
            elif expression["kind"] in {
                "UNION",
                "INTERSECTION",
                "COMPOSE",
            }:
                for child in expression["children"]:
                    visit(child)

        for equation in packet["program"]["equations"]:
            visit(equation["expression"])
        at_least_one_nonlexical_order |= serialized_ids != occurrence_ids
    assert at_least_one_nonlexical_order


def test_program_and_constant_transplants_change_the_causal_result() -> None:
    rows = [
        generate_orbit(split="train", seed=9206, index=index) for index in range(24)
    ]
    pair = next(
        (left, right)
        for left in rows
        for right in rows
        if left is not right
        and left["axes"]["cardinality"] == right["axes"]["cardinality"]
    )
    donor, recipient = pair
    program_swap = transplant_program(donor, recipient)
    constant_swap = transplant_constants(donor, recipient)
    program_result = evaluate_simultaneous(program_swap["simultaneous"])
    constant_result = evaluate_simultaneous(constant_swap["simultaneous"])
    assert program_result == evaluate_nested_mu(program_swap["nested"])
    assert constant_result == evaluate_nested_mu(constant_swap["nested"])
    donor_target = deserialize_targets(donor["targets"]["p"])
    recipient_target = deserialize_targets(recipient["targets"]["p"])
    assert set(program_result) == set(recipient_target)
    assert program_result != recipient_target
    assert constant_result != recipient_target
    assert program_result != donor_target or constant_result != donor_target


def test_split_receipts_are_disjoint_and_generation_is_deterministic() -> None:
    train, development = generate_train_development(
        train_count=8,
        development_count=10,
        seed=9207,
    )
    assert_split_disjoint(train, development)
    repeated_train, repeated_development = generate_train_development(
        train_count=8,
        development_count=10,
        seed=9207,
    )
    assert train == repeated_train
    assert development == repeated_development
    for depth in (2, 3):
        train_motifs = {
            receipt
            for row in train
            for arm in (
                "p",
                "p_prime",
                *ISOLATED_COUNTERFACTUAL_ARMS,
            )
            for receipt in individual_motif_receipts(
                (
                    select_machine_input(
                        row,
                        arm=arm,
                        form="simultaneous",
                    )
                    if arm in {"p", "p_prime"}
                    else select_isolated_counterfactual_input(
                        row,
                        arm=arm,
                        form="simultaneous",
                    )
                )["program"],
                depth,
            )
        }
        development_motifs = {
            receipt
            for row in development
            for arm in (
                "p",
                "p_prime",
                *ISOLATED_COUNTERFACTUAL_ARMS,
            )
            for receipt in individual_motif_receipts(
                (
                    select_machine_input(
                        row,
                        arm=arm,
                        form="simultaneous",
                    )
                    if arm in {"p", "p_prime"}
                    else select_isolated_counterfactual_input(
                        row,
                        arm=arm,
                        form="simultaneous",
                    )
                )["program"],
                depth,
            )
        }
        assert train_motifs.isdisjoint(development_motifs)


def test_development_contract_exercises_all_partition_cells() -> None:
    rows = generate_orbits(
        split="development",
        count=10,
        seed=9208,
    )
    assert {row["axes"]["cell"] for row in rows} == {
        "in_range",
        "motif",
        "scale",
        "depth",
        "joint",
    }
    assert {
        row["axes"]["cardinality"] for row in rows if row["axes"]["cell"] == "scale"
    } == {7}
    assert all(
        row["axes"]["ast_depth"] >= 8
        for row in rows
        if row["axes"]["cell"] in {"depth", "joint"}
    )
    assert all(
        contains_held_out_motif(
            select_machine_input(row, arm="p", form="simultaneous")["program"]
        )
        == (row["axes"]["cell"] == "motif")
        for row in rows
    )
    assert all(
        row["receipts"]["held_out_motif"] == {
            "definition": list(HELD_OUT_MOTIF),
            "required_in_score_bearing_arms": (
                row["axes"]["cell"] == "motif"
            ),
            "present_by_arm": {
                arm: contains_held_out_motif(
                    (
                        select_machine_input(
                            row,
                            arm=arm,
                            form="simultaneous",
                        )
                        if arm in {"p", "p_prime", "p_eq"}
                        else select_isolated_counterfactual_input(
                            row,
                            arm=arm,
                            form="simultaneous",
                        )
                    )["program"]
                )
                for arm in (
                    "p",
                    "p_prime",
                    "p_eq",
                    *ISOLATED_COUNTERFACTUAL_ARMS,
                )
            },
        }
        for row in rows
    )


def test_held_out_motif_is_absent_from_training() -> None:
    rows = generate_orbits(split="train", count=32, seed=9212)
    assert all(
        not contains_held_out_motif(
            (
                select_machine_input(
                    row,
                    arm=arm,
                    form="simultaneous",
                )
                if arm in {"p", "p_prime", "p_eq"}
                else select_isolated_counterfactual_input(
                    row,
                    arm=arm,
                    form="simultaneous",
                )
            )["program"]
        )
        for row in rows
        for arm in (
            "p",
            "p_prime",
            "p_eq",
            *ISOLATED_COUNTERFACTUAL_ARMS,
        )
    )


def test_motif_development_requires_motif_in_every_score_bearing_arm() -> None:
    rows = generate_orbits(split="development", count=7, seed=9213)
    motif = next(row for row in rows if row["axes"]["cell"] == "motif")
    assert all(
        contains_held_out_motif(
            select_machine_input(
                motif,
                arm=arm,
                form="simultaneous",
            )["program"]
        )
        for arm in ("p", "p_prime", "p_eq")
    )


def test_known_motif_seed_is_feasible_after_prior_partition_receipts() -> None:
    seed = 2026072313
    train = generate_orbit(
        split="train",
        seed=seed,
        index=0,
        max_attempts=400,
    )
    forbidden = _receipt_values(train)
    in_range = generate_orbit(
        split="development",
        seed=seed,
        index=0,
        forbidden_receipts=forbidden,
        max_attempts=400,
    )
    forbidden.update(_receipt_values(in_range))
    motif = generate_orbit(
        split="development",
        seed=seed,
        index=1,
        forbidden_receipts=forbidden,
        max_attempts=400,
    )
    assert motif["axes"]["cell"] == "motif"
    assert all(
        contains_held_out_motif(
            select_machine_input(
                motif,
                arm=arm,
                form="simultaneous",
            )["program"]
        )
        for arm in ("p", "p_prime", "p_eq")
    )


def test_confirmation_api_is_permanently_fail_closed() -> None:
    with pytest.raises(ContrastiveOrbitError, match="fail-closed"):
        split_contract("confirmation")
    with pytest.raises(ContrastiveOrbitError, match="fail-closed"):
        generate_orbit(split="confirmation", seed=9209)
    with pytest.raises(ContrastiveOrbitError, match="fail-closed"):
        generate_orbits(split="confirmation", count=1, seed=9209)


def test_machine_inputs_contain_no_target_or_execution_leakage() -> None:
    forbidden = {
        "answer",
        "fixed_point",
        "halt",
        "iteration",
        "query",
        "schedule",
        "target",
        "trajectory",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return {str(key).lower() for key in value} | set().union(
                *(keys(item) for item in value.values()), set()
            )
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    for row in generate_orbits(split="train", count=8, seed=9210):
        for arm in (
            "p",
            "p_prime",
            "p_eq",
            *ISOLATED_COUNTERFACTUAL_ARMS,
        ):
            for form in ("simultaneous", "nested"):
                packet = (
                    select_machine_input(row, arm=arm, form=form)
                    if arm in {"p", "p_prime", "p_eq"}
                    else select_isolated_counterfactual_input(
                        row,
                        arm=arm,
                        form=form,
                    )
                )
                packet_keys = keys(packet)
                assert all(
                    not any(fragment in key for fragment in forbidden)
                    for key in packet_keys
                )
                assert "targets" not in packet
                assert "receipts" not in packet
                assert "axes" not in packet


def test_hash_and_program_tampering_is_detected() -> None:
    row = generate_orbit(split="train", seed=9211)
    tampered = copy.deepcopy(row)
    tampered["counterfactual"]["joint_target_hamming"] = 0.0
    with pytest.raises(ContrastiveOrbitError, match="separation"):
        validate_orbit(tampered)

    tampered = copy.deepcopy(row)
    program = tampered["inputs"]["p"]["simultaneous"]["program"]
    operation = next(
        node
        for equation in program["equations"]
        for node in _nodes(equation["expression"])
        if node.get("kind") == "COMPOSE"
    )
    operation["kind"] = "UNION"
    with pytest.raises(ContrastiveOrbitError):
        validate_orbit(tampered)


def _nodes(expression: dict[str, object]) -> list[dict[str, object]]:
    output = [expression]
    if expression["kind"] == "CONVERSE":
        output.extend(_nodes(expression["child"]))
    elif expression["kind"] in {"UNION", "INTERSECTION", "COMPOSE"}:
        for child in expression["children"]:
            output.extend(_nodes(child))
    return output
