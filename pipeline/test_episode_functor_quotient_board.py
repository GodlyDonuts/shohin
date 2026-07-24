from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline.episode_functor_quotient_board import (
    ACTION_NAMES,
    DEFAULT_FIXTURE_SEED,
    EXHAUSTIVE_DEPTH,
    OBSERVER_NAMES,
    PHYSICAL_STATE_COUNT,
    QUOTIENT_SIZES,
    QuotientBoardError,
    action_intervention,
    audit_machine,
    build_cpu_fixture,
    consume_cpu_fixture,
    empty_merge_future_separator,
    equivalence_relation,
    equivalent_word_witness,
    exhaustive_future_partition,
    fixture_digest,
    noncommuting_witness,
    observer_intervention,
    partition_refinement,
    product_automaton_partition,
    quotient_word_transform,
    shortest_separator,
    state_gauge_audit,
    structural_signature,
    words_through_depth,
)


@pytest.fixture(scope="module")
def cpu_fixture():
    return build_cpu_fixture()


def test_fixture_is_deterministic_cpu_only_and_has_required_shape(cpu_fixture) -> None:
    rebuilt = build_cpu_fixture(DEFAULT_FIXTURE_SEED)
    assert fixture_digest(rebuilt) == fixture_digest(cpu_fixture)
    assert cpu_fixture.status == "exploratory_cpu_only_not_official_not_sealed"
    assert (
        tuple(machine.quotient_size for machine in cpu_fixture.train) == QUOTIENT_SIZES
    )
    assert (
        tuple(machine.quotient_size for machine in cpu_fixture.development)
        == QUOTIENT_SIZES
    )
    for machine in (*cpu_fixture.train, *cpu_fixture.development):
        assert len(machine.physical_keys) == PHYSICAL_STATE_COUNT
        assert machine.action_names == ACTION_NAMES
        assert machine.observer_names == OBSERVER_NAMES
        assert len(machine.key_to_quotient_class) == PHYSICAL_STATE_COUNT
        for observer in machine.observations:
            assert 1 < len(set(observer)) < PHYSICAL_STATE_COUNT


def test_three_independent_quotient_oracles_match_explicit_maps(cpu_fixture) -> None:
    for machine in (*cpu_fixture.train, *cpu_fixture.development):
        expected = equivalence_relation(machine.key_to_quotient_class)
        assert equivalence_relation(partition_refinement(machine)) == expected
        assert equivalence_relation(product_automaton_partition(machine)) == expected
        assert (
            equivalence_relation(
                exhaustive_future_partition(
                    machine,
                    maximum_depth=EXHAUSTIVE_DEPTH,
                )
            )
            == expected
        )


def test_shortest_separators_are_minimal_and_cover_future_only_merges(
    cpu_fixture,
) -> None:
    all_shorter_words = {depth: words_through_depth(depth - 1) for depth in range(1, 8)}
    for machine in (*cpu_fixture.train, *cpu_fixture.development):
        witness = empty_merge_future_separator(machine)
        left = witness["left_state"]
        right = witness["right_state"]
        separator = witness["separator"]
        assert machine.observe(left) == machine.observe(right)
        assert (
            machine.key_to_quotient_class[left]
            != (machine.key_to_quotient_class[right])
        )
        assert separator
        assert machine.response(left, separator) != machine.response(right, separator)
        assert all(
            machine.response(left, word) == machine.response(right, word)
            for word in all_shorter_words[len(separator)]
        )

        for pair_left in range(PHYSICAL_STATE_COUNT):
            for pair_right in range(pair_left + 1, PHYSICAL_STATE_COUNT):
                word = shortest_separator(machine, pair_left, pair_right)
                same_class = (
                    machine.key_to_quotient_class[pair_left]
                    == machine.key_to_quotient_class[pair_right]
                )
                assert (word is None) == same_class
                if word:
                    assert all(
                        machine.response(pair_left, shorter)
                        == machine.response(pair_right, shorter)
                        for shorter in all_shorter_words[len(word)]
                    )


def test_state_gauge_conjugacy_preserves_behavior_classes_and_structure(
    cpu_fixture,
) -> None:
    for machine in (*cpu_fixture.train, *cpu_fixture.development):
        report = state_gauge_audit(machine)
        assert report["behavior_preserved"]
        assert report["class_transport_preserved"]
        assert report["structural_signature_preserved"]
        assert sorted(report["old_to_new"]) == list(range(PHYSICAL_STATE_COUNT))


def test_observer_and_action_interventions_are_isolated_and_causal(
    cpu_fixture,
) -> None:
    for machine in (*cpu_fixture.train, *cpu_fixture.development):
        observed, observer_receipt = observer_intervention(machine)
        assert observed.transitions == machine.transitions
        assert observed.key_to_quotient_class == machine.key_to_quotient_class
        changed_states = set(observer_receipt["changed_states"])
        changed_cells = {
            state
            for state in range(PHYSICAL_STATE_COUNT)
            if observed.observations[observer_receipt["observer"]][state]
            != machine.observations[observer_receipt["observer"]][state]
        }
        assert changed_cells == changed_states
        assert (
            shortest_separator(
                observed,
                observer_receipt["probe_left"],
                observer_receipt["probe_right"],
            )
            == ()
        )

        acted, action_receipt = action_intervention(machine)
        assert acted.observations == machine.observations
        assert acted.key_to_quotient_class == machine.key_to_quotient_class
        action = action_receipt["action"]
        changed_rows = {
            state
            for state in range(PHYSICAL_STATE_COUNT)
            if acted.transitions[action][state] != machine.transitions[action][state]
        }
        assert changed_rows == set(action_receipt["changed_states"])
        probe = action_receipt["probe_state"]
        assert acted.response(probe, (action,)) != machine.response(probe, (action,))


def test_equivalent_words_and_noncommuting_orders_have_explicit_witnesses(
    cpu_fixture,
) -> None:
    for machine in (*cpu_fixture.train, *cpu_fixture.development):
        equivalent = equivalent_word_witness(machine)
        assert equivalent["left"] != equivalent["right"]
        assert quotient_word_transform(machine, equivalent["left"]) == (
            quotient_word_transform(machine, equivalent["right"])
        )

        noncommuting = noncommuting_witness(machine)
        assert noncommuting["left_word"] != noncommuting["right_word"]
        assert noncommuting["left_class"] != noncommuting["right_class"]
        start = machine.key_to_quotient_class.index(noncommuting["start_class"])
        left_response = machine.response(
            start,
            (*noncommuting["left_word"], *noncommuting["distinguishing_suffix"]),
        )
        right_response = machine.response(
            start,
            (*noncommuting["right_word"], *noncommuting["distinguishing_suffix"]),
        )
        assert left_response != right_response


def test_consumed_fixture_has_zero_train_development_structural_overlap(
    cpu_fixture,
) -> None:
    report = consume_cpu_fixture(cpu_fixture)
    assert report["train_development_structural_signature_overlap"] == []
    assert all(report["gates"].values())
    assert report["quotient_coverage"] == {
        "train": list(QUOTIENT_SIZES),
        "development": list(QUOTIENT_SIZES),
    }
    train_signatures = {structural_signature(machine) for machine in cpu_fixture.train}
    development_signatures = {
        structural_signature(machine) for machine in cpu_fixture.development
    }
    assert train_signatures.isdisjoint(development_signatures)
    assert report["claims_excluded"] == (
        "neural_fit",
        "model_capability",
        "gpu_execution",
        "official_sealed_board",
        "pretraining",
    )


def test_auditor_rejects_a_tampered_explicit_class_map(cpu_fixture) -> None:
    machine = cpu_fixture.train[0]
    tampered_map = list(machine.key_to_quotient_class)
    source = 0
    tampered_map[source] = (tampered_map[source] + 1) % machine.quotient_size
    with pytest.raises(QuotientBoardError):
        replace(machine, key_to_quotient_class=tuple(tampered_map))


def test_per_machine_audit_exposes_all_required_evidence(cpu_fixture) -> None:
    for machine in (*cpu_fixture.train, *cpu_fixture.development):
        report = audit_machine(machine)
        assert report["quotient_size"] in QUOTIENT_SIZES
        assert len(report["key_to_quotient_class"]) == PHYSICAL_STATE_COUNT
        assert report["oracle_agreement"]
        assert report["shortest_separators"]
        assert report["empty_merge_future_separator"]["separator"]
        assert report["observer_intervention"]["isolated_and_causal"]
        assert report["action_intervention"]["isolated_and_causal"]
