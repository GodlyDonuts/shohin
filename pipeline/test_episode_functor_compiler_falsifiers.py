from __future__ import annotations

from dataclasses import replace
from itertools import permutations

from pipeline.episode_action_binding_board import (
    ACTION_COUNT,
    generate_cyclic_order_cluster,
    split_world_and_query,
)
from pipeline.episode_functor_compiler_falsifiers import (
    CategoricalEpisodeMachine,
    audit_interventions,
    canonical_query_universe,
    causal_quotient,
    compile_canonical_two_answer_cache,
    compile_leaky_two_answer_cache,
    execute_query_tokens,
    parse_world_machine,
    resource_receipt,
    shortest_separating_words,
)


def _machine_and_queries():
    cluster = generate_cyclic_order_cluster(20260723991, query_depth=6)
    left = cluster.primary.variants[0]
    right = cluster.reordered.variants[0]
    world, left_query = split_world_and_query(left.packet)
    right_world, right_query = split_world_and_query(right.packet)
    assert world == right_world
    return parse_world_machine(world), left, right, left_query, right_query


def test_categorical_machine_executes_both_hidden_query_orders() -> None:
    machine, left, right, left_query, right_query = _machine_and_queries()
    assert execute_query_tokens(machine, left_query) == left.target_token
    assert execute_query_tokens(machine, right_query) == right.target_token


def test_two_answer_cache_needs_hidden_query_identities() -> None:
    machine, left, right, left_query, right_query = _machine_and_queries()
    world_only = compile_canonical_two_answer_cache(machine)
    leaky = compile_leaky_two_answer_cache(
        machine,
        (left_query, right_query),
        (left.target_token, right.target_token),
    )
    assert world_only.lookup(left_query) is None
    assert world_only.lookup(right_query) is None
    assert leaky.lookup(left_query) == left.target_token
    assert leaky.lookup(right_query) == right.target_token


def test_current_query_support_is_8736_not_two() -> None:
    machine, *_ = _machine_and_queries()
    queries = canonical_query_universe(machine)
    assert len(queries) == 8 * sum(3**depth for depth in range(1, 7))
    assert len(queries) == 8_736
    assert len(set(queries)) == len(queries)


def test_resource_receipt_counts_state_keys_and_query_start() -> None:
    machine, *_ = _machine_and_queries()
    receipt = resource_receipt(machine)
    assert receipt.query_count == 8_736
    assert receipt.answer_bits_per_query == 3
    assert receipt.exhaustive_answer_bits == 26_208
    assert receipt.transition_bits == 72
    assert receipt.observer_bits == 24
    assert receipt.opaque_state_key_bits == 120
    assert receipt.opaque_action_key_bits == 45
    assert receipt.machine_semantic_bits == 261


def test_identity_observer_makes_current_eight_states_causally_distinct() -> None:
    machine, *_ = _machine_and_queries()
    classes = causal_quotient(machine.action_next, machine.observer_answer)
    assert len(set(classes)) == 8
    separators = shortest_separating_words(machine)
    assert len(separators) == 28
    assert max(map(len, separators.values())) == 0


def test_partition_refinement_finds_nontrivial_future_separator() -> None:
    machine = CategoricalEpisodeMachine(
        state_keys=(100, 101, 102),
        action_keys=(200,),
        action_next=((1, 2, 2),),
        observer_answer=((0, 0, 1),),
    )
    classes = causal_quotient(machine.action_next, machine.observer_answer)
    assert len(set(classes)) == 3
    separators = shortest_separating_words(machine)
    assert separators[(0, 1)] == (0,)


def test_key_operator_and_compensated_interventions_are_clean() -> None:
    machine, *_ = _machine_and_queries()
    report = audit_interventions(machine, maximum_depth=4)
    assert report["queries_checked"] == 8 * sum(
        ACTION_COUNT**depth for depth in range(1, 5)
    )
    assert report["key_intervention_nontrivial"]
    assert report["operator_intervention_nontrivial"]
    assert report["compensated_invariance"]

    for permutation in permutations(range(ACTION_COUNT)):
        compensated = machine.compensated_action_permutation(permutation)
        for query in canonical_query_universe(machine, maximum_depth=2):
            assert execute_query_tokens(compensated, query) == execute_query_tokens(
                machine,
                query,
            )


def test_one_row_transplant_changes_only_paths_reaching_that_row() -> None:
    machine, *_ = _machine_and_queries()
    action = 0
    state = 0
    original_destination = machine.action_next[action][state]
    donor_destination = next(
        candidate
        for candidate in range(machine.state_count)
        if candidate != original_destination
    )
    transplanted = machine.transplant_transition_row(
        action=action,
        state=state,
        destination=donor_destination,
    )
    assert machine.execute_indices(state, (action,)) == original_destination
    assert transplanted.execute_indices(state, (action,)) == donor_destination
    for other_state in range(1, machine.state_count):
        assert transplanted.execute_indices(
            other_state,
            (action,),
        ) == machine.execute_indices(other_state, (action,))


def test_missing_state_keys_cannot_bind_current_query_start() -> None:
    machine, _, _, left_query, _ = _machine_and_queries()
    broken = replace(machine, state_keys=tuple(range(machine.state_count)))
    try:
        execute_query_tokens(broken, left_query)
    except ValueError as exc:
        assert "unknown opaque key" in str(exc)
    else:
        raise AssertionError("query start unexpectedly bound without retained state keys")
