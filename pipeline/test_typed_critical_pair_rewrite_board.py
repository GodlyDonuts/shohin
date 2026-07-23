from __future__ import annotations

import json

import pytest

import typed_critical_pair_rewrite_board as tcrr


REQUIRED_CLASSES = {
    "independent_redexes",
    "confluent_diamond",
    "nonconfluent_fork",
    "nested_redex_creation_removal",
    "repeated_variable_binding",
    "destructive_cancellation",
    "root_deletion",
    "counterfactual_rhs_pointer",
    "shared_occurrence_redexes",
    "repeated_rhs_pointer_sharing",
    "heterogeneous_valid_typing",
    "capacity_unblocking_order",
    "mixed_cyclic_terminating",
}


def _episodes() -> dict[str, tcrr.RewriteEpisode]:
    return {episode.name: episode for episode in tcrr.build_mechanics_board()}


def _production(episode: tcrr.RewriteEpisode) -> tcrr.OracleResult:
    return tcrr.ProductionRewriteStateOracle().enumerate(
        episode.system,
        episode.initial_graph,
    )


def _reference(
    episode: tcrr.RewriteEpisode,
) -> tcrr.ReferenceOracleResult:
    return tcrr.IndependentNestedReferenceOracle().enumerate(
        episode.system,
        episode.initial_graph,
    )


def _manual_state(
    capacity: int,
    nodes: list[dict[str, object]],
    root: int | None = 0,
) -> str:
    return json.dumps(
        {"capacity": capacity, "root": root, "nodes": nodes},
        sort_keys=True,
        separators=(",", ":"),
    )


def _constructor_record(
    type_id: str,
    constructor_id: str,
    children: list[int],
) -> dict[str, object]:
    return {
        "kind": "constructor",
        "type": type_id,
        "constructor": constructor_id,
        "children": children,
    }


def test_board_covers_every_required_class_with_opaque_ids() -> None:
    episodes = tcrr.build_mechanics_board()
    assert {episode.episode_class for episode in episodes} == REQUIRED_CLASSES
    assert len(episodes) == 14
    for episode in episodes:
        identifiers = [
            constructor.identifier for constructor in episode.system.constructors
        ]
        identifiers.extend(rule.identifier for rule in episode.system.rules)
        assert all(
            len(identifier) == 24
            and set(identifier) <= set("0123456789abcdef")
            for identifier in identifiers
        )
        assert episode.initial_graph.conservation_receipt()["conserved"]
        tcrr.validate_graph(episode.system, episode.initial_graph)


def test_independent_reference_matches_production_on_every_episode() -> None:
    for episode in tcrr.build_mechanics_board():
        production = _production(episode)
        reference = _reference(episode)
        assert production.normal_forms == reference.normal_forms, episode.name
        assert production.transitions == reference.transitions, episode.name
        assert production.cyclic_sccs == reference.cyclic_sccs, episode.name
        assert production.cyclic_states == reference.cyclic_states, episode.name
        assert production.states_explored == reference.states_explored
        assert production.transitions_explored == reference.transitions_explored


def test_reference_oracle_does_not_call_production_matcher_or_canonicalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = _episodes()["diamond"]
    expected = _reference(episode)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("reference oracle called production mechanics")

    monkeypatch.setattr(tcrr, "_match_pattern", forbidden)
    monkeypatch.setattr(tcrr, "canonical_graph_serialization", forbidden)
    assert _reference(episode) == expected


def test_independent_redex_state_graph_retains_both_orders() -> None:
    result = _production(_episodes()["independent"])
    assert len(result.normal_forms) == 1
    assert result.states_explored == 4
    assert result.transitions_explored == 4
    assert result.cyclic_sccs == ()
    terminal = result.normal_forms[0]
    incoming = [edge for edge in result.transitions if edge.target == terminal]
    assert len(incoming) == 2
    assert len({edge.source for edge in incoming}) == 2


def test_diamond_joins_and_fork_preserves_two_exact_terminal_payloads() -> None:
    episodes = _episodes()
    diamond = _production(episodes["diamond"])
    fork_episode = episodes["fork"]
    fork = _production(fork_episode)
    assert len(diamond.normal_forms) == 1
    assert len(fork.normal_forms) == 2

    root = fork_episode.initial_graph.node_map()[fork_episode.initial_graph.root]
    lhs_constructor = root.constructor_id
    rhs_constructors = {
        rule.rhs.constructor_id
        for rule in fork_episode.system.rules
        if isinstance(rule.rhs, tcrr.RhsConstructor)
    }
    assert lhs_constructor not in rhs_constructors
    expected = {
        _manual_state(
            fork_episode.initial_graph.capacity,
            [
                _constructor_record(
                    root.type_id,
                    constructor_id,
                    [],
                )
            ],
        )
        for constructor_id in rhs_constructors
    }
    assert set(fork.normal_forms) == expected


def test_shared_dag_child_rewrites_one_occurrence_not_all_aliases() -> None:
    episode = _episodes()["shared_occurrence"]
    graph = episode.initial_graph
    root = graph.node_map()[graph.root]
    assert root.children[0] == root.children[1]
    reductions = tcrr.legal_reductions(episode.system, graph)
    assert {reduction.target_path for reduction in reductions} == {(0,), (1,)}

    redex = graph.node_map()[root.children[0]]
    normal = episode.system.rules[0].rhs
    assert isinstance(normal, tcrr.RhsConstructor)
    expected_by_path = {
        (0,): _manual_state(
            graph.capacity,
            [
                _constructor_record(
                    root.type_id,
                    str(root.constructor_id),
                    [1, 2],
                ),
                _constructor_record(
                    redex.type_id,
                    normal.constructor_id,
                    [],
                ),
                _constructor_record(
                    redex.type_id,
                    str(redex.constructor_id),
                    [],
                ),
            ],
        ),
        (1,): _manual_state(
            graph.capacity,
            [
                _constructor_record(
                    root.type_id,
                    str(root.constructor_id),
                    [1, 2],
                ),
                _constructor_record(
                    redex.type_id,
                    str(redex.constructor_id),
                    [],
                ),
                _constructor_record(
                    redex.type_id,
                    normal.constructor_id,
                    [],
                ),
            ],
        ),
    }
    for reduction in reductions:
        successor = tcrr.apply_reduction(
            episode.system,
            graph,
            reduction,
        )
        assert tcrr.canonical_graph_serialization(successor) == (
            expected_by_path[reduction.target_path]
        )


def test_repeated_rhs_pointer_preserves_exact_dag_sharing() -> None:
    episode = _episodes()["repeated_rhs_pointer"]
    reduction = tcrr.legal_reductions(
        episode.system,
        episode.initial_graph,
    )[0]
    successor = tcrr.apply_reduction(
        episode.system,
        episode.initial_graph,
        reduction,
    )
    root = successor.node_map()[successor.root]
    assert successor.occupied_count == 2
    assert root.children == (1, 1)
    payload = successor.node_map()[1]
    rhs = episode.system.rules[0].rhs
    assert isinstance(rhs, tcrr.RhsConstructor)
    expected = _manual_state(
        successor.capacity,
        [
            _constructor_record(
                root.type_id,
                rhs.constructor_id,
                [1, 1],
            ),
            _constructor_record(
                payload.type_id,
                str(payload.constructor_id),
                [],
            ),
        ],
    )
    assert tcrr.canonical_graph_serialization(successor) == expected


def test_repeated_variable_binding_kills_slot_identity_mutant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = _episodes()["repeated_variable"]
    correct = _production(episode)
    reference = _reference(episode)
    assert correct.normal_forms == reference.normal_forms
    assert correct.normal_form_graphs[0].occupied_count == 5

    monkeypatch.setattr(
        tcrr,
        "_binding_equal",
        lambda _graph, left, right: left == right,
    )
    identity_mutant = _production(episode)
    assert identity_mutant.normal_forms != reference.normal_forms
    assert identity_mutant.normal_form_graphs[0].occupied_count == 7


def test_greedy_only_enumerator_is_killed_by_exact_fork_payload_set() -> None:
    episode = _episodes()["fork"]
    exhaustive = _production(episode)
    greedy = tcrr.greedy_normal_form(episode.system, episode.initial_graph)
    greedy_set = {tcrr.canonical_graph_serialization(greedy)}
    assert len(exhaustive.normal_forms) == 2
    assert greedy_set < set(exhaustive.normal_forms)


def test_nonconfluent_branch_collapse_mutant_loses_nested_terminal() -> None:
    episode = _episodes()["nested"]
    exhaustive = _production(episode)

    def first_transition_only(
        system: tcrr.RewriteSystem,
        graph: tcrr.TermGraph,
    ) -> set[str]:
        active = graph
        seen = set()
        while True:
            key = tcrr.canonical_graph_serialization(active)
            if key in seen:
                return set()
            seen.add(key)
            reductions = tcrr.legal_reductions(system, active)
            if not reductions:
                return {key}
            active = tcrr.apply_reduction(system, active, reductions[0])

    collapsed = first_transition_only(episode.system, episode.initial_graph)
    assert len(exhaustive.normal_forms) == 2
    assert collapsed < set(exhaustive.normal_forms)


def test_storage_order_mutant_changes_bytes_but_both_oracles_do_not() -> None:
    episode = _episodes()["repeated_variable"]
    graph = episode.initial_graph
    permuted = tcrr.reindex_graph(
        graph,
        tuple(reversed(range(graph.capacity))),
    )

    def physical_slot_mutant(value: tcrr.TermGraph) -> str:
        return json.dumps(
            [
                (node.slot, node.constructor_id, node.children)
                for node in value.nodes
            ],
            separators=(",", ":"),
        )

    assert physical_slot_mutant(graph) != physical_slot_mutant(permuted)
    assert tcrr.canonical_graph_serialization(graph) == (
        tcrr.canonical_graph_serialization(permuted)
    )
    production = tcrr.ProductionRewriteStateOracle().enumerate(
        episode.system,
        permuted,
    )
    reference = tcrr.IndependentNestedReferenceOracle().enumerate(
        episode.system,
        permuted,
    )
    assert production.normal_forms == _production(episode).normal_forms
    assert reference.normal_forms == _reference(episode).normal_forms


def test_capacity_is_part_of_canonical_state_identity() -> None:
    empty_three = tcrr.TermGraph(3, None, ())
    empty_four = tcrr.TermGraph(4, None, ())
    assert tcrr.canonical_graph_serialization(empty_three) == _manual_state(
        3,
        [],
        root=None,
    )
    assert tcrr.canonical_graph_serialization(empty_four) == _manual_state(
        4,
        [],
        root=None,
    )
    assert tcrr.canonical_graph_serialization(empty_three) != (
        tcrr.canonical_graph_serialization(empty_four)
    )


def test_alpha_renaming_is_invariant_and_capture_fails_closed() -> None:
    rule = _episodes()["counterfactual_forward"].system.rules[0]
    swapped = tcrr.alpha_rename_rule(
        rule,
        {"left_pointer": "right_pointer", "right_pointer": "left_pointer"},
    )
    assert tcrr.canonical_rule_payload(rule) == tcrr.canonical_rule_payload(
        swapped
    )
    with pytest.raises(tcrr.RewriteMechanicsError, match="injective"):
        tcrr.alpha_rename_rule(
            rule,
            {"left_pointer": "captured", "right_pointer": "captured"},
        )
    with pytest.raises(tcrr.RewriteMechanicsError, match="injective"):
        tcrr.alpha_rename_rule(rule, {"left_pointer": "right_pointer"})
    with pytest.raises(tcrr.RewriteMechanicsError, match="unbound"):
        tcrr.alpha_rename_rule(rule, {"absent": "new"})


def test_cancellation_reclaims_exact_slot_and_redirects_parent_pointer() -> None:
    episode = _episodes()["destructive"]
    initial = episode.initial_graph
    reduction = tcrr.legal_reductions(episode.system, initial)[0]
    terminal = tcrr.apply_reduction(episode.system, initial, reduction)
    terminal_nodes = terminal.node_map()
    assert initial.occupied_count == 3
    assert terminal.occupied_count == 2
    assert tuple(sorted(terminal_nodes)) == (0, 1)
    assert terminal_nodes[0].children == (1,)
    assert set(range(terminal.capacity)) - set(terminal_nodes) == {2, 3, 4, 5}
    assert terminal.conservation_receipt() == {
        "capacity": 6,
        "occupied": 2,
        "free": 4,
        "conserved": True,
    }


def test_injected_no_deletion_mutant_is_killed_by_exact_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = _episodes()["destructive"]
    reference = _reference(episode)
    original = tcrr._rewrite_occurrence

    def no_deletion_mutant(
        system: tcrr.RewriteSystem,
        graph: tcrr.TermGraph,
        target_slot: int,
        target_path: tuple[int, ...],
        rule: tcrr.RewriteRule,
        bindings: dict[str, int],
    ) -> tcrr.TermGraph:
        if isinstance(rule.rhs, tcrr.RhsVariable):
            return graph
        return original(
            system,
            graph,
            target_slot,
            target_path,
            rule,
            bindings,
        )

    monkeypatch.setattr(tcrr, "_rewrite_occurrence", no_deletion_mutant)
    mutant = _production(episode)
    assert mutant.normal_forms != reference.normal_forms
    assert mutant.normal_forms == (
        tcrr.canonical_graph_serialization(episode.initial_graph),
    )


def test_root_deletion_is_separate_and_returns_every_slot() -> None:
    episode = _episodes()["deletion"]
    assert episode.episode_class == "root_deletion"
    result = _production(episode)
    assert result.normal_forms == (_manual_state(3, [], root=None),)
    terminal = result.normal_form_graphs[0]
    assert terminal.root is None
    assert terminal.conservation_receipt() == {
        "capacity": 3,
        "occupied": 0,
        "free": 3,
        "conserved": True,
    }


def test_injected_reversed_rhs_pointer_mutant_becomes_exact_twin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episodes = _episodes()
    forward = episodes["counterfactual_forward"]
    reverse = episodes["counterfactual_reverse"]
    correct = _production(forward)
    reverse_reference = _reference(reverse)
    forward_rule = forward.system.rules[0]
    reverse_rule = reverse.system.rules[0]
    assert tcrr.rule_shape_payload(forward_rule) == tcrr.rule_shape_payload(
        reverse_rule
    )
    assert correct.normal_forms != reverse_reference.normal_forms

    monkeypatch.setattr(
        tcrr,
        "_production_rhs_children",
        lambda expression: tuple(reversed(expression.children)),
    )
    mutant = _production(forward)
    assert mutant.normal_forms == reverse_reference.normal_forms
    assert mutant.normal_forms != correct.normal_forms


def test_heterogeneous_valid_typing_preserves_exact_edge_types() -> None:
    episode = _episodes()["heterogeneous_typing"]
    initial_types = {node.type_id for node in episode.initial_graph.nodes}
    assert len(initial_types) == 2
    result = _production(episode)
    terminal = result.normal_form_graphs[0]
    tcrr.validate_graph(episode.system, terminal)
    root = terminal.node_map()[terminal.root]
    child = terminal.node_map()[root.children[0]]
    root_spec = episode.system.constructor_map()[str(root.constructor_id)]
    assert root.type_id == root_spec.result_type
    assert (child.type_id,) == root_spec.argument_types
    assert root.type_id != child.type_id


def test_capacity_unblocking_forces_reclamation_before_growth() -> None:
    episode = _episodes()["capacity_unblocking"]
    initial = episode.initial_graph
    reductions = tcrr.legal_reductions(episode.system, initial)
    assert len(reductions) == 1
    assert reductions[0].target_path == (0,)
    reclaimed = tcrr.apply_reduction(episode.system, initial, reductions[0])
    assert reclaimed.occupied_count == 3
    growth = tcrr.legal_reductions(episode.system, reclaimed)
    assert len(growth) == 1
    assert growth[0].target_path == (1,)
    terminal = tcrr.apply_reduction(episode.system, reclaimed, growth[0])
    assert terminal.occupied_count == terminal.capacity == 4
    assert _production(episode).normal_forms == (
        tcrr.canonical_graph_serialization(terminal),
    )


def test_mixed_cycle_reports_scc_and_still_collects_normal_form() -> None:
    episode = _episodes()["mixed_cyclic"]
    result = _production(episode)
    reference = _reference(episode)
    assert len(result.normal_forms) == 1
    assert len(result.traces) == 1
    assert len(result.traces[0].steps) == 1
    assert len(result.cyclic_sccs) == 1
    assert len(result.cyclic_sccs[0]) == 2
    assert set(result.cyclic_states) == set(result.cyclic_sccs[0])
    assert result.cyclic_sccs == reference.cyclic_sccs
    assert result.normal_forms == reference.normal_forms


def test_every_production_transition_conserves_fixed_reservoir() -> None:
    for episode in tcrr.build_mechanics_board():
        frontier = [episode.initial_graph]
        visited = set()
        while frontier:
            graph = frontier.pop()
            key = tcrr.canonical_graph_serialization(graph)
            if key in visited:
                continue
            visited.add(key)
            receipt = graph.conservation_receipt()
            assert receipt["conserved"]
            assert receipt["occupied"] + receipt["free"] == receipt["capacity"]
            for reduction in tcrr.legal_reductions(episode.system, graph):
                successor = tcrr.apply_reduction(
                    episode.system,
                    graph,
                    reduction,
                )
                assert successor.capacity == graph.capacity
                assert successor.conservation_receipt()["conserved"]
                frontier.append(successor)


def test_nonroot_deletion_and_invalid_type_fail_closed() -> None:
    left = tcrr.ConstructorSpec("1" * 24, "left")
    right = tcrr.ConstructorSpec("2" * 24, "right")
    with pytest.raises(tcrr.RewriteMechanicsError, match="preserve"):
        tcrr.RewriteSystem(
            (left, right),
            (
                tcrr.RewriteRule(
                    "3" * 24,
                    tcrr.PatternConstructor(left.identifier),
                    tcrr.RhsConstructor(right.identifier),
                ),
            ),
        )

    episode = _episodes()["destructive"]
    deletion_rule = tcrr.RewriteRule(
        "4" * 24,
        episode.system.rules[0].lhs,
        None,
    )
    deletion_system = tcrr.RewriteSystem(
        episode.system.constructors,
        (deletion_rule,),
    )
    assert tcrr.legal_reductions(
        deletion_system,
        episode.initial_graph,
    ) == ()

