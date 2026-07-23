from __future__ import annotations

import dataclasses
import json
from collections import Counter

import pytest

import neural_tcrr_board as board


def _built() -> board.LocalTransitionSlice:
    return board.build_local_transition_slice()


def _expected_by_digest(
    value: board.LocalTransitionSlice,
) -> dict[str, board.ExpectedTransitionRecord]:
    return {item.packet_sha256: item for item in value.expected_records}


def _fingerprints_by_digest(
    value: board.LocalTransitionSlice,
) -> dict[str, board.PacketFingerprints]:
    return {item.packet_sha256: item for item in value.fingerprints}


def _packet_by_digest(
    value: board.LocalTransitionSlice,
) -> dict[str, board.SourceDeletedPacket]:
    return {board.packet_sha256(item): item for item in value.packets}


def _twin(
    value: board.LocalTransitionSlice,
    kind: str,
) -> board.CausalTwinRecord:
    return next(item for item in value.twins if item.kind == kind)


def _successor_semantic_digest(
    packet: board.SourceDeletedPacket,
    action: board.ExpectedTransition,
) -> str:
    successor_packet = dataclasses.replace(packet, graph=action.successor)
    return board.packet_fingerprints(successor_packet).isomorphic_sha256


def _all_opaque_ids(packet: board.SourceDeletedPacket) -> set[str]:
    output = set(packet.graph.reservoir)
    for constructor in packet.constructors:
        output.add(constructor.identifier)
        output.add(constructor.result_type)
        output.update(constructor.argument_types)

    def collect_term(term: board.RuleTermRecord | None) -> None:
        if term is None:
            return
        if term.constructor_id is not None:
            output.add(term.constructor_id)
        if term.variable_id is not None:
            output.add(term.variable_id)
        output.add(term.type_id)
        for child in term.children:
            collect_term(child)

    for rule in packet.rules:
        output.add(rule.identifier)
        collect_term(rule.lhs)
        collect_term(rule.rhs)
    for node in packet.graph.nodes:
        output.add(node.storage_id)
        output.add(node.type_id)
        if node.constructor_id is not None:
            output.add(node.constructor_id)
        if node.variable_id is not None:
            output.add(node.variable_id)
    return output


def _rule_marginals(rule: board.RuleRecord) -> Counter[tuple[object, ...]]:
    output: Counter[tuple[object, ...]] = Counter()

    def visit(term: board.RuleTermRecord | None, side: str) -> None:
        if term is None:
            output[(side, "delete")] += 1
            return
        output[(side, term.kind, len(term.children))] += 1
        for child in term.children:
            visit(child, side)

    visit(rule.lhs, "lhs")
    visit(rule.rhs, "rhs")
    return output


def test_slice_is_deterministic_and_has_bounded_partitions() -> None:
    first = _built()
    second = _built()
    assert first == second
    assert len(first.packets) == 21
    counts = Counter(item.partition for item in first.split_assignments)
    assert counts == {
        "local_transition_train": 15,
        "local_transition_development": 6,
    }
    assert {item.kind for item in first.twins} == {
        "rhs_pointer",
        "shared_occurrence",
        "capacity",
        "storage_reindex",
        "rule_reindex",
    }


def test_every_identity_namespace_is_fresh_across_packets() -> None:
    value = _built()
    seen: set[str] = set()
    for packet in value.packets:
        identifiers = _all_opaque_ids(packet)
        assert all(
            len(identifier) == board.OPAQUE_ID_LENGTH
            and set(identifier) <= set("0123456789abcdef")
            for identifier in identifiers
        )
        assert not (seen & identifiers)
        seen.update(identifiers)


def test_model_packet_schema_has_no_label_or_generation_metadata() -> None:
    value = _built()
    forbidden = {
        "answer",
        "class",
        "expected",
        "family",
        "label",
        "legal",
        "mask",
        "oracle",
        "schedule",
        "seed",
        "source",
        "split",
        "target",
        "trace",
    }
    assert {item.name for item in dataclasses.fields(board.SourceDeletedPacket)} == {
        "constructors",
        "rules",
        "graph",
    }
    for packet in value.packets:
        payload = json.loads(board.serialize_model_packet(packet))

        def scan(item: object) -> None:
            if isinstance(item, dict):
                for key, nested in item.items():
                    normalized = key.lower().replace("-", "_")
                    assert not any(part in normalized for part in forbidden)
                    scan(nested)
            elif isinstance(item, list):
                for nested in item:
                    scan(nested)

        scan(payload)


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "family",
        "episode_class",
        "source_text",
        "oracle_state",
        "target_graph",
        "answer_schedule",
        "legal_action_mask",
    ],
)
def test_serialization_validator_kills_forbidden_field_mutations(
    forbidden_key: str,
) -> None:
    packet = _built().packets[0]
    payload = dataclasses.asdict(packet)
    payload["graph"][forbidden_key] = "leak"
    with pytest.raises(board.NeuralTcrrBoardError, match="forbidden"):
        board.validate_model_packet_payload(payload)


def test_expected_records_are_physically_separate_and_join_by_digest() -> None:
    value = _built()
    packets = _packet_by_digest(value)
    expected = _expected_by_digest(value)
    assignments = {
        item.packet_sha256: item.partition for item in value.split_assignments
    }
    fingerprints = _fingerprints_by_digest(value)
    assert set(packets) == set(expected) == set(assignments) == set(fingerprints)
    for digest, packet in packets.items():
        serialized = board.serialize_model_packet(packet)
        assert digest not in serialized
        assert "occurrence_path" not in serialized
        assert "successor" not in serialized
        assert expected[digest].packet_sha256 == digest


def test_offline_oracle_mutation_changes_labels_not_packet_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = board._replace_example(2026072991, depth=2)
    packet = board._packet_from_mechanics(
        generated.system,
        generated.graph,
        storage_ids=generated.storage_ids,
        packet_seed=generated.packet_seed,
    )
    original_bytes = board.serialize_model_packet(packet)
    original_expected = board._expected_record(
        generated.system,
        generated.graph,
        packet,
        generated.storage_ids,
        packet_seed=generated.packet_seed,
    )
    assert len(original_expected.transitions) == 1
    monkeypatch.setattr(
        board.mechanics,
        "legal_reductions",
        lambda _system, _graph: (),
    )
    mutated_expected = board._expected_record(
        generated.system,
        generated.graph,
        packet,
        generated.storage_ids,
        packet_seed=generated.packet_seed,
    )
    assert mutated_expected.transitions == ()
    assert board.serialize_model_packet(packet) == original_bytes


def test_split_fingerprints_have_no_exact_isomorphic_or_window_leakage() -> None:
    value = _built()
    board.validate_split_isolation(
        value.split_assignments,
        value.fingerprints,
    )
    by_partition: dict[str, list[board.PacketFingerprints]] = {
        "local_transition_train": [],
        "local_transition_development": [],
    }
    fingerprint_map = _fingerprints_by_digest(value)
    for assignment in value.split_assignments:
        by_partition[assignment.partition].append(
            fingerprint_map[assignment.packet_sha256]
        )
    train = by_partition["local_transition_train"]
    development = by_partition["local_transition_development"]
    assert not (
        {item.exact_sha256 for item in train}
        & {item.exact_sha256 for item in development}
    )
    assert not (
        {item.isomorphic_sha256 for item in train}
        & {item.isomorphic_sha256 for item in development}
    )
    assert not (
        {
            window
            for item in train
            for window in item.normalized_rule_windows
        }
        & {
            window
            for item in development
            for window in item.normalized_rule_windows
        }
    )


def test_cross_split_reindex_mutant_is_rejected() -> None:
    value = _built()
    twin = _twin(value, "storage_reindex")
    assignments = []
    for item in value.split_assignments:
        partition = item.partition
        if item.packet_sha256 == twin.right_packet_sha256:
            partition = "local_transition_development"
        assignments.append(
            board.SplitAssignment(item.packet_sha256, partition)
        )
    with pytest.raises(board.NeuralTcrrBoardError, match="cross-split"):
        board.validate_split_isolation(
            tuple(assignments),
            value.fingerprints,
        )


def test_rhs_pointer_twins_match_marginals_but_change_exact_successor() -> None:
    value = _built()
    twin = _twin(value, "rhs_pointer")
    packets = _packet_by_digest(value)
    expected = _expected_by_digest(value)
    left_packet = packets[twin.left_packet_sha256]
    right_packet = packets[twin.right_packet_sha256]
    left_rule = left_packet.rules[0]
    right_rule = right_packet.rules[0]
    assert _rule_marginals(left_rule) == _rule_marginals(right_rule)
    assert (
        _successor_semantic_digest(
            left_packet,
            expected[twin.left_packet_sha256].transitions[0],
        )
        != _successor_semantic_digest(
            right_packet,
            expected[twin.right_packet_sha256].transitions[0],
        )
    )
    fingerprints = _fingerprints_by_digest(value)
    assert (
        fingerprints[twin.left_packet_sha256].isomorphic_sha256
        != fingerprints[twin.right_packet_sha256].isomorphic_sha256
    )


def test_shared_occurrence_twin_uses_one_slot_two_paths_two_successors() -> None:
    value = _built()
    twin = _twin(value, "shared_occurrence")
    packets = _packet_by_digest(value)
    expected = _expected_by_digest(value)
    assert twin.left_packet_sha256 == twin.right_packet_sha256
    packet = packets[twin.left_packet_sha256]
    actions = expected[twin.left_packet_sha256].transitions
    left = actions[twin.left_transition_index]
    right = actions[twin.right_transition_index]
    assert left.target_storage_id == right.target_storage_id
    assert {left.occurrence_path, right.occurrence_path} == {(0,), (1,)}
    assert _successor_semantic_digest(packet, left) != (
        _successor_semantic_digest(packet, right)
    )


def test_capacity_twin_preserves_window_but_blocks_growth_at_fifteen() -> None:
    value = _built()
    twin = _twin(value, "capacity")
    packets = _packet_by_digest(value)
    expected = _expected_by_digest(value)
    fingerprints = _fingerprints_by_digest(value)
    left_packet = packets[twin.left_packet_sha256]
    right_packet = packets[twin.right_packet_sha256]
    assert {
        len(left_packet.graph.reservoir),
        len(right_packet.graph.reservoir),
    } == {15, 16}
    action_counts = {
        len(expected[twin.left_packet_sha256].transitions),
        len(expected[twin.right_packet_sha256].transitions),
    }
    assert action_counts == {0, 1}
    assert (
        fingerprints[twin.left_packet_sha256].normalized_rule_windows
        == fingerprints[twin.right_packet_sha256].normalized_rule_windows
    )
    assert (
        fingerprints[twin.left_packet_sha256].isomorphic_sha256
        != fingerprints[twin.right_packet_sha256].isomorphic_sha256
    )


@pytest.mark.parametrize("kind", ["storage_reindex", "rule_reindex"])
def test_reindex_twins_change_bytes_but_preserve_semantics(kind: str) -> None:
    value = _built()
    twin = _twin(value, kind)
    packets = _packet_by_digest(value)
    expected = _expected_by_digest(value)
    fingerprints = _fingerprints_by_digest(value)
    left = fingerprints[twin.left_packet_sha256]
    right = fingerprints[twin.right_packet_sha256]
    assert left.exact_sha256 != right.exact_sha256
    assert left.isomorphic_sha256 == right.isomorphic_sha256
    assert left.normalized_rule_windows == right.normalized_rule_windows
    left_successors = {
        _successor_semantic_digest(
            packets[twin.left_packet_sha256],
            action,
        )
        for action in expected[twin.left_packet_sha256].transitions
    }
    right_successors = {
        _successor_semantic_digest(
            packets[twin.right_packet_sha256],
            action,
        )
        for action in expected[twin.right_packet_sha256].transitions
    }
    assert left_successors == right_successors


def test_inline_storage_and_rule_order_mutants_are_canonicalized() -> None:
    packet = _built().packets[10]
    original = board.packet_fingerprints(packet)
    graph_mutant = dataclasses.replace(
        packet.graph,
        reservoir=tuple(reversed(packet.graph.reservoir)),
        nodes=tuple(reversed(packet.graph.nodes)),
    )
    storage_mutant = dataclasses.replace(packet, graph=graph_mutant)
    storage_fingerprint = board.packet_fingerprints(storage_mutant)
    assert original.exact_sha256 != storage_fingerprint.exact_sha256
    assert original.isomorphic_sha256 == storage_fingerprint.isomorphic_sha256
    rule_mutant = dataclasses.replace(packet, rules=tuple(reversed(packet.rules)))
    rule_fingerprint = board.packet_fingerprints(rule_mutant)
    assert original.exact_sha256 != rule_fingerprint.exact_sha256
    assert original.isomorphic_sha256 == rule_fingerprint.isomorphic_sha256


def test_local_slice_contains_deletion_growth_reclamation_and_heterogeneous_types() -> None:
    value = _built()
    packets = _packet_by_digest(value)
    expected = _expected_by_digest(value)
    saw_deletion = False
    saw_growth = False
    saw_reclamation = False
    saw_heterogeneous = False
    for digest, packet in packets.items():
        occupied = len(packet.graph.nodes)
        type_count = {
            item.result_type for item in packet.constructors
        } | {
            type_id
            for item in packet.constructors
            for type_id in item.argument_types
        }
        saw_heterogeneous |= len(type_count) > 1
        for action in expected[digest].transitions:
            successor_occupied = len(action.successor.nodes)
            saw_deletion |= action.successor.root is None
            saw_growth |= successor_occupied > occupied
            saw_reclamation |= successor_occupied < occupied
            assert len(action.occurrence_path) <= board.MAX_PATH_DEPTH
    assert saw_deletion
    assert saw_growth
    assert saw_reclamation
    assert saw_heterogeneous


def test_packet_validation_kills_dangling_pointer_mutant() -> None:
    packet = _built().packets[0]
    root = next(
        node for node in packet.graph.nodes if node.storage_id == packet.graph.root
    )
    mutant_root = dataclasses.replace(root, children=("f" * 24,))
    mutant_nodes = tuple(
        mutant_root if node.storage_id == root.storage_id else node
        for node in packet.graph.nodes
    )
    mutant = dataclasses.replace(
        packet,
        graph=dataclasses.replace(packet.graph, nodes=mutant_nodes),
    )
    with pytest.raises(board.NeuralTcrrBoardError, match="free record"):
        board.validate_source_deleted_packet(mutant)


def test_packet_validation_kills_rhs_unbound_variable_mutant() -> None:
    packet = _built().packets[1]
    rule = packet.rules[0]
    assert rule.rhs is not None
    mutant_rhs = dataclasses.replace(rule.rhs, variable_id="f" * 24)
    mutant = dataclasses.replace(
        packet,
        rules=(dataclasses.replace(rule, rhs=mutant_rhs),),
    )
    with pytest.raises(board.NeuralTcrrBoardError, match="RHS variable"):
        board.validate_source_deleted_packet(mutant)


def test_board_validation_kills_missing_expected_record() -> None:
    value = _built()
    mutant = dataclasses.replace(
        value,
        expected_records=value.expected_records[:-1],
    )
    with pytest.raises(board.NeuralTcrrBoardError, match="one-to-one"):
        board.validate_local_transition_slice(mutant)
