from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from functools import lru_cache
from pathlib import Path
import stat

import pytest

import neural_tcrr_board as board


@lru_cache(maxsize=1)
def _built() -> board.LocalTransitionSlice:
    return board.build_local_transition_slice()


def _packets(
    value: board.LocalTransitionSlice,
) -> dict[str, board.SourceDeletedPacket]:
    return {board.packet_sha256(item): item for item in value.packets}


def _expected(
    value: board.LocalTransitionSlice,
) -> dict[str, board.ExpectedTransitionRecord]:
    return {item.packet_sha256: item for item in value.expected_records}


def _fingerprints(
    value: board.LocalTransitionSlice,
) -> dict[str, board.PacketFingerprints]:
    return {item.packet_sha256: item for item in value.fingerprints}


def _twin(
    value: board.LocalTransitionSlice,
    kind: str,
) -> board.CausalTwinRecord:
    return next(item for item in value.twins if item.kind == kind)


def _all_identifiers(packet: board.SourceDeletedPacket) -> set[str]:
    output = set(packet.graph.reservoir)
    for constructor in packet.constructors:
        output.add(constructor.identifier)
        output.add(constructor.result_type)
        output.update(constructor.argument_types)

    def collect(term: board.RuleTermRecord | None) -> None:
        if term is None:
            return
        output.add(term.type_id)
        if term.constructor_id is not None:
            output.add(term.constructor_id)
        if term.variable_id is not None:
            output.add(term.variable_id)
        for child in term.children:
            collect(child)

    for rule in packet.rules:
        output.add(rule.identifier)
        collect(rule.lhs)
        collect(rule.rhs)
    for node in packet.graph.nodes:
        output.add(node.storage_id)
        output.add(node.type_id)
        if node.constructor_id is not None:
            output.add(node.constructor_id)
        if node.variable_id is not None:
            output.add(node.variable_id)
    return output


def _namespace_identifiers(
    packet: board.SourceDeletedPacket,
) -> dict[str, set[str]]:
    variables = set()

    def collect(term: board.RuleTermRecord | None) -> None:
        if term is None:
            return
        if term.variable_id is not None:
            variables.add(term.variable_id)
        for child in term.children:
            collect(child)

    for rule in packet.rules:
        collect(rule.lhs)
        collect(rule.rhs)
    variables.update(
        node.variable_id for node in packet.graph.nodes if node.variable_id is not None
    )
    return {
        "constructor": {item.identifier for item in packet.constructors},
        "type": {
            type_id
            for item in packet.constructors
            for type_id in (item.result_type, *item.argument_types)
        },
        "rule": {item.identifier for item in packet.rules},
        "storage": set(packet.graph.reservoir),
        "variable": variables,
    }


def _successor_digest(
    packet: board.SourceDeletedPacket,
    action: board.ExpectedTransition,
) -> str:
    return board.packet_fingerprints(
        dataclasses.replace(packet, graph=action.successor)
    ).isomorphic_sha256


def test_repaired_slice_has_exact_frozen_receipts() -> None:
    value = _built()
    assert len(value.packets) == 22
    assert len(value.expected_records) == 22
    assert len(value.oracle_agreements) == 22
    assert len(value.primitive_coverage) == 22
    assert len(value.twins) == 10
    assert sum(len(item.transitions) for item in value.expected_records) == 24
    assert sum(not item.transitions for item in value.expected_records) == 4
    assert {
        partition: sum(item.partition == partition for item in value.split_assignments)
        for partition in (
            "local_transition_train",
            "local_transition_development",
        )
    } == {
        "local_transition_train": 16,
        "local_transition_development": 6,
    }
    assert sum(len(item.normalized_rule_pairs) for item in value.fingerprints) == 51
    assert (
        sum(len(item.reachable_two_rule_compositions) for item in value.fingerprints)
        == 8
    )


def test_board_is_deterministic() -> None:
    assert board.build_local_transition_slice() == _built()


def test_packet_schema_excludes_every_offline_field() -> None:
    forbidden = {
        "answer",
        "class",
        "expected",
        "family",
        "fingerprint",
        "label",
        "legal",
        "mask",
        "oracle",
        "partition",
        "schedule",
        "seed",
        "source",
        "split",
        "successor",
        "target",
        "trace",
        "twin",
    }
    assert {item.name for item in dataclasses.fields(board.SourceDeletedPacket)} == {
        "constructors",
        "rules",
        "graph",
    }
    for packet in _built().packets:
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
    "key",
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
def test_packet_serializer_kills_forbidden_field_mutants(key: str) -> None:
    payload = dataclasses.asdict(_built().packets[0])
    payload["graph"][key] = "leak"
    with pytest.raises(board.NeuralTcrrBoardError, match="forbidden"):
        board.validate_model_packet_payload(payload)


def test_geometry_maxima_are_explicit_and_enforced() -> None:
    packet = _built().packets[0]
    existing_type = packet.constructors[0].result_type
    extra = tuple(
        board.ConstructorRecord(
            hashlib.sha256(f"constructor:{index}".encode()).hexdigest()[:24],
            existing_type,
            (),
        )
        for index in range(board.MAX_CONSTRUCTORS - len(packet.constructors) + 1)
    )
    with pytest.raises(board.NeuralTcrrBoardError, match="constructor count"):
        board.validate_source_deleted_packet(
            dataclasses.replace(
                packet,
                constructors=(*packet.constructors, *extra),
            )
        )

    type_packet = _packets(_built())[
        _twin(_built(), "type_mismatch").left_packet_sha256
    ]
    present_types = {
        type_id
        for item in type_packet.constructors
        for type_id in (item.result_type, *item.argument_types)
    }
    needed = board.MAX_TYPES - len(present_types) + 1
    extra_types = tuple(
        board.ConstructorRecord(
            hashlib.sha256(f"type-constructor:{index}".encode()).hexdigest()[:24],
            hashlib.sha256(f"type:{index}".encode()).hexdigest()[:24],
            (),
        )
        for index in range(needed)
    )
    with pytest.raises(board.NeuralTcrrBoardError, match="type count"):
        board.validate_source_deleted_packet(
            dataclasses.replace(
                type_packet,
                constructors=(*type_packet.constructors, *extra_types),
            )
        )


def test_unknown_graph_and_rule_kinds_fail_closed() -> None:
    packet = _built().packets[0]
    graph_mutant = dataclasses.replace(
        packet,
        graph=dataclasses.replace(
            packet.graph,
            nodes=(
                dataclasses.replace(packet.graph.nodes[0], kind="mystery"),
                *packet.graph.nodes[1:],
            ),
        ),
    )
    with pytest.raises(board.NeuralTcrrBoardError, match="unknown kind"):
        board.validate_source_deleted_packet(graph_mutant)
    rule = packet.rules[0]
    rule_mutant = dataclasses.replace(
        packet,
        rules=(
            dataclasses.replace(
                rule,
                lhs=dataclasses.replace(rule.lhs, kind="mystery"),
            ),
            *packet.rules[1:],
        ),
    )
    with pytest.raises(board.NeuralTcrrBoardError, match="unknown kind"):
        board.validate_source_deleted_packet(rule_mutant)


def test_duplicate_graph_variable_identifiers_fail_closed() -> None:
    twin = _twin(_built(), "rhs_pointer")
    packet = _packets(_built())[twin.left_packet_sha256]
    root = next(
        item for item in packet.graph.nodes if item.storage_id == packet.graph.root
    )
    duplicate_variable = "f" * 24
    child_ids = set(root.children)
    nodes = tuple(
        (
            board.GraphNodeRecord(
                storage_id=item.storage_id,
                kind="variable",
                type_id=item.type_id,
                variable_id=duplicate_variable,
            )
            if item.storage_id in child_ids
            else item
        )
        for item in packet.graph.nodes
    )
    mutant = dataclasses.replace(
        packet,
        graph=dataclasses.replace(packet.graph, nodes=nodes),
    )
    with pytest.raises(board.NeuralTcrrBoardError, match="variable identifiers"):
        board.validate_source_deleted_packet(mutant)


@pytest.mark.parametrize(
    "kind",
    [
        "repeated_variable_equality",
        "partial_nested_match",
        "type_mismatch",
        "capacity",
    ],
)
def test_controlled_no_redex_twins_are_positive_negative_contrasts(
    kind: str,
) -> None:
    value = _built()
    twin = _twin(value, kind)
    expected = _expected(value)
    assert expected[twin.left_packet_sha256].transitions
    assert expected[twin.right_packet_sha256].transitions == ()


def test_type_mismatch_negative_is_also_an_all_distractor_packet() -> None:
    value = _built()
    twin = _twin(value, "type_mismatch")
    packet = _packets(value)[twin.right_packet_sha256]
    assert len(packet.rules) == 2
    assert _expected(value)[twin.right_packet_sha256].transitions == ()


def test_multi_rule_base_has_only_a_strict_legal_subset() -> None:
    value = _built()
    twin = _twin(value, "constructor_reindex")
    packet = _packets(value)[twin.left_packet_sha256]
    actions = _expected(value)[twin.left_packet_sha256].transitions
    legal_rules = {item.rule_id for item in actions}
    assert len(packet.rules) == 3
    assert 0 < len(legal_rules) < len(packet.rules)
    assert len(legal_rules) == 2


def test_every_twin_is_reconstructed_from_its_base_named_axis() -> None:
    value = _built()
    packets = _packets(value)
    for twin in value.twins:
        assert (
            board._apply_twin_mutation(
                packets[twin.left_packet_sha256],
                twin,
            )
            == packets[twin.right_packet_sha256]
        )


def test_non_reindex_twins_reuse_one_identity_namespace() -> None:
    value = _built()
    packets = _packets(value)
    for kind in (
        "rhs_pointer",
        "repeated_variable_equality",
        "partial_nested_match",
        "type_mismatch",
    ):
        twin = _twin(value, kind)
        assert _all_identifiers(packets[twin.left_packet_sha256]) == (
            _all_identifiers(packets[twin.right_packet_sha256])
        )
    capacity = _twin(value, "capacity")
    left = _all_identifiers(packets[capacity.left_packet_sha256])
    right = _all_identifiers(packets[capacity.right_packet_sha256])
    assert len(left - right) == 1
    assert not (right - left)


@pytest.mark.parametrize(
    ("kind", "namespace"),
    [
        ("constructor_reindex", "constructor"),
        ("type_reindex", "type"),
        ("rule_reindex", "rule"),
        ("storage_reindex", "storage"),
    ],
)
def test_each_reindex_changes_only_its_named_namespace(
    kind: str,
    namespace: str,
) -> None:
    value = _built()
    twin = _twin(value, kind)
    packets = _packets(value)
    fingerprints = _fingerprints(value)
    left = packets[twin.left_packet_sha256]
    right = packets[twin.right_packet_sha256]
    left_namespaces = _namespace_identifiers(left)
    right_namespaces = _namespace_identifiers(right)
    for active_namespace in left_namespaces:
        assert left_namespaces[active_namespace] == (right_namespaces[active_namespace])
    assert twin.namespace == namespace
    assert {item.old for item in twin.remap} == left_namespaces[namespace]
    assert {item.new for item in twin.remap} == right_namespaces[namespace]
    assert any(item.old != item.new for item in twin.remap)
    assert board.packet_sha256(left) != board.packet_sha256(right)
    assert board._apply_twin_mutation(left, twin) == right
    assert (
        fingerprints[twin.left_packet_sha256].isomorphic_sha256
        == fingerprints[twin.right_packet_sha256].isomorphic_sha256
    )


def test_rhs_pointer_and_shared_occurrence_twins_remain_causal() -> None:
    value = _built()
    packets = _packets(value)
    expected = _expected(value)
    rhs = _twin(value, "rhs_pointer")
    left_action = expected[rhs.left_packet_sha256].transitions[0]
    right_action = expected[rhs.right_packet_sha256].transitions[0]
    assert _successor_digest(
        packets[rhs.left_packet_sha256],
        left_action,
    ) != _successor_digest(
        packets[rhs.right_packet_sha256],
        right_action,
    )

    shared = _twin(value, "shared_occurrence")
    actions = expected[shared.left_packet_sha256].transitions
    left = actions[shared.left_transition_index]
    right = actions[shared.right_transition_index]
    assert left.target_storage_id == right.target_storage_id
    assert {left.occurrence_path, right.occurrence_path} == {(0,), (1,)}
    assert _successor_digest(
        packets[shared.left_packet_sha256],
        left,
    ) != _successor_digest(
        packets[shared.left_packet_sha256],
        right,
    )


def test_rule_pair_fingerprints_cover_every_pair_without_applicability_filter() -> None:
    value = _built()
    for packet, fingerprints in zip(
        value.packets,
        value.fingerprints,
        strict=True,
    ):
        rule_count = len(packet.rules)
        assert len(fingerprints.normalized_rule_pairs) == (
            rule_count * (rule_count + 1) // 2
        )
    mismatch = _twin(value, "type_mismatch")
    assert not _expected(value)[mismatch.right_packet_sha256].transitions
    assert _fingerprints(value)[mismatch.right_packet_sha256].normalized_rule_pairs


def test_composition_fingerprint_count_matches_reachable_two_step_windows() -> None:
    for packet, fingerprints in zip(
        _built().packets,
        _built().fingerprints,
        strict=True,
    ):
        system, graph, _storage = board._packet_to_mechanics(packet)
        expected_count = 0
        for first in board.mechanics.legal_reductions(system, graph):
            intermediate = board.mechanics.apply_reduction(system, graph, first)
            expected_count += len(
                board.mechanics.legal_reductions(system, intermediate)
            )
        assert len(fingerprints.reachable_two_rule_compositions) == (expected_count)


def test_split_isolation_covers_rule_pairs_and_compositions() -> None:
    value = _built()
    board.validate_split_isolation(
        value.split_assignments,
        value.fingerprints,
    )
    fingerprints = _fingerprints(value)
    split = {item.packet_sha256: item.partition for item in value.split_assignments}
    for field in (
        "normalized_rule_windows",
        "normalized_rule_pairs",
        "reachable_two_rule_compositions",
    ):
        train = {
            item
            for digest, partition in split.items()
            if partition == "local_transition_train"
            for item in getattr(fingerprints[digest], field)
        }
        development = {
            item
            for digest, partition in split.items()
            if partition == "local_transition_development"
            for item in getattr(fingerprints[digest], field)
        }
        assert not (train & development)


def test_cross_split_reindex_mutant_is_rejected() -> None:
    value = _built()
    twin = _twin(value, "constructor_reindex")
    assignments = tuple(
        board.SplitAssignment(
            item.packet_sha256,
            (
                "local_transition_development"
                if item.packet_sha256 == twin.right_packet_sha256
                else item.partition
            ),
        )
        for item in value.split_assignments
    )
    with pytest.raises(board.NeuralTcrrBoardError, match="cross-split"):
        board.validate_split_isolation(assignments, value.fingerprints)


def test_canonicalizer_uses_refinement_not_factorial_permutations() -> None:
    source = inspect.getsource(board._canonical_colored_graph)
    module_source = inspect.getsource(board)
    assert "_refine_colors" in source
    assert "maximum_backtrack_states" in source
    assert "itertools.permutations" not in module_source
    assert board.MAX_CONSTRUCTORS == 16
    assert board.MAX_TYPES == 8
    assert board.MAX_CANONICAL_BACKTRACK_STATES == 50_000


def test_every_development_primitive_has_train_coverage() -> None:
    value = _built()
    split = {item.packet_sha256: item.partition for item in value.split_assignments}
    train = {
        primitive
        for item in value.primitive_coverage
        if split[item.packet_sha256] == "local_transition_train"
        for primitive in item.primitives
    }
    development = {
        primitive
        for item in value.primitive_coverage
        if split[item.packet_sha256] == "local_transition_development"
        for primitive in item.primitives
    }
    assert development - {"no_redex"} <= train


def test_every_packet_has_exact_independent_oracle_agreement() -> None:
    value = _built()
    assert all(item.exact_agreement for item in value.oracle_agreements)
    assert all(
        item.production_sha256 == item.independent_reference_sha256
        for item in value.oracle_agreements
    )
    for packet, receipt in zip(
        value.packets,
        value.oracle_agreements,
        strict=True,
    ):
        assert board._oracle_agreement(packet) == receipt


def test_independent_oracle_disagreement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _built().packets[0]
    original = board.mechanics.IndependentNestedReferenceOracle.enumerate

    def mutant(
        self: object,
        system: object,
        graph: object,
    ) -> object:
        result = original(self, system, graph)
        return dataclasses.replace(
            result,
            transitions=(),
            transitions_explored=0,
        )

    monkeypatch.setattr(
        board.mechanics.IndependentNestedReferenceOracle,
        "enumerate",
        mutant,
    )
    with pytest.raises(board.NeuralTcrrBoardError, match="oracles disagree"):
        board._oracle_agreement(packet)


@pytest.mark.parametrize(
    "ledger_name",
    [
        "expected_records",
        "split_assignments",
        "fingerprints",
        "oracle_agreements",
        "primitive_coverage",
    ],
)
def test_duplicate_ledger_keys_fail_closed(ledger_name: str) -> None:
    value = _built()
    records = getattr(value, ledger_name)
    duplicate = (records[0], records[0], *records[2:])
    mutant = dataclasses.replace(value, **{ledger_name: duplicate})
    with pytest.raises(board.NeuralTcrrBoardError, match="keys must be unique"):
        board.validate_local_transition_slice(mutant)


def test_stale_expected_and_twin_receipts_fail_closed() -> None:
    value = _built()
    first = value.expected_records[0]
    stale_expected = dataclasses.replace(first, transitions=())
    with pytest.raises(board.NeuralTcrrBoardError, match="expected transition"):
        board.validate_local_transition_slice(
            dataclasses.replace(
                value,
                expected_records=(stale_expected, *value.expected_records[1:]),
            )
        )
    twin = _twin(value, "constructor_reindex")
    stale_twin = dataclasses.replace(twin, remap=twin.remap[:-1])
    twins = tuple(
        stale_twin if item.kind == twin.kind else item for item in value.twins
    )
    with pytest.raises(board.NeuralTcrrBoardError):
        board.validate_local_transition_slice(dataclasses.replace(value, twins=twins))


def test_packet_only_export_uses_three_physically_separate_roots(
    tmp_path: Path,
) -> None:
    packet_root = tmp_path / "packets"
    train_label_root = tmp_path / "train-labels"
    assessor_root = tmp_path / "development-assessor"
    receipt = board.export_packet_only_corpus(
        _built(),
        packet_root=packet_root,
        train_label_root=train_label_root,
        development_assessment_root=assessor_root,
    )
    assert receipt.train_packet_count == 16
    assert receipt.development_packet_count == 6
    assert len(board.load_packet_only_partition(packet_root, "train")) == 16
    assert len(board.load_packet_only_partition(packet_root, "development")) == 6
    for path in packet_root.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "occurrence_path" not in text
        assert "successor" not in text
        assert "oracle_agreements" not in text
    assert not list(packet_root.rglob("train_labels.json"))
    assert not list(packet_root.rglob("sealed_development_assessment.json"))


def test_export_never_crosses_train_and_development_label_custody(
    tmp_path: Path,
) -> None:
    value = _built()
    packet_root = tmp_path / "packets"
    label_root = tmp_path / "labels"
    assessor_root = tmp_path / "assessor"
    receipt = board.export_packet_only_corpus(
        value,
        packet_root=packet_root,
        train_label_root=label_root,
        development_assessment_root=assessor_root,
    )
    split = {item.packet_sha256: item.partition for item in value.split_assignments}
    train_digests = {
        digest
        for digest, partition in split.items()
        if partition == "local_transition_train"
    }
    development_digests = set(split) - train_digests
    train_payload = json.loads(
        (label_root / "train_labels.json").read_text(encoding="utf-8")
    )
    assert {item["packet_sha256"] for item in train_payload["records"]} == train_digests
    assert not (
        development_digests
        & {item["packet_sha256"] for item in train_payload["records"]}
    )
    assessor_artifact = assessor_root / "sealed_development_assessment.json"
    assert stat.S_IMODE(assessor_artifact.stat().st_mode) == 0o400
    assessor_payload = board.load_sealed_development_assessment(
        assessor_artifact,
        expected_sha256=receipt.sealed_development_artifact_sha256,
    )
    assert {
        item["packet_sha256"] for item in assessor_payload["records"]
    } == development_digests


def test_packet_loader_cannot_receive_the_offline_slice() -> None:
    source = inspect.getsource(board.load_packet_only_partition)
    assert "LocalTransitionSlice" not in source
    with pytest.raises((AttributeError, TypeError)):
        board.load_packet_only_partition(_built(), "development")


def test_export_rejects_nested_or_reused_roots(tmp_path: Path) -> None:
    with pytest.raises(board.NeuralTcrrBoardError, match="must be disjoint"):
        board.export_packet_only_corpus(
            _built(),
            packet_root=tmp_path / "root",
            train_label_root=tmp_path / "root" / "labels",
            development_assessment_root=tmp_path / "assessor",
        )


def test_sealed_assessor_rejects_writeable_or_tampered_artifact(
    tmp_path: Path,
) -> None:
    receipt = board.export_packet_only_corpus(
        _built(),
        packet_root=tmp_path / "packets",
        train_label_root=tmp_path / "labels",
        development_assessment_root=tmp_path / "assessor",
    )
    artifact = tmp_path / "assessor" / "sealed_development_assessment.json"
    artifact.chmod(0o600)
    with pytest.raises(board.NeuralTcrrBoardError, match="not sealed"):
        board.load_sealed_development_assessment(
            artifact,
            expected_sha256=receipt.sealed_development_artifact_sha256,
        )
    artifact.write_text(
        artifact.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    artifact.chmod(0o400)
    with pytest.raises(board.NeuralTcrrBoardError, match="hash mismatch"):
        board.load_sealed_development_assessment(
            artifact,
            expected_sha256=receipt.sealed_development_artifact_sha256,
        )


def test_packet_round_trip_has_no_offline_dependency() -> None:
    for packet in _built().packets:
        serialized = board.serialize_model_packet(packet)
        assert board.deserialize_model_packet(serialized) == packet
