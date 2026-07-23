from __future__ import annotations

import dataclasses
import inspect
from collections import Counter, defaultdict
from functools import lru_cache

import pytest

import generate_neural_tcrr_corpus as corpus
import neural_tcrr_board as audited
import typed_critical_pair_rewrite_board as mechanics


FIXTURE_TRAIN_PACKETS = 22
FIXTURE_DEVELOPMENT_PACKETS = 10


@lru_cache(maxsize=1)
def _fixture() -> corpus.ProceduralNeuralTcrrCorpus:
    return corpus.generate_neural_tcrr_corpus(
        train_packets=FIXTURE_TRAIN_PACKETS,
        development_packets=FIXTURE_DEVELOPMENT_PACKETS,
    )


def _packet_map() -> dict[str, audited.SourceDeletedPacket]:
    return {audited.packet_sha256(packet): packet for packet in _fixture().packets}


def _metadata_map() -> dict[str, corpus.CorpusPacketMetadata]:
    return {item.packet_sha256: item for item in _fixture().metadata}


def _expected_map() -> dict[str, audited.ExpectedTransitionRecord]:
    return {item.packet_sha256: item for item in _fixture().expected_records}


def _fingerprint_categories(
    partition: str,
) -> dict[str, set[str]]:
    split = {item.packet_sha256: item.partition for item in _fixture().metadata}
    output: dict[str, set[str]] = defaultdict(set)
    for item in _fixture().fingerprints:
        if split[item.packet_sha256] != partition:
            continue
        output["exact"].add(item.exact_sha256)
        output["isomorphic"].add(item.isomorphic_sha256)
        output["rule_window"].update(item.normalized_rule_windows)
        output["rule_pair"].update(item.normalized_rule_pairs)
        output["composition"].update(item.reachable_two_rule_compositions)
    return dict(output)


def _identifier_set(packet: audited.SourceDeletedPacket) -> set[str]:
    output = set(packet.graph.reservoir)
    for constructor in packet.constructors:
        output.add(constructor.identifier)
        output.add(constructor.result_type)
        output.update(constructor.argument_types)

    def collect(term: audited.RuleTermRecord | None) -> None:
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


def _term_variables(term: audited.RuleTermRecord | None) -> tuple[str, ...]:
    if term is None:
        return ()
    own = (term.variable_id,) if term.variable_id is not None else ()
    return (
        *own,
        *(value for child in term.children for value in _term_variables(child)),
    )


def _opaque(index: int) -> str:
    return f"{index:024x}"[-24:]


def _depth_nine_packet(seed: int = 991) -> audited.SourceDeletedPacket:
    factory = audited._EpisodeFactory(seed)  # noqa: SLF001
    factory.constructor("wrap", "value", ("value",))
    factory.constructor("atom", "value")
    value = factory.variable("value")
    system = factory.system(
        (
            factory.rule(
                factory.pattern("wrap", value),
                mechanics.RhsVariable(value.name),
            ),
        )
    )
    term = factory.leaf("atom")
    for _ in range(9):
        term = factory.term("wrap", term)
    generated = audited._pack_example(  # noqa: SLF001
        factory,
        system,
        term,
        capacity=11,
    )
    return audited._packet_from_generated(generated)  # noqa: SLF001


def _oversubscribed_action_packet(
    seed: int = 992,
) -> audited.SourceDeletedPacket:
    factory = audited._EpisodeFactory(seed)  # noqa: SLF001
    factory.constructor("fork", "value", ("value", "value"))
    factory.constructor("leaf", "value")
    for index in range(8):
        factory.constructor(f"outcome_{index}", "value")
    rules = tuple(
        factory.rule(
            factory.pattern("leaf"),
            factory.rhs(f"outcome_{index}"),
        )
        for index in range(8)
    )
    system = factory.system(rules)
    shared = factory.leaf("leaf")
    term = shared
    for _ in range(6):
        term = factory.term("fork", term, term)
    generated = audited._pack_example(  # noqa: SLF001
        factory,
        system,
        term,
        capacity=16,
    )
    return audited._packet_from_generated(generated)  # noqa: SLF001


def test_scale_plan_uses_one_semantics_at_default_and_preregistered_counts() -> None:
    default = corpus.plan_neural_tcrr_corpus()
    full = corpus.plan_neural_tcrr_corpus(
        train_packets=corpus.PREREGISTERED_TRAIN_PACKETS,
        development_packets=corpus.PREREGISTERED_DEVELOPMENT_PACKETS,
    )
    assert (
        default.train_packets,
        default.development_packets,
        default.train_orbits,
        default.development_orbits,
    ) == (256, 64, 128, 32)
    assert (
        full.train_packets,
        full.development_packets,
        full.train_orbits,
        full.development_orbits,
    ) == (48_000, 4_000, 24_000, 2_000)
    assert default.grammar_semantics_sha256 == full.grammar_semantics_sha256
    assert default.packets_per_orbit == full.packets_per_orbit == 2
    full_specs = corpus._build_orbit_specs(  # noqa: SLF001
        train_packets=corpus.PREREGISTERED_TRAIN_PACKETS,
        development_packets=corpus.PREREGISTERED_DEVELOPMENT_PACKETS,
    )
    assert len(full_specs) == 26_000
    assert (
        sum(item.partition == "local_transition_train" for item in full_specs) == 24_000
    )
    assert (
        sum(item.partition == "local_transition_development" for item in full_specs)
        == 2_000
    )
    assert not any(
        item.partition == "local_transition_train"
        and item.family in {"typed_stack", "dataflow"}
        for item in full_specs
    )
    with pytest.raises(corpus.ProceduralCorpusError, match="even"):
        corpus.plan_neural_tcrr_corpus(
            train_packets=23,
            development_packets=10,
        )
    with pytest.raises(corpus.ProceduralCorpusError, match="at least"):
        corpus.plan_neural_tcrr_corpus(
            train_packets=20,
            development_packets=10,
        )


def test_fast_fixture_has_complete_ledgers_cells_and_frozen_geometry() -> None:
    value = _fixture()
    assert (
        corpus.N,
        corpus.C,
        corpus.Y,
        corpus.R,
        corpus.P,
        corpus.A,
        corpus.D,
        corpus.MAX_LEGAL_ACTIONS,
    ) == (16, 16, 8, 8, 12, 3, 8, 128)
    assert value.manifest.packet_count == 32
    assert value.manifest.train_packet_count == FIXTURE_TRAIN_PACKETS
    assert value.manifest.development_packet_count == FIXTURE_DEVELOPMENT_PACKETS
    assert value.manifest.orbit_count == 16
    assert len(value.packets) == len(value.expected_records) == 32
    assert len(value.metadata) == len(value.fingerprints) == 32
    assert len(value.oracle_agreements) == len(value.label_agreements) == 32
    assert all(
        item.count >= value.manifest.minimum_cell_count for item in value.manifest.cells
    )
    assert value.manifest.geometry == corpus.GeometryReceipt()
    assert value.manifest.payload_sha256
    assert value.manifest.packet_manifest_sha256
    assert value.manifest.split_isolation_manifest_sha256


def test_generation_replays_exactly_and_changes_under_a_new_seed() -> None:
    first = _fixture()
    replay = corpus.generate_neural_tcrr_corpus(
        train_packets=FIXTURE_TRAIN_PACKETS,
        development_packets=FIXTURE_DEVELOPMENT_PACKETS,
    )
    alternate = corpus.generate_neural_tcrr_corpus(
        seed=corpus.DEFAULT_SEED + 1,
        train_packets=FIXTURE_TRAIN_PACKETS,
        development_packets=FIXTURE_DEVELOPMENT_PACKETS,
    )
    assert first == replay
    assert first.manifest.payload_sha256 == replay.manifest.payload_sha256
    assert first.manifest.payload_sha256 != alternate.manifest.payload_sha256
    assert (
        first.manifest.grammar_semantics_sha256
        == alternate.manifest.grammar_semantics_sha256
    )


def test_packets_are_source_deleted_and_identifiers_are_opaque() -> None:
    forbidden = {
        "behavior",
        "candidate_seed",
        "family",
        "fingerprint",
        "grammar_lane",
        "label",
        "metadata",
        "oracle",
        "partition",
        "renderer",
        "semantic_seed",
        "split",
        "successor",
    }
    for packet in _fixture().packets:
        audited.validate_source_deleted_packet(packet)
        serialized = audited.serialize_model_packet(packet)
        assert not any(f'"{key}"' in serialized for key in forbidden)
        identifiers = _identifier_set(packet)
        assert identifiers
        assert all(len(item) == 24 for item in identifiers)
        assert all(set(item) <= set("0123456789abcdef") for item in identifiers)


def test_diversity_covers_rule_shapes_types_arities_and_hard_behaviors() -> None:
    value = _fixture()
    diversity = value.manifest.diversity
    assert diversity.unique_exact_packets == 32
    assert diversity.unique_isomorphic_packets >= 16
    assert diversity.unique_rule_windows >= 18
    assert diversity.unique_rule_pairs >= 40
    assert diversity.unique_two_rule_compositions >= 20
    assert diversity.constructor_arities == (0, 1, 2, 3)
    assert diversity.type_cardinalities == (1, 2, 3)
    behaviors = Counter(item.behavior for item in value.metadata)
    assert {
        "replacement",
        "deletion",
        "repeated_variable",
        "expansion",
        "nested_creation",
        "cycle",
        "critical_pair",
        "ternary_replacement",
        "no_redex",
        "shared_occurrence",
        "capacity_sensitive",
    } <= set(behaviors)
    assert all(count >= 2 for count in behaviors.values())
    assert any(item.cyclic_component_count > 0 for item in value.oracle_agreements)
    critical_digests = {
        item.packet_sha256
        for item in value.metadata
        if item.behavior == "critical_pair"
    }
    expected = _expected_map()
    assert any(len(expected[digest].transitions) >= 2 for digest in critical_digests)
    packets = _packet_map()
    metadata = _metadata_map()
    deletion_packets = [
        packets[digest]
        for digest, item in metadata.items()
        if item.behavior == "deletion"
    ]
    repeated_packets = [
        packets[digest]
        for digest, item in metadata.items()
        if item.behavior == "repeated_variable"
    ]
    expansion_packets = [
        packets[digest]
        for digest, item in metadata.items()
        if item.behavior == "expansion"
    ]
    assert any(rule.rhs is None for packet in deletion_packets for rule in packet.rules)
    assert any(
        len(variables) != len(set(variables))
        for packet in repeated_packets
        for rule in packet.rules
        if (variables := _term_variables(rule.lhs))
    )
    assert any(
        rule.rhs is not None
        and audited._rule_term_count(rule.rhs)  # noqa: SLF001
        > audited._rule_term_count(rule.lhs)  # noqa: SLF001
        for packet in expansion_packets
        for rule in packet.rules
    )


def test_optimization_families_are_distinct_model_visible_grammars() -> None:
    packets = tuple(
        corpus._single_sort_packet(  # noqa: SLF001
            family=family,
            behavior="replacement",
            grammar_lane="train_optimization",
            semantic_seed=1_234_567,
            semantic_index=99,
            renderer_seed=7_654_321,
        )
        for family in (
            "algebraic_normalization",
            "boolean_simplification",
            "list_tree_normalization",
        )
    )
    fingerprints = tuple(audited.packet_fingerprints(packet) for packet in packets)
    assert len({item.isomorphic_sha256 for item in fingerprints}) == 3
    arity_signatures = {
        tuple(
            sorted(
                len(constructor.argument_types) for constructor in packet.constructors
            )
        )
        for packet in packets
    }
    assert len(arity_signatures) == 3


def test_development_is_unseen_by_every_split_fingerprint_and_offline_axis() -> None:
    value = _fixture()
    assignments = tuple(
        audited.SplitAssignment(item.packet_sha256, item.partition)
        for item in value.metadata
    )
    audited.validate_split_isolation(assignments, value.fingerprints)
    train_categories = _fingerprint_categories("local_transition_train")
    development_categories = _fingerprint_categories("local_transition_development")
    for category in (
        "exact",
        "isomorphic",
        "rule_window",
        "rule_pair",
        "composition",
    ):
        assert not (
            train_categories.get(category, set())
            & development_categories.get(category, set())
        ), category

    train = [
        item for item in value.metadata if item.partition == "local_transition_train"
    ]
    development = [
        item
        for item in value.metadata
        if item.partition == "local_transition_development"
    ]
    assert {item.family for item in train} == {
        "algebraic_normalization",
        "boolean_simplification",
        "list_tree_normalization",
    }
    assert {item.family for item in development} == {
        "algebraic_normalization",
        "boolean_simplification",
        "list_tree_normalization",
        "typed_stack",
        "dataflow",
    }
    assert not (
        {item.renderer for item in train} & {item.renderer for item in development}
    )
    for family in (
        "algebraic_normalization",
        "boolean_simplification",
        "list_tree_normalization",
    ):
        train_depths = {
            item.max_occurrence_depth for item in train if item.family == family
        }
        development_depths = {
            item.max_occurrence_depth for item in development if item.family == family
        }
        assert development_depths - train_depths


def test_renderer_capacity_no_redex_and_shared_orbits_retain_exact_predicates() -> None:
    value = _fixture()
    packets = _packet_map()
    expected = _expected_map()
    fingerprints = {item.packet_sha256: item for item in value.fingerprints}
    kinds = Counter(item.kind for item in value.orbit_receipts)
    assert kinds["matched_no_redex"] == 1
    assert kinds["matched_capacity"] == 1
    assert kinds["shared_renderer_reindex"] == 1
    assert kinds["renderer_reindex"] >= 1
    for orbit in value.orbit_receipts:
        left = packets[orbit.left_packet_sha256]
        right = packets[orbit.right_packet_sha256]
        if orbit.kind in {
            "renderer_reindex",
            "shared_renderer_reindex",
        }:
            assert orbit.left_packet_sha256 != orbit.right_packet_sha256
            assert (
                fingerprints[orbit.left_packet_sha256].isomorphic_sha256
                == fingerprints[orbit.right_packet_sha256].isomorphic_sha256
            )
            assert not (_identifier_set(left) & _identifier_set(right))
        elif orbit.kind == "matched_no_redex":
            assert expected[orbit.left_packet_sha256].transitions
            assert not expected[orbit.right_packet_sha256].transitions
            assert left.rules == right.rules
        elif orbit.kind == "matched_capacity":
            assert len(left.graph.reservoir) == 16
            assert len(right.graph.reservoir) == 15
            assert expected[orbit.left_packet_sha256].transitions
            assert not expected[orbit.right_packet_sha256].transitions
        if orbit.kind == "shared_renderer_reindex":
            for digest in (
                orbit.left_packet_sha256,
                orbit.right_packet_sha256,
            ):
                actions = expected[digest].transitions
                assert any(
                    left_action.target_storage_id == right_action.target_storage_id
                    and left_action.occurrence_path != right_action.occurrence_path
                    for left_index, left_action in enumerate(actions)
                    for right_action in actions[left_index + 1 :]
                )


def test_complete_labels_and_exported_successor_edges_match_both_oracles() -> None:
    value = _fixture()
    assert all(item.exact_agreement for item in value.oracle_agreements)
    assert all(item.exact_agreement for item in value.label_agreements)
    assert all(
        item.production_sha256 == item.independent_reference_sha256
        for item in value.oracle_agreements
    )
    assert all(
        item.production_sha256 == item.independent_reference_sha256
        for item in value.label_agreements
    )
    corpus.validate_neural_tcrr_corpus(value, recompute_oracles=True)

    candidate_index = next(
        index
        for index, item in enumerate(value.expected_records)
        if len(item.transitions) >= 2
        and item.transitions[0].successor != item.transitions[1].successor
    )
    original_record = value.expected_records[candidate_index]
    corrupted_transition = dataclasses.replace(
        original_record.transitions[0],
        successor=original_record.transitions[1].successor,
    )
    corrupted_record = dataclasses.replace(
        original_record,
        transitions=(corrupted_transition, *original_record.transitions[1:]),
    )
    corrupted_records = list(value.expected_records)
    corrupted_records[candidate_index] = corrupted_record
    with pytest.raises(
        corpus.ProceduralCorpusError,
        match="expected transition ledger is stale",
    ):
        corpus.validate_neural_tcrr_corpus(
            dataclasses.replace(
                value,
                expected_records=tuple(corrupted_records),
            ),
            recompute_oracles=True,
        )


def test_geometry_and_action_limits_fail_closed_without_label_truncation() -> None:
    depth_nine = _depth_nine_packet()
    with pytest.raises(
        corpus.ProceduralCorpusCandidateRejected,
        match="occurrence depth 9",
    ):
        corpus.assess_corpus_packet(depth_nine)

    oversubscribed = _oversubscribed_action_packet()
    complete = audited._expected_record_from_packet(oversubscribed)  # noqa: SLF001
    assert len(complete.transitions) > corpus.MAX_LEGAL_ACTIONS
    with pytest.raises(
        corpus.ProceduralCorpusCandidateRejected,
        match="legal action count",
    ):
        corpus.assess_corpus_packet(oversubscribed)
    source = inspect.getsource(corpus)
    assert "[:MAX_LEGAL_ACTIONS]" not in source

    packet = _fixture().packets[0]
    extra_reservoir = (*packet.graph.reservoir, _opaque(100_000))
    with pytest.raises(
        corpus.ProceduralCorpusCandidateRejected,
        match="reservoir",
    ):
        corpus.assess_corpus_packet(
            dataclasses.replace(
                packet,
                graph=dataclasses.replace(
                    packet.graph,
                    reservoir=extra_reservoir,
                ),
            )
        )


def test_manifest_and_parallel_ledgers_fail_closed_when_tampered() -> None:
    value = _fixture()
    with pytest.raises(corpus.ProceduralCorpusError, match="manifest"):
        corpus.validate_neural_tcrr_corpus(
            dataclasses.replace(
                value,
                manifest=dataclasses.replace(
                    value.manifest,
                    packet_manifest_sha256="0" * 64,
                ),
            ),
            recompute_oracles=False,
        )
    with pytest.raises(corpus.ProceduralCorpusError, match="order"):
        corpus.validate_neural_tcrr_corpus(
            dataclasses.replace(
                value,
                metadata=tuple(reversed(value.metadata)),
            ),
            recompute_oracles=False,
        )
