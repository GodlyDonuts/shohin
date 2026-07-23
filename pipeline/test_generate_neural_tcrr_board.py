from __future__ import annotations

import dataclasses
import inspect
import json
from collections import Counter
from functools import lru_cache

import pytest

import generate_neural_tcrr_board as generated
import neural_tcrr_board as audited
import typed_critical_pair_rewrite_board as mechanics


@lru_cache(maxsize=1)
def _pilot() -> generated.ProceduralNeuralTcrrPilot:
    return generated.generate_neural_tcrr_pilot()


def _by_digest(
    packets: tuple[audited.SourceDeletedPacket, ...],
) -> dict[str, audited.SourceDeletedPacket]:
    return {audited.packet_sha256(packet): packet for packet in packets}


def _expected(
    value: generated.ProceduralNeuralTcrrPilot,
) -> dict[str, audited.ExpectedTransitionRecord]:
    return {item.packet_sha256: item for item in value.expected_records}


def _fingerprints(
    value: generated.ProceduralNeuralTcrrPilot,
) -> dict[str, audited.PacketFingerprints]:
    return {item.packet_sha256: item for item in value.fingerprints}


def _metadata(
    value: generated.ProceduralNeuralTcrrPilot,
) -> dict[str, generated.ProceduralPacketMetadata]:
    return {item.packet_sha256: item for item in value.metadata}


def _identifiers(packet: audited.SourceDeletedPacket) -> set[str]:
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


def _opaque(index: int) -> str:
    return f"{index:024x}"[-24:]


def _geometry_variants(
    packet: audited.SourceDeletedPacket,
) -> dict[str, audited.SourceDeletedPacket]:
    type_id = packet.constructors[0].result_type
    constructors_17 = list(packet.constructors)
    while len(constructors_17) < 17:
        constructors_17.append(
            audited.ConstructorRecord(
                identifier=_opaque(10_000 + len(constructors_17)),
                result_type=type_id,
                argument_types=(),
            )
        )

    type_constructors = list(packet.constructors)
    active_types = {
        active
        for constructor in type_constructors
        for active in (constructor.result_type, *constructor.argument_types)
    }
    while len(active_types) < 9:
        fresh_type = _opaque(20_000 + len(active_types))
        active_types.add(fresh_type)
        type_constructors.append(
            audited.ConstructorRecord(
                identifier=_opaque(30_000 + len(type_constructors)),
                result_type=fresh_type,
                argument_types=(),
            )
        )

    rules_9 = list(packet.rules)
    while len(rules_9) < 9:
        rules_9.append(
            dataclasses.replace(
                packet.rules[0],
                identifier=_opaque(40_000 + len(rules_9)),
            )
        )

    unary = next(
        item
        for item in packet.constructors
        if item.result_type == type_id and item.argument_types == (type_id,)
    )
    oversized_lhs = packet.rules[0].lhs
    while audited._rule_term_count(oversized_lhs) <= 12:  # noqa: SLF001
        oversized_lhs = audited.RuleTermRecord(
            kind="constructor",
            type_id=type_id,
            constructor_id=unary.identifier,
            children=(oversized_lhs,),
        )
    oversized_rule = dataclasses.replace(packet.rules[0], lhs=oversized_lhs)

    arity_four = audited.ConstructorRecord(
        identifier=_opaque(50_000),
        result_type=type_id,
        argument_types=(type_id, type_id, type_id, type_id),
    )

    reservoir_17 = list(packet.graph.reservoir)
    while len(reservoir_17) < 17:
        reservoir_17.append(_opaque(60_000 + len(reservoir_17)))

    return {
        "N17": dataclasses.replace(
            packet,
            graph=dataclasses.replace(
                packet.graph,
                reservoir=tuple(reservoir_17),
            ),
        ),
        "C17": dataclasses.replace(
            packet,
            constructors=tuple(constructors_17),
        ),
        "Y9": dataclasses.replace(
            packet,
            constructors=tuple(type_constructors),
        ),
        "R9": dataclasses.replace(packet, rules=tuple(rules_9)),
        "P13": dataclasses.replace(
            packet,
            rules=(oversized_rule, *packet.rules[1:]),
        ),
        "A4": dataclasses.replace(
            packet,
            constructors=(*packet.constructors, arity_four),
        ),
    }


def _depth_nine_packet(seed: int = 881) -> audited.SourceDeletedPacket:
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
    example = audited._pack_example(  # noqa: SLF001
        factory,
        system,
        term,
        capacity=11,
    )
    return audited._packet_from_generated(example)  # noqa: SLF001


def _oversubscribed_action_packet(seed: int = 882) -> audited.SourceDeletedPacket:
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
    example = audited._pack_example(  # noqa: SLF001
        factory,
        system,
        term,
        capacity=16,
    )
    return audited._packet_from_generated(example)  # noqa: SLF001


def test_pilot_has_frozen_geometry_required_cells_and_no_confirmation() -> None:
    value = _pilot()
    assert (
        generated.N,
        generated.C,
        generated.Y,
        generated.R,
        generated.P,
        generated.A,
        generated.D,
    ) == (16, 16, 8, 8, 12, 3, 8)
    assert generated.MAX_LEGAL_ACTIONS == 128
    assert len(value.packets) == 15
    assert len(value.expected_records) == 15
    assert len(value.metadata) == 15
    assert len(value.oracle_agreements) == 15
    assert len(value.label_agreements) == 15
    assert sum(len(item.transitions) for item in value.expected_records) == 36
    assert value.manifest.packet_count == 15
    assert value.manifest.train_packet_count == 11
    assert value.manifest.development_packet_count == 4
    assert value.manifest.geometry == generated.GeometryReceipt()
    assert set(item.partition for item in value.metadata) == {
        "local_transition_train",
        "local_transition_development",
    }
    assert {
        item.family
        for item in value.metadata
        if item.partition == "local_transition_train"
    } >= {
        "algebraic_normalization",
        "boolean_simplification",
        "list_tree_normalization",
    }
    assert {
        item.family
        for item in value.metadata
        if item.partition == "local_transition_development"
    } == {"typed_stack", "dataflow"}
    assert all(
        "confirmation" not in dataclasses.asdict(item).values()
        for item in value.metadata
    )
    assert value.manifest.cells
    assert all(item.count == 1 for item in value.manifest.cells)


def test_generator_is_reproducible_and_seed_sensitive() -> None:
    first = _pilot()
    second = generated.generate_neural_tcrr_pilot()
    alternate = generated.generate_neural_tcrr_pilot(seed=generated.DEFAULT_SEED + 1)
    assert first == second
    assert first.manifest.payload_sha256 == second.manifest.payload_sha256
    assert first.manifest.payload_sha256 != alternate.manifest.payload_sha256
    assert {audited.packet_sha256(item) for item in first.packets}.isdisjoint(
        {audited.packet_sha256(item) for item in alternate.packets}
    )
    assert [
        (item.partition, item.family, item.renderer, item.composition, item.role)
        for item in first.metadata
    ] == [
        (item.partition, item.family, item.renderer, item.composition, item.role)
        for item in alternate.metadata
    ]


def test_model_packets_are_opaque_and_exclude_every_offline_axis() -> None:
    value = _pilot()
    assert {item.name for item in dataclasses.fields(audited.SourceDeletedPacket)} == {
        "constructors",
        "rules",
        "graph",
    }
    semantic_values = {
        str(field)
        for item in value.metadata
        for field in (
            item.partition,
            item.family,
            item.template,
            item.renderer,
            item.composition,
            item.role,
            item.orbit_id,
        )
    }
    for packet in value.packets:
        serialized = audited.serialize_model_packet(packet)
        audited.validate_model_packet_payload(json.loads(serialized))
        assert not any(value in serialized for value in semantic_values)
        identifiers = _identifiers(packet)
        assert identifiers
        assert all(
            len(identifier) == audited.OPAQUE_ID_LENGTH
            and set(identifier) <= set("0123456789abcdef")
            for identifier in identifiers
        )

    packets = _by_digest(value.packets)
    for twin in value.twins:
        if twin.kind != "render_reindex":
            continue
        assert _identifiers(packets[twin.left_packet_sha256]).isdisjoint(
            _identifiers(packets[twin.right_packet_sha256])
        )


def test_split_isolation_covers_exact_isomorphic_rule_pair_and_composition() -> None:
    value = _pilot()
    assignments = tuple(
        audited.SplitAssignment(item.packet_sha256, item.partition)
        for item in value.metadata
    )
    audited.validate_split_isolation(assignments, value.fingerprints)
    split = {item.packet_sha256: item.partition for item in value.metadata}

    def values(
        partition: str,
        field: str,
    ) -> set[str]:
        output: set[str] = set()
        for item in value.fingerprints:
            if split[item.packet_sha256] != partition:
                continue
            active = getattr(item, field)
            if isinstance(active, str):
                output.add(active)
            else:
                output.update(active)
        return output

    for field in (
        "exact_sha256",
        "isomorphic_sha256",
        "normalized_rule_pairs",
        "reachable_two_rule_compositions",
    ):
        assert values("local_transition_train", field).isdisjoint(
            values("local_transition_development", field)
        )
    assert values(
        "local_transition_train",
        "reachable_two_rule_compositions",
    )
    assert values(
        "local_transition_development",
        "reachable_two_rule_compositions",
    )


def test_all_complete_labels_agree_with_the_independent_reference() -> None:
    value = _pilot()
    for packet, expected, metadata, oracle, labels in zip(
        value.packets,
        value.expected_records,
        value.metadata,
        value.oracle_agreements,
        value.label_agreements,
        strict=True,
    ):
        recomputed = audited._expected_record_from_packet(packet)  # noqa: SLF001
        assert expected == recomputed
        assert metadata.legal_action_count == len(expected.transitions)
        assert metadata.legal_action_count <= generated.MAX_LEGAL_ACTIONS
        assert metadata.depth <= generated.D
        assert oracle.exact_agreement
        assert labels.exact_agreement
        assert labels.action_count == len(expected.transitions)
        assert labels.production_sha256 == labels.independent_reference_sha256
        assert oracle.production_sha256 == oracle.independent_reference_sha256
        assert oracle.state_count < generated.MAX_ORACLE_STATES


def test_exported_successor_bridge_rejects_a_same_count_wrong_label() -> None:
    value = _pilot()
    packet, expected = next(
        (packet, record)
        for packet, record in zip(
            value.packets,
            value.expected_records,
            strict=True,
        )
        if len(record.transitions) >= 2
        and record.transitions[0].successor != record.transitions[1].successor
    )
    wrong_first = dataclasses.replace(
        expected.transitions[0],
        successor=expected.transitions[1].successor,
    )
    corrupted = dataclasses.replace(
        expected,
        transitions=(wrong_first, *expected.transitions[1:]),
    )
    with pytest.raises(
        generated.ProceduralCandidateRejected,
        match="exported opaque transitions differ",
    ):
        generated._oracle_and_label_agreement(  # noqa: SLF001
            packet,
            corrupted,
        )


def test_required_twins_have_exact_causal_predicates() -> None:
    value = _pilot()
    packets = _by_digest(value.packets)
    expected = _expected(value)
    fingerprints = _fingerprints(value)
    metadata = _metadata(value)
    assert Counter(item.kind for item in value.twins) == Counter(
        {
            "render_reindex": 5,
            "no_redex": 1,
            "shared_occurrence": 1,
            "capacity": 1,
        }
    )
    for twin in value.twins:
        left_packet = packets[twin.left_packet_sha256]
        right_packet = packets[twin.right_packet_sha256]
        left_expected = expected[twin.left_packet_sha256]
        right_expected = expected[twin.right_packet_sha256]
        if twin.kind == "render_reindex":
            assert twin.left_packet_sha256 != twin.right_packet_sha256
            assert (
                fingerprints[twin.left_packet_sha256].isomorphic_sha256
                == fingerprints[twin.right_packet_sha256].isomorphic_sha256
            )
            assert (
                metadata[twin.left_packet_sha256].family
                == metadata[twin.right_packet_sha256].family
            )
            assert (
                metadata[twin.left_packet_sha256].depth
                == metadata[twin.right_packet_sha256].depth
            )
            assert {
                metadata[twin.left_packet_sha256].renderer,
                metadata[twin.right_packet_sha256].renderer,
            } == {"renderer_0", "renderer_1"}
        elif twin.kind == "no_redex":
            assert left_expected.transitions
            assert not right_expected.transitions
            assert left_packet.rules == right_packet.rules
        elif twin.kind == "shared_occurrence":
            assert twin.left_packet_sha256 == twin.right_packet_sha256
            assert twin.left_transition_index is not None
            assert twin.right_transition_index is not None
            left_action = left_expected.transitions[twin.left_transition_index]
            right_action = left_expected.transitions[twin.right_transition_index]
            assert left_action.target_storage_id == right_action.target_storage_id
            assert left_action.occurrence_path != right_action.occurrence_path
        elif twin.kind == "capacity":
            assert len(left_packet.graph.reservoir) == 16
            assert len(right_packet.graph.reservoir) == 15
            assert left_expected.transitions
            assert not right_expected.transitions
            assert left_packet.graph.nodes == right_packet.graph.nodes


def test_held_out_templates_really_use_cross_typed_structures() -> None:
    value = _pilot()
    packets = _by_digest(value.packets)
    for item in value.metadata:
        if item.family not in {"typed_stack", "dataflow"}:
            continue
        packet = packets[item.packet_sha256]
        type_ids = {
            active
            for constructor in packet.constructors
            for active in (constructor.result_type, *constructor.argument_types)
        }
        assert len(type_ids) == 2
        if item.family == "typed_stack":
            assert any(
                len(constructor.argument_types) == 2
                and constructor.argument_types[0] != constructor.argument_types[1]
                for constructor in packet.constructors
            )
        else:
            assert any(
                len(constructor.argument_types) == 1
                and constructor.result_type != constructor.argument_types[0]
                for constructor in packet.constructors
            )


def test_every_frozen_geometry_axis_fails_closed() -> None:
    base = next(
        packet
        for packet, item in zip(
            _pilot().packets,
            _pilot().metadata,
            strict=True,
        )
        if item.family == "algebraic_normalization" and item.role == "base"
    )
    variants = _geometry_variants(base)
    variants["D9"] = _depth_nine_packet()
    assert set(variants) == {"N17", "C17", "Y9", "R9", "P13", "A4", "D9"}
    for packet in variants.values():
        with pytest.raises(generated.ProceduralCandidateRejected):
            generated.assess_source_deleted_candidate(packet)


def test_action_cap_rejects_and_resamples_without_truncating() -> None:
    oversubscribed = _oversubscribed_action_packet()
    complete = audited._expected_record_from_packet(oversubscribed)  # noqa: SLF001
    assert len(complete.transitions) == 512
    with pytest.raises(
        generated.ProceduralCandidateRejected,
        match="legal action count 512",
    ):
        generated.assess_source_deleted_candidate(oversubscribed)

    valid = _pilot().packets[0]
    calls = 0

    def candidates(_seed: int) -> tuple[audited.SourceDeletedPacket]:
        nonlocal calls
        calls += 1
        return (oversubscribed if calls == 1 else valid,)

    assessments, receipt = generated.rejection_sample_packet_group(
        candidates,
        seed=91,
        orbit_id="test_rejection_then_acceptance",
        max_attempts=2,
    )
    assert calls == 2
    assert receipt.accepted_attempt == 1
    assert receipt.rejected_attempts == 1
    assert "512" in receipt.rejection_reasons[0]
    assert assessments[0].expected == audited._expected_record_from_packet(  # noqa: SLF001
        valid
    )
    source = inspect.getsource(generated.assess_source_deleted_candidate)
    assert "[:MAX_LEGAL_ACTIONS]" not in source
    assert "action_count > MAX_LEGAL_ACTIONS" in source


def test_manifest_and_parallel_ledgers_fail_closed_when_tampered() -> None:
    value = _pilot()
    generated.validate_neural_tcrr_pilot(value)
    stale_metadata = dataclasses.replace(
        value.metadata[0],
        legal_action_count=value.metadata[0].legal_action_count + 1,
    )
    with pytest.raises(generated.ProceduralBoardError):
        generated.validate_neural_tcrr_pilot(
            dataclasses.replace(
                value,
                metadata=(stale_metadata, *value.metadata[1:]),
            )
        )
    with pytest.raises(generated.ProceduralBoardError, match="manifest"):
        generated.validate_neural_tcrr_pilot(
            dataclasses.replace(
                value,
                manifest=dataclasses.replace(
                    value.manifest,
                    payload_sha256="0" * 64,
                ),
            )
        )
