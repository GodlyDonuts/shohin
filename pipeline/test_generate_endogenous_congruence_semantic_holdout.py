from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from functools import lru_cache

import pytest

from pipeline import generate_endogenous_congruence_semantic_holdout as corpus
from pipeline.endogenous_congruence_board import (
    EndogenousCongruencePacket,
    compute_exhaustive_partition,
    compute_refinement_partition,
    solve_by_refinement,
)


FIXTURE_TRAIN_PACKETS = 64
FIXTURE_DEVELOPMENT_PACKETS = 64


@lru_cache(maxsize=1)
def _fixture() -> corpus.ProceduralSemanticHoldoutCorpus:
    return corpus.generate_endogenous_congruence_semantic_holdout(
        train_packets=FIXTURE_TRAIN_PACKETS,
        development_packets=FIXTURE_DEVELOPMENT_PACKETS,
    )


def _relation(
    blocks: tuple[tuple[str, ...], ...],
) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(block) for block in blocks)


def _observation_equality(
    packet: EndogenousCongruencePacket,
) -> tuple[tuple[bool, ...], ...]:
    values = {
        (item.record, item.query_port): item.value
        for item in packet.observation_witnesses
    }
    return tuple(
        tuple(
            values[(left, query)] == values[(right, query)]
            for left in packet.records
            for right in packet.records
        )
        for query in packet.query_ports
    )


def test_plan_requires_complete_geometry_support() -> None:
    plan = corpus.plan_semantic_holdout_corpus()
    assert (
        plan.train_packets,
        plan.development_packets,
        plan.train_orbits,
        plan.development_orbits,
    ) == (256, 64, 32, 8)
    assert plan.geometry_cell_count == 8
    assert plan.train_combination_count == 4
    assert plan.development_combination_count == 4
    assert plan.packets_per_orbit == 8
    with pytest.raises(corpus.SemanticHoldoutCorpusError, match="at least"):
        corpus.plan_semantic_holdout_corpus(
            train_packets=56,
            development_packets=64,
        )
    with pytest.raises(corpus.SemanticHoldoutCorpusError, match="divisible"):
        corpus.plan_semantic_holdout_corpus(
            train_packets=65,
            development_packets=64,
        )


def test_specs_match_geometry_and_hold_out_only_combinations() -> None:
    specs = corpus._build_orbit_specs(  # noqa: SLF001
        seed=corpus.DEFAULT_SEED,
        train_packets=FIXTURE_TRAIN_PACKETS,
        development_packets=FIXTURE_DEVELOPMENT_PACKETS,
    )
    by_partition: dict[str, list[corpus.SemanticHoldoutOrbitSpec]] = defaultdict(list)
    for item in specs:
        by_partition[item.partition].append(item)
    train = by_partition[corpus.TRAIN_PARTITION]
    development = by_partition[corpus.DEVELOPMENT_PARTITION]
    assert {item.geometry for item in train} == set(corpus.GEOMETRY_CELLS)
    assert {item.geometry for item in development} == set(corpus.GEOMETRY_CELLS)

    train_combinations = {item.semantic_combination for item in train}
    development_combinations = {item.semantic_combination for item in development}
    assert train_combinations == set(corpus.TRAIN_COMBINATIONS)
    assert development_combinations == set(corpus.DEVELOPMENT_COMBINATIONS)
    assert not train_combinations & development_combinations
    for index in range(3):
        assert {item[index] for item in train_combinations} == {
            item[index] for item in development_combinations
        }


def test_semantic_draw_is_reproducible_and_seed_sensitive() -> None:
    first = corpus._build_orbit_specs(  # noqa: SLF001
        seed=corpus.DEFAULT_SEED,
        train_packets=FIXTURE_TRAIN_PACKETS,
        development_packets=FIXTURE_DEVELOPMENT_PACKETS,
    )
    replay = corpus._build_orbit_specs(  # noqa: SLF001
        seed=corpus.DEFAULT_SEED,
        train_packets=FIXTURE_TRAIN_PACKETS,
        development_packets=FIXTURE_DEVELOPMENT_PACKETS,
    )
    alternate = corpus._build_orbit_specs(  # noqa: SLF001
        seed=corpus.DEFAULT_SEED + 1,
        train_packets=FIXTURE_TRAIN_PACKETS,
        development_packets=FIXTURE_DEVELOPMENT_PACKETS,
    )
    assert first == replay
    assert first != alternate
    assert [item.geometry for item in first] == [item.geometry for item in alternate]
    assert [item.semantic_combination for item in first] == [
        item.semantic_combination for item in alternate
    ]
    assert corpus._sample_hidden_automaton(first[0]) == (  # noqa: SLF001
        corpus._sample_hidden_automaton(replay[0])  # noqa: SLF001
    )
    assert corpus._sample_hidden_automaton(first[0]) != (  # noqa: SLF001
        corpus._sample_hidden_automaton(alternate[0])  # noqa: SLF001
    )


def test_model_packets_are_exactly_source_deleted() -> None:
    allowed_fields = {
        "records",
        "generators",
        "query_ports",
        "transition_witnesses",
        "observation_witnesses",
    }
    assert {field.name for field in dataclasses.fields(EndogenousCongruencePacket)} == (
        allowed_fields
    )
    visible = corpus.serialize_model_packets(_fixture().packets)
    assert len(json.loads(visible)) == len(_fixture().packets)
    forbidden = (
        corpus.TRAIN_PARTITION,
        corpus.DEVELOPMENT_PARTITION,
        *corpus.GENERATOR_SEMANTICS,
        *corpus.NONCOMMUTING_MOTIFS,
        "composition_depth",
        "semantic_seed",
        "target_relation",
        "oracle",
        "renderer",
        "latent",
    )
    assert not any(item in visible for item in forbidden)


def test_physical_geometry_is_identical_across_splits() -> None:
    receipt = _fixture().manifest.split_receipt
    assert receipt.train_geometry_cells == receipt.development_geometry_cells
    assert receipt.train_physical_cells == receipt.development_physical_cells
    assert set(receipt.train_geometry_cells) == set(corpus.GEOMETRY_CELLS)
    assert {item[0] for item in receipt.train_physical_cells} == {5, 6, 7, 8}
    assert {item[1] for item in receipt.train_physical_cells} == {3, 4}
    assert {item[2] for item in receipt.train_physical_cells} == {3, 4}


def test_semantic_axes_are_shared_but_triples_are_disjoint() -> None:
    receipt = _fixture().manifest.split_receipt
    assert receipt.generator_semantics_support == tuple(
        sorted(corpus.GENERATOR_SEMANTICS)
    )
    assert receipt.noncommuting_motif_support == tuple(
        sorted(corpus.NONCOMMUTING_MOTIFS)
    )
    assert receipt.composition_depth_support == corpus.COMPOSITION_DEPTHS
    assert receipt.semantic_combination_overlap == 0
    assert set(receipt.train_semantic_combinations) == set(corpus.TRAIN_COMBINATIONS)
    assert set(receipt.development_semantic_combinations) == set(
        corpus.DEVELOPMENT_COMBINATIONS
    )
    assert receipt.exact_packet_overlap == 0


def test_semantic_labels_are_real_mechanical_properties() -> None:
    fixture = _fixture()
    hidden_by_orbit = {item.orbit_id: item for item in fixture.hidden_automata}
    for item in fixture.semantic_orbits:
        hidden = hidden_by_orbit[item.orbit_id]
        context = hidden.base_generators[1]
        motif = hidden.base_generators[2]
        assert not corpus._maps_commute(context, motif)  # noqa: SLF001
        assert item.context_and_motif_noncommute
        if item.generator_semantics == "cyclic_context":
            assert item.context_rank == 4
        else:
            assert item.generator_semantics == "sink_context"
            assert item.context_rank == 3
        if item.noncommuting_motif == "reset_motif":
            assert item.motif_rank == 1
        else:
            assert item.noncommuting_motif == "terminal_swap_motif"
            assert item.motif_rank == 4

        latent = corpus.base._HiddenAutomaton(  # noqa: SLF001
            generators=hidden.base_generators,
            observations=hidden.latent_observations,
            marker_state=hidden.collision_pair[0],
            collision_pair=hidden.collision_pair,
        )
        solution = solve_by_refinement(
            corpus.base._latent_packet(latent),  # noqa: SLF001
            distinction_depth=6,
        )
        measured = corpus._maximum_distinction_depth(solution)  # noqa: SLF001
        assert measured == item.composition_depth
        assert measured == item.measured_maximum_distinction_depth


def test_every_packet_has_independent_oracle_agreement() -> None:
    fixture = _fixture()
    target_by_hash = {item.packet_sha256: item for item in fixture.target_relations}
    oracle_by_hash = {item.packet_sha256: item for item in fixture.oracle_agreements}
    for packet in fixture.packets:
        digest = corpus.packet_sha256(packet)
        refinement = compute_refinement_partition(packet)
        exhaustive = compute_exhaustive_partition(packet)
        assert _relation(refinement) == _relation(exhaustive)
        assert _relation(refinement) == _relation(target_by_hash[digest].blocks)
        assert oracle_by_hash[digest].independent_oracles_agree
    assert fixture.manifest.oracle_agreement_count == len(fixture.packets)


def test_renderer_orbits_and_arbitrary_recodings_remain_intact() -> None:
    fixture = _fixture()
    packet_by_hash = {
        corpus.packet_sha256(packet): packet for packet in fixture.packets
    }
    metadata_by_orbit: dict[str, dict[str, corpus.SemanticHoldoutPacketMetadata]] = (
        defaultdict(dict)
    )
    for item in fixture.metadata:
        metadata_by_orbit[item.orbit_id][item.variant] = item
    recoding_by_hash = {
        item.packet_sha256: item for item in fixture.observation_recodings
    }
    fingerprint_by_hash = {item.packet_sha256: item for item in fixture.fingerprints}

    for audit in fixture.orbit_audits:
        variants = metadata_by_orbit[audit.orbit_id]
        assert set(variants) == set(corpus.ORBIT_VARIANTS)
        base_hash = variants["base"].packet_sha256
        recode_hash = variants["value_recode"].packet_sha256
        base_packet = packet_by_hash[base_hash]
        recoded_packet = packet_by_hash[recode_hash]
        assert _observation_equality(base_packet) == _observation_equality(
            recoded_packet
        )
        assert recoding_by_hash[base_hash] != recoding_by_hash[recode_hash]
        preserved = (
            "base",
            "opaque_reindex",
            "value_recode",
            "bisimilar_split",
            "bisimilar_merge",
        )
        assert (
            len(
                {
                    fingerprint_by_hash[variants[name].packet_sha256].latent_sha256
                    for name in preserved
                }
            )
            == 1
        )
        assert audit.reindex_naturality
        assert audit.split_naturality
        assert audit.merge_naturality
        assert audit.value_recode_preserves_quotient
        assert audit.minimal_twin_separates_quotient
        assert audit.collision_separates_path

    values = {
        witness.value
        for packet in fixture.packets
        for witness in packet.observation_witnesses
    }
    assert any(value < 0 for value in values)
    assert any(value > 0 for value in values)
    assert all(abs(value) >= 10_000 for value in values)


def test_manifest_closes_all_orbit_and_semantic_gates() -> None:
    manifest = _fixture().manifest
    assert manifest.packet_count == 128
    assert manifest.unique_packet_count == 128
    assert manifest.train_packet_count == 64
    assert manifest.development_packet_count == 64
    assert manifest.orbit_count == 16
    assert manifest.hidden_minimal_count == 16
    assert manifest.composition_depth_agreement_count == 16
    assert manifest.noncommuting_motif_count == 16
    assert manifest.reindex_orbit_count == 16
    assert manifest.split_orbit_count == 16
    assert manifest.merge_orbit_count == 16
    assert manifest.collision_orbit_count == 16
    assert sum(item.orbit_count for item in manifest.cells) == 16
    assert manifest.packet_manifest_sha256
    assert manifest.offline_ledger_sha256
    assert manifest.payload_sha256
    assert manifest.split_receipt.receipt_sha256
