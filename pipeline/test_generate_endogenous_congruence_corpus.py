from __future__ import annotations

import dataclasses
import json
from collections import Counter, defaultdict
from functools import lru_cache

import pytest

from pipeline import generate_endogenous_congruence_corpus as corpus
from pipeline.endogenous_congruence_board import (
    EndogenousCongruencePacket,
    compute_exhaustive_partition,
    compute_refinement_partition,
    solve_by_refinement,
)


FIXTURE_TRAIN_PACKETS = 32
FIXTURE_DEVELOPMENT_PACKETS = 32


@lru_cache(maxsize=1)
def _fixture() -> corpus.ProceduralEndogenousCongruenceCorpus:
    return corpus.generate_endogenous_congruence_corpus(
        train_packets=FIXTURE_TRAIN_PACKETS,
        development_packets=FIXTURE_DEVELOPMENT_PACKETS,
    )


def _packet_map() -> dict[str, EndogenousCongruencePacket]:
    return {corpus.packet_sha256(packet): packet for packet in _fixture().packets}


def _metadata_map() -> dict[str, corpus.CorpusPacketMetadata]:
    return {item.packet_sha256: item for item in _fixture().metadata}


def _fingerprint_map() -> dict[str, corpus.QuotientFingerprintLedger]:
    return {item.packet_sha256: item for item in _fixture().fingerprints}


def _target_map() -> dict[str, corpus.TargetRelationLedger]:
    return {item.packet_sha256: item for item in _fixture().target_relations}


def _renderer_map() -> dict[str, corpus.RendererLedger]:
    return {item.packet_sha256: item for item in _fixture().renderers}


def _oracle_map() -> dict[str, corpus.OracleAgreementLedger]:
    return {item.packet_sha256: item for item in _fixture().oracle_agreements}


def _variant_map(
    orbit_id: str,
) -> dict[str, tuple[EndogenousCongruencePacket, str]]:
    packets = _packet_map()
    return {
        item.variant: (packets[item.packet_sha256], item.packet_sha256)
        for item in _fixture().metadata
        if item.orbit_id == orbit_id
    }


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


def test_default_and_preregistered_scale_use_one_frozen_semantics() -> None:
    default = corpus.plan_endogenous_congruence_corpus()
    full = corpus.plan_endogenous_congruence_corpus(
        train_packets=corpus.PREREGISTERED_TRAIN_PACKETS,
        development_packets=corpus.PREREGISTERED_DEVELOPMENT_PACKETS,
    )
    assert (
        default.train_packets,
        default.development_packets,
        default.train_orbits,
        default.development_orbits,
    ) == (256, 64, 32, 8)
    assert (
        full.train_packets,
        full.development_packets,
        full.train_orbits,
        full.development_orbits,
    ) == (48_000, 4_000, 6_000, 500)
    assert default.packets_per_orbit == full.packets_per_orbit == 8
    assert default.semantics_sha256 == full.semantics_sha256
    specs = corpus._build_orbit_specs(  # noqa: SLF001
        seed=corpus.DEFAULT_SEED,
        train_packets=FIXTURE_TRAIN_PACKETS,
        development_packets=FIXTURE_DEVELOPMENT_PACKETS,
    )
    train = [item for item in specs if item.partition == corpus.TRAIN_PARTITION]
    development = [
        item for item in specs if item.partition == corpus.DEVELOPMENT_PARTITION
    ]
    assert {(item.generators, item.query_ports) for item in train} == {
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    }
    assert {(item.generators, item.query_ports) for item in development} == {
        (3, 3),
        (3, 4),
        (4, 3),
        (4, 4),
    }
    with pytest.raises(corpus.EndogenousCongruenceCorpusError, match="divisible"):
        corpus.plan_endogenous_congruence_corpus(
            train_packets=31,
            development_packets=32,
        )
    with pytest.raises(corpus.EndogenousCongruenceCorpusError, match="at least"):
        corpus.plan_endogenous_congruence_corpus(
            train_packets=0,
            development_packets=8,
        )


def test_generation_is_reproducible_and_seed_sensitive() -> None:
    first = _fixture()
    replay = corpus.generate_endogenous_congruence_corpus(
        train_packets=FIXTURE_TRAIN_PACKETS,
        development_packets=FIXTURE_DEVELOPMENT_PACKETS,
    )
    alternate = corpus.generate_endogenous_congruence_corpus(
        seed=corpus.DEFAULT_SEED + 1,
        train_packets=FIXTURE_TRAIN_PACKETS,
        development_packets=FIXTURE_DEVELOPMENT_PACKETS,
    )
    assert first == replay
    assert first.manifest.payload_sha256 == replay.manifest.payload_sha256
    assert first.manifest.payload_sha256 != alternate.manifest.payload_sha256
    assert first.manifest.semantics_sha256 == alternate.manifest.semantics_sha256


def test_model_packets_are_source_deleted_and_opaque() -> None:
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
    forbidden = {
        "family",
        "motif",
        "oracle",
        "partition",
        "renderer",
        "seed",
        "target_relation",
        "latent",
        "solution",
        "recode",
    }
    serialized = corpus.serialize_model_packets(_fixture().packets)
    parsed = json.loads(serialized)
    assert len(parsed) == len(_fixture().packets)
    assert not any(f'"{key}"' in serialized for key in forbidden)
    for packet in _fixture().packets:
        identifiers = (
            *packet.records,
            *packet.generators,
            *packet.query_ports,
        )
        assert all(len(item) == 24 for item in identifiers)
        assert all(set(item) <= set("0123456789abcdef") for item in identifiers)
        values = {item.value for item in packet.observation_witnesses}
        assert any(value < 0 for value in values)
        assert any(value > 0 for value in values)
        assert all(abs(value) >= 10_000 for value in values)


def test_every_packet_has_aligned_target_and_independent_oracles() -> None:
    targets = _target_map()
    oracles = _oracle_map()
    for digest, packet in _packet_map().items():
        refinement = compute_refinement_partition(packet)
        exhaustive = compute_exhaustive_partition(packet)
        solution = solve_by_refinement(packet)
        target = targets[digest]
        oracle = oracles[digest]
        assert _relation(refinement) == _relation(exhaustive)
        assert _relation(refinement) == _relation(target.blocks)
        assert target.record_class == solution.record_class
        assert _relation(oracle.refinement_blocks) == _relation(refinement)
        assert _relation(oracle.exhaustive_blocks) == _relation(exhaustive)
        assert oracle.independent_oracles_agree
        assert len(packet.records) <= 8


def test_hidden_automata_are_minimal_and_not_model_visible() -> None:
    fixture = _fixture()
    visible = corpus.serialize_model_packets(fixture.packets)
    assert all(item.family not in visible for item in fixture.hidden_automata)
    assert all(item.motif not in visible for item in fixture.hidden_automata)
    for item in fixture.hidden_automata:
        base = corpus._HiddenAutomaton(  # noqa: SLF001
            generators=item.base_generators,
            observations=item.latent_observations,
            marker_state=item.collision_pair[0],
            collision_pair=item.collision_pair,
        )
        collision = dataclasses.replace(
            base,
            generators=item.collision_generators,
        )
        for hidden in (base, collision):
            packet = corpus._latent_packet(hidden)  # noqa: SLF001
            refinement = compute_refinement_partition(packet)
            exhaustive = compute_exhaustive_partition(packet)
            assert _relation(refinement) == _relation(exhaustive)
            assert all(len(block) == 1 for block in refinement)


def test_hidden_target_relation_aligns_except_for_the_declared_negative_twin() -> None:
    targets = _target_map()
    renderers = _renderer_map()
    for item in _fixture().metadata:
        renderer = renderers[item.packet_sha256]
        expected: dict[int, set[str]] = defaultdict(set)
        for record, state in renderer.record_hidden_state:
            expected[state].add(record)
        expected_relation = frozenset(frozenset(block) for block in expected.values())
        actual = _relation(targets[item.packet_sha256].blocks)
        if item.variant == "minimal_noncongruent":
            assert actual != expected_relation
            assert not item.hidden_relation_applicable
        else:
            assert actual == expected_relation
            assert item.hidden_relation_applicable


def test_split_isolation_covers_exact_latent_action_and_path_signatures() -> None:
    receipt = _fixture().manifest.split_isolation
    assert receipt.train_exact_packets == FIXTURE_TRAIN_PACKETS
    assert receipt.development_exact_packets == FIXTURE_DEVELOPMENT_PACKETS
    assert (
        receipt.exact_packet_overlap,
        receipt.latent_signature_overlap,
        receipt.action_signature_overlap,
        receipt.path_signature_overlap,
    ) == (0, 0, 0, 0)
    metadata = _metadata_map()
    categories: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for item in _fixture().fingerprints:
        partition = metadata[item.packet_sha256].partition
        categories[partition]["latent"].add(item.latent_sha256)
        categories[partition]["action"].add(item.action_sha256)
        categories[partition]["path"].add(item.path_sha256)
    for category in ("latent", "action", "path"):
        assert not (
            categories[corpus.TRAIN_PARTITION][category]
            & categories[corpus.DEVELOPMENT_PARTITION][category]
        )
    train_motifs = {
        item.motif
        for item in _fixture().metadata
        if item.partition == corpus.TRAIN_PARTITION
    }
    development_motifs = {
        item.motif
        for item in _fixture().metadata
        if item.partition == corpus.DEVELOPMENT_PARTITION
    }
    assert not train_motifs & development_motifs


def test_renderer_split_merge_and_value_recode_orbits_preserve_quotient() -> None:
    fingerprints = _fingerprint_map()
    recodings = {item.packet_sha256: item for item in _fixture().observation_recodings}
    for orbit in _fixture().orbit_audits:
        variants = _variant_map(orbit.orbit_id)
        base, base_hash = variants["base"]
        reindexed, reindexed_hash = variants["opaque_reindex"]
        recoded, recoded_hash = variants["value_recode"]
        split, split_hash = variants["bisimilar_split"]
        merge, merge_hash = variants["bisimilar_merge"]
        hashes = (
            base_hash,
            reindexed_hash,
            recoded_hash,
            split_hash,
            merge_hash,
        )
        assert len({fingerprints[item].latent_sha256 for item in hashes}) == 1
        assert len({fingerprints[item].action_sha256 for item in hashes}) == 1
        assert len({fingerprints[item].path_sha256 for item in hashes}) == 1
        assert len(split.records) == len(base.records) + 1
        assert len(merge.records) == len(base.records) - 1
        assert orbit.reindex_naturality
        assert orbit.split_naturality
        assert orbit.merge_naturality
        assert orbit.value_recode_preserves_quotient
        assert recodings[base_hash].query_recodings != (
            recodings[recoded_hash].query_recodings
        )
        assert _observation_equality(base) == _observation_equality(recoded)
        assert len(reindexed.records) == len(base.records)


def test_collision_twins_preserve_marginals_and_separate_behavior() -> None:
    fingerprints = _fingerprint_map()
    for orbit in _fixture().orbit_audits:
        variants = _variant_map(orbit.orbit_id)
        base, base_hash = variants["base"]
        minimal, minimal_hash = variants["minimal_noncongruent"]
        collision, collision_hash = variants["path_collision"]
        base_transitions = {
            (item.source, item.generator): item.target
            for item in base.transition_witnesses
        }
        minimal_transitions = {
            (item.source, item.generator): item.target
            for item in minimal.transition_witnesses
        }
        assert (
            sum(
                base_transitions[key] != minimal_transitions[key]
                for key in base_transitions
            )
            == 2
        )
        assert corpus._simple_marginals(base) == corpus._simple_marginals(minimal)  # noqa: SLF001
        assert corpus._simple_marginals(base) == corpus._simple_marginals(collision)  # noqa: SLF001
        assert (
            fingerprints[base_hash].latent_sha256
            != fingerprints[minimal_hash].latent_sha256
        )
        assert (
            fingerprints[base_hash].path_sha256
            != fingerprints[collision_hash].path_sha256
        )
        assert orbit.minimal_twin_separates_quotient
        assert orbit.minimal_twin_preserves_simple_marginals
        assert orbit.collision_separates_path
        assert orbit.collision_preserves_simple_marginals
        if len(base.generators) >= 2:
            assert orbit.base_commutes is True
            assert orbit.collision_commutes is False
        else:
            assert orbit.base_commutes is None
            assert orbit.collision_commutes is None


def test_all_packets_have_exactly_one_complete_offline_receipt() -> None:
    fixture = _fixture()
    packet_hashes = {corpus.packet_sha256(packet) for packet in fixture.packets}
    assert len(packet_hashes) == fixture.manifest.unique_packet_count == 64
    ledgers = (
        fixture.metadata,
        fixture.renderers,
        fixture.observation_recodings,
        fixture.target_relations,
        fixture.oracle_agreements,
        fixture.fingerprints,
    )
    for ledger in ledgers:
        hashes = [item.packet_sha256 for item in ledger]
        assert len(hashes) == len(packet_hashes)
        assert len(set(hashes)) == len(packet_hashes)
        assert set(hashes) == packet_hashes
    assert len(fixture.hidden_automata) == fixture.manifest.orbit_count == 8
    assert len(fixture.orbit_audits) == fixture.manifest.orbit_count
    assert sum(item.count for item in fixture.manifest.cells) == len(packet_hashes)
    assert fixture.manifest.oracle_agreement_count == len(packet_hashes)
    assert fixture.manifest.packet_manifest_sha256 == corpus._sha256(  # noqa: SLF001
        tuple(corpus.packet_sha256(packet) for packet in fixture.packets)
    )
    assert fixture.manifest.payload_sha256
    assert fixture.manifest.offline_ledger_sha256
    assert fixture.manifest.split_isolation.receipt_sha256


def test_geometry_receipts_cover_requested_train_and_development_regimes() -> None:
    packets = _packet_map()
    metadata = _metadata_map()
    geometry: dict[str, set[tuple[int, int, int]]] = defaultdict(set)
    for digest, item in metadata.items():
        packet = packets[digest]
        geometry[item.partition].add(
            (
                len(packet.records),
                len(packet.generators),
                len(packet.query_ports),
            )
        )
    train = geometry[corpus.TRAIN_PARTITION]
    development = geometry[corpus.DEVELOPMENT_PARTITION]
    assert {records for records, _, _ in train} == {4, 5, 6}
    assert {generators for _, generators, _ in train} == {1, 2}
    assert {queries for _, _, queries in train} == {1, 2}
    assert {6, 7, 8} <= {records for records, _, _ in development}
    assert {generators for _, generators, _ in development} == {3, 4}
    assert {queries for _, _, queries in development} == {3, 4}


def test_nonminimal_capacity_duplicate_overlap_and_oracle_failures_close() -> None:
    nonminimal = corpus._HiddenAutomaton(  # noqa: SLF001
        generators=((0, 1, 2),),
        observations=((0, 0, 0),),
        marker_state=0,
        collision_pair=(0, 1),
    )
    with pytest.raises(
        corpus.EndogenousCongruenceCorpusError,
        match="not minimal",
    ):
        corpus._validate_hidden_automaton(nonminimal)  # noqa: SLF001

    valid = corpus._sample_hidden_automaton(  # noqa: SLF001
        corpus._build_orbit_specs(  # noqa: SLF001
            seed=corpus.DEFAULT_SEED,
            train_packets=8,
            development_packets=8,
        )[0]
    )
    with pytest.raises(
        corpus.EndogenousCongruenceCorpusError,
        match="capacity",
    ):
        corpus._lift_hidden_automaton(  # noqa: SLF001
            valid,
            alias_counts=(3, 3, 3),
            renderer_seed=1,
            recodings=corpus._signed_value_recode(valid, seed=2),  # noqa: SLF001
        )

    packet = _fixture().packets[0]
    with pytest.raises(
        corpus.EndogenousCongruenceCorpusError,
        match="duplicate",
    ):
        corpus._verify_complete_ledgers(  # noqa: SLF001
            packets=(packet, packet),
            metadata=(),
            renderers=(),
            recodings=(),
            targets=(),
            oracles=(),
            fingerprints=(),
        )

    train_meta = _fixture().metadata[0]
    train_fingerprint = _fixture().fingerprints[0]
    development_meta = dataclasses.replace(
        train_meta,
        packet_sha256="f" * 64,
        partition=corpus.DEVELOPMENT_PARTITION,
    )
    development_fingerprint = dataclasses.replace(
        train_fingerprint,
        packet_sha256=development_meta.packet_sha256,
    )
    with pytest.raises(
        corpus.EndogenousCongruenceCorpusError,
        match="split overlap",
    ):
        corpus._split_isolation(  # noqa: SLF001
            (train_meta, development_meta),
            (train_fingerprint, development_fingerprint),
        )


def test_oracle_disagreement_is_detected_before_labels_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = corpus._build_orbit_specs(  # noqa: SLF001
        seed=corpus.DEFAULT_SEED,
        train_packets=8,
        development_packets=8,
    )[0]
    presentations, _ = corpus._build_orbit(spec)  # noqa: SLF001
    presentation = presentations[0]
    refinement = compute_refinement_partition(presentation.packet)
    false_exhaustive = tuple((record,) for record in presentation.packet.records)
    assert _relation(false_exhaustive) != _relation(refinement)
    monkeypatch.setattr(
        corpus,
        "compute_exhaustive_partition",
        lambda _: false_exhaustive,
    )
    with pytest.raises(
        corpus.EndogenousCongruenceCorpusError,
        match="disagree",
    ):
        corpus._audit_packet(  # noqa: SLF001
            presentation,
            require_hidden_relation=True,
        )


def test_observation_recoding_is_arbitrary_signed_and_query_local() -> None:
    recoding_by_hash = {
        item.packet_sha256: item for item in _fixture().observation_recodings
    }
    for digest, packet in _packet_map().items():
        receipt = recoding_by_hash[digest]
        assert {query for query, _ in receipt.query_recodings} == set(
            packet.query_ports
        )
        visible_receipt_values = {
            value for _, mapping in receipt.query_recodings for _, value in mapping
        }
        visible_packet_values = {item.value for item in packet.observation_witnesses}
        assert visible_packet_values <= visible_receipt_values
        assert any(value < 0 for value in visible_receipt_values)
        assert any(value > 0 for value in visible_receipt_values)
        assert not visible_receipt_values & {-1, 0, 1}


def test_receipt_distribution_has_no_missing_variants_or_orbits() -> None:
    variants = Counter(item.variant for item in _fixture().metadata)
    assert set(variants) == set(corpus.ORBIT_VARIANTS)
    assert set(variants.values()) == {_fixture().manifest.orbit_count}
    by_orbit: dict[str, set[str]] = defaultdict(set)
    for item in _fixture().metadata:
        by_orbit[item.orbit_id].add(item.variant)
    assert all(value == set(corpus.ORBIT_VARIANTS) for value in by_orbit.values())
