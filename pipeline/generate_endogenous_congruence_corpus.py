"""Scalable procedural corpus for endogenous congruence induction.

The model-visible side of this module is deliberately narrow: every example is
an :class:`EndogenousCongruencePacket` containing only complete physical
transition and observation witnesses. Hidden latent automata, quotient labels,
renderer maps, value recodings, generation seeds, motifs, and independent
oracle products are kept in separate assessor-side ledgers.

This is a mechanics and custody artifact. It does not claim neural quotient
induction or general reasoning competence.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from pipeline.endogenous_congruence_board import (
    MAX_RECORDS,
    CongruenceSolution,
    EndogenousCongruencePacket,
    ObservationWitness,
    PresentationMorphism,
    TransitionWitness,
    compute_exhaustive_partition,
    compute_refinement_partition,
    model_packet_payload,
    solve_by_refinement,
    validate_packet,
    validate_presentation_naturality,
    validate_solution,
)


DEFAULT_SEED = 202_607_230_1
DEFAULT_TRAIN_PACKETS = 256
DEFAULT_DEVELOPMENT_PACKETS = 64
PREREGISTERED_TRAIN_PACKETS = 48_000
PREREGISTERED_DEVELOPMENT_PACKETS = 4_000
PACKETS_PER_ORBIT = 8
PATH_FINGERPRINT_DEPTH = 4

TRAIN_PARTITION = "quotient_induction_train"
DEVELOPMENT_PARTITION = "quotient_induction_development"

ORBIT_VARIANTS = (
    "base",
    "opaque_reindex",
    "value_recode",
    "bisimilar_split",
    "bisimilar_merge",
    "minimal_noncongruent",
    "path_collision",
    "path_collision_reindex",
)

TRAIN_FAMILIES = (
    "finite_dynamics",
    "symbolic_process",
    "causal_machine",
    "typed_protocol",
)
TRAIN_MOTIFS = (
    "periodic_reachability",
    "commuting_identity_cycle",
)
DEVELOPMENT_FAMILIES = (
    "unseen_multiaction_system",
    "unseen_multisensor_system",
)
DEVELOPMENT_MOTIFS = (
    "higher_arity_composition",
    "deep_path_interference",
    "noncommuting_context",
)

_SEMANTICS_DESCRIPTOR = {
    "version": 1,
    "packet_type": "EndogenousCongruencePacket",
    "complete_unary_transitions": True,
    "complete_observations": True,
    "packets_per_orbit": PACKETS_PER_ORBIT,
    "orbit_variants": ORBIT_VARIANTS,
    "path_fingerprint_depth": PATH_FINGERPRINT_DEPTH,
    "train_geometry": {"N": [4, 6], "G": [1, 2], "Q": [1, 2]},
    "development_geometry": {"N": [6, 8], "G": [3, 4], "Q": [3, 4]},
    "independent_oracles": ("partition_refinement", "bell_partition_search"),
}


class EndogenousCongruenceCorpusError(ValueError):
    """The procedural corpus failed a frozen generation or custody invariant."""


@dataclass(frozen=True)
class EndogenousCongruenceScalePlan:
    train_packets: int
    development_packets: int
    train_orbits: int
    development_orbits: int
    packets_per_orbit: int
    semantics_sha256: str


@dataclass(frozen=True)
class OrbitSpec:
    partition: str
    orbit_index: int
    semantic_seed: int
    base_records: int
    latent_states: int
    generators: int
    query_ports: int
    family: str
    motif: str


@dataclass(frozen=True)
class HiddenLatentAutomatonLedger:
    orbit_id: str
    partition: str
    semantic_seed: int
    family: str
    motif: str
    latent_state_count: int
    generator_count: int
    query_count: int
    base_alias_counts: tuple[int, ...]
    base_generators: tuple[tuple[int, ...], ...]
    collision_generators: tuple[tuple[int, ...], ...]
    latent_observations: tuple[tuple[int, ...], ...]
    collision_pair: tuple[int, int]
    base_latent_sha256: str
    collision_latent_sha256: str


@dataclass(frozen=True)
class CorpusPacketMetadata:
    packet_sha256: str
    partition: str
    orbit_id: str
    variant: str
    family: str
    motif: str
    semantic_seed: int
    renderer_seed: int
    hidden_relation_applicable: bool


@dataclass(frozen=True)
class RendererLedger:
    packet_sha256: str
    record_hidden_state: tuple[tuple[str, int], ...]
    generator_hidden_index: tuple[tuple[str, int], ...]
    query_hidden_index: tuple[tuple[str, int], ...]
    parent_packet_sha256: str | None
    parent_record_map: tuple[tuple[str, str], ...]
    parent_generator_map: tuple[tuple[str, str], ...]
    parent_query_map: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ObservationRecodingLedger:
    packet_sha256: str
    query_recodings: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]


@dataclass(frozen=True)
class TargetRelationLedger:
    packet_sha256: str
    blocks: tuple[tuple[str, ...], ...]
    record_class: tuple[int, ...]


@dataclass(frozen=True)
class OracleAgreementLedger:
    packet_sha256: str
    refinement_blocks: tuple[tuple[str, ...], ...]
    exhaustive_blocks: tuple[tuple[str, ...], ...]
    induced_generators: tuple[tuple[int, ...], ...]
    query_readers: tuple[tuple[int, ...], ...]
    merge_certificate_count: int
    distinction_certificate_count: int
    maximum_distinction_depth: int
    independent_oracles_agree: bool


@dataclass(frozen=True)
class QuotientFingerprintLedger:
    packet_sha256: str
    latent_sha256: str
    action_sha256: str
    path_sha256: str
    quotient_states: int
    generator_count: int
    query_count: int


@dataclass(frozen=True)
class OrbitAuditLedger:
    orbit_id: str
    partition: str
    packet_hashes: tuple[tuple[str, str], ...]
    reindex_naturality: bool
    split_naturality: bool
    merge_naturality: bool
    value_recode_preserves_quotient: bool
    minimal_twin_separates_quotient: bool
    minimal_twin_preserves_simple_marginals: bool
    collision_separates_path: bool
    collision_preserves_simple_marginals: bool
    base_commutes: bool | None
    collision_commutes: bool | None


@dataclass(frozen=True)
class CorpusCellReceipt:
    partition: str
    physical_records: int
    generators: int
    query_ports: int
    variant: str
    count: int


@dataclass(frozen=True)
class SplitIsolationReceipt:
    train_exact_packets: int
    development_exact_packets: int
    exact_packet_overlap: int
    latent_signature_overlap: int
    action_signature_overlap: int
    path_signature_overlap: int
    receipt_sha256: str


@dataclass(frozen=True)
class EndogenousCongruenceCorpusManifest:
    seed: int
    packet_count: int
    train_packet_count: int
    development_packet_count: int
    orbit_count: int
    train_orbit_count: int
    development_orbit_count: int
    packets_per_orbit: int
    unique_packet_count: int
    unique_latent_signatures: int
    unique_action_signatures: int
    unique_path_signatures: int
    observation_value_minimum: int
    observation_value_maximum: int
    negative_observation_values: int
    positive_observation_values: int
    oracle_agreement_count: int
    hidden_minimal_count: int
    reindex_orbit_count: int
    split_orbit_count: int
    merge_orbit_count: int
    collision_orbit_count: int
    semantics_sha256: str
    packet_manifest_sha256: str
    offline_ledger_sha256: str
    split_isolation: SplitIsolationReceipt
    cells: tuple[CorpusCellReceipt, ...]
    payload_sha256: str
    preregistered_train_packets: int
    preregistered_development_packets: int


@dataclass(frozen=True)
class ProceduralEndogenousCongruenceCorpus:
    packets: tuple[EndogenousCongruencePacket, ...]
    hidden_automata: tuple[HiddenLatentAutomatonLedger, ...]
    metadata: tuple[CorpusPacketMetadata, ...]
    renderers: tuple[RendererLedger, ...]
    observation_recodings: tuple[ObservationRecodingLedger, ...]
    target_relations: tuple[TargetRelationLedger, ...]
    oracle_agreements: tuple[OracleAgreementLedger, ...]
    fingerprints: tuple[QuotientFingerprintLedger, ...]
    orbit_audits: tuple[OrbitAuditLedger, ...]
    manifest: EndogenousCongruenceCorpusManifest


@dataclass(frozen=True)
class _HiddenAutomaton:
    generators: tuple[tuple[int, ...], ...]
    observations: tuple[tuple[int, ...], ...]
    marker_state: int
    collision_pair: tuple[int, int]


@dataclass(frozen=True)
class _Presentation:
    packet: EndogenousCongruencePacket
    record_hidden_state: tuple[tuple[str, int], ...]
    generator_hidden_index: tuple[tuple[str, int], ...]
    query_hidden_index: tuple[tuple[str, int], ...]
    query_recodings: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    renderer_seed: int
    parent_record_map: tuple[tuple[str, str], ...] = ()
    parent_generator_map: tuple[tuple[str, str], ...] = ()
    parent_query_map: tuple[tuple[str, str], ...] = ()
    parent_packet_sha256: str | None = None


@dataclass(frozen=True)
class _AuditedPacket:
    packet_sha256: str
    solution: CongruenceSolution
    target: TargetRelationLedger
    oracle: OracleAgreementLedger
    fingerprint: QuotientFingerprintLedger


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def packet_sha256(packet: EndogenousCongruencePacket) -> str:
    """Hash only the model-visible physical packet."""

    return _sha256(model_packet_payload(packet))


def serialize_model_packets(
    packets: Sequence[EndogenousCongruencePacket],
) -> str:
    """Serialize only model-visible packets, excluding all assessor ledgers."""

    return _canonical_json(tuple(model_packet_payload(packet) for packet in packets))


def _derive_seed(seed: int, *parts: object) -> int:
    payload = ":".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big")


def _opaque(seed: int, domain: str, index: int) -> str:
    payload = f"{seed}:{domain}:{index}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:24]


def plan_endogenous_congruence_corpus(
    *,
    train_packets: int = DEFAULT_TRAIN_PACKETS,
    development_packets: int = DEFAULT_DEVELOPMENT_PACKETS,
) -> EndogenousCongruenceScalePlan:
    """Validate counts without constructing any automata or packets."""

    for name, count in (
        ("train_packets", train_packets),
        ("development_packets", development_packets),
    ):
        if count < PACKETS_PER_ORBIT:
            raise EndogenousCongruenceCorpusError(
                f"{name} must contain at least one complete orbit"
            )
        if count % PACKETS_PER_ORBIT:
            raise EndogenousCongruenceCorpusError(
                f"{name} must be divisible by {PACKETS_PER_ORBIT}"
            )
    return EndogenousCongruenceScalePlan(
        train_packets=train_packets,
        development_packets=development_packets,
        train_orbits=train_packets // PACKETS_PER_ORBIT,
        development_orbits=development_packets // PACKETS_PER_ORBIT,
        packets_per_orbit=PACKETS_PER_ORBIT,
        semantics_sha256=_sha256(_SEMANTICS_DESCRIPTOR),
    )


def _build_orbit_specs(
    *,
    seed: int,
    train_packets: int,
    development_packets: int,
) -> tuple[OrbitSpec, ...]:
    plan = plan_endogenous_congruence_corpus(
        train_packets=train_packets,
        development_packets=development_packets,
    )
    output: list[OrbitSpec] = []
    global_index = 0
    for partition, orbit_count in (
        (TRAIN_PARTITION, plan.train_orbits),
        (DEVELOPMENT_PARTITION, plan.development_orbits),
    ):
        for local_index in range(orbit_count):
            semantic_seed = _derive_seed(seed, partition, local_index, "semantic")
            if partition == TRAIN_PARTITION:
                generator_count = 1 + local_index % 2
                query_count = 1 + (local_index // 2) % 2
                family = TRAIN_FAMILIES[local_index % len(TRAIN_FAMILIES)]
                motif = TRAIN_MOTIFS[generator_count - 1]
                base_records = 5
                latent_states = 3
            else:
                generator_count = 3 + local_index % 2
                query_count = 3 + (local_index // 2) % 2
                family = DEVELOPMENT_FAMILIES[local_index % len(DEVELOPMENT_FAMILIES)]
                motif = DEVELOPMENT_MOTIFS[local_index % len(DEVELOPMENT_MOTIFS)]
                base_records = 7
                latent_states = 4
            output.append(
                OrbitSpec(
                    partition=partition,
                    orbit_index=global_index,
                    semantic_seed=semantic_seed,
                    base_records=base_records,
                    latent_states=latent_states,
                    generators=generator_count,
                    query_ports=query_count,
                    family=family,
                    motif=motif,
                )
            )
            global_index += 1
    return tuple(output)


def _blocks_signature(
    blocks: Sequence[Sequence[str]],
) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(block) for block in blocks)


def _latent_packet(hidden: _HiddenAutomaton) -> EndogenousCongruencePacket:
    state_count = len(hidden.generators[0])
    records = tuple(f"latent_state_{index}" for index in range(state_count))
    generators = tuple(
        f"latent_generator_{index}" for index in range(len(hidden.generators))
    )
    queries = tuple(
        f"latent_query_{index}" for index in range(len(hidden.observations))
    )
    return EndogenousCongruencePacket(
        records=records,
        generators=generators,
        query_ports=queries,
        transition_witnesses=tuple(
            TransitionWitness(
                records[source],
                generators[generator],
                records[hidden.generators[generator][source]],
            )
            for source in range(state_count)
            for generator in range(len(generators))
        ),
        observation_witnesses=tuple(
            ObservationWitness(
                records[state],
                queries[query],
                hidden.observations[query][state],
            )
            for state in range(state_count)
            for query in range(len(queries))
        ),
    )


def _validate_hidden_automaton(hidden: _HiddenAutomaton) -> None:
    packet = _latent_packet(hidden)
    refinement = compute_refinement_partition(packet)
    exhaustive = compute_exhaustive_partition(packet)
    if _blocks_signature(refinement) != _blocks_signature(exhaustive):
        raise EndogenousCongruenceCorpusError(
            "hidden refinement and exhaustive quotient disagree"
        )
    if any(len(block) != 1 for block in refinement):
        raise EndogenousCongruenceCorpusError("sampled hidden automaton is not minimal")


def _sample_hidden_automaton(spec: OrbitSpec) -> _HiddenAutomaton:
    rng = random.Random(spec.semantic_seed)
    state_count = spec.latent_states
    cycle_order = list(range(state_count))
    rng.shuffle(cycle_order)
    cycle = [0] * state_count
    for index, source in enumerate(cycle_order):
        cycle[source] = cycle_order[(index + 1) % state_count]

    if spec.generators == 1:
        generators: list[tuple[int, ...]] = [tuple(cycle)]
    else:
        generators = [tuple(range(state_count)), tuple(cycle)]
        for _ in range(2, spec.generators):
            generators.append(
                tuple(rng.randrange(state_count) for _ in range(state_count))
            )

    marker_state = cycle_order[0]
    observations: list[tuple[int, ...]] = [
        tuple(1 if state == marker_state else 0 for state in range(state_count))
    ]
    for query_index in range(1, spec.query_ports):
        cardinality = 2 + query_index % min(2, state_count - 1)
        values = [rng.randrange(cardinality) for _ in range(state_count)]
        if len(set(values)) < 2:
            values[cycle_order[-1]] = (values[cycle_order[0]] + 1) % cardinality
        observations.append(tuple(values))

    partner = cycle_order[1]
    hidden = _HiddenAutomaton(
        generators=tuple(generators),
        observations=tuple(observations),
        marker_state=marker_state,
        collision_pair=(marker_state, partner),
    )
    _validate_hidden_automaton(hidden)
    return hidden


def _alias_counts(
    *,
    state_count: int,
    physical_count: int,
    collision_pair: tuple[int, int],
    rng: random.Random,
) -> tuple[int, ...]:
    if physical_count < state_count + 2:
        raise EndogenousCongruenceCorpusError(
            "base presentation needs two aliases beyond the latent state count"
        )
    counts = [1] * state_count
    counts[collision_pair[0]] += 1
    counts[collision_pair[1]] += 1
    remaining = physical_count - sum(counts)
    candidates = [state for state in range(state_count) if state not in collision_pair]
    rng.shuffle(candidates)
    for offset in range(remaining):
        counts[candidates[offset % len(candidates)]] += 1
    if counts[collision_pair[0]] != counts[collision_pair[1]]:
        raise EndogenousCongruenceCorpusError(
            "collision states must have matched alias counts"
        )
    return tuple(counts)


def _signed_value_recode(
    hidden: _HiddenAutomaton,
    *,
    seed: int,
) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    rng = random.Random(seed)
    output: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    used: set[int] = set()
    for query, values in enumerate(hidden.observations):
        categories = sorted(set(values))
        mapping: list[tuple[int, int]] = []
        for category_index, category in enumerate(categories):
            sign = -1 if category_index % 2 == 0 else 1
            while True:
                magnitude = rng.randrange(10_000, 2_000_000_000)
                visible = sign * magnitude
                if visible not in used:
                    used.add(visible)
                    break
            mapping.append((category, visible))
        rng.shuffle(mapping)
        output.append((query, tuple(sorted(mapping))))
    return tuple(output)


def _recode_lookup(
    recodings: Sequence[tuple[int, Sequence[tuple[int, int]]]],
) -> dict[int, dict[int, int]]:
    return {query: dict(mapping) for query, mapping in recodings}


def _lift_hidden_automaton(
    hidden: _HiddenAutomaton,
    *,
    alias_counts: Sequence[int],
    renderer_seed: int,
    recodings: tuple[tuple[int, tuple[tuple[int, int], ...]], ...],
) -> _Presentation:
    rng = random.Random(renderer_seed)
    state_count = len(hidden.generators[0])
    if len(alias_counts) != state_count or any(count < 1 for count in alias_counts):
        raise EndogenousCongruenceCorpusError("invalid physical alias allocation")
    physical_count = sum(alias_counts)
    if physical_count > MAX_RECORDS:
        raise EndogenousCongruenceCorpusError("physical lift exceeds record capacity")

    hidden_assignments = [
        state for state, count in enumerate(alias_counts) for _ in range(count)
    ]
    rng.shuffle(hidden_assignments)
    records = tuple(
        _opaque(renderer_seed, "record", index) for index in range(physical_count)
    )
    generators = tuple(
        _opaque(renderer_seed, "generator", index)
        for index in range(len(hidden.generators))
    )
    queries = tuple(
        _opaque(renderer_seed, "query", index)
        for index in range(len(hidden.observations))
    )
    record_hidden = tuple(zip(records, hidden_assignments, strict=True))
    records_by_state: dict[int, list[str]] = defaultdict(list)
    for record, state in record_hidden:
        records_by_state[state].append(record)

    representative: dict[tuple[int, int], str] = {}
    for state in range(state_count):
        for generator in range(len(hidden.generators)):
            target_state = hidden.generators[generator][state]
            representative[(state, generator)] = rng.choice(
                records_by_state[target_state]
            )

    recode = _recode_lookup(recodings)
    packet = EndogenousCongruencePacket(
        records=records,
        generators=generators,
        query_ports=queries,
        transition_witnesses=tuple(
            TransitionWitness(
                record,
                generator_id,
                representative[(state, generator)],
            )
            for record, state in record_hidden
            for generator, generator_id in enumerate(generators)
        ),
        observation_witnesses=tuple(
            ObservationWitness(
                record,
                query_id,
                recode[query][hidden.observations[query][state]],
            )
            for record, state in record_hidden
            for query, query_id in enumerate(queries)
        ),
    )
    validate_packet(packet)
    return _Presentation(
        packet=packet,
        record_hidden_state=record_hidden,
        generator_hidden_index=tuple(
            (identifier, index) for index, identifier in enumerate(generators)
        ),
        query_hidden_index=tuple(
            (identifier, index) for index, identifier in enumerate(queries)
        ),
        query_recodings=recodings,
        renderer_seed=renderer_seed,
    )


def _packet_tables(
    packet: EndogenousCongruencePacket,
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], int]]:
    transition = {
        (item.source, item.generator): item.target
        for item in packet.transition_witnesses
    }
    observation = {
        (item.record, item.query_port): item.value
        for item in packet.observation_witnesses
    }
    return transition, observation


def _rename_presentation(
    source: _Presentation,
    *,
    renderer_seed: int,
) -> _Presentation:
    rng = random.Random(renderer_seed)
    record_items = list(source.packet.records)
    generator_items = list(source.packet.generators)
    query_items = list(source.packet.query_ports)
    rng.shuffle(record_items)
    rng.shuffle(generator_items)
    rng.shuffle(query_items)
    record_map = {
        old: _opaque(renderer_seed, "record", index)
        for index, old in enumerate(record_items)
    }
    generator_map = {
        old: _opaque(renderer_seed, "generator", index)
        for index, old in enumerate(generator_items)
    }
    query_map = {
        old: _opaque(renderer_seed, "query", index)
        for index, old in enumerate(query_items)
    }
    transition, observation = _packet_tables(source.packet)
    packet = EndogenousCongruencePacket(
        records=tuple(record_map[item] for item in record_items),
        generators=tuple(generator_map[item] for item in generator_items),
        query_ports=tuple(query_map[item] for item in query_items),
        transition_witnesses=tuple(
            TransitionWitness(
                record_map[source_record],
                generator_map[generator],
                record_map[transition[(source_record, generator)]],
            )
            for source_record in record_items
            for generator in generator_items
        ),
        observation_witnesses=tuple(
            ObservationWitness(
                record_map[record],
                query_map[query],
                observation[(record, query)],
            )
            for record in record_items
            for query in query_items
        ),
    )
    hidden_by_record = dict(source.record_hidden_state)
    generator_hidden = dict(source.generator_hidden_index)
    query_hidden = dict(source.query_hidden_index)
    validate_packet(packet)
    return _Presentation(
        packet=packet,
        record_hidden_state=tuple(
            (record_map[record], hidden_by_record[record]) for record in record_items
        ),
        generator_hidden_index=tuple(
            (generator_map[item], generator_hidden[item]) for item in generator_items
        ),
        query_hidden_index=tuple(
            (query_map[item], query_hidden[item]) for item in query_items
        ),
        query_recodings=source.query_recodings,
        renderer_seed=renderer_seed,
        parent_record_map=tuple(record_map.items()),
        parent_generator_map=tuple(generator_map.items()),
        parent_query_map=tuple(query_map.items()),
        parent_packet_sha256=packet_sha256(source.packet),
    )


def _recode_presentation(
    source: _Presentation,
    hidden: _HiddenAutomaton,
    *,
    renderer_seed: int,
) -> _Presentation:
    recodings = _signed_value_recode(hidden, seed=renderer_seed)
    if recodings == source.query_recodings:
        raise EndogenousCongruenceCorpusError("value recoding replayed the base codes")
    record_hidden = dict(source.record_hidden_state)
    query_hidden = dict(source.query_hidden_index)
    recode = _recode_lookup(recodings)
    transition, _ = _packet_tables(source.packet)
    packet = EndogenousCongruencePacket(
        records=source.packet.records,
        generators=source.packet.generators,
        query_ports=source.packet.query_ports,
        transition_witnesses=tuple(
            TransitionWitness(record, generator, transition[(record, generator)])
            for record in source.packet.records
            for generator in source.packet.generators
        ),
        observation_witnesses=tuple(
            ObservationWitness(
                record,
                query,
                recode[query_hidden[query]][
                    hidden.observations[query_hidden[query]][record_hidden[record]]
                ],
            )
            for record in source.packet.records
            for query in source.packet.query_ports
        ),
    )
    validate_packet(packet)
    return dataclasses.replace(
        source,
        packet=packet,
        query_recodings=recodings,
        renderer_seed=renderer_seed,
        parent_packet_sha256=packet_sha256(source.packet),
    )


def _split_presentation(
    source: _Presentation,
    *,
    renderer_seed: int,
) -> _Presentation:
    if len(source.packet.records) >= MAX_RECORDS:
        raise EndogenousCongruenceCorpusError("split would exceed record capacity")
    rng = random.Random(renderer_seed)
    hidden_by_record = dict(source.record_hidden_state)
    counts = Counter(hidden_by_record.values())
    duplicate = rng.choice(
        [record for record in source.packet.records if counts[hidden_by_record[record]]]
    )
    new_record = _opaque(renderer_seed, "split_record", 0)
    if new_record in source.packet.records:
        raise EndogenousCongruenceCorpusError("split renderer identifier collision")
    records = list(source.packet.records)
    records.insert(rng.randrange(len(records) + 1), new_record)
    transition, observation = _packet_tables(source.packet)
    new_transitions: dict[tuple[str, str], str] = {}
    for record in source.packet.records:
        for generator in source.packet.generators:
            target = transition[(record, generator)]
            if target == duplicate and rng.randrange(2):
                target = new_record
            new_transitions[(record, generator)] = target
    for generator in source.packet.generators:
        target = transition[(duplicate, generator)]
        if target == duplicate and rng.randrange(2):
            target = new_record
        new_transitions[(new_record, generator)] = target
    packet = EndogenousCongruencePacket(
        records=tuple(records),
        generators=source.packet.generators,
        query_ports=source.packet.query_ports,
        transition_witnesses=tuple(
            TransitionWitness(record, generator, new_transitions[(record, generator)])
            for record in records
            for generator in source.packet.generators
        ),
        observation_witnesses=tuple(
            ObservationWitness(
                record,
                query,
                observation[(duplicate if record == new_record else record, query)],
            )
            for record in records
            for query in source.packet.query_ports
        ),
    )
    validate_packet(packet)
    split_hidden = hidden_by_record[duplicate]
    return _Presentation(
        packet=packet,
        record_hidden_state=tuple(
            (
                record,
                split_hidden if record == new_record else hidden_by_record[record],
            )
            for record in records
        ),
        generator_hidden_index=source.generator_hidden_index,
        query_hidden_index=source.query_hidden_index,
        query_recodings=source.query_recodings,
        renderer_seed=renderer_seed,
        parent_record_map=tuple(
            (record, duplicate if record == new_record else record)
            for record in records
        ),
        parent_generator_map=tuple((item, item) for item in source.packet.generators),
        parent_query_map=tuple((item, item) for item in source.packet.query_ports),
        parent_packet_sha256=packet_sha256(source.packet),
    )


def _merge_presentation(
    source: _Presentation,
    *,
    renderer_seed: int,
) -> _Presentation:
    rng = random.Random(renderer_seed)
    hidden_by_record = dict(source.record_hidden_state)
    records_by_state: dict[int, list[str]] = defaultdict(list)
    for record, state in source.record_hidden_state:
        records_by_state[state].append(record)
    merge_states = [
        state for state, records in records_by_state.items() if len(records) > 1
    ]
    if not merge_states:
        raise EndogenousCongruenceCorpusError(
            "merge requires a bisimilar physical alias pair"
        )
    state = rng.choice(merge_states)
    removed, survivor = rng.sample(records_by_state[state], 2)
    record_map = {
        record: survivor if record == removed else record
        for record in source.packet.records
    }
    records = tuple(record for record in source.packet.records if record != removed)
    transition, observation = _packet_tables(source.packet)
    packet = EndogenousCongruencePacket(
        records=records,
        generators=source.packet.generators,
        query_ports=source.packet.query_ports,
        transition_witnesses=tuple(
            TransitionWitness(
                record,
                generator,
                record_map[transition[(record, generator)]],
            )
            for record in records
            for generator in source.packet.generators
        ),
        observation_witnesses=tuple(
            ObservationWitness(record, query, observation[(record, query)])
            for record in records
            for query in source.packet.query_ports
        ),
    )
    validate_packet(packet)
    return _Presentation(
        packet=packet,
        record_hidden_state=tuple(
            (record, hidden_by_record[record]) for record in records
        ),
        generator_hidden_index=source.generator_hidden_index,
        query_hidden_index=source.query_hidden_index,
        query_recodings=source.query_recodings,
        renderer_seed=renderer_seed,
        parent_record_map=tuple(record_map.items()),
        parent_generator_map=tuple((item, item) for item in source.packet.generators),
        parent_query_map=tuple((item, item) for item in source.packet.query_ports),
        parent_packet_sha256=packet_sha256(source.packet),
    )


def _equivalence_relation(
    blocks: Sequence[Sequence[str]],
) -> frozenset[frozenset[str]]:
    return frozenset(
        frozenset((left, right))
        for block in blocks
        for left in block
        for right in block
    )


def _minimal_noncongruent_presentation(
    source: _Presentation,
    *,
    renderer_seed: int,
) -> _Presentation:
    base_blocks = compute_refinement_partition(source.packet)
    base_relation = _equivalence_relation(base_blocks)
    transition, observation = _packet_tables(source.packet)
    record_class = {
        record: block_index
        for block_index, block in enumerate(base_blocks)
        for record in block
    }
    rng = random.Random(renderer_seed)
    candidates: list[tuple[str, str, str]] = []
    for block in base_blocks:
        if len(block) < 2:
            continue
        for first in block:
            for second in source.packet.records:
                if record_class[first] == record_class[second]:
                    continue
                for generator in source.packet.generators:
                    first_target = transition[(first, generator)]
                    second_target = transition[(second, generator)]
                    if record_class[first_target] != record_class[second_target]:
                        candidates.append((first, second, generator))
    rng.shuffle(candidates)
    for first, second, generator in candidates:
        changed = dict(transition)
        changed[(first, generator)], changed[(second, generator)] = (
            changed[(second, generator)],
            changed[(first, generator)],
        )
        packet = EndogenousCongruencePacket(
            records=source.packet.records,
            generators=source.packet.generators,
            query_ports=source.packet.query_ports,
            transition_witnesses=tuple(
                TransitionWitness(record, item, changed[(record, item)])
                for record in source.packet.records
                for item in source.packet.generators
            ),
            observation_witnesses=tuple(
                ObservationWitness(record, query, observation[(record, query)])
                for record in source.packet.records
                for query in source.packet.query_ports
            ),
        )
        refinement = compute_refinement_partition(packet)
        exhaustive = compute_exhaustive_partition(packet)
        if _blocks_signature(refinement) != _blocks_signature(exhaustive):
            continue
        if _equivalence_relation(refinement) == base_relation:
            continue
        if len(refinement) <= len(base_blocks):
            continue
        validate_packet(packet)
        return dataclasses.replace(
            source,
            packet=packet,
            renderer_seed=renderer_seed,
            parent_packet_sha256=packet_sha256(source.packet),
        )
    raise EndogenousCongruenceCorpusError(
        "could not construct a two-edge minimal noncongruence twin"
    )


def _collision_hidden(hidden: _HiddenAutomaton) -> _HiddenAutomaton:
    state_count = len(hidden.generators[0])
    left, right = hidden.collision_pair
    if len(hidden.generators) == 1:
        collision = list(range(state_count))
        collision[left], collision[right] = collision[right], collision[left]
        generators = (tuple(collision),)
    else:
        collision = list(range(state_count))
        collision[left], collision[right] = collision[right], collision[left]
        generators = (tuple(collision), *hidden.generators[1:])
    output = dataclasses.replace(hidden, generators=tuple(generators))
    _validate_hidden_automaton(output)
    return output


def _collision_presentation(
    source: _Presentation,
    base_hidden: _HiddenAutomaton,
    collision_hidden: _HiddenAutomaton,
    *,
    renderer_seed: int,
) -> _Presentation:
    rng = random.Random(renderer_seed)
    record_hidden = dict(source.record_hidden_state)
    generator_hidden = dict(source.generator_hidden_index)
    records_by_state: dict[int, list[str]] = defaultdict(list)
    for record, state in source.record_hidden_state:
        records_by_state[state].append(record)
    base_transition, observation = _packet_tables(source.packet)
    transition: dict[tuple[str, str], str] = {}
    representative: dict[tuple[int, int], str] = {}
    for state in range(len(collision_hidden.generators[0])):
        for generator in range(len(collision_hidden.generators)):
            target_state = collision_hidden.generators[generator][state]
            representative[(state, generator)] = rng.choice(
                records_by_state[target_state]
            )
    for record in source.packet.records:
        state = record_hidden[record]
        for generator_id in source.packet.generators:
            generator = generator_hidden[generator_id]
            if (
                base_hidden.generators[generator][state]
                == collision_hidden.generators[generator][state]
            ):
                target = base_transition[(record, generator_id)]
            else:
                target = representative[(state, generator)]
            transition[(record, generator_id)] = target
    packet = EndogenousCongruencePacket(
        records=source.packet.records,
        generators=source.packet.generators,
        query_ports=source.packet.query_ports,
        transition_witnesses=tuple(
            TransitionWitness(record, generator, transition[(record, generator)])
            for record in source.packet.records
            for generator in source.packet.generators
        ),
        observation_witnesses=tuple(
            ObservationWitness(record, query, observation[(record, query)])
            for record in source.packet.records
            for query in source.packet.query_ports
        ),
    )
    validate_packet(packet)
    return dataclasses.replace(
        source,
        packet=packet,
        renderer_seed=renderer_seed,
        parent_packet_sha256=packet_sha256(source.packet),
    )


def _expected_blocks(
    presentation: _Presentation,
) -> frozenset[frozenset[str]]:
    blocks: dict[int, set[str]] = defaultdict(set)
    for record, state in presentation.record_hidden_state:
        blocks[state].add(record)
    return frozenset(frozenset(block) for block in blocks.values())


def _matrix_map(matrix: Sequence[Sequence[int]]) -> tuple[int, ...]:
    output: list[int] = []
    for row in matrix:
        if sum(row) != 1:
            raise EndogenousCongruenceCorpusError(
                "oracle generator matrix is not row-one-hot"
            )
        output.append(tuple(row).index(1))
    return tuple(output)


def _canonicalize_values(values: Sequence[int]) -> tuple[int, ...]:
    owner: dict[int, int] = {}
    return tuple(owner.setdefault(value, len(owner)) for value in values)


def _canonical_quotient(
    solution: CongruenceSolution,
) -> tuple[
    tuple[object, ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    state_count = len(solution.blocks)
    generators = tuple(_matrix_map(matrix) for matrix in solution.induced_generators)
    readers = solution.query_readers
    best: tuple[object, ...] | None = None
    best_generators: tuple[tuple[int, ...], ...] | None = None
    best_readers: tuple[tuple[int, ...], ...] | None = None
    for order in permutations(range(state_count)):
        inverse = [0] * state_count
        for new, old in enumerate(order):
            inverse[old] = new
        transformed_generators = tuple(
            sorted(
                tuple(inverse[generator[old]] for old in order)
                for generator in generators
            )
        )
        transformed_readers = tuple(
            sorted(
                _canonicalize_values(tuple(reader[old] for old in order))
                for reader in readers
            )
        )
        candidate: tuple[object, ...] = (
            state_count,
            len(generators),
            len(readers),
            transformed_generators,
            transformed_readers,
        )
        if best is None or candidate < best:
            best = candidate
            best_generators = transformed_generators
            best_readers = transformed_readers
    if best is None or best_generators is None or best_readers is None:
        raise EndogenousCongruenceCorpusError("empty quotient cannot be canonicalized")
    return best, best_generators, best_readers


def _compose(
    left: Sequence[int],
    right: Sequence[int],
) -> tuple[int, ...]:
    return tuple(right[left[state]] for state in range(len(left)))


def _bounded_actions(
    generators: Sequence[Sequence[int]],
    *,
    depth: int,
) -> tuple[tuple[int, ...], ...]:
    state_count = len(generators[0])
    identity = tuple(range(state_count))
    actions = {identity}
    frontier = {identity}
    for _ in range(depth):
        next_frontier = {
            _compose(action, generator)
            for action in frontier
            for generator in generators
        }
        actions.update(next_frontier)
        frontier = next_frontier
    return tuple(sorted(actions))


def _path_signature(
    generators: tuple[tuple[int, ...], ...],
    *,
    depth: int,
) -> tuple[object, ...]:
    state_count = len(generators[0])
    identity = tuple(range(state_count))
    actions = _bounded_actions(generators, depth=depth)
    action_index = {action: index for index, action in enumerate(actions)}
    word_classes: list[tuple[int, int]] = [(0, action_index[identity])]
    for word_depth in range(1, depth + 1):
        for word in product(range(len(generators)), repeat=word_depth):
            action = identity
            for generator in word:
                action = _compose(action, generators[generator])
            word_classes.append((word_depth, action_index[action]))
    commuting = tuple(
        int(_compose(left, right) == _compose(right, left))
        for left in generators
        for right in generators
    )
    return (
        state_count,
        len(generators),
        depth,
        actions,
        tuple(word_classes),
        commuting,
    )


def _fingerprint_solution(
    packet_digest: str,
    solution: CongruenceSolution,
) -> QuotientFingerprintLedger:
    canonical, generators, readers = _canonical_quotient(solution)
    actions = _bounded_actions(generators, depth=2)
    paths = _path_signature(generators, depth=PATH_FINGERPRINT_DEPTH)
    return QuotientFingerprintLedger(
        packet_sha256=packet_digest,
        latent_sha256=_sha256(canonical),
        action_sha256=_sha256(
            (len(solution.blocks), len(generators), generators, actions)
        ),
        path_sha256=_sha256(paths),
        quotient_states=len(solution.blocks),
        generator_count=len(generators),
        query_count=len(readers),
    )


def _audit_packet(
    presentation: _Presentation,
    *,
    require_hidden_relation: bool,
) -> _AuditedPacket:
    packet = presentation.packet
    validate_packet(packet)
    refinement = compute_refinement_partition(packet)
    exhaustive = compute_exhaustive_partition(packet)
    if _blocks_signature(refinement) != _blocks_signature(exhaustive):
        raise EndogenousCongruenceCorpusError(
            "partition refinement and exhaustive reference disagree"
        )
    if require_hidden_relation and _blocks_signature(refinement) != _expected_blocks(
        presentation
    ):
        raise EndogenousCongruenceCorpusError(
            "physical quotient does not align with the hidden target relation"
        )
    solution = solve_by_refinement(
        packet,
        path_depth=PATH_FINGERPRINT_DEPTH,
        distinction_depth=6,
    )
    validate_solution(packet, solution)
    digest = packet_sha256(packet)
    target = TargetRelationLedger(
        packet_sha256=digest,
        blocks=solution.blocks,
        record_class=solution.record_class,
    )
    oracle = OracleAgreementLedger(
        packet_sha256=digest,
        refinement_blocks=refinement,
        exhaustive_blocks=exhaustive,
        induced_generators=tuple(
            _matrix_map(matrix) for matrix in solution.induced_generators
        ),
        query_readers=solution.query_readers,
        merge_certificate_count=len(solution.merge_certificates),
        distinction_certificate_count=len(solution.distinction_certificates),
        maximum_distinction_depth=max(
            (
                len(certificate.continuation)
                for certificate in solution.distinction_certificates
            ),
            default=0,
        ),
        independent_oracles_agree=True,
    )
    return _AuditedPacket(
        packet_sha256=digest,
        solution=solution,
        target=target,
        oracle=oracle,
        fingerprint=_fingerprint_solution(digest, solution),
    )


def _simple_marginals(packet: EndogenousCongruencePacket) -> tuple[object, ...]:
    transition, observation = _packet_tables(packet)
    incoming_by_generator = tuple(
        sorted(
            Counter(transition[(record, generator)] for record in packet.records)[
                target
            ]
            for target in packet.records
        )
        for generator in packet.generators
    )
    observations_by_query = tuple(
        sorted(observation[(record, query)] for record in packet.records)
        for query in packet.query_ports
    )
    return (
        len(packet.records),
        len(packet.generators),
        len(packet.query_ports),
        tuple(sorted(incoming_by_generator)),
        tuple(sorted(observations_by_query)),
    )


def _generators_commute(solution: CongruenceSolution) -> bool | None:
    generators = tuple(_matrix_map(matrix) for matrix in solution.induced_generators)
    if len(generators) < 2:
        return None
    return _compose(generators[0], generators[1]) == _compose(
        generators[1],
        generators[0],
    )


def _renderer_ledger(
    presentation: _Presentation,
    digest: str,
) -> RendererLedger:
    return RendererLedger(
        packet_sha256=digest,
        record_hidden_state=presentation.record_hidden_state,
        generator_hidden_index=presentation.generator_hidden_index,
        query_hidden_index=presentation.query_hidden_index,
        parent_packet_sha256=presentation.parent_packet_sha256,
        parent_record_map=presentation.parent_record_map,
        parent_generator_map=presentation.parent_generator_map,
        parent_query_map=presentation.parent_query_map,
    )


def _recoding_ledger(
    presentation: _Presentation,
    digest: str,
) -> ObservationRecodingLedger:
    query_hidden = dict(presentation.query_hidden_index)
    recodings = dict(presentation.query_recodings)
    return ObservationRecodingLedger(
        packet_sha256=digest,
        query_recodings=tuple(
            (query, recodings[query_hidden[query]])
            for query in presentation.packet.query_ports
        ),
    )


def _build_orbit(
    spec: OrbitSpec,
) -> tuple[
    tuple[_Presentation, ...],
    HiddenLatentAutomatonLedger,
]:
    hidden = _sample_hidden_automaton(spec)
    collision_hidden = _collision_hidden(hidden)
    alias_rng = random.Random(_derive_seed(spec.semantic_seed, "aliases"))
    alias_counts = _alias_counts(
        state_count=spec.latent_states,
        physical_count=spec.base_records,
        collision_pair=hidden.collision_pair,
        rng=alias_rng,
    )
    base_recode = _signed_value_recode(
        hidden,
        seed=_derive_seed(spec.semantic_seed, "base_recode"),
    )
    base = _lift_hidden_automaton(
        hidden,
        alias_counts=alias_counts,
        renderer_seed=_derive_seed(spec.semantic_seed, "base_renderer"),
        recodings=base_recode,
    )
    reindexed = _rename_presentation(
        base,
        renderer_seed=_derive_seed(spec.semantic_seed, "reindex_renderer"),
    )
    value_recode = _recode_presentation(
        base,
        hidden,
        renderer_seed=_derive_seed(spec.semantic_seed, "value_recode"),
    )
    split = _split_presentation(
        base,
        renderer_seed=_derive_seed(spec.semantic_seed, "split_renderer"),
    )
    merge = _merge_presentation(
        base,
        renderer_seed=_derive_seed(spec.semantic_seed, "merge_renderer"),
    )
    noncongruent = _minimal_noncongruent_presentation(
        base,
        renderer_seed=_derive_seed(spec.semantic_seed, "noncongruent_renderer"),
    )
    collision = _collision_presentation(
        base,
        hidden,
        collision_hidden,
        renderer_seed=_derive_seed(spec.semantic_seed, "collision_renderer"),
    )
    collision_reindex = _rename_presentation(
        collision,
        renderer_seed=_derive_seed(spec.semantic_seed, "collision_reindex"),
    )
    presentations = (
        base,
        reindexed,
        value_recode,
        split,
        merge,
        noncongruent,
        collision,
        collision_reindex,
    )
    if len(presentations) != PACKETS_PER_ORBIT:
        raise EndogenousCongruenceCorpusError("orbit packet count drifted")
    orbit_id = f"{spec.partition}:{spec.orbit_index:06d}"
    hidden_ledger = HiddenLatentAutomatonLedger(
        orbit_id=orbit_id,
        partition=spec.partition,
        semantic_seed=spec.semantic_seed,
        family=spec.family,
        motif=spec.motif,
        latent_state_count=spec.latent_states,
        generator_count=spec.generators,
        query_count=spec.query_ports,
        base_alias_counts=alias_counts,
        base_generators=hidden.generators,
        collision_generators=collision_hidden.generators,
        latent_observations=hidden.observations,
        collision_pair=hidden.collision_pair,
        base_latent_sha256=_sha256((hidden.generators, hidden.observations)),
        collision_latent_sha256=_sha256(
            (collision_hidden.generators, collision_hidden.observations)
        ),
    )
    return presentations, hidden_ledger


def _presentation_morphism(
    presentation: _Presentation,
) -> PresentationMorphism:
    return PresentationMorphism(
        record_map=presentation.parent_record_map,
        generator_map=presentation.parent_generator_map,
        query_map=presentation.parent_query_map,
    )


def _verify_orbit(
    *,
    orbit_id: str,
    partition: str,
    variants: Mapping[str, _Presentation],
    audited: Mapping[str, _AuditedPacket],
) -> OrbitAuditLedger:
    base = variants["base"]
    reindexed = variants["opaque_reindex"]
    split = variants["bisimilar_split"]
    merge = variants["bisimilar_merge"]
    collision = variants["path_collision"]
    collision_reindex = variants["path_collision_reindex"]

    validate_presentation_naturality(
        base.packet,
        reindexed.packet,
        _presentation_morphism(reindexed),
        source_solution=audited["base"].solution,
        target_solution=audited["opaque_reindex"].solution,
    )
    validate_presentation_naturality(
        split.packet,
        base.packet,
        _presentation_morphism(split),
        source_solution=audited["bisimilar_split"].solution,
        target_solution=audited["base"].solution,
    )
    validate_presentation_naturality(
        base.packet,
        merge.packet,
        _presentation_morphism(merge),
        source_solution=audited["base"].solution,
        target_solution=audited["bisimilar_merge"].solution,
    )
    validate_presentation_naturality(
        collision.packet,
        collision_reindex.packet,
        _presentation_morphism(collision_reindex),
        source_solution=audited["path_collision"].solution,
        target_solution=audited["path_collision_reindex"].solution,
    )

    base_fingerprint = audited["base"].fingerprint
    if not (
        base_fingerprint.latent_sha256
        == audited["opaque_reindex"].fingerprint.latent_sha256
        == audited["value_recode"].fingerprint.latent_sha256
        == audited["bisimilar_split"].fingerprint.latent_sha256
        == audited["bisimilar_merge"].fingerprint.latent_sha256
    ):
        raise EndogenousCongruenceCorpusError(
            "presentation/value orbit changed the causal quotient"
        )
    if (
        audited["minimal_noncongruent"].fingerprint.latent_sha256
        == base_fingerprint.latent_sha256
    ):
        raise EndogenousCongruenceCorpusError(
            "minimal noncongruence twin did not change the quotient"
        )
    if _simple_marginals(base.packet) != _simple_marginals(
        variants["minimal_noncongruent"].packet
    ):
        raise EndogenousCongruenceCorpusError(
            "minimal noncongruence twin changed simple physical marginals"
        )
    if (
        audited["path_collision"].fingerprint.path_sha256
        == base_fingerprint.path_sha256
    ):
        raise EndogenousCongruenceCorpusError(
            "path collision did not separate bounded path behavior"
        )
    if _simple_marginals(base.packet) != _simple_marginals(collision.packet):
        raise EndogenousCongruenceCorpusError(
            "path collision changed simple physical marginals"
        )
    base_commutes = _generators_commute(audited["base"].solution)
    collision_commutes = _generators_commute(audited["path_collision"].solution)
    if base_commutes is not None and (
        not base_commutes or collision_commutes is not False
    ):
        raise EndogenousCongruenceCorpusError(
            "commutation collision did not separate the matched pair"
        )
    return OrbitAuditLedger(
        orbit_id=orbit_id,
        partition=partition,
        packet_hashes=tuple(
            (variant, audited[variant].packet_sha256) for variant in ORBIT_VARIANTS
        ),
        reindex_naturality=True,
        split_naturality=True,
        merge_naturality=True,
        value_recode_preserves_quotient=True,
        minimal_twin_separates_quotient=True,
        minimal_twin_preserves_simple_marginals=True,
        collision_separates_path=True,
        collision_preserves_simple_marginals=True,
        base_commutes=base_commutes,
        collision_commutes=collision_commutes,
    )


def _split_isolation(
    metadata: Sequence[CorpusPacketMetadata],
    fingerprints: Sequence[QuotientFingerprintLedger],
) -> SplitIsolationReceipt:
    partition_by_hash = {item.packet_sha256: item.partition for item in metadata}
    exact: dict[str, set[str]] = {
        TRAIN_PARTITION: set(),
        DEVELOPMENT_PARTITION: set(),
    }
    latent: dict[str, set[str]] = {
        TRAIN_PARTITION: set(),
        DEVELOPMENT_PARTITION: set(),
    }
    actions: dict[str, set[str]] = {
        TRAIN_PARTITION: set(),
        DEVELOPMENT_PARTITION: set(),
    }
    paths: dict[str, set[str]] = {
        TRAIN_PARTITION: set(),
        DEVELOPMENT_PARTITION: set(),
    }
    for item in fingerprints:
        partition = partition_by_hash[item.packet_sha256]
        exact[partition].add(item.packet_sha256)
        latent[partition].add(item.latent_sha256)
        actions[partition].add(item.action_sha256)
        paths[partition].add(item.path_sha256)
    overlaps = (
        exact[TRAIN_PARTITION] & exact[DEVELOPMENT_PARTITION],
        latent[TRAIN_PARTITION] & latent[DEVELOPMENT_PARTITION],
        actions[TRAIN_PARTITION] & actions[DEVELOPMENT_PARTITION],
        paths[TRAIN_PARTITION] & paths[DEVELOPMENT_PARTITION],
    )
    if any(overlaps):
        raise EndogenousCongruenceCorpusError(
            "train/development canonical split overlap detected"
        )
    payload = {
        "train_exact": sorted(exact[TRAIN_PARTITION]),
        "development_exact": sorted(exact[DEVELOPMENT_PARTITION]),
        "train_latent": sorted(latent[TRAIN_PARTITION]),
        "development_latent": sorted(latent[DEVELOPMENT_PARTITION]),
        "train_actions": sorted(actions[TRAIN_PARTITION]),
        "development_actions": sorted(actions[DEVELOPMENT_PARTITION]),
        "train_paths": sorted(paths[TRAIN_PARTITION]),
        "development_paths": sorted(paths[DEVELOPMENT_PARTITION]),
    }
    return SplitIsolationReceipt(
        train_exact_packets=len(exact[TRAIN_PARTITION]),
        development_exact_packets=len(exact[DEVELOPMENT_PARTITION]),
        exact_packet_overlap=0,
        latent_signature_overlap=0,
        action_signature_overlap=0,
        path_signature_overlap=0,
        receipt_sha256=_sha256(payload),
    )


def _cell_receipts(
    packets: Sequence[EndogenousCongruencePacket],
    metadata: Sequence[CorpusPacketMetadata],
) -> tuple[CorpusCellReceipt, ...]:
    packet_by_hash = {packet_sha256(packet): packet for packet in packets}
    counts = Counter(
        (
            item.partition,
            len(packet_by_hash[item.packet_sha256].records),
            len(packet_by_hash[item.packet_sha256].generators),
            len(packet_by_hash[item.packet_sha256].query_ports),
            item.variant,
        )
        for item in metadata
    )
    return tuple(
        CorpusCellReceipt(
            partition=partition,
            physical_records=records,
            generators=generators,
            query_ports=queries,
            variant=variant,
            count=count,
        )
        for (
            partition,
            records,
            generators,
            queries,
            variant,
        ), count in sorted(counts.items())
    )


def _verify_complete_ledgers(
    *,
    packets: Sequence[EndogenousCongruencePacket],
    metadata: Sequence[CorpusPacketMetadata],
    renderers: Sequence[RendererLedger],
    recodings: Sequence[ObservationRecodingLedger],
    targets: Sequence[TargetRelationLedger],
    oracles: Sequence[OracleAgreementLedger],
    fingerprints: Sequence[QuotientFingerprintLedger],
) -> tuple[str, ...]:
    packet_hashes = tuple(packet_sha256(packet) for packet in packets)
    if len(packet_hashes) != len(set(packet_hashes)):
        raise EndogenousCongruenceCorpusError("duplicate model-visible packet")
    expected = set(packet_hashes)
    ledgers: tuple[tuple[str, Iterable[str]], ...] = (
        ("metadata", (item.packet_sha256 for item in metadata)),
        ("renderers", (item.packet_sha256 for item in renderers)),
        ("recodings", (item.packet_sha256 for item in recodings)),
        ("targets", (item.packet_sha256 for item in targets)),
        ("oracles", (item.packet_sha256 for item in oracles)),
        ("fingerprints", (item.packet_sha256 for item in fingerprints)),
    )
    for name, values in ledgers:
        hashes = tuple(values)
        if len(hashes) != len(expected) or set(hashes) != expected:
            raise EndogenousCongruenceCorpusError(
                f"{name} does not contain exactly one receipt per packet"
            )
    return packet_hashes


def generate_endogenous_congruence_corpus(
    *,
    seed: int = DEFAULT_SEED,
    train_packets: int = DEFAULT_TRAIN_PACKETS,
    development_packets: int = DEFAULT_DEVELOPMENT_PACKETS,
) -> ProceduralEndogenousCongruenceCorpus:
    """Generate and fully audit a deterministic source-deleted ECCR corpus."""

    plan = plan_endogenous_congruence_corpus(
        train_packets=train_packets,
        development_packets=development_packets,
    )
    specs = _build_orbit_specs(
        seed=seed,
        train_packets=train_packets,
        development_packets=development_packets,
    )

    packets: list[EndogenousCongruencePacket] = []
    hidden_ledgers: list[HiddenLatentAutomatonLedger] = []
    metadata: list[CorpusPacketMetadata] = []
    renderers: list[RendererLedger] = []
    recodings: list[ObservationRecodingLedger] = []
    targets: list[TargetRelationLedger] = []
    oracles: list[OracleAgreementLedger] = []
    fingerprints: list[QuotientFingerprintLedger] = []
    orbit_audits: list[OrbitAuditLedger] = []

    for spec in specs:
        presentations, hidden_ledger = _build_orbit(spec)
        hidden_ledgers.append(hidden_ledger)
        orbit_id = hidden_ledger.orbit_id
        variants = dict(zip(ORBIT_VARIANTS, presentations, strict=True))
        audited: dict[str, _AuditedPacket] = {}
        for variant, presentation in variants.items():
            audited_packet = _audit_packet(
                presentation,
                require_hidden_relation=variant != "minimal_noncongruent",
            )
            audited[variant] = audited_packet
            packets.append(presentation.packet)
            metadata.append(
                CorpusPacketMetadata(
                    packet_sha256=audited_packet.packet_sha256,
                    partition=spec.partition,
                    orbit_id=orbit_id,
                    variant=variant,
                    family=spec.family,
                    motif=spec.motif,
                    semantic_seed=spec.semantic_seed,
                    renderer_seed=presentation.renderer_seed,
                    hidden_relation_applicable=variant != "minimal_noncongruent",
                )
            )
            renderers.append(
                _renderer_ledger(presentation, audited_packet.packet_sha256)
            )
            recodings.append(
                _recoding_ledger(presentation, audited_packet.packet_sha256)
            )
            targets.append(audited_packet.target)
            oracles.append(audited_packet.oracle)
            fingerprints.append(audited_packet.fingerprint)
        orbit_audits.append(
            _verify_orbit(
                orbit_id=orbit_id,
                partition=spec.partition,
                variants=variants,
                audited=audited,
            )
        )

    packet_hashes = _verify_complete_ledgers(
        packets=packets,
        metadata=metadata,
        renderers=renderers,
        recodings=recodings,
        targets=targets,
        oracles=oracles,
        fingerprints=fingerprints,
    )
    split_isolation = _split_isolation(metadata, fingerprints)
    cells = _cell_receipts(packets, metadata)
    metadata_by_hash = {item.packet_sha256: item for item in metadata}
    train_count = sum(
        metadata_by_hash[digest].partition == TRAIN_PARTITION
        for digest in packet_hashes
    )
    development_count = len(packet_hashes) - train_count
    if train_count != train_packets or development_count != development_packets:
        raise EndogenousCongruenceCorpusError("partition packet count drifted")

    all_values = [
        item.value for packet in packets for item in packet.observation_witnesses
    ]
    packet_manifest_sha256 = _sha256(packet_hashes)
    offline_payload = {
        "hidden_automata": hidden_ledgers,
        "metadata": metadata,
        "renderers": renderers,
        "observation_recodings": recodings,
        "target_relations": targets,
        "oracle_agreements": oracles,
        "fingerprints": fingerprints,
        "orbit_audits": orbit_audits,
        "cells": cells,
        "split_isolation": split_isolation,
    }
    offline_ledger_sha256 = _sha256(offline_payload)
    payload_sha256 = _sha256(
        {
            "model_packets": tuple(model_packet_payload(packet) for packet in packets),
            "offline_ledgers": offline_payload,
            "seed": seed,
            "semantics_sha256": plan.semantics_sha256,
        }
    )
    manifest = EndogenousCongruenceCorpusManifest(
        seed=seed,
        packet_count=len(packets),
        train_packet_count=train_count,
        development_packet_count=development_count,
        orbit_count=len(specs),
        train_orbit_count=plan.train_orbits,
        development_orbit_count=plan.development_orbits,
        packets_per_orbit=PACKETS_PER_ORBIT,
        unique_packet_count=len(set(packet_hashes)),
        unique_latent_signatures=len({item.latent_sha256 for item in fingerprints}),
        unique_action_signatures=len({item.action_sha256 for item in fingerprints}),
        unique_path_signatures=len({item.path_sha256 for item in fingerprints}),
        observation_value_minimum=min(all_values),
        observation_value_maximum=max(all_values),
        negative_observation_values=len({value for value in all_values if value < 0}),
        positive_observation_values=len({value for value in all_values if value > 0}),
        oracle_agreement_count=sum(item.independent_oracles_agree for item in oracles),
        hidden_minimal_count=len(hidden_ledgers),
        reindex_orbit_count=sum(item.reindex_naturality for item in orbit_audits),
        split_orbit_count=sum(item.split_naturality for item in orbit_audits),
        merge_orbit_count=sum(item.merge_naturality for item in orbit_audits),
        collision_orbit_count=sum(
            item.collision_separates_path for item in orbit_audits
        ),
        semantics_sha256=plan.semantics_sha256,
        packet_manifest_sha256=packet_manifest_sha256,
        offline_ledger_sha256=offline_ledger_sha256,
        split_isolation=split_isolation,
        cells=cells,
        payload_sha256=payload_sha256,
        preregistered_train_packets=PREREGISTERED_TRAIN_PACKETS,
        preregistered_development_packets=PREREGISTERED_DEVELOPMENT_PACKETS,
    )
    return ProceduralEndogenousCongruenceCorpus(
        packets=tuple(packets),
        hidden_automata=tuple(hidden_ledgers),
        metadata=tuple(metadata),
        renderers=tuple(renderers),
        observation_recodings=tuple(recodings),
        target_relations=tuple(targets),
        oracle_agreements=tuple(oracles),
        fingerprints=tuple(fingerprints),
        orbit_audits=tuple(orbit_audits),
        manifest=manifest,
    )


def write_endogenous_congruence_corpus(
    corpus: ProceduralEndogenousCongruenceCorpus,
    output_path: Path,
) -> None:
    """Atomically publish model packets and clearly separated offline ledgers."""

    payload = _canonical_json(corpus)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.write_text(f"{payload}\n", encoding="ascii")
    temporary.replace(output_path)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-packets", type=int, default=DEFAULT_TRAIN_PACKETS)
    parser.add_argument(
        "--development-packets",
        type=int,
        default=DEFAULT_DEVELOPMENT_PACKETS,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    corpus = generate_endogenous_congruence_corpus(
        seed=args.seed,
        train_packets=args.train_packets,
        development_packets=args.development_packets,
    )
    if args.output is not None:
        write_endogenous_congruence_corpus(corpus, args.output)
    print(_canonical_json(corpus.manifest))


if __name__ == "__main__":
    _main()
