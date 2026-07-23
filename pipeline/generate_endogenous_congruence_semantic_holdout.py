"""Cardinality-matched semantic holdout corpus for congruence induction.

This generator keeps train and development geometry identical. Generalization
is tested by holding out combinations of three assessor-only semantic axes:

* generator context: a cyclic permutation or a noninvertible sink chain;
* noncommuting motif: a reset or a terminal transposition;
* composition depth: the exact shortest generator-word depth needed to
  distinguish every pair of latent states.

Every individual axis value occurs in both partitions, but no development
triple occurs in training. Model-visible packets contain only complete
physical transition and observation tables. Semantic construction records,
targets, renderer maps, recodings, and independent oracle products remain in
offline ledgers.

This is a corpus and custody artifact, not a neural reasoning claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

from pipeline import generate_endogenous_congruence_corpus as base
from pipeline.endogenous_congruence_board import (
    CongruenceSolution,
    EndogenousCongruencePacket,
    compute_exhaustive_partition,
    compute_refinement_partition,
    model_packet_payload,
    solve_by_refinement,
)


DEFAULT_SEED = 202_607_230_7
DEFAULT_TRAIN_PACKETS = 256
DEFAULT_DEVELOPMENT_PACKETS = 64

PACKETS_PER_ORBIT = base.PACKETS_PER_ORBIT
ORBIT_VARIANTS = base.ORBIT_VARIANTS
TRAIN_PARTITION = "semantic_holdout_train"
DEVELOPMENT_PARTITION = "semantic_holdout_development"

LATENT_STATE_COUNT = 4
BASE_RECORD_COUNTS = (6, 7)
GENERATOR_COUNTS = (3, 4)
QUERY_COUNTS = (3, 4)
GENERATOR_SEMANTICS = ("cyclic_context", "sink_context")
NONCOMMUTING_MOTIFS = ("reset_motif", "terminal_swap_motif")
COMPOSITION_DEPTHS = (1, 2)

GEOMETRY_CELLS = tuple(product(BASE_RECORD_COUNTS, GENERATOR_COUNTS, QUERY_COUNTS))
SEMANTIC_COMBINATIONS = tuple(
    product(GENERATOR_SEMANTICS, NONCOMMUTING_MOTIFS, COMPOSITION_DEPTHS)
)


def _combination_is_development(
    combination: tuple[str, str, int],
) -> bool:
    semantic, motif, depth = combination
    semantic_index = GENERATOR_SEMANTICS.index(semantic)
    motif_index = NONCOMMUTING_MOTIFS.index(motif)
    depth_index = COMPOSITION_DEPTHS.index(depth)
    return (semantic_index + motif_index + depth_index) % 2 == 0


TRAIN_COMBINATIONS = tuple(
    item for item in SEMANTIC_COMBINATIONS if not _combination_is_development(item)
)
DEVELOPMENT_COMBINATIONS = tuple(
    item for item in SEMANTIC_COMBINATIONS if _combination_is_development(item)
)

_SEMANTICS_DESCRIPTOR = {
    "version": 1,
    "packet_type": "EndogenousCongruencePacket",
    "source_deleted_model_payload": True,
    "complete_unary_transitions": True,
    "complete_observations": True,
    "latent_state_count": LATENT_STATE_COUNT,
    "base_record_counts": BASE_RECORD_COUNTS,
    "generator_counts": GENERATOR_COUNTS,
    "query_counts": QUERY_COUNTS,
    "generator_semantics": GENERATOR_SEMANTICS,
    "noncommuting_motifs": NONCOMMUTING_MOTIFS,
    "composition_depths": COMPOSITION_DEPTHS,
    "semantic_split_rule": "parity_even_development",
    "packets_per_orbit": PACKETS_PER_ORBIT,
    "orbit_variants": ORBIT_VARIANTS,
    "independent_oracles": ("partition_refinement", "bell_partition_search"),
}


class SemanticHoldoutCorpusError(ValueError):
    """A frozen semantic-holdout invariant failed."""


@dataclass(frozen=True)
class SemanticHoldoutScalePlan:
    train_packets: int
    development_packets: int
    train_orbits: int
    development_orbits: int
    packets_per_orbit: int
    geometry_cell_count: int
    train_combination_count: int
    development_combination_count: int
    semantics_sha256: str


@dataclass(frozen=True)
class SemanticHoldoutOrbitSpec:
    partition: str
    orbit_index: int
    local_index: int
    semantic_seed: int
    base_records: int
    latent_states: int
    generators: int
    query_ports: int
    generator_semantics: str
    noncommuting_motif: str
    composition_depth: int

    @property
    def geometry(self) -> tuple[int, int, int]:
        return (self.base_records, self.generators, self.query_ports)

    @property
    def semantic_combination(self) -> tuple[str, str, int]:
        return (
            self.generator_semantics,
            self.noncommuting_motif,
            self.composition_depth,
        )


@dataclass(frozen=True)
class SemanticHoldoutPacketMetadata:
    packet_sha256: str
    partition: str
    orbit_id: str
    variant: str
    semantic_seed: int
    renderer_seed: int
    hidden_relation_applicable: bool
    generator_semantics: str
    noncommuting_motif: str
    composition_depth: int
    base_records: int
    latent_states: int
    generators: int
    query_ports: int


@dataclass(frozen=True)
class SemanticHoldoutOrbitLedger:
    orbit_id: str
    partition: str
    generator_semantics: str
    noncommuting_motif: str
    composition_depth: int
    base_records: int
    latent_states: int
    generators: int
    query_ports: int
    context_rank: int
    motif_rank: int
    context_and_motif_noncommute: bool
    measured_maximum_distinction_depth: int
    base_packet_sha256: str


@dataclass(frozen=True)
class SemanticHoldoutSplitReceipt:
    train_exact_packets: int
    development_exact_packets: int
    exact_packet_overlap: int
    train_geometry_cells: tuple[tuple[int, int, int], ...]
    development_geometry_cells: tuple[tuple[int, int, int], ...]
    train_physical_cells: tuple[tuple[int, int, int, str], ...]
    development_physical_cells: tuple[tuple[int, int, int, str], ...]
    generator_semantics_support: tuple[str, ...]
    noncommuting_motif_support: tuple[str, ...]
    composition_depth_support: tuple[int, ...]
    train_semantic_combinations: tuple[tuple[str, str, int], ...]
    development_semantic_combinations: tuple[tuple[str, str, int], ...]
    semantic_combination_overlap: int
    latent_signature_overlap: int
    action_signature_overlap: int
    path_signature_overlap: int
    receipt_sha256: str


@dataclass(frozen=True)
class SemanticHoldoutCellReceipt:
    partition: str
    base_records: int
    generators: int
    query_ports: int
    generator_semantics: str
    noncommuting_motif: str
    composition_depth: int
    orbit_count: int


@dataclass(frozen=True)
class SemanticHoldoutManifest:
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
    oracle_agreement_count: int
    hidden_minimal_count: int
    composition_depth_agreement_count: int
    noncommuting_motif_count: int
    reindex_orbit_count: int
    split_orbit_count: int
    merge_orbit_count: int
    collision_orbit_count: int
    observation_value_minimum: int
    observation_value_maximum: int
    negative_observation_values: int
    positive_observation_values: int
    semantics_sha256: str
    packet_manifest_sha256: str
    offline_ledger_sha256: str
    payload_sha256: str
    split_receipt: SemanticHoldoutSplitReceipt
    cells: tuple[SemanticHoldoutCellReceipt, ...]


@dataclass(frozen=True)
class ProceduralSemanticHoldoutCorpus:
    packets: tuple[EndogenousCongruencePacket, ...]
    hidden_automata: tuple[base.HiddenLatentAutomatonLedger, ...]
    metadata: tuple[SemanticHoldoutPacketMetadata, ...]
    semantic_orbits: tuple[SemanticHoldoutOrbitLedger, ...]
    renderers: tuple[base.RendererLedger, ...]
    observation_recodings: tuple[base.ObservationRecodingLedger, ...]
    target_relations: tuple[base.TargetRelationLedger, ...]
    oracle_agreements: tuple[base.OracleAgreementLedger, ...]
    fingerprints: tuple[base.QuotientFingerprintLedger, ...]
    orbit_audits: tuple[base.OrbitAuditLedger, ...]
    manifest: SemanticHoldoutManifest


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


def _derive_seed(seed: int, *parts: object) -> int:
    payload = ":".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big")


def packet_sha256(packet: EndogenousCongruencePacket) -> str:
    """Hash only the model-visible physical packet."""

    return _sha256(model_packet_payload(packet))


def serialize_model_packets(
    packets: Sequence[EndogenousCongruencePacket],
) -> str:
    """Serialize model packets without assessor-side construction records."""

    return _canonical_json(tuple(model_packet_payload(packet) for packet in packets))


def plan_semantic_holdout_corpus(
    *,
    train_packets: int = DEFAULT_TRAIN_PACKETS,
    development_packets: int = DEFAULT_DEVELOPMENT_PACKETS,
) -> SemanticHoldoutScalePlan:
    """Validate scale before drawing any semantic or renderer seed."""

    minimum_packets = len(GEOMETRY_CELLS) * PACKETS_PER_ORBIT
    for name, count in (
        ("train_packets", train_packets),
        ("development_packets", development_packets),
    ):
        if count < minimum_packets:
            raise SemanticHoldoutCorpusError(
                f"{name} must contain at least {minimum_packets} packets "
                "to cover every geometry cell"
            )
        if count % PACKETS_PER_ORBIT:
            raise SemanticHoldoutCorpusError(
                f"{name} must be divisible by {PACKETS_PER_ORBIT}"
            )
    return SemanticHoldoutScalePlan(
        train_packets=train_packets,
        development_packets=development_packets,
        train_orbits=train_packets // PACKETS_PER_ORBIT,
        development_orbits=development_packets // PACKETS_PER_ORBIT,
        packets_per_orbit=PACKETS_PER_ORBIT,
        geometry_cell_count=len(GEOMETRY_CELLS),
        train_combination_count=len(TRAIN_COMBINATIONS),
        development_combination_count=len(DEVELOPMENT_COMBINATIONS),
        semantics_sha256=_sha256(_SEMANTICS_DESCRIPTOR),
    )


def _build_orbit_specs(
    *,
    seed: int,
    train_packets: int,
    development_packets: int,
) -> tuple[SemanticHoldoutOrbitSpec, ...]:
    plan = plan_semantic_holdout_corpus(
        train_packets=train_packets,
        development_packets=development_packets,
    )
    output: list[SemanticHoldoutOrbitSpec] = []
    global_index = 0
    for partition, orbit_count, combinations in (
        (TRAIN_PARTITION, plan.train_orbits, TRAIN_COMBINATIONS),
        (
            DEVELOPMENT_PARTITION,
            plan.development_orbits,
            DEVELOPMENT_COMBINATIONS,
        ),
    ):
        for local_index in range(orbit_count):
            geometry_index = local_index % len(GEOMETRY_CELLS)
            repetition = local_index // len(GEOMETRY_CELLS)
            combination_index = (geometry_index + repetition) % len(combinations)
            base_records, generators, query_ports = GEOMETRY_CELLS[geometry_index]
            semantic, motif, depth = combinations[combination_index]
            semantic_seed = _derive_seed(
                seed,
                partition,
                local_index,
                geometry_index,
                semantic,
                motif,
                depth,
                "semantic",
            )
            output.append(
                SemanticHoldoutOrbitSpec(
                    partition=partition,
                    orbit_index=global_index,
                    local_index=local_index,
                    semantic_seed=semantic_seed,
                    base_records=base_records,
                    latent_states=LATENT_STATE_COUNT,
                    generators=generators,
                    query_ports=query_ports,
                    generator_semantics=semantic,
                    noncommuting_motif=motif,
                    composition_depth=depth,
                )
            )
            global_index += 1
    return tuple(output)


def _compose_maps(
    left: Sequence[int],
    right: Sequence[int],
) -> tuple[int, ...]:
    return tuple(right[left[state]] for state in range(len(left)))


def _maps_commute(left: Sequence[int], right: Sequence[int]) -> bool:
    return _compose_maps(left, right) == _compose_maps(right, left)


def _map_rank(mapping: Sequence[int]) -> int:
    return len(set(mapping))


def _maximum_distinction_depth(solution: CongruenceSolution) -> int:
    return max(
        (
            len(certificate.continuation)
            for certificate in solution.distinction_certificates
        ),
        default=0,
    )


def _semantic_generators(
    spec: SemanticHoldoutOrbitSpec,
) -> tuple[tuple[int, ...], ...]:
    identity = tuple(range(LATENT_STATE_COUNT))
    if spec.generator_semantics == "cyclic_context":
        context = (1, 2, 3, 0)
    elif spec.generator_semantics == "sink_context":
        context = (1, 2, 3, 3)
    else:
        raise SemanticHoldoutCorpusError("unknown generator semantics")

    if spec.noncommuting_motif == "reset_motif":
        # Reset to a nonterminal anchor. Resetting the sink context to its
        # fixed point would commute and would make the motif label false.
        motif = (0,) * LATENT_STATE_COUNT
    elif spec.noncommuting_motif == "terminal_swap_motif":
        motif = (0, 1, 3, 2)
    else:
        raise SemanticHoldoutCorpusError("unknown noncommuting motif")
    if _maps_commute(context, motif):
        raise SemanticHoldoutCorpusError("declared motif commutes with its context")

    generators = [identity, context, motif]
    if spec.generators == 4:
        generators.append(_compose_maps(motif, context))
    if len(generators) != spec.generators:
        raise SemanticHoldoutCorpusError("generator cardinality drifted")
    return tuple(generators)


def _direct_maximum_distinction_depth(
    generators: Sequence[Sequence[int]],
    observations: Sequence[Sequence[int]],
) -> int | None:
    """Compute exact shortest distinguishing words on the latent pair graph."""

    maximum = 0
    for left in range(LATENT_STATE_COUNT):
        for right in range(left + 1, LATENT_STATE_COUNT):
            frontier = {(left, right)}
            seen: set[tuple[int, int]] = set()
            depth = 0
            while frontier:
                if any(
                    any(values[first] != values[second] for values in observations)
                    for first, second in frontier
                ):
                    maximum = max(maximum, depth)
                    break
                seen.update(frontier)
                next_frontier = {
                    (generator[first], generator[second])
                    for first, second in frontier
                    for generator in generators
                }
                frontier = next_frontier - seen
                depth += 1
            else:
                return None
    return maximum


@lru_cache(maxsize=None)
def _select_observation_pattern(
    generators: tuple[tuple[int, ...], ...],
    *,
    desired_depth: int,
) -> tuple[int, ...]:
    """Find a canonical binary reader with the requested exact path depth."""

    for values in product((0, 1), repeat=LATENT_STATE_COUNT):
        if values[0] != 0 or len(set(values)) != 2:
            continue
        hidden = base._HiddenAutomaton(  # noqa: SLF001
            generators=generators,
            observations=(values,),
            marker_state=0,
            collision_pair=(0, 1),
        )
        packet = base._latent_packet(hidden)  # noqa: SLF001
        refinement = compute_refinement_partition(packet)
        exhaustive = compute_exhaustive_partition(packet)
        if base._blocks_signature(refinement) != base._blocks_signature(exhaustive):  # noqa: SLF001
            raise SemanticHoldoutCorpusError(
                "observation search found oracle disagreement"
            )
        if any(len(block) != 1 for block in refinement):
            continue
        measured_depth = _direct_maximum_distinction_depth(generators, (values,))
        if measured_depth == desired_depth:
            return tuple(values)
    raise SemanticHoldoutCorpusError(
        f"no minimal observation pattern realizes composition depth {desired_depth}"
    )


def _permute_hidden_states(
    hidden: base._HiddenAutomaton,  # noqa: SLF001
    *,
    seed: int,
) -> base._HiddenAutomaton:  # noqa: SLF001
    rng = random.Random(seed)
    old_by_new = list(range(LATENT_STATE_COUNT))
    rng.shuffle(old_by_new)
    new_by_old = [0] * LATENT_STATE_COUNT
    for new, old in enumerate(old_by_new):
        new_by_old[old] = new

    def transform(mapping: Sequence[int]) -> tuple[int, ...]:
        return tuple(
            new_by_old[mapping[old_by_new[new]]] for new in range(LATENT_STATE_COUNT)
        )

    return base._HiddenAutomaton(  # noqa: SLF001
        generators=tuple(transform(item) for item in hidden.generators),
        observations=tuple(
            tuple(values[old_by_new[new]] for new in range(LATENT_STATE_COUNT))
            for values in hidden.observations
        ),
        marker_state=new_by_old[hidden.marker_state],
        collision_pair=tuple(new_by_old[item] for item in hidden.collision_pair),
    )


def _sample_hidden_automaton(
    spec: SemanticHoldoutOrbitSpec,
) -> base._HiddenAutomaton:  # noqa: SLF001
    generators = _semantic_generators(spec)
    pattern = _select_observation_pattern(
        generators,
        desired_depth=spec.composition_depth,
    )
    marker = 0 if spec.generator_semantics == "cyclic_context" else 3
    hidden = base._HiddenAutomaton(  # noqa: SLF001
        generators=generators,
        observations=tuple(
            pattern if query % 2 == 0 else tuple(1 - value for value in pattern)
            for query in range(spec.query_ports)
        ),
        marker_state=marker,
        collision_pair=(0, 1),
    )
    base._validate_hidden_automaton(hidden)  # noqa: SLF001
    collision = base._collision_hidden(hidden)  # noqa: SLF001
    base._validate_hidden_automaton(collision)  # noqa: SLF001

    permuted = _permute_hidden_states(
        hidden,
        seed=_derive_seed(spec.semantic_seed, "latent_permutation"),
    )
    base._validate_hidden_automaton(permuted)  # noqa: SLF001
    solution = solve_by_refinement(
        base._latent_packet(permuted),  # noqa: SLF001
        distinction_depth=6,
    )
    measured_depth = _maximum_distinction_depth(solution)
    if measured_depth != spec.composition_depth:
        raise SemanticHoldoutCorpusError(
            "state permutation changed the requested composition depth"
        )
    if _maps_commute(permuted.generators[1], permuted.generators[2]):
        raise SemanticHoldoutCorpusError(
            "state permutation erased declared noncommutation"
        )
    return permuted


def _build_orbit(
    spec: SemanticHoldoutOrbitSpec,
) -> tuple[
    tuple[base._Presentation, ...],  # noqa: SLF001
    base.HiddenLatentAutomatonLedger,
    base._HiddenAutomaton,  # noqa: SLF001
]:
    hidden = _sample_hidden_automaton(spec)
    collision_hidden = base._collision_hidden(hidden)  # noqa: SLF001
    alias_rng = random.Random(_derive_seed(spec.semantic_seed, "aliases"))
    alias_counts = base._alias_counts(  # noqa: SLF001
        state_count=spec.latent_states,
        physical_count=spec.base_records,
        collision_pair=hidden.collision_pair,
        rng=alias_rng,
    )
    base_recode = base._signed_value_recode(  # noqa: SLF001
        hidden,
        seed=_derive_seed(spec.semantic_seed, "base_recode"),
    )
    presentation = base._lift_hidden_automaton(  # noqa: SLF001
        hidden,
        alias_counts=alias_counts,
        renderer_seed=_derive_seed(spec.semantic_seed, "base_renderer"),
        recodings=base_recode,
    )
    reindexed = base._rename_presentation(  # noqa: SLF001
        presentation,
        renderer_seed=_derive_seed(spec.semantic_seed, "reindex_renderer"),
    )
    value_recode = base._recode_presentation(  # noqa: SLF001
        presentation,
        hidden,
        renderer_seed=_derive_seed(spec.semantic_seed, "value_recode"),
    )
    split = base._split_presentation(  # noqa: SLF001
        presentation,
        renderer_seed=_derive_seed(spec.semantic_seed, "split_renderer"),
    )
    merge = base._merge_presentation(  # noqa: SLF001
        presentation,
        renderer_seed=_derive_seed(spec.semantic_seed, "merge_renderer"),
    )
    noncongruent = base._minimal_noncongruent_presentation(  # noqa: SLF001
        presentation,
        renderer_seed=_derive_seed(spec.semantic_seed, "noncongruent_renderer"),
    )
    collision = base._collision_presentation(  # noqa: SLF001
        presentation,
        hidden,
        collision_hidden,
        renderer_seed=_derive_seed(spec.semantic_seed, "collision_renderer"),
    )
    collision_reindex = base._rename_presentation(  # noqa: SLF001
        collision,
        renderer_seed=_derive_seed(spec.semantic_seed, "collision_reindex"),
    )
    presentations = (
        presentation,
        reindexed,
        value_recode,
        split,
        merge,
        noncongruent,
        collision,
        collision_reindex,
    )
    orbit_id = f"{spec.partition}:{spec.orbit_index:06d}"
    hidden_ledger = base.HiddenLatentAutomatonLedger(
        orbit_id=orbit_id,
        partition=spec.partition,
        semantic_seed=spec.semantic_seed,
        family="cardinality_matched_semantic_holdout",
        motif=spec.noncommuting_motif,
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
    return presentations, hidden_ledger, hidden


def _sets_by_partition(
    metadata: Sequence[SemanticHoldoutPacketMetadata],
    fingerprints: Sequence[base.QuotientFingerprintLedger],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    partition_by_hash = {item.packet_sha256: item.partition for item in metadata}
    latent = {TRAIN_PARTITION: set(), DEVELOPMENT_PARTITION: set()}
    actions = {TRAIN_PARTITION: set(), DEVELOPMENT_PARTITION: set()}
    paths = {TRAIN_PARTITION: set(), DEVELOPMENT_PARTITION: set()}
    for item in fingerprints:
        partition = partition_by_hash[item.packet_sha256]
        latent[partition].add(item.latent_sha256)
        actions[partition].add(item.action_sha256)
        paths[partition].add(item.path_sha256)
    return latent, actions, paths


def _split_receipt(
    *,
    packets: Sequence[EndogenousCongruencePacket],
    metadata: Sequence[SemanticHoldoutPacketMetadata],
    semantic_orbits: Sequence[SemanticHoldoutOrbitLedger],
    fingerprints: Sequence[base.QuotientFingerprintLedger],
) -> SemanticHoldoutSplitReceipt:
    packet_by_hash = {packet_sha256(packet): packet for packet in packets}
    exact = {TRAIN_PARTITION: set(), DEVELOPMENT_PARTITION: set()}
    physical = {TRAIN_PARTITION: set(), DEVELOPMENT_PARTITION: set()}
    combinations = {TRAIN_PARTITION: set(), DEVELOPMENT_PARTITION: set()}
    geometry = {TRAIN_PARTITION: set(), DEVELOPMENT_PARTITION: set()}
    axis_semantics = {TRAIN_PARTITION: set(), DEVELOPMENT_PARTITION: set()}
    axis_motifs = {TRAIN_PARTITION: set(), DEVELOPMENT_PARTITION: set()}
    axis_depths = {TRAIN_PARTITION: set(), DEVELOPMENT_PARTITION: set()}

    for item in metadata:
        packet = packet_by_hash[item.packet_sha256]
        exact[item.partition].add(item.packet_sha256)
        physical[item.partition].add(
            (
                len(packet.records),
                len(packet.generators),
                len(packet.query_ports),
                item.variant,
            )
        )
    for item in semantic_orbits:
        geometry[item.partition].add(
            (item.base_records, item.generators, item.query_ports)
        )
        combinations[item.partition].add(
            (
                item.generator_semantics,
                item.noncommuting_motif,
                item.composition_depth,
            )
        )
        axis_semantics[item.partition].add(item.generator_semantics)
        axis_motifs[item.partition].add(item.noncommuting_motif)
        axis_depths[item.partition].add(item.composition_depth)

    exact_overlap = exact[TRAIN_PARTITION] & exact[DEVELOPMENT_PARTITION]
    combination_overlap = (
        combinations[TRAIN_PARTITION] & combinations[DEVELOPMENT_PARTITION]
    )
    if exact_overlap:
        raise SemanticHoldoutCorpusError("exact train/development packet overlap")
    if combination_overlap:
        raise SemanticHoldoutCorpusError("semantic holdout combination overlap")
    if geometry[TRAIN_PARTITION] != geometry[DEVELOPMENT_PARTITION]:
        raise SemanticHoldoutCorpusError("base geometry differs across partitions")
    if physical[TRAIN_PARTITION] != physical[DEVELOPMENT_PARTITION]:
        raise SemanticHoldoutCorpusError("physical dimensions differ across partitions")
    for name, support in (
        ("generator semantics", axis_semantics),
        ("noncommuting motif", axis_motifs),
        ("composition depth", axis_depths),
    ):
        if support[TRAIN_PARTITION] != support[DEVELOPMENT_PARTITION]:
            raise SemanticHoldoutCorpusError(f"{name} support differs across splits")

    latent, actions, paths = _sets_by_partition(metadata, fingerprints)
    payload = {
        "train_exact": sorted(exact[TRAIN_PARTITION]),
        "development_exact": sorted(exact[DEVELOPMENT_PARTITION]),
        "geometry": sorted(geometry[TRAIN_PARTITION]),
        "physical": sorted(physical[TRAIN_PARTITION]),
        "train_combinations": sorted(combinations[TRAIN_PARTITION]),
        "development_combinations": sorted(combinations[DEVELOPMENT_PARTITION]),
        "latent_overlap": sorted(
            latent[TRAIN_PARTITION] & latent[DEVELOPMENT_PARTITION]
        ),
        "action_overlap": sorted(
            actions[TRAIN_PARTITION] & actions[DEVELOPMENT_PARTITION]
        ),
        "path_overlap": sorted(paths[TRAIN_PARTITION] & paths[DEVELOPMENT_PARTITION]),
    }
    return SemanticHoldoutSplitReceipt(
        train_exact_packets=len(exact[TRAIN_PARTITION]),
        development_exact_packets=len(exact[DEVELOPMENT_PARTITION]),
        exact_packet_overlap=0,
        train_geometry_cells=tuple(sorted(geometry[TRAIN_PARTITION])),
        development_geometry_cells=tuple(sorted(geometry[DEVELOPMENT_PARTITION])),
        train_physical_cells=tuple(sorted(physical[TRAIN_PARTITION])),
        development_physical_cells=tuple(sorted(physical[DEVELOPMENT_PARTITION])),
        generator_semantics_support=tuple(sorted(axis_semantics[TRAIN_PARTITION])),
        noncommuting_motif_support=tuple(sorted(axis_motifs[TRAIN_PARTITION])),
        composition_depth_support=tuple(sorted(axis_depths[TRAIN_PARTITION])),
        train_semantic_combinations=tuple(sorted(combinations[TRAIN_PARTITION])),
        development_semantic_combinations=tuple(
            sorted(combinations[DEVELOPMENT_PARTITION])
        ),
        semantic_combination_overlap=0,
        latent_signature_overlap=len(
            latent[TRAIN_PARTITION] & latent[DEVELOPMENT_PARTITION]
        ),
        action_signature_overlap=len(
            actions[TRAIN_PARTITION] & actions[DEVELOPMENT_PARTITION]
        ),
        path_signature_overlap=len(
            paths[TRAIN_PARTITION] & paths[DEVELOPMENT_PARTITION]
        ),
        receipt_sha256=_sha256(payload),
    )


def _cell_receipts(
    specs: Sequence[SemanticHoldoutOrbitSpec],
) -> tuple[SemanticHoldoutCellReceipt, ...]:
    counts = Counter(
        (
            item.partition,
            item.base_records,
            item.generators,
            item.query_ports,
            item.generator_semantics,
            item.noncommuting_motif,
            item.composition_depth,
        )
        for item in specs
    )
    return tuple(
        SemanticHoldoutCellReceipt(
            partition=partition,
            base_records=records,
            generators=generators,
            query_ports=queries,
            generator_semantics=semantic,
            noncommuting_motif=motif,
            composition_depth=depth,
            orbit_count=count,
        )
        for (
            partition,
            records,
            generators,
            queries,
            semantic,
            motif,
            depth,
        ), count in sorted(counts.items())
    )


def _verify_source_deletion(
    packets: Sequence[EndogenousCongruencePacket],
) -> None:
    visible = serialize_model_packets(packets)
    forbidden = (
        TRAIN_PARTITION,
        DEVELOPMENT_PARTITION,
        *GENERATOR_SEMANTICS,
        *NONCOMMUTING_MOTIFS,
        "composition_depth",
        "semantic_seed",
        "target_relation",
        "oracle",
        "renderer",
        "latent",
    )
    if any(item in visible for item in forbidden):
        raise SemanticHoldoutCorpusError("assessor metadata leaked into model packets")


def generate_endogenous_congruence_semantic_holdout(
    *,
    seed: int = DEFAULT_SEED,
    train_packets: int = DEFAULT_TRAIN_PACKETS,
    development_packets: int = DEFAULT_DEVELOPMENT_PACKETS,
) -> ProceduralSemanticHoldoutCorpus:
    """Generate and fully audit a cardinality-matched semantic split."""

    plan = plan_semantic_holdout_corpus(
        train_packets=train_packets,
        development_packets=development_packets,
    )
    specs = _build_orbit_specs(
        seed=seed,
        train_packets=train_packets,
        development_packets=development_packets,
    )

    packets: list[EndogenousCongruencePacket] = []
    hidden_ledgers: list[base.HiddenLatentAutomatonLedger] = []
    metadata: list[SemanticHoldoutPacketMetadata] = []
    semantic_orbits: list[SemanticHoldoutOrbitLedger] = []
    renderers: list[base.RendererLedger] = []
    recodings: list[base.ObservationRecodingLedger] = []
    targets: list[base.TargetRelationLedger] = []
    oracles: list[base.OracleAgreementLedger] = []
    fingerprints: list[base.QuotientFingerprintLedger] = []
    orbit_audits: list[base.OrbitAuditLedger] = []

    for spec in specs:
        presentations, hidden_ledger, hidden = _build_orbit(spec)
        hidden_ledgers.append(hidden_ledger)
        orbit_id = hidden_ledger.orbit_id
        variants = dict(zip(ORBIT_VARIANTS, presentations, strict=True))
        audited: dict[str, base._AuditedPacket] = {}  # noqa: SLF001
        for variant, presentation in variants.items():
            audited_packet = base._audit_packet(  # noqa: SLF001
                presentation,
                require_hidden_relation=variant != "minimal_noncongruent",
            )
            audited[variant] = audited_packet
            packets.append(presentation.packet)
            metadata.append(
                SemanticHoldoutPacketMetadata(
                    packet_sha256=audited_packet.packet_sha256,
                    partition=spec.partition,
                    orbit_id=orbit_id,
                    variant=variant,
                    semantic_seed=spec.semantic_seed,
                    renderer_seed=presentation.renderer_seed,
                    hidden_relation_applicable=variant != "minimal_noncongruent",
                    generator_semantics=spec.generator_semantics,
                    noncommuting_motif=spec.noncommuting_motif,
                    composition_depth=spec.composition_depth,
                    base_records=spec.base_records,
                    latent_states=spec.latent_states,
                    generators=spec.generators,
                    query_ports=spec.query_ports,
                )
            )
            renderers.append(
                base._renderer_ledger(  # noqa: SLF001
                    presentation,
                    audited_packet.packet_sha256,
                )
            )
            recodings.append(
                base._recoding_ledger(  # noqa: SLF001
                    presentation,
                    audited_packet.packet_sha256,
                )
            )
            targets.append(audited_packet.target)
            oracles.append(audited_packet.oracle)
            fingerprints.append(audited_packet.fingerprint)

        orbit_audit = base._verify_orbit(  # noqa: SLF001
            orbit_id=orbit_id,
            partition=spec.partition,
            variants=variants,
            audited=audited,
        )
        orbit_audits.append(orbit_audit)
        measured_depth = _maximum_distinction_depth(
            solve_by_refinement(
                base._latent_packet(hidden),  # noqa: SLF001
                distinction_depth=6,
            )
        )
        if audited["base"].oracle.maximum_distinction_depth != measured_depth:
            raise SemanticHoldoutCorpusError(
                "physical and latent composition-depth receipts disagree"
            )
        semantic_orbits.append(
            SemanticHoldoutOrbitLedger(
                orbit_id=orbit_id,
                partition=spec.partition,
                generator_semantics=spec.generator_semantics,
                noncommuting_motif=spec.noncommuting_motif,
                composition_depth=spec.composition_depth,
                base_records=spec.base_records,
                latent_states=spec.latent_states,
                generators=spec.generators,
                query_ports=spec.query_ports,
                context_rank=_map_rank(hidden.generators[1]),
                motif_rank=_map_rank(hidden.generators[2]),
                context_and_motif_noncommute=not _maps_commute(
                    hidden.generators[1],
                    hidden.generators[2],
                ),
                measured_maximum_distinction_depth=measured_depth,
                base_packet_sha256=audited["base"].packet_sha256,
            )
        )

    packet_hashes = base._verify_complete_ledgers(  # noqa: SLF001
        packets=packets,
        metadata=metadata,
        renderers=renderers,
        recodings=recodings,
        targets=targets,
        oracles=oracles,
        fingerprints=fingerprints,
    )
    _verify_source_deletion(packets)
    split_receipt = _split_receipt(
        packets=packets,
        metadata=metadata,
        semantic_orbits=semantic_orbits,
        fingerprints=fingerprints,
    )
    cells = _cell_receipts(specs)
    train_count = sum(item.partition == TRAIN_PARTITION for item in metadata)
    development_count = len(metadata) - train_count
    if (train_count, development_count) != (
        train_packets,
        development_packets,
    ):
        raise SemanticHoldoutCorpusError("partition packet count drifted")

    all_values = [
        item.value for packet in packets for item in packet.observation_witnesses
    ]
    packet_manifest_sha256 = _sha256(packet_hashes)
    offline_payload = {
        "hidden_automata": hidden_ledgers,
        "metadata": metadata,
        "semantic_orbits": semantic_orbits,
        "renderers": renderers,
        "observation_recodings": recodings,
        "target_relations": targets,
        "oracle_agreements": oracles,
        "fingerprints": fingerprints,
        "orbit_audits": orbit_audits,
        "split_receipt": split_receipt,
        "cells": cells,
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
    manifest = SemanticHoldoutManifest(
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
        oracle_agreement_count=sum(item.independent_oracles_agree for item in oracles),
        hidden_minimal_count=len(hidden_ledgers),
        composition_depth_agreement_count=sum(
            item.composition_depth == item.measured_maximum_distinction_depth
            for item in semantic_orbits
        ),
        noncommuting_motif_count=sum(
            item.context_and_motif_noncommute for item in semantic_orbits
        ),
        reindex_orbit_count=sum(item.reindex_naturality for item in orbit_audits),
        split_orbit_count=sum(item.split_naturality for item in orbit_audits),
        merge_orbit_count=sum(item.merge_naturality for item in orbit_audits),
        collision_orbit_count=sum(
            item.collision_separates_path for item in orbit_audits
        ),
        observation_value_minimum=min(all_values),
        observation_value_maximum=max(all_values),
        negative_observation_values=len({value for value in all_values if value < 0}),
        positive_observation_values=len({value for value in all_values if value > 0}),
        semantics_sha256=plan.semantics_sha256,
        packet_manifest_sha256=packet_manifest_sha256,
        offline_ledger_sha256=offline_ledger_sha256,
        payload_sha256=payload_sha256,
        split_receipt=split_receipt,
        cells=cells,
    )
    return ProceduralSemanticHoldoutCorpus(
        packets=tuple(packets),
        hidden_automata=tuple(hidden_ledgers),
        metadata=tuple(metadata),
        semantic_orbits=tuple(semantic_orbits),
        renderers=tuple(renderers),
        observation_recodings=tuple(recodings),
        target_relations=tuple(targets),
        oracle_agreements=tuple(oracles),
        fingerprints=tuple(fingerprints),
        orbit_audits=tuple(orbit_audits),
        manifest=manifest,
    )


def write_endogenous_congruence_semantic_holdout(
    corpus: ProceduralSemanticHoldoutCorpus,
    output_path: Path,
) -> None:
    """Atomically write model packets and separated assessor ledgers."""

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
    corpus = generate_endogenous_congruence_semantic_holdout(
        seed=args.seed,
        train_packets=args.train_packets,
        development_packets=args.development_packets,
    )
    if args.output is not None:
        write_endogenous_congruence_semantic_holdout(corpus, args.output)
    print(_canonical_json(corpus.manifest))


if __name__ == "__main__":
    _main()
