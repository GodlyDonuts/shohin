"""Finite CPU mechanics for endogenous causal congruence.

This module implements the deterministic unary slice of ECCR described in
``R12_GENERAL_REASONING_MECHANISM_THEORY.md``.  It is an offline mechanics
board, not a neural runtime and not a reasoning claim.

The model-visible packet contains only anonymous records, generators, query
ports, transition witnesses, and observation witnesses.  Quotients, induced
actions, certificates, path equations, orbit provenance, and presentation
morphisms are assessor-side products.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence


MAX_RECORDS = 8
MAX_GENERATORS = 4
MAX_QUERY_PORTS = 4
MAX_PATH_DEPTH = 6
MAX_REFERENCE_RECORDS = 8

Matrix = tuple[tuple[int, ...], ...]
Word = tuple[str, ...]


class CongruenceBoardError(ValueError):
    """Base class for fail-closed board errors."""


class EpisodeUnderidentifiedError(CongruenceBoardError):
    """The physical witnesses do not identify a complete finite episode."""


class EpisodeAmbiguityError(CongruenceBoardError):
    """The packet contains conflicting or multiply asserted evidence."""


class BoardCapacityError(CongruenceBoardError):
    """The packet exceeds the frozen finite mechanics budget."""


class CongruenceInvariantError(CongruenceBoardError):
    """A quotient, certificate, path relation, or naturality square failed."""


@dataclass(frozen=True)
class TransitionWitness:
    source: str
    generator: str
    target: str


@dataclass(frozen=True)
class ObservationWitness:
    record: str
    query_port: str
    value: int


@dataclass(frozen=True)
class EndogenousCongruencePacket:
    """The complete source-deleted, model-visible physical packet."""

    records: tuple[str, ...]
    generators: tuple[str, ...]
    query_ports: tuple[str, ...]
    transition_witnesses: tuple[TransitionWitness, ...]
    observation_witnesses: tuple[ObservationWitness, ...]


@dataclass(frozen=True)
class MergeCertificate:
    left_record: str
    right_record: str
    relation: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DistinctionCertificate:
    left_class: int
    right_class: int
    left_record: str
    right_record: str
    continuation: Word
    query_port: str
    left_value: int
    right_value: int


@dataclass(frozen=True)
class PathCongruence:
    words: tuple[Word, ...]
    class_assignment: tuple[int, ...]
    action_signatures: tuple[tuple[int, ...], ...]
    max_depth: int


@dataclass(frozen=True)
class CongruenceSolution:
    blocks: tuple[tuple[str, ...], ...]
    record_class: tuple[int, ...]
    quotient: Matrix
    physical_generators: tuple[Matrix, ...]
    induced_generators: tuple[Matrix, ...]
    physical_observations: tuple[tuple[int, ...], ...]
    query_readers: tuple[tuple[int, ...], ...]
    merge_certificates: tuple[MergeCertificate, ...]
    distinction_certificates: tuple[DistinctionCertificate, ...]
    path_congruence: PathCongruence


@dataclass(frozen=True)
class PresentationMorphism:
    """A total typed map from one physical presentation to another."""

    record_map: tuple[tuple[str, str], ...]
    generator_map: tuple[tuple[str, str], ...]
    query_map: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class NaturalityWitness:
    source_to_target_class: tuple[int, ...]


@dataclass(frozen=True)
class CongruenceCollisionOrbit:
    """Assessor-side matched presentations; names never enter a model packet."""

    base: EndogenousCongruencePacket
    reindexed: EndogenousCongruencePacket
    split_bisimilar: EndogenousCongruencePacket
    merged: EndogenousCongruencePacket
    minimal_noncongruent: EndogenousCongruencePacket
    commuting_path_twin: EndogenousCongruencePacket
    noncommuting_twin: EndogenousCongruencePacket
    base_to_reindexed: PresentationMorphism
    split_to_base: PresentationMorphism
    base_to_merged: PresentationMorphism
    commuting_pair: tuple[Word, Word]


@dataclass(frozen=True)
class _EpisodeTables:
    transition: Mapping[tuple[str, str], str]
    observation: Mapping[tuple[str, str], int]
    record_index: Mapping[str, int]
    generator_index: Mapping[str, int]
    query_index: Mapping[str, int]


def model_packet_payload(packet: EndogenousCongruencePacket) -> dict[str, object]:
    """Return the complete model-visible payload without assessor products."""

    validate_packet(packet)
    return {
        "records": packet.records,
        "generators": packet.generators,
        "query_ports": packet.query_ports,
        "transition_witnesses": tuple(
            (item.source, item.generator, item.target)
            for item in packet.transition_witnesses
        ),
        "observation_witnesses": tuple(
            (item.record, item.query_port, item.value)
            for item in packet.observation_witnesses
        ),
    }


def _require_unique_nonempty(values: Sequence[str], field: str) -> None:
    if not values:
        raise EpisodeUnderidentifiedError(f"{field} must not be empty")
    if any(not isinstance(value, str) or not value for value in values):
        raise EpisodeAmbiguityError(f"{field} must contain opaque nonempty strings")
    if len(set(values)) != len(values):
        raise EpisodeAmbiguityError(f"{field} contains duplicate identifiers")


def validate_packet(packet: EndogenousCongruencePacket) -> _EpisodeTables:
    """Validate completeness and return exact witness lookup tables."""

    _require_unique_nonempty(packet.records, "records")
    _require_unique_nonempty(packet.generators, "generators")
    _require_unique_nonempty(packet.query_ports, "query_ports")
    if len(packet.records) < 2:
        raise EpisodeUnderidentifiedError("at least two physical records are required")
    if len(packet.records) > MAX_RECORDS:
        raise BoardCapacityError(f"record count exceeds {MAX_RECORDS}")
    if len(packet.generators) > MAX_GENERATORS:
        raise BoardCapacityError(f"generator count exceeds {MAX_GENERATORS}")
    if len(packet.query_ports) > MAX_QUERY_PORTS:
        raise BoardCapacityError(f"query-port count exceeds {MAX_QUERY_PORTS}")

    record_set = set(packet.records)
    generator_set = set(packet.generators)
    query_set = set(packet.query_ports)
    transition: dict[tuple[str, str], str] = {}
    for witness in packet.transition_witnesses:
        if witness.source not in record_set or witness.target not in record_set:
            raise EpisodeAmbiguityError("transition references an unknown record")
        if witness.generator not in generator_set:
            raise EpisodeAmbiguityError("transition references an unknown generator")
        key = (witness.source, witness.generator)
        if key in transition:
            prior = transition[key]
            qualifier = "conflicting" if prior != witness.target else "duplicate"
            raise EpisodeAmbiguityError(f"{qualifier} transition witness for {key}")
        transition[key] = witness.target

    expected_transition_keys = {
        (record, generator)
        for record in packet.records
        for generator in packet.generators
    }
    missing_transitions = expected_transition_keys - transition.keys()
    if missing_transitions:
        raise EpisodeUnderidentifiedError(
            f"missing {len(missing_transitions)} transition witnesses"
        )
    if transition.keys() - expected_transition_keys:
        raise EpisodeAmbiguityError("transition table contains out-of-domain entries")

    observation: dict[tuple[str, str], int] = {}
    for witness in packet.observation_witnesses:
        if witness.record not in record_set:
            raise EpisodeAmbiguityError("observation references an unknown record")
        if witness.query_port not in query_set:
            raise EpisodeAmbiguityError("observation references an unknown query port")
        if not isinstance(witness.value, int):
            raise EpisodeAmbiguityError("observation values must be finite integers")
        key = (witness.record, witness.query_port)
        if key in observation:
            prior = observation[key]
            qualifier = "conflicting" if prior != witness.value else "duplicate"
            raise EpisodeAmbiguityError(f"{qualifier} observation witness for {key}")
        observation[key] = witness.value

    expected_observation_keys = {
        (record, query) for record in packet.records for query in packet.query_ports
    }
    missing_observations = expected_observation_keys - observation.keys()
    if missing_observations:
        raise EpisodeUnderidentifiedError(
            f"missing {len(missing_observations)} observation witnesses"
        )
    if observation.keys() - expected_observation_keys:
        raise EpisodeAmbiguityError("observation table contains out-of-domain entries")

    return _EpisodeTables(
        transition=transition,
        observation=observation,
        record_index={record: index for index, record in enumerate(packet.records)},
        generator_index={
            generator: index for index, generator in enumerate(packet.generators)
        },
        query_index={query: index for index, query in enumerate(packet.query_ports)},
    )


def _canonicalize_blocks(
    blocks: Iterable[Iterable[str]],
    packet: EndogenousCongruencePacket,
) -> tuple[tuple[str, ...], ...]:
    order = {record: index for index, record in enumerate(packet.records)}
    normalized = [
        tuple(sorted(block, key=order.__getitem__)) for block in blocks if tuple(block)
    ]
    normalized.sort(key=lambda block: order[block[0]])
    return tuple(normalized)


def _partition_assignment(
    packet: EndogenousCongruencePacket,
    blocks: Sequence[Sequence[str]],
) -> tuple[int, ...]:
    owner: dict[str, int] = {}
    for block_index, block in enumerate(blocks):
        if not block:
            raise CongruenceInvariantError("partition contains an empty block")
        for record in block:
            if record in owner:
                raise CongruenceInvariantError("partition repeats a physical record")
            owner[record] = block_index
    if set(owner) != set(packet.records):
        raise CongruenceInvariantError("partition does not cover the packet records")
    return tuple(owner[record] for record in packet.records)


def _equivalence_signature(
    blocks: Sequence[Sequence[str]],
) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(block) for block in blocks)


def compute_refinement_partition(
    packet: EndogenousCongruencePacket,
) -> tuple[tuple[str, ...], ...]:
    """Compute the coarsest congruence by iterative signature refinement."""

    tables = validate_packet(packet)
    blocks_by_observation: dict[tuple[int, ...], list[str]] = {}
    for record in packet.records:
        signature = tuple(
            tables.observation[(record, query)] for query in packet.query_ports
        )
        blocks_by_observation.setdefault(signature, []).append(record)
    blocks = _canonicalize_blocks(blocks_by_observation.values(), packet)

    while True:
        assignment = _partition_assignment(packet, blocks)
        record_class = {
            record: assignment[index] for index, record in enumerate(packet.records)
        }
        refined: dict[tuple[tuple[int, ...], tuple[int, ...]], list[str]] = {}
        for record in packet.records:
            observations = tuple(
                tables.observation[(record, query)] for query in packet.query_ports
            )
            successors = tuple(
                record_class[tables.transition[(record, generator)]]
                for generator in packet.generators
            )
            refined.setdefault((observations, successors), []).append(record)
        next_blocks = _canonicalize_blocks(refined.values(), packet)
        if _equivalence_signature(next_blocks) == _equivalence_signature(blocks):
            return next_blocks
        blocks = next_blocks


def _restricted_growth_assignments(size: int) -> Iterable[tuple[int, ...]]:
    labels = [0] * size

    def visit(position: int, maximum: int) -> Iterable[tuple[int, ...]]:
        if position == size:
            yield tuple(labels)
            return
        for label in range(maximum + 2):
            labels[position] = label
            yield from visit(position + 1, max(maximum, label))

    if size:
        yield from visit(1, 0)


def _blocks_from_assignment(
    packet: EndogenousCongruencePacket,
    assignment: Sequence[int],
) -> tuple[tuple[str, ...], ...]:
    blocks: dict[int, list[str]] = {}
    for record, block in zip(packet.records, assignment, strict=True):
        blocks.setdefault(block, []).append(record)
    return _canonicalize_blocks(blocks.values(), packet)


def _reference_partition_is_congruence(
    packet: EndogenousCongruencePacket,
    tables: _EpisodeTables,
    assignment: Sequence[int],
) -> bool:
    for left_index, left in enumerate(packet.records):
        for right_index in range(left_index + 1, len(packet.records)):
            if assignment[left_index] != assignment[right_index]:
                continue
            right = packet.records[right_index]
            if any(
                tables.observation[(left, query)] != tables.observation[(right, query)]
                for query in packet.query_ports
            ):
                return False
            for generator in packet.generators:
                left_target = tables.record_index[tables.transition[(left, generator)]]
                right_target = tables.record_index[
                    tables.transition[(right, generator)]
                ]
                if assignment[left_target] != assignment[right_target]:
                    return False
    return True


def compute_exhaustive_partition(
    packet: EndogenousCongruencePacket,
) -> tuple[tuple[str, ...], ...]:
    """Independent Bell-partition search for the coarsest valid congruence."""

    tables = validate_packet(packet)
    if len(packet.records) > MAX_REFERENCE_RECORDS:
        raise BoardCapacityError(
            f"reference search is capped at {MAX_REFERENCE_RECORDS} records"
        )

    best_size = len(packet.records) + 1
    best: dict[frozenset[frozenset[str]], tuple[tuple[str, ...], ...]] = {}
    for assignment in _restricted_growth_assignments(len(packet.records)):
        class_count = max(assignment) + 1
        if class_count > best_size:
            continue
        if not _reference_partition_is_congruence(packet, tables, assignment):
            continue
        blocks = _blocks_from_assignment(packet, assignment)
        signature = _equivalence_signature(blocks)
        if class_count < best_size:
            best_size = class_count
            best = {signature: blocks}
        elif class_count == best_size:
            best[signature] = blocks

    if not best:
        raise CongruenceInvariantError("no observation-preserving congruence exists")
    if len(best) != 1:
        raise EpisodeAmbiguityError(
            "physical evidence admits multiple incomparable coarsest congruences"
        )
    return next(iter(best.values()))


def validate_candidate_partition(
    packet: EndogenousCongruencePacket,
    blocks: Sequence[Sequence[str]],
    *,
    require_coarsest: bool = True,
) -> None:
    """Validate a proposed hard quotient and optionally enforce coarseness."""

    tables = validate_packet(packet)
    canonical = _canonicalize_blocks(blocks, packet)
    assignment = _partition_assignment(packet, canonical)
    if not _reference_partition_is_congruence(packet, tables, assignment):
        raise CongruenceInvariantError(
            "candidate partition violates observation preservation or descent"
        )
    if require_coarsest:
        expected = compute_refinement_partition(packet)
        if _equivalence_signature(canonical) != _equivalence_signature(expected):
            raise CongruenceInvariantError("candidate partition is not coarsest")


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    if not left or not right:
        raise CongruenceInvariantError("matrix operands must not be empty")
    width = len(left[0])
    if any(len(row) != width for row in left):
        raise CongruenceInvariantError("left matrix is ragged")
    if any(len(row) != len(right[0]) for row in right):
        raise CongruenceInvariantError("right matrix is ragged")
    if width != len(right):
        raise CongruenceInvariantError("matrix dimensions do not compose")
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(width))
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _one_hot(index: int, width: int) -> tuple[int, ...]:
    return tuple(1 if position == index else 0 for position in range(width))


def _physical_generator_matrix(
    packet: EndogenousCongruencePacket,
    tables: _EpisodeTables,
    generator: str,
) -> Matrix:
    width = len(packet.records)
    return tuple(
        _one_hot(
            tables.record_index[tables.transition[(record, generator)]],
            width,
        )
        for record in packet.records
    )


def _induced_generator_matrix(
    packet: EndogenousCongruencePacket,
    tables: _EpisodeTables,
    blocks: Sequence[Sequence[str]],
    assignment: Sequence[int],
    generator: str,
) -> Matrix:
    class_count = len(blocks)
    rows: list[tuple[int, ...]] = []
    for block in blocks:
        representative = block[0]
        target = tables.transition[(representative, generator)]
        rows.append(_one_hot(assignment[tables.record_index[target]], class_count))
    return tuple(rows)


def validate_descent(
    solution: CongruenceSolution,
    *,
    induced_override: Sequence[Matrix] | None = None,
) -> None:
    induced = (
        tuple(induced_override)
        if induced_override is not None
        else solution.induced_generators
    )
    if len(induced) != len(solution.physical_generators):
        raise CongruenceInvariantError("generator matrix count mismatch")
    for physical, private in zip(
        solution.physical_generators,
        induced,
        strict=True,
    ):
        if _matmul(physical, solution.quotient) != _matmul(
            solution.quotient,
            private,
        ):
            raise CongruenceInvariantError("T_g C != C A_g")


def validate_observation_factorization(solution: CongruenceSolution) -> None:
    for physical, reader in zip(
        solution.physical_observations,
        solution.query_readers,
        strict=True,
    ):
        reconstructed = tuple(
            sum(
                solution.quotient[row][column] * reader[column]
                for column in range(len(reader))
            )
            for row in range(len(solution.quotient))
        )
        if reconstructed != physical:
            raise CongruenceInvariantError("o_q != C r_q")


def _ordered_pair(
    left: str,
    right: str,
    record_index: Mapping[str, int],
) -> tuple[str, str]:
    if record_index[left] <= record_index[right]:
        return left, right
    return right, left


def _build_merge_certificate(
    packet: EndogenousCongruencePacket,
    tables: _EpisodeTables,
    left: str,
    right: str,
) -> MergeCertificate:
    seed = _ordered_pair(left, right, tables.record_index)
    relation = {seed}
    queue = deque([seed])
    while queue:
        current_left, current_right = queue.popleft()
        for generator in packet.generators:
            successor = _ordered_pair(
                tables.transition[(current_left, generator)],
                tables.transition[(current_right, generator)],
                tables.record_index,
            )
            if successor not in relation:
                relation.add(successor)
                queue.append(successor)
    ordered = tuple(
        sorted(
            relation,
            key=lambda pair: (
                tables.record_index[pair[0]],
                tables.record_index[pair[1]],
            ),
        )
    )
    certificate = MergeCertificate(left, right, ordered)
    validate_merge_certificate(packet, certificate)
    return certificate


def validate_merge_certificate(
    packet: EndogenousCongruencePacket,
    certificate: MergeCertificate,
) -> None:
    tables = validate_packet(packet)
    relation = set(certificate.relation)
    seed = _ordered_pair(
        certificate.left_record,
        certificate.right_record,
        tables.record_index,
    )
    if seed not in relation:
        raise CongruenceInvariantError("merge certificate omits its claimed pair")
    for left, right in relation:
        if left not in tables.record_index or right not in tables.record_index:
            raise CongruenceInvariantError(
                "merge certificate references unknown records"
            )
        for query in packet.query_ports:
            if tables.observation[(left, query)] != tables.observation[(right, query)]:
                raise CongruenceInvariantError(
                    "merge certificate violates observation bisimulation"
                )
        for generator in packet.generators:
            successor = _ordered_pair(
                tables.transition[(left, generator)],
                tables.transition[(right, generator)],
                tables.record_index,
            )
            if successor not in relation:
                raise CongruenceInvariantError(
                    "merge certificate is not generator-closed"
                )


def _apply_record_word(
    tables: _EpisodeTables,
    start: str,
    word: Sequence[str],
) -> str:
    current = start
    for generator in word:
        current = tables.transition[(current, generator)]
    return current


def _shortest_distinction(
    packet: EndogenousCongruencePacket,
    tables: _EpisodeTables,
    left: str,
    right: str,
    bound: int,
) -> tuple[Word, str, int, int]:
    queue: deque[tuple[str, str, Word]] = deque([(left, right, ())])
    visited = {(left, right)}
    while queue:
        current_left, current_right, word = queue.popleft()
        for query in packet.query_ports:
            left_value = tables.observation[(current_left, query)]
            right_value = tables.observation[(current_right, query)]
            if left_value != right_value:
                return word, query, left_value, right_value
        if len(word) >= bound:
            continue
        for generator in packet.generators:
            successor = (
                tables.transition[(current_left, generator)],
                tables.transition[(current_right, generator)],
            )
            if successor in visited:
                continue
            visited.add(successor)
            queue.append((*successor, (*word, generator)))
    raise EpisodeUnderidentifiedError(
        f"no distinction continuation found within depth {bound}"
    )


def validate_distinction_certificate(
    packet: EndogenousCongruencePacket,
    certificate: DistinctionCertificate,
) -> None:
    tables = validate_packet(packet)
    if certificate.query_port not in tables.query_index:
        raise CongruenceInvariantError("distinction uses an unknown query port")
    left_terminal = _apply_record_word(
        tables,
        certificate.left_record,
        certificate.continuation,
    )
    right_terminal = _apply_record_word(
        tables,
        certificate.right_record,
        certificate.continuation,
    )
    left_value = tables.observation[(left_terminal, certificate.query_port)]
    right_value = tables.observation[(right_terminal, certificate.query_port)]
    if (left_value, right_value) != (
        certificate.left_value,
        certificate.right_value,
    ):
        raise CongruenceInvariantError("distinction certificate values are stale")
    if left_value == right_value:
        raise CongruenceInvariantError("distinction continuation does not distinguish")

    for depth in range(len(certificate.continuation)):
        for word in product(packet.generators, repeat=depth):
            left_probe = _apply_record_word(tables, certificate.left_record, word)
            right_probe = _apply_record_word(tables, certificate.right_record, word)
            if any(
                tables.observation[(left_probe, query)]
                != tables.observation[(right_probe, query)]
                for query in packet.query_ports
            ):
                raise CongruenceInvariantError(
                    "distinction continuation is not shortest"
                )


def _enumerate_words(generators: Sequence[str], max_depth: int) -> tuple[Word, ...]:
    if max_depth < 0 or max_depth > MAX_PATH_DEPTH:
        raise BoardCapacityError(f"path depth must be in [0, {MAX_PATH_DEPTH}]")
    return tuple(
        word
        for depth in range(max_depth + 1)
        for word in product(generators, repeat=depth)
    )


def _apply_private_word(
    induced_generators: Sequence[Matrix],
    generator_index: Mapping[str, int],
    start_class: int,
    word: Sequence[str],
) -> int:
    current = start_class
    for generator in word:
        row = induced_generators[generator_index[generator]][current]
        if sum(row) != 1:
            raise CongruenceInvariantError(
                "deterministic mechanics require one-hot induced actions"
            )
        current = row.index(1)
    return current


def _build_path_congruence(
    packet: EndogenousCongruencePacket,
    induced_generators: Sequence[Matrix],
    class_count: int,
    max_depth: int,
) -> PathCongruence:
    words = _enumerate_words(packet.generators, max_depth)
    generator_index = {
        generator: index for index, generator in enumerate(packet.generators)
    }
    signatures = tuple(
        tuple(
            _apply_private_word(
                induced_generators,
                generator_index,
                start_class,
                word,
            )
            for start_class in range(class_count)
        )
        for word in words
    )
    signature_class: dict[tuple[int, ...], int] = {}
    assignment: list[int] = []
    for signature in signatures:
        assignment.append(signature_class.setdefault(signature, len(signature_class)))
    result = PathCongruence(
        words=words,
        class_assignment=tuple(assignment),
        action_signatures=signatures,
        max_depth=max_depth,
    )
    validate_path_partition(result, result.class_assignment)
    return result


def validate_path_partition(
    path_congruence: PathCongruence,
    candidate_assignment: Sequence[int],
) -> None:
    """Require exact action equality and bounded two-sided contextual closure."""

    if len(candidate_assignment) != len(path_congruence.words):
        raise CongruenceInvariantError("path assignment has the wrong size")
    word_index = {word: index for index, word in enumerate(path_congruence.words)}
    for left in range(len(path_congruence.words)):
        for right in range(left, len(path_congruence.words)):
            claimed_equal = candidate_assignment[left] == candidate_assignment[right]
            action_equal = (
                path_congruence.action_signatures[left]
                == path_congruence.action_signatures[right]
            )
            if claimed_equal != action_equal:
                raise CongruenceInvariantError(
                    "path quotient is not the exact induced-action congruence"
                )
            if not claimed_equal:
                continue
            left_word = path_congruence.words[left]
            right_word = path_congruence.words[right]
            generator_set = {item for word in path_congruence.words for item in word}
            for generator in generator_set:
                for contextual_left, contextual_right in (
                    ((*left_word, generator), (*right_word, generator)),
                    ((generator, *left_word), (generator, *right_word)),
                ):
                    if (
                        len(contextual_left) > path_congruence.max_depth
                        or len(contextual_right) > path_congruence.max_depth
                    ):
                        continue
                    left_index = word_index[contextual_left]
                    right_index = word_index[contextual_right]
                    if (
                        candidate_assignment[left_index]
                        != candidate_assignment[right_index]
                    ):
                        raise CongruenceInvariantError(
                            "path congruence is not contextually closed"
                        )


def path_equivalent(
    path_congruence: PathCongruence,
    left: Word,
    right: Word,
) -> bool:
    word_index = {word: index for index, word in enumerate(path_congruence.words)}
    if left not in word_index or right not in word_index:
        raise BoardCapacityError("path is outside the frozen bounded bank")
    return (
        path_congruence.class_assignment[word_index[left]]
        == path_congruence.class_assignment[word_index[right]]
    )


def _build_solution(
    packet: EndogenousCongruencePacket,
    blocks: tuple[tuple[str, ...], ...],
    *,
    path_depth: int,
    distinction_depth: int,
) -> CongruenceSolution:
    tables = validate_packet(packet)
    validate_candidate_partition(packet, blocks, require_coarsest=True)
    assignment = _partition_assignment(packet, blocks)
    class_count = len(blocks)
    quotient = tuple(_one_hot(block, class_count) for block in assignment)
    physical_generators = tuple(
        _physical_generator_matrix(packet, tables, generator)
        for generator in packet.generators
    )
    induced_generators = tuple(
        _induced_generator_matrix(
            packet,
            tables,
            blocks,
            assignment,
            generator,
        )
        for generator in packet.generators
    )
    physical_observations = tuple(
        tuple(tables.observation[(record, query)] for record in packet.records)
        for query in packet.query_ports
    )
    query_readers = tuple(
        tuple(tables.observation[(block[0], query)] for block in blocks)
        for query in packet.query_ports
    )

    merge_certificates = tuple(
        _build_merge_certificate(packet, tables, left, right)
        for block in blocks
        for left_index, left in enumerate(block)
        for right in block[left_index + 1 :]
    )
    distinctions: list[DistinctionCertificate] = []
    for left_class in range(class_count):
        for right_class in range(left_class + 1, class_count):
            left = blocks[left_class][0]
            right = blocks[right_class][0]
            word, query, left_value, right_value = _shortest_distinction(
                packet,
                tables,
                left,
                right,
                distinction_depth,
            )
            certificate = DistinctionCertificate(
                left_class=left_class,
                right_class=right_class,
                left_record=left,
                right_record=right,
                continuation=word,
                query_port=query,
                left_value=left_value,
                right_value=right_value,
            )
            validate_distinction_certificate(packet, certificate)
            distinctions.append(certificate)

    path_congruence = _build_path_congruence(
        packet,
        induced_generators,
        class_count,
        path_depth,
    )
    solution = CongruenceSolution(
        blocks=blocks,
        record_class=assignment,
        quotient=quotient,
        physical_generators=physical_generators,
        induced_generators=induced_generators,
        physical_observations=physical_observations,
        query_readers=query_readers,
        merge_certificates=merge_certificates,
        distinction_certificates=tuple(distinctions),
        path_congruence=path_congruence,
    )
    validate_solution(packet, solution)
    return solution


def solve_by_refinement(
    packet: EndogenousCongruencePacket,
    *,
    path_depth: int = 4,
    distinction_depth: int = 6,
) -> CongruenceSolution:
    return _build_solution(
        packet,
        compute_refinement_partition(packet),
        path_depth=path_depth,
        distinction_depth=distinction_depth,
    )


def solve_by_exhaustive_search(
    packet: EndogenousCongruencePacket,
    *,
    path_depth: int = 4,
    distinction_depth: int = 6,
) -> CongruenceSolution:
    return _build_solution(
        packet,
        compute_exhaustive_partition(packet),
        path_depth=path_depth,
        distinction_depth=distinction_depth,
    )


def solve_with_independent_crosscheck(
    packet: EndogenousCongruencePacket,
    *,
    path_depth: int = 4,
    distinction_depth: int = 6,
) -> CongruenceSolution:
    production = solve_by_refinement(
        packet,
        path_depth=path_depth,
        distinction_depth=distinction_depth,
    )
    reference = solve_by_exhaustive_search(
        packet,
        path_depth=path_depth,
        distinction_depth=distinction_depth,
    )
    if _equivalence_signature(production.blocks) != _equivalence_signature(
        reference.blocks
    ):
        raise CongruenceInvariantError(
            "partition refinement and exhaustive reference disagree"
        )
    if production.induced_generators != reference.induced_generators:
        raise CongruenceInvariantError("independent induced generators disagree")
    if production.query_readers != reference.query_readers:
        raise CongruenceInvariantError("independent query readers disagree")
    if (
        production.path_congruence.action_signatures
        != reference.path_congruence.action_signatures
    ):
        raise CongruenceInvariantError("independent path actions disagree")
    return production


def validate_solution(
    packet: EndogenousCongruencePacket,
    solution: CongruenceSolution,
) -> None:
    tables = validate_packet(packet)
    validate_candidate_partition(packet, solution.blocks, require_coarsest=True)
    expected_assignment = _partition_assignment(packet, solution.blocks)
    if solution.record_class != expected_assignment:
        raise CongruenceInvariantError("record-class assignment is stale")
    if solution.quotient != tuple(
        _one_hot(block, len(solution.blocks)) for block in solution.record_class
    ):
        raise CongruenceInvariantError("hard quotient is not row-one-hot")
    expected_physical_generators = tuple(
        _physical_generator_matrix(packet, tables, generator)
        for generator in packet.generators
    )
    if solution.physical_generators != expected_physical_generators:
        raise CongruenceInvariantError(
            "physical generator matrices do not match the sealed witnesses"
        )
    expected_induced_generators = tuple(
        _induced_generator_matrix(
            packet,
            tables,
            solution.blocks,
            solution.record_class,
            generator,
        )
        for generator in packet.generators
    )
    if solution.induced_generators != expected_induced_generators:
        raise CongruenceInvariantError(
            "induced generator matrices do not match the hard quotient"
        )
    expected_physical_observations = tuple(
        tuple(tables.observation[(record, query)] for record in packet.records)
        for query in packet.query_ports
    )
    if solution.physical_observations != expected_physical_observations:
        raise CongruenceInvariantError(
            "physical observations do not match the sealed witnesses"
        )
    expected_readers = tuple(
        tuple(tables.observation[(block[0], query)] for block in solution.blocks)
        for query in packet.query_ports
    )
    if solution.query_readers != expected_readers:
        raise CongruenceInvariantError(
            "query readers do not factor the sealed observations"
        )
    validate_descent(solution)
    validate_observation_factorization(solution)
    for certificate in solution.merge_certificates:
        validate_merge_certificate(packet, certificate)
    expected_merge_pairs = {
        frozenset((left, right))
        for block in solution.blocks
        for left_index, left in enumerate(block)
        for right in block[left_index + 1 :]
    }
    actual_merge_pairs = {
        frozenset((certificate.left_record, certificate.right_record))
        for certificate in solution.merge_certificates
    }
    if actual_merge_pairs != expected_merge_pairs:
        raise CongruenceInvariantError(
            "merge certificates do not cover every nontrivial quotient pair"
        )
    expected_distinctions = {
        (left, right)
        for left in range(len(solution.blocks))
        for right in range(left + 1, len(solution.blocks))
    }
    actual_distinctions = {
        (certificate.left_class, certificate.right_class)
        for certificate in solution.distinction_certificates
    }
    if actual_distinctions != expected_distinctions:
        raise CongruenceInvariantError(
            "distinction certificates do not cover every class pair"
        )
    for certificate in solution.distinction_certificates:
        if (
            solution.record_class[tables.record_index[certificate.left_record]]
            != certificate.left_class
            or solution.record_class[tables.record_index[certificate.right_record]]
            != certificate.right_class
        ):
            raise CongruenceInvariantError(
                "distinction certificate points at the wrong quotient classes"
            )
        validate_distinction_certificate(packet, certificate)
    expected_path = _build_path_congruence(
        packet,
        solution.induced_generators,
        len(solution.blocks),
        solution.path_congruence.max_depth,
    )
    if solution.path_congruence != expected_path:
        raise CongruenceInvariantError(
            "path congruence does not match the induced generator actions"
        )
    validate_path_partition(
        solution.path_congruence,
        solution.path_congruence.class_assignment,
    )


def _mapping(items: Sequence[tuple[str, str]], field: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for source, target in items:
        if source in result:
            raise CongruenceInvariantError(f"{field} repeats a source identifier")
        result[source] = target
    return result


def validate_presentation_morphism(
    source: EndogenousCongruencePacket,
    target: EndogenousCongruencePacket,
    morphism: PresentationMorphism,
) -> None:
    source_tables = validate_packet(source)
    target_tables = validate_packet(target)
    record_map = _mapping(morphism.record_map, "record map")
    generator_map = _mapping(morphism.generator_map, "generator map")
    query_map = _mapping(morphism.query_map, "query map")
    if set(record_map) != set(source.records):
        raise CongruenceInvariantError("record morphism is not total")
    if set(generator_map) != set(source.generators):
        raise CongruenceInvariantError("generator morphism is not total")
    if set(query_map) != set(source.query_ports):
        raise CongruenceInvariantError("query morphism is not total")
    if not set(record_map.values()) <= set(target.records):
        raise CongruenceInvariantError("record morphism leaves the target packet")
    if set(generator_map.values()) != set(target.generators):
        raise CongruenceInvariantError("generator morphism is not onto")
    if set(query_map.values()) != set(target.query_ports):
        raise CongruenceInvariantError("query morphism is not onto")

    for record in source.records:
        for generator in source.generators:
            mapped_source = record_map[record]
            mapped_generator = generator_map[generator]
            mapped_target = record_map[source_tables.transition[(record, generator)]]
            if (
                target_tables.transition[(mapped_source, mapped_generator)]
                != mapped_target
            ):
                raise CongruenceInvariantError(
                    "presentation morphism does not preserve transitions"
                )
        for query in source.query_ports:
            if (
                source_tables.observation[(record, query)]
                != target_tables.observation[(record_map[record], query_map[query])]
            ):
                raise CongruenceInvariantError(
                    "presentation morphism does not preserve observations"
                )


def validate_presentation_naturality(
    source: EndogenousCongruencePacket,
    target: EndogenousCongruencePacket,
    morphism: PresentationMorphism,
    *,
    source_solution: CongruenceSolution | None = None,
    target_solution: CongruenceSolution | None = None,
) -> NaturalityWitness:
    """Verify quotient, action, and reader squares for a presentation map."""

    validate_presentation_morphism(source, target, morphism)
    source_solution = source_solution or solve_with_independent_crosscheck(source)
    target_solution = target_solution or solve_with_independent_crosscheck(target)
    record_map = _mapping(morphism.record_map, "record map")
    generator_map = _mapping(morphism.generator_map, "generator map")
    query_map = _mapping(morphism.query_map, "query map")
    source_record_class = dict(
        zip(source.records, source_solution.record_class, strict=True)
    )
    target_record_class = dict(
        zip(target.records, target_solution.record_class, strict=True)
    )

    class_map: dict[int, int] = {}
    for record in source.records:
        source_class = source_record_class[record]
        target_class = target_record_class[record_map[record]]
        prior = class_map.setdefault(source_class, target_class)
        if prior != target_class:
            raise CongruenceInvariantError(
                "physical morphism does not descend to the quotient"
            )
    if set(class_map) != set(range(len(source_solution.blocks))):
        raise CongruenceInvariantError("induced private map is not total")
    if set(class_map.values()) != set(range(len(target_solution.blocks))):
        raise CongruenceInvariantError("induced private map is not onto")
    if len(set(class_map.values())) != len(class_map):
        raise CongruenceInvariantError(
            "equivalent presentations changed the causal quotient"
        )

    source_generator_index = {
        generator: index for index, generator in enumerate(source.generators)
    }
    target_generator_index = {
        generator: index for index, generator in enumerate(target.generators)
    }
    for generator in source.generators:
        source_matrix = source_solution.induced_generators[
            source_generator_index[generator]
        ]
        target_matrix = target_solution.induced_generators[
            target_generator_index[generator_map[generator]]
        ]
        for source_class, target_class in class_map.items():
            source_target = source_matrix[source_class].index(1)
            target_target = target_matrix[target_class].index(1)
            if class_map[source_target] != target_target:
                raise CongruenceInvariantError(
                    "generator action is not natural across presentations"
                )

    source_query_index = {
        query: index for index, query in enumerate(source.query_ports)
    }
    target_query_index = {
        query: index for index, query in enumerate(target.query_ports)
    }
    for query in source.query_ports:
        source_reader = source_solution.query_readers[source_query_index[query]]
        target_reader = target_solution.query_readers[
            target_query_index[query_map[query]]
        ]
        for source_class, target_class in class_map.items():
            if source_reader[source_class] != target_reader[target_class]:
                raise CongruenceInvariantError(
                    "query reader is not natural across presentations"
                )

    target_word_index = {
        word: index for index, word in enumerate(target_solution.path_congruence.words)
    }
    for source_word, source_signature in zip(
        source_solution.path_congruence.words,
        source_solution.path_congruence.action_signatures,
        strict=True,
    ):
        mapped_word = tuple(generator_map[item] for item in source_word)
        if mapped_word not in target_word_index:
            raise CongruenceInvariantError(
                "presentation map leaves the bounded path bank"
            )
        target_signature = target_solution.path_congruence.action_signatures[
            target_word_index[mapped_word]
        ]
        for source_class, source_destination in enumerate(source_signature):
            target_class = class_map[source_class]
            if class_map[source_destination] != target_signature[target_class]:
                raise CongruenceInvariantError(
                    "path action is not natural across presentations"
                )
    return NaturalityWitness(
        source_to_target_class=tuple(
            class_map[index] for index in range(len(source_solution.blocks))
        )
    )


def _build_packet(
    records: Sequence[str],
    generators: Sequence[str],
    query_ports: Sequence[str],
    state_of: Mapping[str, int],
    generator_functions: Mapping[str, Sequence[int]],
    representative: Mapping[int, str],
    *,
    transition_overrides: Mapping[tuple[str, str], str] | None = None,
) -> EndogenousCongruencePacket:
    overrides = dict(transition_overrides or {})
    transitions = []
    for record in records:
        state = state_of[record]
        for generator in generators:
            target_state = generator_functions[generator][state]
            target = overrides.get((record, generator), representative[target_state])
            transitions.append(TransitionWitness(record, generator, target))
    observations = []
    for record in records:
        first_bit = state_of[record] // 2
        values = (first_bit, 1 - first_bit)
        for query, value in zip(query_ports, values, strict=True):
            observations.append(ObservationWitness(record, query, value))
    packet = EndogenousCongruencePacket(
        records=tuple(records),
        generators=tuple(generators),
        query_ports=tuple(query_ports),
        transition_witnesses=tuple(transitions),
        observation_witnesses=tuple(observations),
    )
    validate_packet(packet)
    return packet


def _reindex_packet(
    packet: EndogenousCongruencePacket,
    record_map: Mapping[str, str],
    generator_map: Mapping[str, str],
    query_map: Mapping[str, str],
    *,
    record_order: Sequence[str],
    generator_order: Sequence[str],
    query_order: Sequence[str],
) -> tuple[EndogenousCongruencePacket, PresentationMorphism]:
    tables = validate_packet(packet)
    if set(record_map) != set(packet.records):
        raise CongruenceInvariantError("reindex record map is not total")
    if set(generator_map) != set(packet.generators):
        raise CongruenceInvariantError("reindex generator map is not total")
    if set(query_map) != set(packet.query_ports):
        raise CongruenceInvariantError("reindex query map is not total")
    inverse_record = {value: key for key, value in record_map.items()}
    inverse_generator = {value: key for key, value in generator_map.items()}
    inverse_query = {value: key for key, value in query_map.items()}
    if (
        len(inverse_record) != len(record_map)
        or len(inverse_generator) != len(generator_map)
        or len(inverse_query) != len(query_map)
    ):
        raise CongruenceInvariantError("reindex maps must be bijections")
    if set(record_order) != set(inverse_record):
        raise CongruenceInvariantError("reindexed record order is incomplete")
    if set(generator_order) != set(inverse_generator):
        raise CongruenceInvariantError("reindexed generator order is incomplete")
    if set(query_order) != set(inverse_query):
        raise CongruenceInvariantError("reindexed query order is incomplete")

    transitions = tuple(
        TransitionWitness(
            target_record,
            target_generator,
            record_map[
                tables.transition[
                    (
                        inverse_record[target_record],
                        inverse_generator[target_generator],
                    )
                ]
            ],
        )
        for target_record in record_order
        for target_generator in generator_order
    )
    observations = tuple(
        ObservationWitness(
            target_record,
            target_query,
            tables.observation[
                (
                    inverse_record[target_record],
                    inverse_query[target_query],
                )
            ],
        )
        for target_record in record_order
        for target_query in query_order
    )
    result = EndogenousCongruencePacket(
        records=tuple(record_order),
        generators=tuple(generator_order),
        query_ports=tuple(query_order),
        transition_witnesses=transitions,
        observation_witnesses=observations,
    )
    morphism = PresentationMorphism(
        record_map=tuple(record_map.items()),
        generator_map=tuple(generator_map.items()),
        query_map=tuple(query_map.items()),
    )
    validate_presentation_morphism(packet, result, morphism)
    return result, morphism


def _collapse_packet(
    packet: EndogenousCongruencePacket,
    record_map: Mapping[str, str],
    target_records: Sequence[str],
) -> tuple[EndogenousCongruencePacket, PresentationMorphism]:
    tables = validate_packet(packet)
    transitions: dict[tuple[str, str], str] = {}
    observations: dict[tuple[str, str], int] = {}
    for source in packet.records:
        mapped_source = record_map[source]
        for generator in packet.generators:
            key = (mapped_source, generator)
            value = record_map[tables.transition[(source, generator)]]
            if key in transitions and transitions[key] != value:
                raise CongruenceInvariantError(
                    "record collapse is not generator-compatible"
                )
            transitions[key] = value
        for query in packet.query_ports:
            key = (mapped_source, query)
            value = tables.observation[(source, query)]
            if key in observations and observations[key] != value:
                raise CongruenceInvariantError(
                    "record collapse is not observation-compatible"
                )
            observations[key] = value
    result = EndogenousCongruencePacket(
        records=tuple(target_records),
        generators=packet.generators,
        query_ports=packet.query_ports,
        transition_witnesses=tuple(
            TransitionWitness(record, generator, transitions[(record, generator)])
            for record in target_records
            for generator in packet.generators
        ),
        observation_witnesses=tuple(
            ObservationWitness(record, query, observations[(record, query)])
            for record in target_records
            for query in packet.query_ports
        ),
    )
    morphism = PresentationMorphism(
        record_map=tuple(record_map.items()),
        generator_map=tuple((item, item) for item in packet.generators),
        query_map=tuple((item, item) for item in packet.query_ports),
    )
    validate_presentation_morphism(packet, result, morphism)
    return result, morphism


def build_congruence_collision_orbit() -> CongruenceCollisionOrbit:
    """Build one deterministic seven-presentation collision orbit."""

    records = ("x_17", "x_03", "x_29", "x_11", "x_23", "x_05")
    generators = ("u_41", "u_07", "u_23")
    query_ports = ("v_19", "v_31")
    state_of = {
        "x_17": 0,
        "x_03": 0,
        "x_29": 1,
        "x_11": 2,
        "x_23": 3,
        "x_05": 3,
    }
    representative = {0: "x_17", 1: "x_29", 2: "x_11", 3: "x_23"}
    generator_functions = {
        "u_41": (2, 3, 0, 1),
        "u_07": (1, 0, 3, 2),
        "u_23": (0, 2, 1, 3),
    }
    base = _build_packet(
        records,
        generators,
        query_ports,
        state_of,
        generator_functions,
        representative,
    )

    reindexed, base_to_reindexed = _reindex_packet(
        base,
        {
            "x_17": "z_44",
            "x_03": "z_08",
            "x_29": "z_51",
            "x_11": "z_13",
            "x_23": "z_37",
            "x_05": "z_02",
        },
        {"u_41": "w_62", "u_07": "w_14", "u_23": "w_35"},
        {"v_19": "k_27", "v_31": "k_09"},
        record_order=("z_13", "z_02", "z_44", "z_37", "z_08", "z_51"),
        generator_order=("w_35", "w_62", "w_14"),
        query_order=("k_09", "k_27"),
    )

    split_records = (*records, "x_37")
    split_state = {**state_of, "x_37": 1}
    split = _build_packet(
        split_records,
        generators,
        query_ports,
        split_state,
        generator_functions,
        representative,
    )
    split_to_base = PresentationMorphism(
        record_map=tuple(
            (record, "x_29" if record == "x_37" else record) for record in split_records
        ),
        generator_map=tuple((item, item) for item in generators),
        query_map=tuple((item, item) for item in query_ports),
    )
    validate_presentation_morphism(split, base, split_to_base)

    merged, base_to_merged = _collapse_packet(
        base,
        {
            "x_17": "x_17",
            "x_03": "x_17",
            "x_29": "x_29",
            "x_11": "x_11",
            "x_23": "x_23",
            "x_05": "x_05",
        },
        ("x_17", "x_29", "x_11", "x_23", "x_05"),
    )

    minimal_noncongruent = _build_packet(
        records,
        generators,
        query_ports,
        state_of,
        generator_functions,
        representative,
        transition_overrides={
            ("x_03", "u_23"): "x_11",
            ("x_29", "u_23"): "x_17",
        },
    )

    commuting_path_twin, _ = _reindex_packet(
        base,
        {
            "x_17": "x_03",
            "x_03": "x_17",
            "x_29": "x_29",
            "x_11": "x_11",
            "x_23": "x_23",
            "x_05": "x_05",
        },
        {item: item for item in generators},
        {item: item for item in query_ports},
        record_order=records,
        generator_order=generators,
        query_order=query_ports,
    )

    noncommuting_twin = _build_packet(
        records,
        generators,
        query_ports,
        state_of,
        generator_functions,
        representative,
        transition_overrides={
            ("x_17", "u_07"): "x_11",
            ("x_23", "u_07"): "x_29",
        },
    )
    return CongruenceCollisionOrbit(
        base=base,
        reindexed=reindexed,
        split_bisimilar=split,
        merged=merged,
        minimal_noncongruent=minimal_noncongruent,
        commuting_path_twin=commuting_path_twin,
        noncommuting_twin=noncommuting_twin,
        base_to_reindexed=base_to_reindexed,
        split_to_base=split_to_base,
        base_to_merged=base_to_merged,
        commuting_pair=(("u_41", "u_07"), ("u_07", "u_41")),
    )


def audit_collision_orbit(
    orbit: CongruenceCollisionOrbit | None = None,
) -> dict[str, int | bool]:
    """Run every finite invariant and return a compact mechanics receipt."""

    orbit = orbit or build_congruence_collision_orbit()
    packets = (
        orbit.base,
        orbit.reindexed,
        orbit.split_bisimilar,
        orbit.merged,
        orbit.minimal_noncongruent,
        orbit.commuting_path_twin,
        orbit.noncommuting_twin,
    )
    solutions = tuple(solve_with_independent_crosscheck(packet) for packet in packets)
    validate_presentation_naturality(
        orbit.base,
        orbit.reindexed,
        orbit.base_to_reindexed,
        source_solution=solutions[0],
        target_solution=solutions[1],
    )
    validate_presentation_naturality(
        orbit.split_bisimilar,
        orbit.base,
        orbit.split_to_base,
        source_solution=solutions[2],
        target_solution=solutions[0],
    )
    validate_presentation_naturality(
        orbit.base,
        orbit.merged,
        orbit.base_to_merged,
        source_solution=solutions[0],
        target_solution=solutions[3],
    )
    left, right = orbit.commuting_pair
    if not path_equivalent(solutions[5].path_congruence, left, right):
        raise CongruenceInvariantError("commuting twin lost its path equation")
    if path_equivalent(solutions[6].path_congruence, left, right):
        raise CongruenceInvariantError("noncommuting twin retained a false equation")
    if len(solutions[4].blocks) <= len(solutions[0].blocks):
        raise CongruenceInvariantError(
            "minimal noncongruent twin did not force a quotient split"
        )
    return {
        "presentations": len(packets),
        "physical_records": sum(len(packet.records) for packet in packets),
        "transition_witnesses": sum(
            len(packet.transition_witnesses) for packet in packets
        ),
        "observation_witnesses": sum(
            len(packet.observation_witnesses) for packet in packets
        ),
        "merge_certificates": sum(
            len(solution.merge_certificates) for solution in solutions
        ),
        "distinction_certificates": sum(
            len(solution.distinction_certificates) for solution in solutions
        ),
        "independent_oracles_agree": True,
        "split_naturality": True,
        "merge_naturality": True,
        "commuting_separated": True,
    }
