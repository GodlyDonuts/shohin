"""Deterministic bounded procedural generator for the neural TCRR board.

This module is an offline generation and custody boundary.  It reuses the
audited source-deleted packet schema and the independent nested reference
mechanics, but keeps semantic family names, templates, renderer identities,
composition classes, seeds, legal actions, and oracle receipts outside every
model-visible packet.

The generator deliberately materializes only a small pilot.  It opens train
and development partitions, never a confirmation partition.  Every candidate
is checked against the frozen N16/C16/Y8/R8/P12/A3/D8 geometry and the explicit
128-action compute budget.  Inadmissible candidates are rejected atomically;
labels are never truncated.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import neural_tcrr_board as audited
import typed_critical_pair_rewrite_board as mechanics


GENERATOR_VERSION = "neural-tcrr-procedural-pilot-v1"
DEFAULT_SEED = 2026072302
DEFAULT_MAX_ATTEMPTS = 8
MAX_LEGAL_ACTIONS = 128
MAX_ORACLE_STATES = 10_000

N = audited.MAX_CAPACITY
C = audited.MAX_CONSTRUCTORS
Y = audited.MAX_TYPES
R = audited.MAX_RULES
P = audited.MAX_RULE_SIDE_NODES
A = audited.MAX_ARITY
D = audited.MAX_PATH_DEPTH

_FROZEN_GEOMETRY = (16, 16, 8, 8, 12, 3, 8)
_ACTIVE_GEOMETRY = (N, C, Y, R, P, A, D)

Partition = Literal["local_transition_train", "local_transition_development"]
Family = Literal[
    "algebraic_normalization",
    "boolean_simplification",
    "list_tree_normalization",
    "typed_stack",
    "dataflow",
]
TwinKind = Literal[
    "render_reindex",
    "no_redex",
    "shared_occurrence",
    "capacity",
]
PacketGroupFactory: TypeAlias = Callable[
    [int],
    Sequence[audited.SourceDeletedPacket],
]
PacketGroupValidator: TypeAlias = Callable[
    [tuple["CandidateAssessment", ...]],
    None,
]

_TRAIN_FAMILIES = frozenset(
    {
        "algebraic_normalization",
        "boolean_simplification",
        "list_tree_normalization",
    }
)
_DEVELOPMENT_FAMILIES = frozenset({"typed_stack", "dataflow"})
_CORE_FAMILIES = (
    "algebraic_normalization",
    "boolean_simplification",
    "list_tree_normalization",
    "typed_stack",
    "dataflow",
)


class ProceduralBoardError(ValueError):
    """Raised when generated custody or a board-level invariant fails."""


class ProceduralCandidateRejected(ProceduralBoardError):
    """Raised when one candidate must be rejected rather than truncated."""


@dataclass(frozen=True)
class GeometryReceipt:
    """Frozen tensor and action geometry used for candidate admission."""

    reservoir_slots: int = N
    constructors: int = C
    types: int = Y
    rules: int = R
    rule_side_nodes: int = P
    arity: int = A
    path_depth: int = D
    legal_actions: int = MAX_LEGAL_ACTIONS


@dataclass(frozen=True)
class OneStepLabelAgreement:
    """Independent agreement on initial legal actions and successors."""

    packet_sha256: str
    action_count: int
    production_sha256: str
    independent_reference_sha256: str
    exact_agreement: bool


@dataclass(frozen=True)
class CandidateAssessment:
    """One fully admitted packet plus offline labels and receipts."""

    packet: audited.SourceDeletedPacket
    expected: audited.ExpectedTransitionRecord
    fingerprints: audited.PacketFingerprints
    oracle_agreement: audited.OracleAgreementRecord
    label_agreement: OneStepLabelAgreement
    max_occurrence_depth: int


@dataclass(frozen=True)
class SamplingReceipt:
    """Deterministic rejection-sampling receipt for one packet orbit."""

    orbit_id: str
    accepted_attempt: int
    accepted_candidate_seed: int
    rejected_attempts: int
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ProceduralPacketMetadata:
    """Offline semantics that are never embedded in a model packet."""

    packet_sha256: str
    partition: Partition
    family: Family
    template: str
    depth: int
    renderer: str
    composition: str
    role: str
    orbit_id: str
    candidate_seed: int
    renderer_seed: int
    legal_action_count: int


@dataclass(frozen=True)
class ProceduralTwinRecord:
    """Offline matched intervention or renderer-orbit receipt."""

    kind: TwinKind
    group_id: str
    left_packet_sha256: str
    right_packet_sha256: str
    left_transition_index: int | None = None
    right_transition_index: int | None = None


@dataclass(frozen=True)
class CellCount:
    """One explicit family/depth/renderer/composition pilot cell."""

    partition: Partition
    family: Family
    depth: int
    renderer: str
    composition: str
    role: str
    count: int


@dataclass(frozen=True)
class ProceduralManifestReceipt:
    """Hash-bound in-memory manifest for the generated pilot."""

    generator_version: str
    master_seed: int
    geometry: GeometryReceipt
    max_attempts: int
    packet_count: int
    train_packet_count: int
    development_packet_count: int
    rejected_candidate_count: int
    cells: tuple[CellCount, ...]
    packet_manifest_sha256: str
    metadata_manifest_sha256: str
    expected_manifest_sha256: str
    fingerprint_manifest_sha256: str
    oracle_manifest_sha256: str
    label_agreement_manifest_sha256: str
    twin_manifest_sha256: str
    sampling_manifest_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class ProceduralNeuralTcrrPilot:
    """Bounded generated corpus and physically separable offline ledgers."""

    packets: tuple[audited.SourceDeletedPacket, ...]
    expected_records: tuple[audited.ExpectedTransitionRecord, ...]
    metadata: tuple[ProceduralPacketMetadata, ...]
    fingerprints: tuple[audited.PacketFingerprints, ...]
    oracle_agreements: tuple[audited.OracleAgreementRecord, ...]
    label_agreements: tuple[OneStepLabelAgreement, ...]
    twins: tuple[ProceduralTwinRecord, ...]
    sampling_receipts: tuple[SamplingReceipt, ...]
    manifest: ProceduralManifestReceipt


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    material = _canonical_json(value).encode()
    return hashlib.sha256(material).hexdigest()


def _derive_seed(seed: int, *parts: object) -> int:
    payload = ":".join((str(seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _assert_frozen_geometry() -> None:
    if _ACTIVE_GEOMETRY != _FROZEN_GEOMETRY:
        raise ProceduralBoardError(
            "audited neural TCRR geometry changed: "
            f"expected {_FROZEN_GEOMETRY}, got {_ACTIVE_GEOMETRY}"
        )


def _oracle_payload(result: object) -> dict[str, object]:
    return {
        "normal_forms": getattr(result, "normal_forms"),
        "transitions": [
            dataclasses.asdict(item) for item in getattr(result, "transitions")
        ],
        "cyclic_sccs": getattr(result, "cyclic_sccs"),
        "cyclic_states": getattr(result, "cyclic_states"),
        "states_explored": getattr(result, "states_explored"),
        "transitions_explored": getattr(result, "transitions_explored"),
    }


def _oracle_and_label_agreement(
    packet: audited.SourceDeletedPacket,
    expected: audited.ExpectedTransitionRecord,
) -> tuple[audited.OracleAgreementRecord, OneStepLabelAgreement]:
    system, graph, _storage = audited._packet_to_mechanics(packet)  # noqa: SLF001
    production = mechanics.ProductionRewriteStateOracle(
        maximum_states=MAX_ORACLE_STATES
    ).enumerate(system, graph)
    reference = mechanics.IndependentNestedReferenceOracle(
        maximum_states=MAX_ORACLE_STATES
    ).enumerate(system, graph)
    production_payload = _oracle_payload(production)
    reference_payload = _oracle_payload(reference)
    if production_payload != reference_payload:
        raise ProceduralCandidateRejected(
            "production and independent reference state graphs disagree"
        )
    digest = audited.packet_sha256(packet)
    oracle = audited.OracleAgreementRecord(
        packet_sha256=digest,
        production_sha256=_sha256(production_payload),
        independent_reference_sha256=_sha256(reference_payload),
        state_count=production.states_explored,
        transition_count=production.transitions_explored,
        normal_form_count=len(production.normal_forms),
        cyclic_component_count=len(production.cyclic_sccs),
        exact_agreement=True,
    )
    initial = mechanics.canonical_graph_serialization(graph)
    production_edges = tuple(
        dataclasses.asdict(item)
        for item in production.transitions
        if item.source == initial
    )
    reference_edges = tuple(
        dataclasses.asdict(item)
        for item in reference.transitions
        if item.source == initial
    )
    if production_edges != reference_edges:
        raise ProceduralCandidateRejected(
            "production and independent reference one-step labels disagree"
        )
    storage_index = {
        storage_id: index for index, storage_id in enumerate(packet.graph.reservoir)
    }
    expected_edges = []
    for transition in expected.transitions:
        successor_packet = dataclasses.replace(
            packet,
            graph=transition.successor,
        )
        _successor_system, successor_graph, _successor_storage = (
            audited._packet_to_mechanics(successor_packet)  # noqa: SLF001
        )
        reduction = mechanics.Reduction(
            rule_id=transition.rule_id,
            target_slot=storage_index[transition.target_storage_id],
            target_path=transition.occurrence_path,
        )
        expected_edges.append(
            dataclasses.asdict(
                mechanics.OracleTransition(
                    source=initial,
                    reduction=reduction.trace_token,
                    target=mechanics.canonical_graph_serialization(successor_graph),
                )
            )
        )
    expected_edges_tuple = tuple(
        sorted(
            expected_edges,
            key=lambda item: (
                str(item["source"]),
                str(item["reduction"]),
                str(item["target"]),
            ),
        )
    )
    if expected_edges_tuple != production_edges:
        raise ProceduralCandidateRejected(
            "exported opaque transitions differ from independent initial edges"
        )
    label_agreement = OneStepLabelAgreement(
        packet_sha256=digest,
        action_count=len(expected.transitions),
        production_sha256=_sha256(production_edges),
        independent_reference_sha256=_sha256(reference_edges),
        exact_agreement=True,
    )
    return oracle, label_agreement


def _maximum_occurrence_depth(packet: audited.SourceDeletedPacket) -> int:
    occurrences = audited._graph_occurrences(packet.graph)  # noqa: SLF001
    return max((len(path) for path, _storage in occurrences), default=0)


def assess_source_deleted_candidate(
    packet: audited.SourceDeletedPacket,
) -> CandidateAssessment:
    """Admit one full candidate or fail closed without slicing its labels."""

    _assert_frozen_geometry()
    try:
        audited.validate_source_deleted_packet(packet)
        occurrence_depth = _maximum_occurrence_depth(packet)
        if occurrence_depth > D:
            raise ProceduralCandidateRejected(
                f"occurrence depth {occurrence_depth} exceeds frozen D={D}"
            )
        expected = audited._expected_record_from_packet(packet)  # noqa: SLF001
        action_count = len(expected.transitions)
        if action_count > MAX_LEGAL_ACTIONS:
            raise ProceduralCandidateRejected(
                f"legal action count {action_count} exceeds "
                f"the fail-closed cap {MAX_LEGAL_ACTIONS}"
            )
        if any(len(item.occurrence_path) > D for item in expected.transitions):
            raise ProceduralCandidateRejected(
                "one legal action exceeds the frozen path depth"
            )
        fingerprints = audited.packet_fingerprints(packet)
        oracle, labels = _oracle_and_label_agreement(packet, expected)
    except ProceduralCandidateRejected:
        raise
    except (audited.NeuralTcrrBoardError, mechanics.RewriteMechanicsError) as exc:
        raise ProceduralCandidateRejected(str(exc)) from exc
    return CandidateAssessment(
        packet=packet,
        expected=expected,
        fingerprints=fingerprints,
        oracle_agreement=oracle,
        label_agreement=labels,
        max_occurrence_depth=occurrence_depth,
    )


def rejection_sample_packet_group(
    candidate_factory: PacketGroupFactory,
    *,
    seed: int,
    orbit_id: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    group_validator: PacketGroupValidator | None = None,
) -> tuple[tuple[CandidateAssessment, ...], SamplingReceipt]:
    """Sample one whole matched group; reject every invalid attempt atomically."""

    if max_attempts <= 0:
        raise ProceduralBoardError("max_attempts must be positive")
    rejected: list[str] = []
    for attempt in range(max_attempts):
        candidate_seed = _derive_seed(seed, orbit_id, attempt)
        try:
            packets = tuple(candidate_factory(candidate_seed))
            if not packets:
                raise ProceduralCandidateRejected("candidate group is empty")
            assessments = tuple(
                assess_source_deleted_candidate(packet) for packet in packets
            )
            packet_digests = [item.expected.packet_sha256 for item in assessments]
            if len(packet_digests) != len(set(packet_digests)):
                raise ProceduralCandidateRejected(
                    "candidate group contains exact duplicate packets"
                )
            if group_validator is not None:
                group_validator(assessments)
        except (
            ProceduralCandidateRejected,
            audited.NeuralTcrrBoardError,
            mechanics.RewriteMechanicsError,
        ) as exc:
            rejected.append(str(exc))
            continue
        return assessments, SamplingReceipt(
            orbit_id=orbit_id,
            accepted_attempt=attempt,
            accepted_candidate_seed=candidate_seed,
            rejected_attempts=len(rejected),
            rejection_reasons=tuple(rejected),
        )
    reason = rejected[-1] if rejected else "no candidate was produced"
    raise ProceduralBoardError(
        f"orbit {orbit_id!r} exhausted {max_attempts} attempts: {reason}"
    )


def _factory(seed: int) -> audited._EpisodeFactory:  # noqa: SLF001
    return audited._EpisodeFactory(seed)  # noqa: SLF001


def _pack(
    factory: audited._EpisodeFactory,  # noqa: SLF001
    system: mechanics.RewriteSystem,
    term: mechanics.GroundTerm,
    *,
    capacity: int,
) -> audited.SourceDeletedPacket:
    generated = audited._pack_example(  # noqa: SLF001
        factory,
        system,
        term,
        capacity=capacity,
    )
    return audited._packet_from_generated(generated)  # noqa: SLF001


def _wrap(
    factory: audited._EpisodeFactory,  # noqa: SLF001
    constructor: str,
    term: mechanics.GroundTerm,
    depth: int,
) -> mechanics.GroundTerm:
    output = term
    for _ in range(depth):
        output = factory.term(constructor, output)
    return output


def _build_algebraic_normalization(
    semantic_seed: int,
    renderer_seed: int,
) -> audited.SourceDeletedPacket:
    factory = _factory(renderer_seed)
    factory.constructor("compose", "expression", ("expression", "expression"))
    factory.constructor("identity", "expression")
    factory.constructor("atom", "expression")
    factory.constructor("wrap", "expression", ("expression",))
    left = factory.variable("expression")
    right = factory.variable("expression")
    wrapped = factory.variable("expression")
    rules = (
        factory.rule(
            factory.pattern("compose", factory.pattern("identity"), left),
            mechanics.RhsVariable(left.name),
        ),
        factory.rule(
            factory.pattern("compose", right, factory.pattern("identity")),
            mechanics.RhsVariable(right.name),
        ),
        factory.rule(
            factory.pattern(
                "wrap",
                factory.pattern("wrap", wrapped),
            ),
            mechanics.RhsVariable(wrapped.name),
        ),
    )
    system = factory.system(rules)
    wrap_depth = 2 + semantic_seed % 2
    core = _wrap(factory, "wrap", factory.leaf("atom"), wrap_depth)
    term = factory.term(
        "compose",
        factory.leaf("identity"),
        factory.term(
            "compose",
            core,
            factory.leaf("identity"),
        ),
    )
    return _pack(factory, system, term, capacity=12)


def _build_boolean_simplification(
    semantic_seed: int,
    renderer_seed: int,
) -> audited.SourceDeletedPacket:
    factory = _factory(renderer_seed)
    factory.constructor("and", "boolean", ("boolean", "boolean"))
    factory.constructor("or", "boolean", ("boolean", "boolean"))
    factory.constructor("not", "boolean", ("boolean",))
    factory.constructor("true", "boolean")
    factory.constructor("false", "boolean")
    factory.constructor("atom", "boolean")
    and_value = factory.variable("boolean")
    or_value = factory.variable("boolean")
    negated = factory.variable("boolean")
    rules = (
        factory.rule(
            factory.pattern("and", factory.pattern("true"), and_value),
            mechanics.RhsVariable(and_value.name),
        ),
        factory.rule(
            factory.pattern("or", factory.pattern("false"), or_value),
            mechanics.RhsVariable(or_value.name),
        ),
        factory.rule(
            factory.pattern("not", factory.pattern("not", negated)),
            mechanics.RhsVariable(negated.name),
        ),
    )
    system = factory.system(rules)
    negation_depth = 2 + 2 * (semantic_seed % 2)
    term = factory.term(
        "and",
        factory.leaf("true"),
        factory.term(
            "or",
            factory.leaf("false"),
            _wrap(
                factory,
                "not",
                factory.leaf("atom"),
                negation_depth,
            ),
        ),
    )
    return _pack(factory, system, term, capacity=14)


def _build_list_tree_normalization(
    semantic_seed: int,
    renderer_seed: int,
) -> audited.SourceDeletedPacket:
    factory = _factory(renderer_seed)
    factory.constructor("node", "tree", ("tree", "tree"))
    factory.constructor("empty", "tree")
    factory.constructor("leaf", "tree")
    factory.constructor("mark", "tree", ("tree",))
    left = factory.variable("tree")
    right = factory.variable("tree")
    marked = factory.variable("tree")
    rules = (
        factory.rule(
            factory.pattern("node", factory.pattern("empty"), left),
            mechanics.RhsVariable(left.name),
        ),
        factory.rule(
            factory.pattern("node", right, factory.pattern("empty")),
            mechanics.RhsVariable(right.name),
        ),
        factory.rule(
            factory.pattern("mark", factory.pattern("mark", marked)),
            mechanics.RhsVariable(marked.name),
        ),
    )
    system = factory.system(rules)
    mark_depth = 2 + semantic_seed % 2
    branch = factory.term(
        "node",
        factory.leaf("leaf"),
        factory.leaf("empty"),
    )
    term = factory.term(
        "node",
        factory.leaf("empty"),
        _wrap(factory, "mark", branch, mark_depth),
    )
    return _pack(factory, system, term, capacity=13)


def _build_typed_stack(
    semantic_seed: int,
    renderer_seed: int,
) -> audited.SourceDeletedPacket:
    factory = _factory(renderer_seed)
    factory.constructor("nil", "stack")
    factory.constructor("push", "stack", ("value", "stack"))
    factory.constructor("pop", "stack", ("stack",))
    factory.constructor("dedup", "stack", ("stack",))
    factory.constructor("a", "value")
    factory.constructor("b", "value")
    popped_value = factory.variable("value")
    popped_stack = factory.variable("stack")
    duplicate_value = factory.variable("value")
    duplicate_stack = factory.variable("stack")
    rules = (
        factory.rule(
            factory.pattern(
                "pop",
                factory.pattern("push", popped_value, popped_stack),
            ),
            mechanics.RhsVariable(popped_stack.name),
        ),
        factory.rule(
            factory.pattern(
                "dedup",
                factory.pattern(
                    "push",
                    duplicate_value,
                    factory.pattern(
                        "push",
                        duplicate_value,
                        duplicate_stack,
                    ),
                ),
            ),
            factory.rhs(
                "push",
                mechanics.RhsVariable(duplicate_value.name),
                mechanics.RhsVariable(duplicate_stack.name),
            ),
        ),
    )
    system = factory.system(rules)
    duplicate = "a" if semantic_seed % 2 else "b"
    inner = factory.term(
        "dedup",
        factory.term(
            "push",
            factory.leaf(duplicate),
            factory.term(
                "push",
                factory.leaf(duplicate),
                factory.leaf("nil"),
            ),
        ),
    )
    term = factory.term(
        "pop",
        factory.term(
            "push",
            factory.leaf("a"),
            inner,
        ),
    )
    return _pack(factory, system, term, capacity=14)


def _build_dataflow(
    semantic_seed: int,
    renderer_seed: int,
) -> audited.SourceDeletedPacket:
    factory = _factory(renderer_seed)
    factory.constructor("token", "payload")
    factory.constructor("emit", "signal", ("payload",))
    factory.constructor("route", "signal", ("signal",))
    factory.constructor("fuse", "signal", ("signal", "signal"))
    factory.constructor("buffer", "signal", ("signal",))
    routed = factory.variable("signal")
    fused = factory.variable("signal")
    buffered = factory.variable("signal")
    rules = (
        factory.rule(
            factory.pattern(
                "route",
                factory.pattern("buffer", routed),
            ),
            mechanics.RhsVariable(routed.name),
        ),
        factory.rule(
            factory.pattern("fuse", fused, fused),
            mechanics.RhsVariable(fused.name),
        ),
        factory.rule(
            factory.pattern(
                "buffer",
                factory.pattern("buffer", buffered),
            ),
            factory.rhs(
                "buffer",
                mechanics.RhsVariable(buffered.name),
            ),
        ),
    )
    system = factory.system(rules)

    def branch() -> mechanics.GroundTerm:
        signal = factory.term("emit", factory.leaf("token"))
        return _wrap(
            factory,
            "buffer",
            signal,
            2 + semantic_seed % 2,
        )

    term = factory.term(
        "route",
        factory.term(
            "buffer",
            factory.term("fuse", branch(), branch()),
        ),
    )
    return _pack(factory, system, term, capacity=15)


_CORE_BUILDERS: Mapping[
    Family,
    Callable[[int, int], audited.SourceDeletedPacket],
] = {
    "algebraic_normalization": _build_algebraic_normalization,
    "boolean_simplification": _build_boolean_simplification,
    "list_tree_normalization": _build_list_tree_normalization,
    "typed_stack": _build_typed_stack,
    "dataflow": _build_dataflow,
}

_CORE_SPEC: Mapping[Family, tuple[Partition, str, str]] = {
    "algebraic_normalization": (
        "local_transition_train",
        "algebraic_identity_chain",
        "overlapping_normalization",
    ),
    "boolean_simplification": (
        "local_transition_train",
        "boolean_nested_simplification",
        "nested_multi_rule",
    ),
    "list_tree_normalization": (
        "local_transition_train",
        "list_tree_branch_normalization",
        "branch_elimination",
    ),
    "typed_stack": (
        "local_transition_development",
        "typed_stack_pop_dedup",
        "typed_stack_projection",
    ),
    "dataflow": (
        "local_transition_development",
        "dataflow_route_fuse",
        "pipeline_fusion",
    ),
}


def _build_boolean_no_redex_pair(
    candidate_seed: int,
) -> tuple[audited.SourceDeletedPacket, audited.SourceDeletedPacket]:
    renderer_seed = _derive_seed(candidate_seed, "matched_renderer")
    factory = _factory(renderer_seed)
    factory.constructor("gate", "boolean", ("boolean", "boolean"))
    factory.constructor("enabled", "boolean")
    factory.constructor("disabled", "boolean")
    factory.constructor("atom", "boolean")
    value = factory.variable("boolean")
    rule = factory.rule(
        factory.pattern("gate", factory.pattern("enabled"), value),
        mechanics.RhsVariable(value.name),
    )
    system = factory.system((rule,))
    positive = _pack(
        factory,
        system,
        factory.term(
            "gate",
            factory.leaf("enabled"),
            factory.leaf("atom"),
        ),
        capacity=7,
    )
    negative = _pack(
        factory,
        system,
        factory.term(
            "gate",
            factory.leaf("disabled"),
            factory.leaf("atom"),
        ),
        capacity=7,
    )
    return positive, negative


def _build_shared_occurrence_group(
    candidate_seed: int,
) -> tuple[audited.SourceDeletedPacket]:
    renderer_seed = _derive_seed(candidate_seed, "shared_renderer")
    return (audited._shared_cancellation_base(renderer_seed),)  # noqa: SLF001


def _build_capacity_pair(
    candidate_seed: int,
) -> tuple[audited.SourceDeletedPacket, audited.SourceDeletedPacket]:
    renderer_seed = _derive_seed(candidate_seed, "capacity_renderer")
    full = audited._capacity_base(renderer_seed)  # noqa: SLF001
    occupied = {item.storage_id for item in full.graph.nodes}
    free = sorted(set(full.graph.reservoir) - occupied)
    if len(free) != 2:
        raise ProceduralCandidateRejected(
            "capacity template did not expose exactly two free records"
        )
    reduced = audited._mutate_capacity(full, free[0])  # noqa: SLF001
    return full, reduced


def _render_group_validator(
    assessments: tuple[CandidateAssessment, ...],
) -> None:
    if len(assessments) != 2:
        raise ProceduralCandidateRejected("render orbit must contain two packets")
    left, right = assessments
    if left.expected.packet_sha256 == right.expected.packet_sha256:
        raise ProceduralCandidateRejected("render orbit did not change packet bytes")
    if left.fingerprints.isomorphic_sha256 != right.fingerprints.isomorphic_sha256:
        raise ProceduralCandidateRejected(
            "matched rerender changed canonical packet semantics"
        )


def _no_redex_group_validator(
    assessments: tuple[CandidateAssessment, ...],
) -> None:
    if len(assessments) != 2:
        raise ProceduralCandidateRejected("no-redex group must contain two packets")
    positive, negative = assessments
    if not positive.expected.transitions or negative.expected.transitions:
        raise ProceduralCandidateRejected(
            "matched no-redex intervention lacks positive/negative separation"
        )
    if positive.packet.rules != negative.packet.rules:
        raise ProceduralCandidateRejected(
            "matched no-redex intervention changed its rule bank"
        )


def _shared_group_validator(
    assessments: tuple[CandidateAssessment, ...],
) -> None:
    if len(assessments) != 1:
        raise ProceduralCandidateRejected(
            "shared-occurrence group must contain one packet"
        )
    actions = assessments[0].expected.transitions
    matches = [
        (left_index, right_index)
        for left_index, left in enumerate(actions)
        for right_index, right in enumerate(actions[left_index + 1 :], left_index + 1)
        if left.target_storage_id == right.target_storage_id
        and left.occurrence_path != right.occurrence_path
    ]
    if not matches:
        raise ProceduralCandidateRejected(
            "shared-occurrence packet lacks two paths to one storage record"
        )


def _capacity_group_validator(
    assessments: tuple[CandidateAssessment, ...],
) -> None:
    if len(assessments) != 2:
        raise ProceduralCandidateRejected("capacity group must contain two packets")
    full, reduced = assessments
    if len(full.packet.graph.reservoir) != N:
        raise ProceduralCandidateRejected("full capacity twin is not N=16")
    if len(reduced.packet.graph.reservoir) != N - 1:
        raise ProceduralCandidateRejected("reduced capacity twin is not N=15")
    if not full.expected.transitions or reduced.expected.transitions:
        raise ProceduralCandidateRejected(
            "capacity intervention lacks legal/blocked action separation"
        )
    if full.packet.rules != reduced.packet.rules:
        raise ProceduralCandidateRejected("capacity intervention changed rules")
    if full.packet.graph.nodes != reduced.packet.graph.nodes:
        raise ProceduralCandidateRejected("capacity intervention changed graph nodes")


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


def _successor_isomorphic_digest(
    packet: audited.SourceDeletedPacket,
    action: audited.ExpectedTransition,
) -> str:
    successor_packet = dataclasses.replace(packet, graph=action.successor)
    return audited.packet_fingerprints(successor_packet).isomorphic_sha256


def _manifest_receipt(
    *,
    seed: int,
    max_attempts: int,
    packets: tuple[audited.SourceDeletedPacket, ...],
    expected: tuple[audited.ExpectedTransitionRecord, ...],
    metadata: tuple[ProceduralPacketMetadata, ...],
    fingerprints: tuple[audited.PacketFingerprints, ...],
    oracles: tuple[audited.OracleAgreementRecord, ...],
    labels: tuple[OneStepLabelAgreement, ...],
    twins: tuple[ProceduralTwinRecord, ...],
    sampling: tuple[SamplingReceipt, ...],
) -> ProceduralManifestReceipt:
    cells_counter = Counter(
        (
            item.partition,
            item.family,
            item.depth,
            item.renderer,
            item.composition,
            item.role,
        )
        for item in metadata
    )
    cells = tuple(
        CellCount(
            partition=partition,
            family=family,
            depth=depth,
            renderer=renderer,
            composition=composition,
            role=role,
            count=count,
        )
        for (
            partition,
            family,
            depth,
            renderer,
            composition,
            role,
        ), count in sorted(cells_counter.items())
    )
    packet_payload = tuple(
        {
            "packet_sha256": audited.packet_sha256(packet),
            "serialized_packet": audited.serialize_model_packet(packet),
        }
        for packet in packets
    )
    base: dict[str, object] = {
        "generator_version": GENERATOR_VERSION,
        "master_seed": seed,
        "geometry": dataclasses.asdict(GeometryReceipt()),
        "max_attempts": max_attempts,
        "packet_count": len(packets),
        "train_packet_count": sum(
            item.partition == "local_transition_train" for item in metadata
        ),
        "development_packet_count": sum(
            item.partition == "local_transition_development" for item in metadata
        ),
        "rejected_candidate_count": sum(item.rejected_attempts for item in sampling),
        "cells": tuple(dataclasses.asdict(item) for item in cells),
        "packet_manifest_sha256": _sha256(packet_payload),
        "metadata_manifest_sha256": _sha256(
            tuple(dataclasses.asdict(item) for item in metadata)
        ),
        "expected_manifest_sha256": _sha256(
            tuple(dataclasses.asdict(item) for item in expected)
        ),
        "fingerprint_manifest_sha256": _sha256(
            tuple(dataclasses.asdict(item) for item in fingerprints)
        ),
        "oracle_manifest_sha256": _sha256(
            tuple(dataclasses.asdict(item) for item in oracles)
        ),
        "label_agreement_manifest_sha256": _sha256(
            tuple(dataclasses.asdict(item) for item in labels)
        ),
        "twin_manifest_sha256": _sha256(
            tuple(dataclasses.asdict(item) for item in twins)
        ),
        "sampling_manifest_sha256": _sha256(
            tuple(dataclasses.asdict(item) for item in sampling)
        ),
    }
    return ProceduralManifestReceipt(
        generator_version=GENERATOR_VERSION,
        master_seed=seed,
        geometry=GeometryReceipt(),
        max_attempts=max_attempts,
        packet_count=len(packets),
        train_packet_count=int(base["train_packet_count"]),
        development_packet_count=int(base["development_packet_count"]),
        rejected_candidate_count=int(base["rejected_candidate_count"]),
        cells=cells,
        packet_manifest_sha256=str(base["packet_manifest_sha256"]),
        metadata_manifest_sha256=str(base["metadata_manifest_sha256"]),
        expected_manifest_sha256=str(base["expected_manifest_sha256"]),
        fingerprint_manifest_sha256=str(base["fingerprint_manifest_sha256"]),
        oracle_manifest_sha256=str(base["oracle_manifest_sha256"]),
        label_agreement_manifest_sha256=str(base["label_agreement_manifest_sha256"]),
        twin_manifest_sha256=str(base["twin_manifest_sha256"]),
        sampling_manifest_sha256=str(base["sampling_manifest_sha256"]),
        payload_sha256=_sha256(base),
    )


def _core_renderer_seeds(candidate_seed: int) -> tuple[int, int, int]:
    semantic_seed = _derive_seed(candidate_seed, "semantic")
    renderer_zero = _derive_seed(candidate_seed, "renderer", 0)
    renderer_one = _derive_seed(candidate_seed, "renderer", 1)
    return semantic_seed, renderer_zero, renderer_one


def generate_neural_tcrr_pilot(
    seed: int = DEFAULT_SEED,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> ProceduralNeuralTcrrPilot:
    """Generate and validate the bounded train/development procedural pilot."""

    _assert_frozen_geometry()
    packets: list[audited.SourceDeletedPacket] = []
    expected: list[audited.ExpectedTransitionRecord] = []
    metadata: list[ProceduralPacketMetadata] = []
    fingerprints: list[audited.PacketFingerprints] = []
    oracles: list[audited.OracleAgreementRecord] = []
    labels: list[OneStepLabelAgreement] = []
    twins: list[ProceduralTwinRecord] = []
    sampling: list[SamplingReceipt] = []

    def append_assessment(
        assessment: CandidateAssessment,
        *,
        partition: Partition,
        family: Family,
        template: str,
        renderer: str,
        composition: str,
        role: str,
        orbit_id: str,
        candidate_seed: int,
        renderer_seed: int,
    ) -> None:
        packets.append(assessment.packet)
        expected.append(assessment.expected)
        fingerprints.append(assessment.fingerprints)
        oracles.append(assessment.oracle_agreement)
        labels.append(assessment.label_agreement)
        metadata.append(
            ProceduralPacketMetadata(
                packet_sha256=assessment.expected.packet_sha256,
                partition=partition,
                family=family,
                template=template,
                depth=assessment.max_occurrence_depth,
                renderer=renderer,
                composition=composition,
                role=role,
                orbit_id=orbit_id,
                candidate_seed=candidate_seed,
                renderer_seed=renderer_seed,
                legal_action_count=len(assessment.expected.transitions),
            )
        )

    for family in _CORE_FAMILIES:
        typed_family: Family = family
        partition, template, composition = _CORE_SPEC[typed_family]
        builder = _CORE_BUILDERS[typed_family]
        orbit_id = f"{partition}:{typed_family}:render_orbit"

        def core_factory(
            candidate_seed: int,
            active_builder: Callable[
                [int, int],
                audited.SourceDeletedPacket,
            ] = builder,
        ) -> tuple[audited.SourceDeletedPacket, audited.SourceDeletedPacket]:
            semantic_seed, renderer_zero, renderer_one = _core_renderer_seeds(
                candidate_seed
            )
            return (
                active_builder(semantic_seed, renderer_zero),
                active_builder(semantic_seed, renderer_one),
            )

        assessments, receipt = rejection_sample_packet_group(
            core_factory,
            seed=seed,
            orbit_id=orbit_id,
            max_attempts=max_attempts,
            group_validator=_render_group_validator,
        )
        sampling.append(receipt)
        semantic_seed, renderer_zero, renderer_one = _core_renderer_seeds(
            receipt.accepted_candidate_seed
        )
        del semantic_seed
        append_assessment(
            assessments[0],
            partition=partition,
            family=typed_family,
            template=template,
            renderer="renderer_0",
            composition=composition,
            role="base",
            orbit_id=orbit_id,
            candidate_seed=receipt.accepted_candidate_seed,
            renderer_seed=renderer_zero,
        )
        append_assessment(
            assessments[1],
            partition=partition,
            family=typed_family,
            template=template,
            renderer="renderer_1",
            composition=composition,
            role="render_twin",
            orbit_id=orbit_id,
            candidate_seed=receipt.accepted_candidate_seed,
            renderer_seed=renderer_one,
        )
        twins.append(
            ProceduralTwinRecord(
                kind="render_reindex",
                group_id=orbit_id,
                left_packet_sha256=assessments[0].expected.packet_sha256,
                right_packet_sha256=assessments[1].expected.packet_sha256,
            )
        )

    no_redex_orbit = "local_transition_train:boolean_simplification:no_redex"
    no_redex_assessments, no_redex_receipt = rejection_sample_packet_group(
        _build_boolean_no_redex_pair,
        seed=seed,
        orbit_id=no_redex_orbit,
        max_attempts=max_attempts,
        group_validator=_no_redex_group_validator,
    )
    sampling.append(no_redex_receipt)
    no_redex_renderer_seed = _derive_seed(
        no_redex_receipt.accepted_candidate_seed,
        "matched_renderer",
    )
    for assessment, role in zip(
        no_redex_assessments,
        ("positive", "no_redex"),
        strict=True,
    ):
        append_assessment(
            assessment,
            partition="local_transition_train",
            family="boolean_simplification",
            template="boolean_matched_no_redex",
            renderer="matched_renderer",
            composition="matched_no_redex",
            role=role,
            orbit_id=no_redex_orbit,
            candidate_seed=no_redex_receipt.accepted_candidate_seed,
            renderer_seed=no_redex_renderer_seed,
        )
    twins.append(
        ProceduralTwinRecord(
            kind="no_redex",
            group_id=no_redex_orbit,
            left_packet_sha256=no_redex_assessments[0].expected.packet_sha256,
            right_packet_sha256=no_redex_assessments[1].expected.packet_sha256,
        )
    )

    shared_orbit = "local_transition_train:list_tree:shared_occurrence"
    shared_assessments, shared_receipt = rejection_sample_packet_group(
        _build_shared_occurrence_group,
        seed=seed,
        orbit_id=shared_orbit,
        max_attempts=max_attempts,
        group_validator=_shared_group_validator,
    )
    sampling.append(shared_receipt)
    shared = shared_assessments[0]
    append_assessment(
        shared,
        partition="local_transition_train",
        family="list_tree_normalization",
        template="shared_alias_rewrite",
        renderer="shared_dag",
        composition="shared_occurrence",
        role="shared_occurrence",
        orbit_id=shared_orbit,
        candidate_seed=shared_receipt.accepted_candidate_seed,
        renderer_seed=_derive_seed(
            shared_receipt.accepted_candidate_seed,
            "shared_renderer",
        ),
    )
    shared_pair = next(
        (
            (left_index, right_index)
            for left_index, left in enumerate(shared.expected.transitions)
            for right_index, right in enumerate(
                shared.expected.transitions[left_index + 1 :],
                left_index + 1,
            )
            if left.target_storage_id == right.target_storage_id
            and left.occurrence_path != right.occurrence_path
        ),
        None,
    )
    if shared_pair is None:
        raise ProceduralBoardError("admitted shared occurrence lost its action pair")
    twins.append(
        ProceduralTwinRecord(
            kind="shared_occurrence",
            group_id=shared_orbit,
            left_packet_sha256=shared.expected.packet_sha256,
            right_packet_sha256=shared.expected.packet_sha256,
            left_transition_index=shared_pair[0],
            right_transition_index=shared_pair[1],
        )
    )

    capacity_orbit = "local_transition_train:algebraic:capacity"
    capacity_assessments, capacity_receipt = rejection_sample_packet_group(
        _build_capacity_pair,
        seed=seed,
        orbit_id=capacity_orbit,
        max_attempts=max_attempts,
        group_validator=_capacity_group_validator,
    )
    sampling.append(capacity_receipt)
    capacity_renderer_seed = _derive_seed(
        capacity_receipt.accepted_candidate_seed,
        "capacity_renderer",
    )
    for assessment, role in zip(
        capacity_assessments,
        ("capacity_16", "capacity_15"),
        strict=True,
    ):
        append_assessment(
            assessment,
            partition="local_transition_train",
            family="algebraic_normalization",
            template="reservoir_sensitive_expansion",
            renderer="capacity_matched",
            composition="capacity_sensitive",
            role=role,
            orbit_id=capacity_orbit,
            candidate_seed=capacity_receipt.accepted_candidate_seed,
            renderer_seed=capacity_renderer_seed,
        )
    twins.append(
        ProceduralTwinRecord(
            kind="capacity",
            group_id=capacity_orbit,
            left_packet_sha256=capacity_assessments[0].expected.packet_sha256,
            right_packet_sha256=capacity_assessments[1].expected.packet_sha256,
        )
    )

    packets_tuple = tuple(packets)
    expected_tuple = tuple(expected)
    metadata_tuple = tuple(metadata)
    fingerprints_tuple = tuple(fingerprints)
    oracles_tuple = tuple(oracles)
    labels_tuple = tuple(labels)
    twins_tuple = tuple(twins)
    sampling_tuple = tuple(sampling)
    manifest = _manifest_receipt(
        seed=seed,
        max_attempts=max_attempts,
        packets=packets_tuple,
        expected=expected_tuple,
        metadata=metadata_tuple,
        fingerprints=fingerprints_tuple,
        oracles=oracles_tuple,
        labels=labels_tuple,
        twins=twins_tuple,
        sampling=sampling_tuple,
    )
    result = ProceduralNeuralTcrrPilot(
        packets=packets_tuple,
        expected_records=expected_tuple,
        metadata=metadata_tuple,
        fingerprints=fingerprints_tuple,
        oracle_agreements=oracles_tuple,
        label_agreements=labels_tuple,
        twins=twins_tuple,
        sampling_receipts=sampling_tuple,
        manifest=manifest,
    )
    validate_neural_tcrr_pilot(result)
    return result


def _unique_ledger(
    records: Sequence[object],
    *,
    packet_digests: tuple[str, ...],
    name: str,
) -> dict[str, object]:
    keys = [str(getattr(item, "packet_sha256")) for item in records]
    if len(keys) != len(set(keys)):
        raise ProceduralBoardError(f"{name} ledger contains duplicate keys")
    if tuple(keys) != packet_digests:
        raise ProceduralBoardError(f"{name} ledger order or membership differs")
    return dict(zip(keys, records, strict=True))


def _validate_required_cells(metadata: tuple[ProceduralPacketMetadata, ...]) -> None:
    train_families = {
        item.family for item in metadata if item.partition == "local_transition_train"
    }
    development_families = {
        item.family
        for item in metadata
        if item.partition == "local_transition_development"
    }
    if not _TRAIN_FAMILIES <= train_families:
        raise ProceduralBoardError("one required train family is absent")
    if development_families != _DEVELOPMENT_FAMILIES:
        raise ProceduralBoardError(
            "development must contain only held-out typed-stack and dataflow"
        )
    if train_families & _DEVELOPMENT_FAMILIES:
        raise ProceduralBoardError("held-out development family leaked into train")
    for family in _CORE_FAMILIES:
        core = [
            item
            for item in metadata
            if item.family == family and item.role in {"base", "render_twin"}
        ]
        if {item.renderer for item in core} != {"renderer_0", "renderer_1"}:
            raise ProceduralBoardError(
                f"family {family!r} lacks its matched renderer orbit"
            )
        if len({item.depth for item in core}) != 1:
            raise ProceduralBoardError(f"family {family!r} renderer twins change depth")
        if len({item.composition for item in core}) != 1:
            raise ProceduralBoardError(
                f"family {family!r} renderer twins change composition"
            )


def _validate_twins(
    value: ProceduralNeuralTcrrPilot,
    packets: Mapping[str, audited.SourceDeletedPacket],
    expected: Mapping[str, audited.ExpectedTransitionRecord],
    fingerprints: Mapping[str, audited.PacketFingerprints],
) -> None:
    kinds = Counter(item.kind for item in value.twins)
    if kinds != Counter(
        {
            "render_reindex": 5,
            "no_redex": 1,
            "shared_occurrence": 1,
            "capacity": 1,
        }
    ):
        raise ProceduralBoardError("required twin inventory is incomplete")
    for twin in value.twins:
        if twin.left_packet_sha256 not in packets:
            raise ProceduralBoardError("twin names an absent left packet")
        if twin.right_packet_sha256 not in packets:
            raise ProceduralBoardError("twin names an absent right packet")
        left_packet = packets[twin.left_packet_sha256]
        right_packet = packets[twin.right_packet_sha256]
        left_expected = expected[twin.left_packet_sha256]
        right_expected = expected[twin.right_packet_sha256]
        if twin.kind == "render_reindex":
            if twin.left_packet_sha256 == twin.right_packet_sha256:
                raise ProceduralBoardError("render twin retained exact packet bytes")
            if (
                fingerprints[twin.left_packet_sha256].isomorphic_sha256
                != fingerprints[twin.right_packet_sha256].isomorphic_sha256
            ):
                raise ProceduralBoardError("render twin changed canonical semantics")
            if _identifier_set(left_packet) & _identifier_set(right_packet):
                raise ProceduralBoardError(
                    "render twin retained a supposedly opaque identifier"
                )
        elif twin.kind == "no_redex":
            if not left_expected.transitions or right_expected.transitions:
                raise ProceduralBoardError("no-redex twin lost exact separation")
            if left_packet.rules != right_packet.rules:
                raise ProceduralBoardError("no-redex twin changed its rule bank")
        elif twin.kind == "shared_occurrence":
            if twin.left_packet_sha256 != twin.right_packet_sha256:
                raise ProceduralBoardError(
                    "shared-occurrence selector must use one packet"
                )
            if (
                twin.left_transition_index is None
                or twin.right_transition_index is None
            ):
                raise ProceduralBoardError(
                    "shared-occurrence twin lacks action selectors"
                )
            left_action = left_expected.transitions[twin.left_transition_index]
            right_action = left_expected.transitions[twin.right_transition_index]
            if left_action.target_storage_id != right_action.target_storage_id:
                raise ProceduralBoardError(
                    "shared-occurrence actions target different storage"
                )
            if left_action.occurrence_path == right_action.occurrence_path:
                raise ProceduralBoardError(
                    "shared-occurrence actions use the same path"
                )
            if _successor_isomorphic_digest(
                left_packet,
                left_action,
            ) == _successor_isomorphic_digest(left_packet, right_action):
                raise ProceduralBoardError(
                    "shared-occurrence paths collapse to one successor"
                )
        elif twin.kind == "capacity":
            if len(left_packet.graph.reservoir) != N:
                raise ProceduralBoardError("capacity left arm is not full")
            if len(right_packet.graph.reservoir) != N - 1:
                raise ProceduralBoardError("capacity right arm is not reduced")
            if not left_expected.transitions or right_expected.transitions:
                raise ProceduralBoardError(
                    "capacity twin lost legal/blocked separation"
                )


def validate_neural_tcrr_pilot(value: ProceduralNeuralTcrrPilot) -> None:
    """Recompute every source-deleted, split, oracle, twin, and hash gate."""

    _assert_frozen_geometry()
    packet_digests = tuple(audited.packet_sha256(packet) for packet in value.packets)
    if len(packet_digests) != len(set(packet_digests)):
        raise ProceduralBoardError("pilot contains exact duplicate packets")
    packets = dict(zip(packet_digests, value.packets, strict=True))
    expected_raw = _unique_ledger(
        value.expected_records,
        packet_digests=packet_digests,
        name="expected",
    )
    metadata_raw = _unique_ledger(
        value.metadata,
        packet_digests=packet_digests,
        name="metadata",
    )
    fingerprints_raw = _unique_ledger(
        value.fingerprints,
        packet_digests=packet_digests,
        name="fingerprint",
    )
    oracles_raw = _unique_ledger(
        value.oracle_agreements,
        packet_digests=packet_digests,
        name="oracle",
    )
    labels_raw = _unique_ledger(
        value.label_agreements,
        packet_digests=packet_digests,
        name="label agreement",
    )
    expected = {
        key: item
        for key, item in expected_raw.items()
        if isinstance(item, audited.ExpectedTransitionRecord)
    }
    metadata = {
        key: item
        for key, item in metadata_raw.items()
        if isinstance(item, ProceduralPacketMetadata)
    }
    fingerprints = {
        key: item
        for key, item in fingerprints_raw.items()
        if isinstance(item, audited.PacketFingerprints)
    }
    if not (len(expected) == len(metadata) == len(fingerprints) == len(packet_digests)):
        raise ProceduralBoardError("typed ledger conversion failed")

    for digest, packet in packets.items():
        recomputed = assess_source_deleted_candidate(packet)
        if expected[digest] != recomputed.expected:
            raise ProceduralBoardError("expected transition ledger is stale")
        if fingerprints[digest] != recomputed.fingerprints:
            raise ProceduralBoardError("fingerprint ledger is stale")
        if oracles_raw[digest] != recomputed.oracle_agreement:
            raise ProceduralBoardError("oracle agreement ledger is stale")
        if labels_raw[digest] != recomputed.label_agreement:
            raise ProceduralBoardError("one-step label agreement ledger is stale")
        packet_metadata = metadata[digest]
        if packet_metadata.depth != recomputed.max_occurrence_depth:
            raise ProceduralBoardError("metadata depth differs from packet depth")
        if packet_metadata.legal_action_count != len(recomputed.expected.transitions):
            raise ProceduralBoardError(
                "metadata action count differs from complete labels"
            )
        if packet_metadata.legal_action_count > MAX_LEGAL_ACTIONS:
            raise ProceduralBoardError("manifest contains a truncated action set")

    assignments = tuple(
        audited.SplitAssignment(item.packet_sha256, item.partition)
        for item in value.metadata
    )
    audited.validate_split_isolation(assignments, value.fingerprints)
    _validate_required_cells(value.metadata)
    _validate_twins(value, packets, expected, fingerprints)

    recomputed_manifest = _manifest_receipt(
        seed=value.manifest.master_seed,
        max_attempts=value.manifest.max_attempts,
        packets=value.packets,
        expected=value.expected_records,
        metadata=value.metadata,
        fingerprints=value.fingerprints,
        oracles=value.oracle_agreements,
        labels=value.label_agreements,
        twins=value.twins,
        sampling=value.sampling_receipts,
    )
    if value.manifest != recomputed_manifest:
        raise ProceduralBoardError("procedural manifest receipt is stale")


__all__ = [
    "A",
    "C",
    "D",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_SEED",
    "GENERATOR_VERSION",
    "MAX_LEGAL_ACTIONS",
    "N",
    "P",
    "R",
    "Y",
    "CandidateAssessment",
    "CellCount",
    "GeometryReceipt",
    "OneStepLabelAgreement",
    "ProceduralBoardError",
    "ProceduralCandidateRejected",
    "ProceduralManifestReceipt",
    "ProceduralNeuralTcrrPilot",
    "ProceduralPacketMetadata",
    "ProceduralTwinRecord",
    "SamplingReceipt",
    "assess_source_deleted_candidate",
    "generate_neural_tcrr_pilot",
    "rejection_sample_packet_group",
    "validate_neural_tcrr_pilot",
]
