"""Scalable deterministic procedural corpus for neural TCRR.

This module is an offline generation boundary.  Model-visible values are only
``SourceDeletedPacket`` records.  Family names, generation seeds, renderer
identities, split assignments, fingerprints, labels, oracle receipts, and
matched-intervention receipts remain in parallel offline ledgers.

The grammar and admission semantics are independent of corpus size.  The
default 256/64 board and the preregistered 48k/4k board use the same orbit
planner, grammar builders, complete one-step labels, independent reference
oracle, split-isolation gates, and fail-closed N16/C16/Y8/R8/P12/A3/D8 plus
128-action geometry.  Larger boards take longer to materialize but do not
change the task.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import generate_neural_tcrr_board as pilot
import neural_tcrr_board as audited
import typed_critical_pair_rewrite_board as mechanics


GENERATOR_VERSION = "neural-tcrr-procedural-corpus-v1"
GRAMMAR_SEMANTICS_VERSION = "typed-rewrite-orbits-v1"
DEFAULT_SEED = 2026072303
DEFAULT_TRAIN_PACKETS = 256
DEFAULT_DEVELOPMENT_PACKETS = 64
PREREGISTERED_TRAIN_PACKETS = 48_000
PREREGISTERED_DEVELOPMENT_PACKETS = 4_000
DEFAULT_MAX_ATTEMPTS = 32
MAX_LEGAL_ACTIONS = pilot.MAX_LEGAL_ACTIONS
MAX_ORACLE_STATES = pilot.MAX_ORACLE_STATES

N = audited.MAX_CAPACITY
C = audited.MAX_CONSTRUCTORS
Y = audited.MAX_TYPES
R = audited.MAX_RULES
P = audited.MAX_RULE_SIDE_NODES
A = audited.MAX_ARITY
D = audited.MAX_PATH_DEPTH

_FROZEN_GEOMETRY = (16, 16, 8, 8, 12, 3, 8, 128)
_ACTIVE_GEOMETRY = (N, C, Y, R, P, A, D, MAX_LEGAL_ACTIONS)

Partition = Literal["local_transition_train", "local_transition_development"]
Family = Literal[
    "algebraic_normalization",
    "boolean_simplification",
    "list_tree_normalization",
    "typed_stack",
    "dataflow",
]
GrammarLane = Literal[
    "train_optimization",
    "development_optimization",
    "heldout_family",
]
Behavior = Literal[
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
]
OrbitKind = Literal[
    "renderer_reindex",
    "matched_no_redex",
    "shared_renderer_reindex",
    "matched_capacity",
]

_OPTIMIZATION_FAMILIES: tuple[Family, ...] = (
    "algebraic_normalization",
    "boolean_simplification",
    "list_tree_normalization",
)
_HELDOUT_FAMILIES: tuple[Family, ...] = ("typed_stack", "dataflow")
_ALL_FAMILIES = (*_OPTIMIZATION_FAMILIES, *_HELDOUT_FAMILIES)

_TRAIN_REQUIRED: tuple[tuple[Family, Behavior], ...] = (
    ("algebraic_normalization", "replacement"),
    ("boolean_simplification", "repeated_variable"),
    ("list_tree_normalization", "deletion"),
    ("algebraic_normalization", "expansion"),
    ("boolean_simplification", "nested_creation"),
    ("list_tree_normalization", "critical_pair"),
    ("algebraic_normalization", "cycle"),
    ("boolean_simplification", "no_redex"),
    ("list_tree_normalization", "shared_occurrence"),
    ("algebraic_normalization", "capacity_sensitive"),
    ("boolean_simplification", "ternary_replacement"),
)
_DEVELOPMENT_REQUIRED: tuple[tuple[Family, Behavior], ...] = (
    ("algebraic_normalization", "ternary_replacement"),
    ("boolean_simplification", "expansion"),
    ("list_tree_normalization", "nested_creation"),
    ("typed_stack", "repeated_variable"),
    ("dataflow", "critical_pair"),
)
_TRAIN_FILL_BEHAVIORS: tuple[Behavior, ...] = (
    "replacement",
    "deletion",
    "repeated_variable",
    "expansion",
    "nested_creation",
    "cycle",
    "critical_pair",
    "ternary_replacement",
)
_DEVELOPMENT_FILL_BEHAVIORS: tuple[Behavior, ...] = (
    "replacement",
    "repeated_variable",
    "expansion",
    "nested_creation",
    "critical_pair",
    "cycle",
    "ternary_replacement",
)

PatternExpression: TypeAlias = mechanics.Pattern
RhsExpression: TypeAlias = mechanics.RhsExpression | None
GroundExpression: TypeAlias = mechanics.GroundTerm


class ProceduralCorpusError(ValueError):
    """Raised when generation, custody, or corpus invariants fail."""


class ProceduralCorpusCandidateRejected(ProceduralCorpusError):
    """Raised when an orbit attempt is rejected atomically."""


@dataclass(frozen=True)
class GeometryReceipt:
    """Frozen packet and action geometry."""

    reservoir_slots: int = N
    constructors: int = C
    types: int = Y
    rules: int = R
    rule_side_nodes: int = P
    arity: int = A
    path_depth: int = D
    legal_actions: int = MAX_LEGAL_ACTIONS


@dataclass(frozen=True)
class CorpusScalePlan:
    """Count-only proof that one grammar policy serves pilot and full scales."""

    train_packets: int
    development_packets: int
    train_orbits: int
    development_orbits: int
    packets_per_orbit: int
    grammar_semantics_sha256: str


@dataclass(frozen=True)
class CorpusPacketMetadata:
    """Offline semantics and custody fields excluded from model packets."""

    packet_sha256: str
    partition: Partition
    family: Family
    grammar_lane: GrammarLane
    behavior: Behavior
    orbit_kind: OrbitKind
    orbit_id: str
    semantic_index: int
    semantic_seed: int
    renderer: str
    renderer_seed: int
    role: str
    max_occurrence_depth: int
    legal_action_count: int
    constructor_count: int
    type_count: int
    rule_count: int
    constructor_arities: tuple[int, ...]


@dataclass(frozen=True)
class CorpusOrbitReceipt:
    """Matched renderer or intervention relationship for one accepted orbit."""

    orbit_id: str
    kind: OrbitKind
    left_packet_sha256: str
    right_packet_sha256: str


@dataclass(frozen=True)
class CorpusSamplingReceipt:
    """Deterministic rejection history for one accepted orbit."""

    orbit_id: str
    accepted_attempt: int
    accepted_candidate_seed: int
    rejected_attempts: int
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CorpusCellReceipt:
    """Explicit minimum-count cell used to prevent scale from erasing cases."""

    partition: Partition
    family: Family
    behavior: Behavior
    count: int


@dataclass(frozen=True)
class CorpusDiversityReceipt:
    """Offline structural diversity summary."""

    unique_exact_packets: int
    unique_isomorphic_packets: int
    unique_rule_windows: int
    unique_rule_pairs: int
    unique_two_rule_compositions: int
    constructor_arities: tuple[int, ...]
    type_cardinalities: tuple[int, ...]
    maximum_legal_actions: int
    maximum_occurrence_depth: int


@dataclass(frozen=True)
class CorpusManifestReceipt:
    """Hash-bound aggregate receipt for every parallel ledger."""

    generator_version: str
    grammar_semantics_version: str
    grammar_semantics_sha256: str
    master_seed: int
    geometry: GeometryReceipt
    train_packet_count: int
    development_packet_count: int
    packet_count: int
    orbit_count: int
    rejected_candidate_count: int
    minimum_cell_count: int
    cells: tuple[CorpusCellReceipt, ...]
    diversity: CorpusDiversityReceipt
    packet_manifest_sha256: str
    metadata_manifest_sha256: str
    expected_manifest_sha256: str
    fingerprint_manifest_sha256: str
    oracle_manifest_sha256: str
    label_agreement_manifest_sha256: str
    orbit_manifest_sha256: str
    sampling_manifest_sha256: str
    split_isolation_manifest_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class ProceduralNeuralTcrrCorpus:
    """Source-deleted packets plus physically separable offline ledgers."""

    packets: tuple[audited.SourceDeletedPacket, ...]
    expected_records: tuple[audited.ExpectedTransitionRecord, ...]
    metadata: tuple[CorpusPacketMetadata, ...]
    fingerprints: tuple[audited.PacketFingerprints, ...]
    oracle_agreements: tuple[audited.OracleAgreementRecord, ...]
    label_agreements: tuple[pilot.OneStepLabelAgreement, ...]
    orbit_receipts: tuple[CorpusOrbitReceipt, ...]
    sampling_receipts: tuple[CorpusSamplingReceipt, ...]
    manifest: CorpusManifestReceipt


@dataclass(frozen=True)
class _OrbitSpec:
    partition: Partition
    family: Family
    grammar_lane: GrammarLane
    behavior: Behavior
    semantic_index: int
    orbit_id: str


@dataclass(frozen=True)
class _AcceptedOrbit:
    spec: _OrbitSpec
    assessments: tuple[pilot.CandidateAssessment, pilot.CandidateAssessment]
    kind: OrbitKind
    semantic_seed: int
    renderer_names: tuple[str, str]
    renderer_seeds: tuple[int, int]
    sampling: CorpusSamplingReceipt


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _derive_seed(seed: int, *parts: object) -> int:
    payload = ":".join((str(seed), *(str(item) for item in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _grammar_semantics_payload() -> dict[str, object]:
    return {
        "version": GRAMMAR_SEMANTICS_VERSION,
        "geometry": _FROZEN_GEOMETRY,
        "train_required": _TRAIN_REQUIRED,
        "development_required": _DEVELOPMENT_REQUIRED,
        "train_fill": _TRAIN_FILL_BEHAVIORS,
        "development_fill": _DEVELOPMENT_FILL_BEHAVIORS,
        "optimization_families": _OPTIMIZATION_FAMILIES,
        "heldout_families": _HELDOUT_FAMILIES,
        "orbit_size": 2,
    }


def _assert_frozen_geometry() -> None:
    if _ACTIVE_GEOMETRY != _FROZEN_GEOMETRY:
        raise ProceduralCorpusError(
            "audited neural TCRR geometry changed: "
            f"expected {_FROZEN_GEOMETRY}, got {_ACTIVE_GEOMETRY}"
        )


def plan_neural_tcrr_corpus(
    *,
    train_packets: int = DEFAULT_TRAIN_PACKETS,
    development_packets: int = DEFAULT_DEVELOPMENT_PACKETS,
) -> CorpusScalePlan:
    """Validate counts and return the exact orbit scale without sampling."""

    _assert_frozen_geometry()
    if train_packets % 2 or development_packets % 2:
        raise ProceduralCorpusError("packet counts must be even atomic orbit sizes")
    if train_packets < 2 * len(_TRAIN_REQUIRED):
        raise ProceduralCorpusError(
            f"train requires at least {2 * len(_TRAIN_REQUIRED)} packets"
        )
    if development_packets < 2 * len(_DEVELOPMENT_REQUIRED):
        raise ProceduralCorpusError(
            f"development requires at least {2 * len(_DEVELOPMENT_REQUIRED)} packets"
        )
    return CorpusScalePlan(
        train_packets=train_packets,
        development_packets=development_packets,
        train_orbits=train_packets // 2,
        development_orbits=development_packets // 2,
        packets_per_orbit=2,
        grammar_semantics_sha256=_sha256(_grammar_semantics_payload()),
    )


def _lane(partition: Partition, family: Family) -> GrammarLane:
    if partition == "local_transition_train":
        return "train_optimization"
    if family in _HELDOUT_FAMILIES:
        return "heldout_family"
    return "development_optimization"


def _build_orbit_specs(
    *,
    train_packets: int,
    development_packets: int,
) -> tuple[_OrbitSpec, ...]:
    scale = plan_neural_tcrr_corpus(
        train_packets=train_packets,
        development_packets=development_packets,
    )
    output: list[_OrbitSpec] = []

    def append_partition(
        partition: Partition,
        orbit_count: int,
        required: tuple[tuple[Family, Behavior], ...],
    ) -> None:
        pairs = list(required)
        fill_index = 0
        while len(pairs) < orbit_count:
            if partition == "local_transition_train":
                family = _OPTIMIZATION_FAMILIES[
                    fill_index % len(_OPTIMIZATION_FAMILIES)
                ]
                behavior = _TRAIN_FILL_BEHAVIORS[
                    (fill_index // len(_OPTIMIZATION_FAMILIES))
                    % len(_TRAIN_FILL_BEHAVIORS)
                ]
            else:
                family = _ALL_FAMILIES[fill_index % len(_ALL_FAMILIES)]
                behavior = _DEVELOPMENT_FILL_BEHAVIORS[
                    (fill_index // len(_ALL_FAMILIES))
                    % len(_DEVELOPMENT_FILL_BEHAVIORS)
                ]
            pairs.append((family, behavior))
            fill_index += 1
        for local_index, (family, behavior) in enumerate(pairs):
            output.append(
                _OrbitSpec(
                    partition=partition,
                    family=family,
                    grammar_lane=_lane(partition, family),
                    behavior=behavior,
                    semantic_index=local_index,
                    orbit_id=(f"{partition}:{local_index:08d}:{family}:{behavior}"),
                )
            )

    append_partition(
        "local_transition_train",
        scale.train_orbits,
        _TRAIN_REQUIRED,
    )
    append_partition(
        "local_transition_development",
        scale.development_orbits,
        _DEVELOPMENT_REQUIRED,
    )
    return tuple(output)


def _factory(seed: int) -> audited._EpisodeFactory:  # noqa: SLF001
    return audited._EpisodeFactory(seed)  # noqa: SLF001


def _pack(
    factory: audited._EpisodeFactory,  # noqa: SLF001
    system: mechanics.RewriteSystem,
    term: GroundExpression,
    *,
    capacity: int = N,
) -> audited.SourceDeletedPacket:
    generated = audited._pack_example(  # noqa: SLF001
        factory,
        system,
        term,
        capacity=capacity,
    )
    return audited._packet_from_generated(generated)  # noqa: SLF001


def _pattern_chain(
    factory: audited._EpisodeFactory,  # noqa: SLF001
    aliases: Sequence[str],
    child: PatternExpression,
) -> PatternExpression:
    output = child
    for alias in reversed(aliases):
        output = factory.pattern(alias, output)
    return output


def _ground_chain(
    factory: audited._EpisodeFactory,  # noqa: SLF001
    aliases: Sequence[str],
    child: GroundExpression,
) -> GroundExpression:
    output = child
    for alias in reversed(aliases):
        output = factory.term(alias, output)
    return output


def _signature_code(
    semantic_seed: int,
    semantic_index: int,
    *,
    development: bool,
) -> tuple[str, ...]:
    width = 2 if development else 1 + ((semantic_seed ^ semantic_index) % 3)
    value = _derive_seed(semantic_seed, "signature", semantic_index)
    aliases = []
    for _ in range(width):
        aliases.append(("sig0", "sig1", "sig2")[value % 3])
        value //= 3
    return tuple(aliases)


def _single_sort_packet(
    *,
    family: Family,
    behavior: Behavior,
    grammar_lane: GrammarLane,
    semantic_seed: int,
    semantic_index: int,
    renderer_seed: int,
) -> audited.SourceDeletedPacket:
    factory = _factory(renderer_seed)
    factory.constructor("unit", "value")
    factory.constructor("atom_a", "value")
    factory.constructor("atom_b", "value")
    factory.constructor("u0", "value", ("value",))
    factory.constructor("u1", "value", ("value",))
    factory.constructor("sig0", "value", ("value",))
    factory.constructor("sig1", "value", ("value",))
    factory.constructor("sig2", "value", ("value",))
    factory.constructor("pair", "value", ("value", "value"))
    factory.constructor("triple", "value", ("value", "value", "value"))
    factory.constructor("pad", "value", ("value",))
    factory.constructor("tag0", "marker")
    factory.constructor("tag1", "marker")
    factory.constructor(
        "guard",
        "value",
        ("marker", "marker", "value"),
    )
    if family == "boolean_simplification":
        factory.constructor("family_tag", "truth_marker")
        factory.constructor(
            "family_context",
            "value",
            ("truth_marker", "value"),
        )
    elif family == "list_tree_normalization":
        factory.constructor("family_item", "element")
        factory.constructor(
            "family_context",
            "value",
            ("element", "element", "value"),
        )
    elif family != "algebraic_normalization":
        raise ProceduralCorpusCandidateRejected(
            f"unsupported optimization family {family!r}"
        )
    development = grammar_lane != "train_optimization"

    def family_pattern(value: PatternExpression) -> PatternExpression:
        if family == "boolean_simplification":
            return factory.pattern(
                "family_context",
                factory.pattern("family_tag"),
                value,
            )
        if family == "list_tree_normalization":
            return factory.pattern(
                "family_context",
                factory.pattern("family_item"),
                factory.pattern("family_item"),
                value,
            )
        return value

    def family_rhs(value: RhsExpression) -> RhsExpression:
        if value is None:
            return None
        if family == "boolean_simplification":
            return factory.rhs(
                "family_context",
                factory.rhs("family_tag"),
                value,
            )
        if family == "list_tree_normalization":
            return factory.rhs(
                "family_context",
                factory.rhs("family_item"),
                factory.rhs("family_item"),
                value,
            )
        return value

    def family_ground(value: GroundExpression) -> GroundExpression:
        if family == "boolean_simplification":
            return factory.term(
                "family_context",
                factory.leaf("family_tag"),
                value,
            )
        if family == "list_tree_normalization":
            return factory.term(
                "family_context",
                factory.leaf("family_item"),
                factory.leaf("family_item"),
                value,
            )
        return value

    def lane_pattern(value: PatternExpression) -> PatternExpression:
        if not development:
            return value
        return factory.pattern(
            "guard",
            factory.pattern("tag0"),
            factory.pattern("tag1"),
            value,
        )

    def lane_rhs(value: RhsExpression) -> RhsExpression:
        if value is None or not development:
            return value
        return factory.rhs(
            "guard",
            factory.rhs("tag0"),
            factory.rhs("tag1"),
            value,
        )

    def guard_pattern(value: PatternExpression) -> PatternExpression:
        return lane_pattern(family_pattern(value))

    def guard_rhs(value: RhsExpression) -> RhsExpression:
        return lane_rhs(family_rhs(value))

    x = factory.variable("value")
    y = factory.variable("value")
    rules: list[mechanics.RewriteRule] = []
    if behavior == "replacement":
        rules.append(
            factory.rule(
                guard_pattern(factory.pattern("pair", factory.pattern("unit"), x)),
                guard_rhs(mechanics.RhsVariable(x.name)),
            )
        )
        core = factory.term(
            "pair",
            factory.leaf("unit"),
            factory.leaf("atom_a"),
        )
    elif behavior == "deletion":
        rules.append(factory.rule(guard_pattern(factory.pattern("u0", x)), None))
        core = factory.term("u0", factory.leaf("atom_a"))
    elif behavior == "repeated_variable":
        rules.append(
            factory.rule(
                guard_pattern(factory.pattern("pair", x, x)),
                guard_rhs(mechanics.RhsVariable(x.name)),
            )
        )
        core = factory.term(
            "pair",
            factory.leaf("atom_a"),
            factory.leaf("atom_a"),
        )
    elif behavior == "expansion":
        rules.extend(
            (
                factory.rule(
                    guard_pattern(factory.pattern("u0", x)),
                    guard_rhs(
                        factory.rhs(
                            "pair",
                            mechanics.RhsVariable(x.name),
                            mechanics.RhsVariable(x.name),
                        )
                    ),
                ),
                factory.rule(
                    guard_pattern(factory.pattern("pair", y, y)),
                    guard_rhs(factory.rhs("u1", mechanics.RhsVariable(y.name))),
                ),
            )
        )
        core = factory.term("u0", factory.leaf("atom_a"))
    elif behavior == "nested_creation":
        rules.extend(
            (
                factory.rule(
                    guard_pattern(factory.pattern("u0", x)),
                    guard_rhs(
                        factory.rhs(
                            "u1",
                            factory.rhs(
                                "u1",
                                mechanics.RhsVariable(x.name),
                            ),
                        )
                    ),
                ),
                factory.rule(
                    guard_pattern(
                        factory.pattern(
                            "u1",
                            factory.pattern("u1", y),
                        )
                    ),
                    guard_rhs(mechanics.RhsVariable(y.name)),
                ),
            )
        )
        core = factory.term("u0", factory.leaf("atom_a"))
    elif behavior == "cycle":
        rules.extend(
            (
                factory.rule(
                    guard_pattern(factory.pattern("u0", x)),
                    guard_rhs(factory.rhs("u1", mechanics.RhsVariable(x.name))),
                ),
                factory.rule(
                    guard_pattern(factory.pattern("u1", y)),
                    guard_rhs(factory.rhs("u0", mechanics.RhsVariable(y.name))),
                ),
            )
        )
        core = factory.term("u0", factory.leaf("atom_a"))
    elif behavior == "critical_pair":
        rules.extend(
            (
                factory.rule(
                    guard_pattern(factory.pattern("pair", factory.pattern("unit"), x)),
                    guard_rhs(mechanics.RhsVariable(x.name)),
                ),
                factory.rule(
                    guard_pattern(factory.pattern("pair", y, factory.pattern("unit"))),
                    guard_rhs(mechanics.RhsVariable(y.name)),
                ),
            )
        )
        core = factory.term(
            "pair",
            factory.leaf("unit"),
            factory.leaf("unit"),
        )
    elif behavior == "ternary_replacement":
        rules.append(
            factory.rule(
                guard_pattern(
                    factory.pattern(
                        "triple",
                        factory.pattern("unit"),
                        x,
                        y,
                    )
                ),
                guard_rhs(
                    factory.rhs(
                        "pair",
                        mechanics.RhsVariable(x.name),
                        mechanics.RhsVariable(y.name),
                    )
                ),
            )
        )
        core = factory.term(
            "triple",
            factory.leaf("unit"),
            factory.leaf("atom_a"),
            factory.leaf("atom_b"),
        )
    else:
        raise ProceduralCorpusCandidateRejected(
            f"unsupported generic behavior {behavior!r}"
        )

    if (
        not development
        and semantic_index >= len(_TRAIN_REQUIRED)
        and behavior != "deletion"
    ):
        branch_selector = (semantic_index - len(_TRAIN_REQUIRED)) % 3
        if branch_selector == 0:
            core = factory.term("triple", core, core, core)
        elif branch_selector == 1:
            core = factory.term("pair", core, core)

    code = _signature_code(
        semantic_seed,
        semantic_index,
        development=development,
    )
    signature_variable = factory.variable("value")
    rules.append(
        factory.rule(
            lane_pattern(_pattern_chain(factory, code, signature_variable)),
            lane_rhs(mechanics.RhsVariable(signature_variable.name)),
        )
    )
    system = factory.system(tuple(rules))
    term = _ground_chain(factory, code, family_ground(core))
    if development:
        term = factory.term(
            "guard",
            factory.leaf("tag0"),
            factory.leaf("tag1"),
            term,
        )
        term = _ground_chain(factory, ("pad", "pad", "pad"), term)
    elif behavior != "deletion" and semantic_seed % 2:
        term = factory.term("pad", term)
    return _pack(factory, system, term)


def _typed_stack_packet(
    *,
    behavior: Behavior,
    semantic_seed: int,
    semantic_index: int,
    renderer_seed: int,
) -> audited.SourceDeletedPacket:
    factory = _factory(renderer_seed)
    factory.constructor("nil", "stack")
    factory.constructor("item_a", "item")
    factory.constructor("item_b", "item")
    factory.constructor("push", "stack", ("item", "stack"))
    factory.constructor("pop", "stack", ("stack",))
    factory.constructor("dedup", "stack", ("stack",))
    factory.constructor("merge", "stack", ("stack", "stack"))
    factory.constructor("sig0", "stack", ("stack",))
    factory.constructor("sig1", "stack", ("stack",))
    factory.constructor("sig2", "stack", ("stack",))
    factory.constructor("pad", "stack", ("stack",))
    factory.constructor("tag0", "marker")
    factory.constructor("tag1", "marker")
    factory.constructor("guard", "stack", ("marker", "marker", "stack"))

    def guard_pattern(value: PatternExpression) -> PatternExpression:
        return factory.pattern(
            "guard",
            factory.pattern("tag0"),
            factory.pattern("tag1"),
            value,
        )

    def guard_rhs(value: mechanics.RhsExpression) -> mechanics.RhsExpression:
        return factory.rhs(
            "guard",
            factory.rhs("tag0"),
            factory.rhs("tag1"),
            value,
        )

    item = factory.variable("item")
    stack = factory.variable("stack")
    other = factory.variable("stack")
    rules: list[mechanics.RewriteRule] = []
    if behavior == "replacement":
        rules.append(
            factory.rule(
                guard_pattern(
                    factory.pattern(
                        "pop",
                        factory.pattern("push", item, stack),
                    )
                ),
                guard_rhs(mechanics.RhsVariable(stack.name)),
            )
        )
        core = factory.term(
            "pop",
            factory.term(
                "push",
                factory.leaf("item_a"),
                factory.leaf("nil"),
            ),
        )
    elif behavior == "repeated_variable":
        rules.append(
            factory.rule(
                guard_pattern(
                    factory.pattern(
                        "dedup",
                        factory.pattern(
                            "push",
                            item,
                            factory.pattern("push", item, stack),
                        ),
                    )
                ),
                guard_rhs(
                    factory.rhs(
                        "push",
                        mechanics.RhsVariable(item.name),
                        mechanics.RhsVariable(stack.name),
                    )
                ),
            )
        )
        core = factory.term(
            "dedup",
            factory.term(
                "push",
                factory.leaf("item_a"),
                factory.term(
                    "push",
                    factory.leaf("item_a"),
                    factory.leaf("nil"),
                ),
            ),
        )
    elif behavior == "expansion":
        rules.append(
            factory.rule(
                guard_pattern(factory.pattern("dedup", stack)),
                guard_rhs(
                    factory.rhs(
                        "push",
                        factory.rhs("item_a"),
                        mechanics.RhsVariable(stack.name),
                    )
                ),
            )
        )
        core = factory.term("dedup", factory.leaf("nil"))
    elif behavior == "nested_creation":
        rules.extend(
            (
                factory.rule(
                    guard_pattern(factory.pattern("pop", stack)),
                    guard_rhs(
                        factory.rhs(
                            "dedup",
                            factory.rhs(
                                "dedup",
                                mechanics.RhsVariable(stack.name),
                            ),
                        )
                    ),
                ),
                factory.rule(
                    guard_pattern(
                        factory.pattern(
                            "dedup",
                            factory.pattern("dedup", other),
                        )
                    ),
                    guard_rhs(mechanics.RhsVariable(other.name)),
                ),
            )
        )
        core = factory.term("pop", factory.leaf("nil"))
    elif behavior == "critical_pair":
        rules.extend(
            (
                factory.rule(
                    guard_pattern(
                        factory.pattern(
                            "merge",
                            factory.pattern("nil"),
                            stack,
                        )
                    ),
                    guard_rhs(mechanics.RhsVariable(stack.name)),
                ),
                factory.rule(
                    guard_pattern(
                        factory.pattern(
                            "merge",
                            other,
                            factory.pattern("nil"),
                        )
                    ),
                    guard_rhs(mechanics.RhsVariable(other.name)),
                ),
            )
        )
        core = factory.term(
            "merge",
            factory.leaf("nil"),
            factory.leaf("nil"),
        )
    elif behavior == "cycle":
        rules.extend(
            (
                factory.rule(
                    guard_pattern(factory.pattern("pop", stack)),
                    guard_rhs(
                        factory.rhs(
                            "dedup",
                            mechanics.RhsVariable(stack.name),
                        )
                    ),
                ),
                factory.rule(
                    guard_pattern(factory.pattern("dedup", other)),
                    guard_rhs(
                        factory.rhs(
                            "pop",
                            mechanics.RhsVariable(other.name),
                        )
                    ),
                ),
            )
        )
        core = factory.term("pop", factory.leaf("nil"))
    elif behavior == "ternary_replacement":
        factory.constructor(
            "select",
            "stack",
            ("stack", "stack", "stack"),
        )
        rules.append(
            factory.rule(
                guard_pattern(
                    factory.pattern(
                        "select",
                        factory.pattern("nil"),
                        stack,
                        other,
                    )
                ),
                guard_rhs(mechanics.RhsVariable(stack.name)),
            )
        )
        core = factory.term(
            "select",
            factory.leaf("nil"),
            factory.leaf("nil"),
            factory.leaf("nil"),
        )
    elif behavior == "deletion":
        rules.append(
            factory.rule(
                guard_pattern(factory.pattern("pop", stack)),
                None,
            )
        )
        core = factory.term("pop", factory.leaf("nil"))
    else:
        raise ProceduralCorpusCandidateRejected(
            f"unsupported typed-stack behavior {behavior!r}"
        )

    code = _signature_code(
        semantic_seed,
        semantic_index,
        development=True,
    )
    signature = factory.variable("stack")
    rules.append(
        factory.rule(
            guard_pattern(_pattern_chain(factory, code, signature)),
            guard_rhs(mechanics.RhsVariable(signature.name)),
        )
    )
    system = factory.system(tuple(rules))
    term = _ground_chain(factory, code, core)
    term = factory.term(
        "guard",
        factory.leaf("tag0"),
        factory.leaf("tag1"),
        term,
    )
    term = _ground_chain(factory, ("pad", "pad"), term)
    return _pack(factory, system, term)


def _dataflow_packet(
    *,
    behavior: Behavior,
    semantic_seed: int,
    semantic_index: int,
    renderer_seed: int,
) -> audited.SourceDeletedPacket:
    factory = _factory(renderer_seed)
    factory.constructor("token_a", "payload")
    factory.constructor("token_b", "payload")
    factory.constructor("emit", "signal", ("payload",))
    factory.constructor("buffer", "signal", ("signal",))
    factory.constructor("route", "signal", ("signal",))
    factory.constructor("fan", "signal", ("signal",))
    factory.constructor("fuse", "signal", ("signal", "signal"))
    factory.constructor("select", "signal", ("signal", "signal", "signal"))
    factory.constructor("sig0", "signal", ("signal",))
    factory.constructor("sig1", "signal", ("signal",))
    factory.constructor("sig2", "signal", ("signal",))
    factory.constructor("pad", "signal", ("signal",))
    factory.constructor("tag0", "marker")
    factory.constructor("tag1", "marker")
    factory.constructor("guard", "signal", ("marker", "marker", "signal"))

    def guard_pattern(value: PatternExpression) -> PatternExpression:
        return factory.pattern(
            "guard",
            factory.pattern("tag0"),
            factory.pattern("tag1"),
            value,
        )

    def guard_rhs(value: mechanics.RhsExpression) -> mechanics.RhsExpression:
        return factory.rhs(
            "guard",
            factory.rhs("tag0"),
            factory.rhs("tag1"),
            value,
        )

    signal = factory.variable("signal")
    other = factory.variable("signal")
    rules: list[mechanics.RewriteRule] = []
    base = factory.term("emit", factory.leaf("token_a"))
    if behavior == "replacement":
        rules.append(
            factory.rule(
                guard_pattern(
                    factory.pattern(
                        "route",
                        factory.pattern("buffer", signal),
                    )
                ),
                guard_rhs(mechanics.RhsVariable(signal.name)),
            )
        )
        core = factory.term("route", factory.term("buffer", base))
    elif behavior == "repeated_variable":
        rules.append(
            factory.rule(
                guard_pattern(factory.pattern("fuse", signal, signal)),
                guard_rhs(mechanics.RhsVariable(signal.name)),
            )
        )
        core = factory.term("fuse", base, base)
    elif behavior == "expansion":
        rules.extend(
            (
                factory.rule(
                    guard_pattern(factory.pattern("fan", signal)),
                    guard_rhs(
                        factory.rhs(
                            "fuse",
                            mechanics.RhsVariable(signal.name),
                            mechanics.RhsVariable(signal.name),
                        )
                    ),
                ),
                factory.rule(
                    guard_pattern(factory.pattern("fuse", other, other)),
                    guard_rhs(
                        factory.rhs(
                            "buffer",
                            mechanics.RhsVariable(other.name),
                        )
                    ),
                ),
            )
        )
        core = factory.term("fan", base)
    elif behavior == "nested_creation":
        rules.extend(
            (
                factory.rule(
                    guard_pattern(factory.pattern("route", signal)),
                    guard_rhs(
                        factory.rhs(
                            "buffer",
                            factory.rhs(
                                "buffer",
                                mechanics.RhsVariable(signal.name),
                            ),
                        )
                    ),
                ),
                factory.rule(
                    guard_pattern(
                        factory.pattern(
                            "buffer",
                            factory.pattern("buffer", other),
                        )
                    ),
                    guard_rhs(mechanics.RhsVariable(other.name)),
                ),
            )
        )
        core = factory.term("route", base)
    elif behavior == "critical_pair":
        left_signal = factory.variable("signal")
        left_other = factory.variable("signal")
        right_signal = factory.variable("signal")
        right_other = factory.variable("signal")
        rules.extend(
            (
                factory.rule(
                    guard_pattern(
                        factory.pattern(
                            "fuse",
                            factory.pattern("buffer", left_signal),
                            left_other,
                        )
                    ),
                    guard_rhs(mechanics.RhsVariable(left_other.name)),
                ),
                factory.rule(
                    guard_pattern(
                        factory.pattern(
                            "fuse",
                            right_signal,
                            factory.pattern("buffer", right_other),
                        )
                    ),
                    guard_rhs(mechanics.RhsVariable(right_signal.name)),
                ),
            )
        )
        core = factory.term(
            "fuse",
            factory.term("buffer", base),
            factory.term("buffer", base),
        )
    elif behavior == "cycle":
        rules.extend(
            (
                factory.rule(
                    guard_pattern(factory.pattern("route", signal)),
                    guard_rhs(
                        factory.rhs(
                            "buffer",
                            mechanics.RhsVariable(signal.name),
                        )
                    ),
                ),
                factory.rule(
                    guard_pattern(factory.pattern("buffer", other)),
                    guard_rhs(
                        factory.rhs(
                            "route",
                            mechanics.RhsVariable(other.name),
                        )
                    ),
                ),
            )
        )
        core = factory.term("route", base)
    elif behavior == "ternary_replacement":
        rules.append(
            factory.rule(
                guard_pattern(
                    factory.pattern(
                        "select",
                        factory.pattern(
                            "emit",
                            factory.pattern("token_a"),
                        ),
                        signal,
                        other,
                    )
                ),
                guard_rhs(mechanics.RhsVariable(signal.name)),
            )
        )
        core = factory.term("select", base, base, base)
    elif behavior == "deletion":
        rules.append(
            factory.rule(
                guard_pattern(factory.pattern("buffer", signal)),
                None,
            )
        )
        core = factory.term("buffer", base)
    else:
        raise ProceduralCorpusCandidateRejected(
            f"unsupported dataflow behavior {behavior!r}"
        )

    code = _signature_code(
        semantic_seed,
        semantic_index,
        development=True,
    )
    signature = factory.variable("signal")
    rules.append(
        factory.rule(
            guard_pattern(_pattern_chain(factory, code, signature)),
            guard_rhs(mechanics.RhsVariable(signature.name)),
        )
    )
    system = factory.system(tuple(rules))
    term = _ground_chain(factory, code, core)
    term = factory.term(
        "guard",
        factory.leaf("tag0"),
        factory.leaf("tag1"),
        term,
    )
    term = _ground_chain(factory, ("pad", "pad"), term)
    return _pack(factory, system, term)


def _build_generic_packet(
    spec: _OrbitSpec,
    *,
    semantic_seed: int,
    renderer_seed: int,
) -> audited.SourceDeletedPacket:
    if spec.family in _OPTIMIZATION_FAMILIES:
        return _single_sort_packet(
            family=spec.family,
            behavior=spec.behavior,
            grammar_lane=spec.grammar_lane,
            semantic_seed=semantic_seed,
            semantic_index=spec.semantic_index,
            renderer_seed=renderer_seed,
        )
    if spec.family == "typed_stack":
        return _typed_stack_packet(
            behavior=spec.behavior,
            semantic_seed=semantic_seed,
            semantic_index=spec.semantic_index,
            renderer_seed=renderer_seed,
        )
    if spec.family == "dataflow":
        return _dataflow_packet(
            behavior=spec.behavior,
            semantic_seed=semantic_seed,
            semantic_index=spec.semantic_index,
            renderer_seed=renderer_seed,
        )
    raise ProceduralCorpusCandidateRejected(f"unknown family {spec.family!r}")


def _identifier_set(packet: audited.SourceDeletedPacket) -> set[str]:
    identifiers = set(packet.graph.reservoir)
    for constructor in packet.constructors:
        identifiers.add(constructor.identifier)
        identifiers.add(constructor.result_type)
        identifiers.update(constructor.argument_types)

    def collect(term: audited.RuleTermRecord | None) -> None:
        if term is None:
            return
        identifiers.add(term.type_id)
        if term.constructor_id is not None:
            identifiers.add(term.constructor_id)
        if term.variable_id is not None:
            identifiers.add(term.variable_id)
        for child in term.children:
            collect(child)

    for rule in packet.rules:
        identifiers.add(rule.identifier)
        collect(rule.lhs)
        collect(rule.rhs)
    for node in packet.graph.nodes:
        identifiers.add(node.storage_id)
        identifiers.add(node.type_id)
        if node.constructor_id is not None:
            identifiers.add(node.constructor_id)
        if node.variable_id is not None:
            identifiers.add(node.variable_id)
    return identifiers


def _fingerprint_categories(
    fingerprints: Sequence[audited.PacketFingerprints],
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for item in fingerprints:
        output["exact"].add(item.exact_sha256)
        output["isomorphic"].add(item.isomorphic_sha256)
        output["rule_window"].update(item.normalized_rule_windows)
        output["rule_pair"].update(item.normalized_rule_pairs)
        output["composition"].update(item.reachable_two_rule_compositions)
    return dict(output)


def _cross_split_overlap(
    train_categories: Mapping[str, set[str]],
    assessments: Sequence[pilot.CandidateAssessment],
) -> tuple[str, ...]:
    candidate = _fingerprint_categories(
        tuple(item.fingerprints for item in assessments)
    )
    overlaps = []
    for category in (
        "exact",
        "isomorphic",
        "rule_window",
        "rule_pair",
        "composition",
    ):
        if train_categories.get(category, set()) & candidate.get(category, set()):
            overlaps.append(category)
    return tuple(overlaps)


def _validate_renderer_pair(
    assessments: tuple[pilot.CandidateAssessment, ...],
) -> None:
    if len(assessments) != 2:
        raise ProceduralCorpusCandidateRejected(
            "renderer orbit must contain two packets"
        )
    left, right = assessments
    if left.expected.packet_sha256 == right.expected.packet_sha256:
        raise ProceduralCorpusCandidateRejected(
            "renderer orbit retained exact packet bytes"
        )
    if left.fingerprints.isomorphic_sha256 != right.fingerprints.isomorphic_sha256:
        raise ProceduralCorpusCandidateRejected(
            "renderer orbit changed canonical packet semantics"
        )
    if _identifier_set(left.packet) & _identifier_set(right.packet):
        raise ProceduralCorpusCandidateRejected(
            "renderer orbit retained an opaque identifier"
        )


def _validate_no_redex_pair(
    assessments: tuple[pilot.CandidateAssessment, ...],
) -> None:
    if len(assessments) != 2:
        raise ProceduralCorpusCandidateRejected(
            "no-redex intervention must contain two packets"
        )
    positive, negative = assessments
    if not positive.expected.transitions or negative.expected.transitions:
        raise ProceduralCorpusCandidateRejected(
            "no-redex intervention lacks positive/negative separation"
        )
    if positive.packet.rules != negative.packet.rules:
        raise ProceduralCorpusCandidateRejected(
            "no-redex intervention changed its rule bank"
        )


def _validate_capacity_pair(
    assessments: tuple[pilot.CandidateAssessment, ...],
) -> None:
    if len(assessments) != 2:
        raise ProceduralCorpusCandidateRejected(
            "capacity intervention must contain two packets"
        )
    full, reduced = assessments
    if len(full.packet.graph.reservoir) != N:
        raise ProceduralCorpusCandidateRejected("capacity left arm is not N=16")
    if len(reduced.packet.graph.reservoir) != N - 1:
        raise ProceduralCorpusCandidateRejected("capacity right arm is not N=15")
    if not full.expected.transitions or reduced.expected.transitions:
        raise ProceduralCorpusCandidateRejected(
            "capacity intervention lacks legal/blocked separation"
        )
    if full.packet.rules != reduced.packet.rules:
        raise ProceduralCorpusCandidateRejected("capacity intervention changed rules")
    if full.packet.graph.nodes != reduced.packet.graph.nodes:
        raise ProceduralCorpusCandidateRejected(
            "capacity intervention changed graph nodes"
        )


def _validate_shared_pair(
    assessments: tuple[pilot.CandidateAssessment, ...],
) -> None:
    _validate_renderer_pair(assessments)
    for assessment in assessments:
        actions = assessment.expected.transitions
        if not any(
            left.target_storage_id == right.target_storage_id
            and left.occurrence_path != right.occurrence_path
            for left_index, left in enumerate(actions)
            for right in actions[left_index + 1 :]
        ):
            raise ProceduralCorpusCandidateRejected(
                "shared packet lacks distinct paths to one storage record"
            )


def _orbit_candidates(
    spec: _OrbitSpec,
    *,
    candidate_seed: int,
) -> tuple[
    tuple[audited.SourceDeletedPacket, audited.SourceDeletedPacket],
    OrbitKind,
    int,
    tuple[str, str],
    tuple[int, int],
]:
    semantic_seed = _derive_seed(candidate_seed, "semantic")
    if spec.partition == "local_transition_train":
        renderer_names = ("train_renderer_0", "train_renderer_1")
    else:
        renderer_names = (
            "development_renderer_2",
            "development_renderer_3",
        )
    renderer_seeds = (
        _derive_seed(candidate_seed, "renderer", 0),
        _derive_seed(candidate_seed, "renderer", 1),
    )
    if spec.behavior == "no_redex":
        packets = pilot._build_boolean_no_redex_pair(candidate_seed)  # noqa: SLF001
        return (
            packets,
            "matched_no_redex",
            semantic_seed,
            ("matched_positive", "matched_negative"),
            (renderer_seeds[0], renderer_seeds[0]),
        )
    if spec.behavior == "capacity_sensitive":
        packets = pilot._build_capacity_pair(candidate_seed)  # noqa: SLF001
        return (
            packets,
            "matched_capacity",
            semantic_seed,
            ("capacity_16", "capacity_15"),
            (renderer_seeds[0], renderer_seeds[0]),
        )
    if spec.behavior == "shared_occurrence":
        packets = (
            audited._shared_cancellation_base(renderer_seeds[0]),  # noqa: SLF001
            audited._shared_cancellation_base(renderer_seeds[1]),  # noqa: SLF001
        )
        return (
            packets,
            "shared_renderer_reindex",
            semantic_seed,
            renderer_names,
            renderer_seeds,
        )
    packets = (
        _build_generic_packet(
            spec,
            semantic_seed=semantic_seed,
            renderer_seed=renderer_seeds[0],
        ),
        _build_generic_packet(
            spec,
            semantic_seed=semantic_seed,
            renderer_seed=renderer_seeds[1],
        ),
    )
    return (
        packets,
        "renderer_reindex",
        semantic_seed,
        renderer_names,
        renderer_seeds,
    )


def _sample_orbit(
    spec: _OrbitSpec,
    *,
    master_seed: int,
    maximum_attempts: int,
    occupied_digests: set[str],
    train_categories: Mapping[str, set[str]],
) -> _AcceptedOrbit:
    if maximum_attempts <= 0:
        raise ProceduralCorpusError("maximum_attempts must be positive")
    rejected: list[str] = []
    for attempt in range(maximum_attempts):
        candidate_seed = _derive_seed(master_seed, spec.orbit_id, attempt)
        try:
            (
                packets,
                kind,
                semantic_seed,
                renderer_names,
                renderer_seeds,
            ) = _orbit_candidates(spec, candidate_seed=candidate_seed)
            assessments = tuple(
                pilot.assess_source_deleted_candidate(packet) for packet in packets
            )
            if len(assessments) != 2:
                raise ProceduralCorpusCandidateRejected(
                    "every corpus orbit must contain exactly two packets"
                )
            digests = tuple(item.expected.packet_sha256 for item in assessments)
            if len(set(digests)) != 2:
                raise ProceduralCorpusCandidateRejected(
                    "orbit contains exact duplicate packets"
                )
            if occupied_digests & set(digests):
                raise ProceduralCorpusCandidateRejected(
                    "orbit duplicates an already accepted packet"
                )
            if kind == "matched_no_redex":
                _validate_no_redex_pair(assessments)
            elif kind == "matched_capacity":
                _validate_capacity_pair(assessments)
            elif kind == "shared_renderer_reindex":
                _validate_shared_pair(assessments)
            else:
                _validate_renderer_pair(assessments)
            if spec.partition == "local_transition_development":
                overlap = _cross_split_overlap(train_categories, assessments)
                if overlap:
                    raise ProceduralCorpusCandidateRejected(
                        "cross-split fingerprints overlap in " + ",".join(overlap)
                    )
        except (
            ProceduralCorpusCandidateRejected,
            pilot.ProceduralBoardError,
            audited.NeuralTcrrBoardError,
            mechanics.RewriteMechanicsError,
        ) as exc:
            rejected.append(str(exc))
            continue
        typed_assessments = (assessments[0], assessments[1])
        return _AcceptedOrbit(
            spec=spec,
            assessments=typed_assessments,
            kind=kind,
            semantic_seed=semantic_seed,
            renderer_names=renderer_names,
            renderer_seeds=renderer_seeds,
            sampling=CorpusSamplingReceipt(
                orbit_id=spec.orbit_id,
                accepted_attempt=attempt,
                accepted_candidate_seed=candidate_seed,
                rejected_attempts=len(rejected),
                rejection_reasons=tuple(rejected),
            ),
        )
    final_reason = rejected[-1] if rejected else "no candidate was produced"
    raise ProceduralCorpusError(
        f"orbit {spec.orbit_id!r} exhausted {maximum_attempts} attempts: {final_reason}"
    )


def assess_corpus_packet(
    packet: audited.SourceDeletedPacket,
) -> pilot.CandidateAssessment:
    """Apply complete geometry, label-bridge, and independent-oracle admission."""

    try:
        return pilot.assess_source_deleted_candidate(packet)
    except pilot.ProceduralCandidateRejected as exc:
        raise ProceduralCorpusCandidateRejected(str(exc)) from exc


def _metadata_from_orbit(
    orbit: _AcceptedOrbit,
) -> tuple[CorpusPacketMetadata, CorpusPacketMetadata]:
    roles = {
        "renderer_reindex": ("renderer_left", "renderer_right"),
        "shared_renderer_reindex": ("shared_left", "shared_right"),
        "matched_no_redex": ("positive", "no_redex"),
        "matched_capacity": ("capacity_16", "capacity_15"),
    }[orbit.kind]
    output = []
    for index, assessment in enumerate(orbit.assessments):
        packet = assessment.packet
        type_ids = {
            type_id
            for constructor in packet.constructors
            for type_id in (
                constructor.result_type,
                *constructor.argument_types,
            )
        }
        output.append(
            CorpusPacketMetadata(
                packet_sha256=assessment.expected.packet_sha256,
                partition=orbit.spec.partition,
                family=orbit.spec.family,
                grammar_lane=orbit.spec.grammar_lane,
                behavior=orbit.spec.behavior,
                orbit_kind=orbit.kind,
                orbit_id=orbit.spec.orbit_id,
                semantic_index=orbit.spec.semantic_index,
                semantic_seed=orbit.semantic_seed,
                renderer=orbit.renderer_names[index],
                renderer_seed=orbit.renderer_seeds[index],
                role=roles[index],
                max_occurrence_depth=assessment.max_occurrence_depth,
                legal_action_count=len(assessment.expected.transitions),
                constructor_count=len(packet.constructors),
                type_count=len(type_ids),
                rule_count=len(packet.rules),
                constructor_arities=tuple(
                    sorted(
                        {
                            len(constructor.argument_types)
                            for constructor in packet.constructors
                        }
                    )
                ),
            )
        )
    return output[0], output[1]


def _cell_receipts(
    metadata: Sequence[CorpusPacketMetadata],
) -> tuple[CorpusCellReceipt, ...]:
    counts = Counter((item.partition, item.family, item.behavior) for item in metadata)
    return tuple(
        CorpusCellReceipt(
            partition=partition,
            family=family,
            behavior=behavior,
            count=count,
        )
        for (partition, family, behavior), count in sorted(counts.items())
    )


def _diversity_receipt(
    packets: Sequence[audited.SourceDeletedPacket],
    metadata: Sequence[CorpusPacketMetadata],
    fingerprints: Sequence[audited.PacketFingerprints],
) -> CorpusDiversityReceipt:
    return CorpusDiversityReceipt(
        unique_exact_packets=len({item.exact_sha256 for item in fingerprints}),
        unique_isomorphic_packets=len(
            {item.isomorphic_sha256 for item in fingerprints}
        ),
        unique_rule_windows=len(
            {value for item in fingerprints for value in item.normalized_rule_windows}
        ),
        unique_rule_pairs=len(
            {value for item in fingerprints for value in item.normalized_rule_pairs}
        ),
        unique_two_rule_compositions=len(
            {
                value
                for item in fingerprints
                for value in item.reachable_two_rule_compositions
            }
        ),
        constructor_arities=tuple(
            sorted(
                {
                    len(constructor.argument_types)
                    for packet in packets
                    for constructor in packet.constructors
                }
            )
        ),
        type_cardinalities=tuple(sorted({item.type_count for item in metadata})),
        maximum_legal_actions=max(
            (item.legal_action_count for item in metadata),
            default=0,
        ),
        maximum_occurrence_depth=max(
            (item.max_occurrence_depth for item in metadata),
            default=0,
        ),
    )


def _split_isolation_payload(
    metadata: Sequence[CorpusPacketMetadata],
    fingerprints: Sequence[audited.PacketFingerprints],
) -> dict[str, object]:
    by_digest = {item.packet_sha256: item for item in fingerprints}
    output: dict[str, object] = {}
    for partition in (
        "local_transition_train",
        "local_transition_development",
    ):
        active = [
            by_digest[item.packet_sha256]
            for item in metadata
            if item.partition == partition
        ]
        categories = _fingerprint_categories(active)
        output[partition] = {
            key: tuple(sorted(value)) for key, value in sorted(categories.items())
        }
    return output


def _manifest(
    *,
    seed: int,
    minimum_cell_count: int,
    packets: tuple[audited.SourceDeletedPacket, ...],
    expected: tuple[audited.ExpectedTransitionRecord, ...],
    metadata: tuple[CorpusPacketMetadata, ...],
    fingerprints: tuple[audited.PacketFingerprints, ...],
    oracles: tuple[audited.OracleAgreementRecord, ...],
    labels: tuple[pilot.OneStepLabelAgreement, ...],
    orbits: tuple[CorpusOrbitReceipt, ...],
    sampling: tuple[CorpusSamplingReceipt, ...],
) -> CorpusManifestReceipt:
    cells = _cell_receipts(metadata)
    diversity = _diversity_receipt(packets, metadata, fingerprints)
    split_payload = _split_isolation_payload(metadata, fingerprints)
    base: dict[str, object] = {
        "generator_version": GENERATOR_VERSION,
        "grammar_semantics_version": GRAMMAR_SEMANTICS_VERSION,
        "grammar_semantics_sha256": _sha256(_grammar_semantics_payload()),
        "master_seed": seed,
        "geometry": dataclasses.asdict(GeometryReceipt()),
        "train_packet_count": sum(
            item.partition == "local_transition_train" for item in metadata
        ),
        "development_packet_count": sum(
            item.partition == "local_transition_development" for item in metadata
        ),
        "packet_count": len(packets),
        "orbit_count": len(orbits),
        "rejected_candidate_count": sum(item.rejected_attempts for item in sampling),
        "minimum_cell_count": minimum_cell_count,
        "cells": tuple(dataclasses.asdict(item) for item in cells),
        "diversity": dataclasses.asdict(diversity),
        "packet_manifest_sha256": _sha256(
            tuple(
                {
                    "packet_sha256": audited.packet_sha256(packet),
                    "serialized_packet": audited.serialize_model_packet(packet),
                }
                for packet in packets
            )
        ),
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
        "orbit_manifest_sha256": _sha256(
            tuple(dataclasses.asdict(item) for item in orbits)
        ),
        "sampling_manifest_sha256": _sha256(
            tuple(dataclasses.asdict(item) for item in sampling)
        ),
        "split_isolation_manifest_sha256": _sha256(split_payload),
    }
    return CorpusManifestReceipt(
        generator_version=GENERATOR_VERSION,
        grammar_semantics_version=GRAMMAR_SEMANTICS_VERSION,
        grammar_semantics_sha256=str(base["grammar_semantics_sha256"]),
        master_seed=seed,
        geometry=GeometryReceipt(),
        train_packet_count=int(base["train_packet_count"]),
        development_packet_count=int(base["development_packet_count"]),
        packet_count=len(packets),
        orbit_count=len(orbits),
        rejected_candidate_count=int(base["rejected_candidate_count"]),
        minimum_cell_count=minimum_cell_count,
        cells=cells,
        diversity=diversity,
        packet_manifest_sha256=str(base["packet_manifest_sha256"]),
        metadata_manifest_sha256=str(base["metadata_manifest_sha256"]),
        expected_manifest_sha256=str(base["expected_manifest_sha256"]),
        fingerprint_manifest_sha256=str(base["fingerprint_manifest_sha256"]),
        oracle_manifest_sha256=str(base["oracle_manifest_sha256"]),
        label_agreement_manifest_sha256=str(base["label_agreement_manifest_sha256"]),
        orbit_manifest_sha256=str(base["orbit_manifest_sha256"]),
        sampling_manifest_sha256=str(base["sampling_manifest_sha256"]),
        split_isolation_manifest_sha256=str(base["split_isolation_manifest_sha256"]),
        payload_sha256=_sha256(base),
    )


def generate_neural_tcrr_corpus(
    seed: int = DEFAULT_SEED,
    *,
    train_packets: int = DEFAULT_TRAIN_PACKETS,
    development_packets: int = DEFAULT_DEVELOPMENT_PACKETS,
    maximum_attempts: int = DEFAULT_MAX_ATTEMPTS,
    minimum_cell_count: int = 2,
) -> ProceduralNeuralTcrrCorpus:
    """Generate a deterministic source-deleted train/development corpus."""

    if minimum_cell_count <= 0:
        raise ProceduralCorpusError("minimum_cell_count must be positive")
    specs = _build_orbit_specs(
        train_packets=train_packets,
        development_packets=development_packets,
    )
    accepted: list[_AcceptedOrbit] = []
    occupied: set[str] = set()
    train_fingerprints: list[audited.PacketFingerprints] = []
    train_categories: dict[str, set[str]] = {}
    for spec in specs:
        orbit = _sample_orbit(
            spec,
            master_seed=seed,
            maximum_attempts=maximum_attempts,
            occupied_digests=occupied,
            train_categories=train_categories,
        )
        accepted.append(orbit)
        for assessment in orbit.assessments:
            occupied.add(assessment.expected.packet_sha256)
        if spec.partition == "local_transition_train":
            train_fingerprints.extend(
                assessment.fingerprints for assessment in orbit.assessments
            )
            train_categories = _fingerprint_categories(train_fingerprints)

    assessments = tuple(
        assessment for orbit in accepted for assessment in orbit.assessments
    )
    packets = tuple(item.packet for item in assessments)
    expected = tuple(item.expected for item in assessments)
    fingerprints = tuple(item.fingerprints for item in assessments)
    oracles = tuple(item.oracle_agreement for item in assessments)
    labels = tuple(item.label_agreement for item in assessments)
    metadata = tuple(item for orbit in accepted for item in _metadata_from_orbit(orbit))
    orbit_receipts = tuple(
        CorpusOrbitReceipt(
            orbit_id=orbit.spec.orbit_id,
            kind=orbit.kind,
            left_packet_sha256=orbit.assessments[0].expected.packet_sha256,
            right_packet_sha256=orbit.assessments[1].expected.packet_sha256,
        )
        for orbit in accepted
    )
    sampling = tuple(orbit.sampling for orbit in accepted)
    manifest = _manifest(
        seed=seed,
        minimum_cell_count=minimum_cell_count,
        packets=packets,
        expected=expected,
        metadata=metadata,
        fingerprints=fingerprints,
        oracles=oracles,
        labels=labels,
        orbits=orbit_receipts,
        sampling=sampling,
    )
    result = ProceduralNeuralTcrrCorpus(
        packets=packets,
        expected_records=expected,
        metadata=metadata,
        fingerprints=fingerprints,
        oracle_agreements=oracles,
        label_agreements=labels,
        orbit_receipts=orbit_receipts,
        sampling_receipts=sampling,
        manifest=manifest,
    )
    validate_neural_tcrr_corpus(result, recompute_oracles=False)
    return result


def _unique_ledger(
    records: Sequence[object],
    *,
    packet_digests: tuple[str, ...],
    name: str,
) -> dict[str, object]:
    keys = tuple(str(getattr(item, "packet_sha256")) for item in records)
    if len(keys) != len(set(keys)):
        raise ProceduralCorpusError(f"{name} ledger contains duplicate keys")
    if keys != packet_digests:
        raise ProceduralCorpusError(f"{name} ledger order or membership differs")
    return dict(zip(keys, records, strict=True))


def _validate_family_and_depth_holds(
    metadata: Sequence[CorpusPacketMetadata],
) -> None:
    train = [item for item in metadata if item.partition == "local_transition_train"]
    development = [
        item for item in metadata if item.partition == "local_transition_development"
    ]
    train_families = {item.family for item in train}
    development_families = {item.family for item in development}
    if train_families != set(_OPTIMIZATION_FAMILIES):
        raise ProceduralCorpusError(
            "train must contain exactly the three optimization families"
        )
    if development_families != set(_ALL_FAMILIES):
        raise ProceduralCorpusError(
            "development must contain optimization and held-out families"
        )
    if any(item.family in _HELDOUT_FAMILIES for item in train):
        raise ProceduralCorpusError("held-out family leaked into train")
    train_renderers = {item.renderer for item in train}
    development_renderers = {item.renderer for item in development}
    if train_renderers & development_renderers:
        raise ProceduralCorpusError("development renderer leaked into train")
    for family in _OPTIMIZATION_FAMILIES:
        train_depths = {
            item.max_occurrence_depth for item in train if item.family == family
        }
        development_depths = {
            item.max_occurrence_depth for item in development if item.family == family
        }
        if not development_depths - train_depths:
            raise ProceduralCorpusError(
                f"development family {family!r} lacks an unseen depth"
            )


def _validate_orbits(
    value: ProceduralNeuralTcrrCorpus,
    *,
    packet_map: Mapping[str, audited.SourceDeletedPacket],
    expected_map: Mapping[str, audited.ExpectedTransitionRecord],
    fingerprint_map: Mapping[str, audited.PacketFingerprints],
) -> None:
    if len(value.orbit_receipts) * 2 != len(value.packets):
        raise ProceduralCorpusError("orbit ledger does not cover packet pairs")
    metadata_by_orbit: dict[str, list[CorpusPacketMetadata]] = defaultdict(list)
    for item in value.metadata:
        metadata_by_orbit[item.orbit_id].append(item)
    for orbit in value.orbit_receipts:
        if orbit.left_packet_sha256 not in packet_map:
            raise ProceduralCorpusError("orbit names absent left packet")
        if orbit.right_packet_sha256 not in packet_map:
            raise ProceduralCorpusError("orbit names absent right packet")
        if len(metadata_by_orbit[orbit.orbit_id]) != 2:
            raise ProceduralCorpusError("orbit metadata cardinality is not two")
        left_packet = packet_map[orbit.left_packet_sha256]
        right_packet = packet_map[orbit.right_packet_sha256]
        left_expected = expected_map[orbit.left_packet_sha256]
        right_expected = expected_map[orbit.right_packet_sha256]
        if orbit.kind in {
            "renderer_reindex",
            "shared_renderer_reindex",
        }:
            if orbit.left_packet_sha256 == orbit.right_packet_sha256:
                raise ProceduralCorpusError("renderer orbit retained exact bytes")
            if (
                fingerprint_map[orbit.left_packet_sha256].isomorphic_sha256
                != fingerprint_map[orbit.right_packet_sha256].isomorphic_sha256
            ):
                raise ProceduralCorpusError(
                    "renderer orbit changed canonical semantics"
                )
            if _identifier_set(left_packet) & _identifier_set(right_packet):
                raise ProceduralCorpusError(
                    "renderer orbit retained opaque identifiers"
                )
        if orbit.kind == "matched_no_redex":
            if not left_expected.transitions or right_expected.transitions:
                raise ProceduralCorpusError("no-redex orbit lost separation")
            if left_packet.rules != right_packet.rules:
                raise ProceduralCorpusError("no-redex orbit changed rules")
        if orbit.kind == "matched_capacity":
            if len(left_packet.graph.reservoir) != N:
                raise ProceduralCorpusError("capacity orbit left arm is not N=16")
            if len(right_packet.graph.reservoir) != N - 1:
                raise ProceduralCorpusError("capacity orbit right arm is not N=15")
            if not left_expected.transitions or right_expected.transitions:
                raise ProceduralCorpusError("capacity orbit lost separation")
        if orbit.kind == "shared_renderer_reindex":
            for expected in (left_expected, right_expected):
                if not any(
                    left.target_storage_id == right.target_storage_id
                    and left.occurrence_path != right.occurrence_path
                    for left_index, left in enumerate(expected.transitions)
                    for right in expected.transitions[left_index + 1 :]
                ):
                    raise ProceduralCorpusError(
                        "shared orbit lost path-distinct aliases"
                    )


def validate_neural_tcrr_corpus(
    value: ProceduralNeuralTcrrCorpus,
    *,
    recompute_oracles: bool = True,
) -> None:
    """Recompute alignment, split, orbit, manifest, and optional oracle gates."""

    _assert_frozen_geometry()
    plan = plan_neural_tcrr_corpus(
        train_packets=value.manifest.train_packet_count,
        development_packets=value.manifest.development_packet_count,
    )
    if plan.grammar_semantics_sha256 != value.manifest.grammar_semantics_sha256:
        raise ProceduralCorpusError("grammar semantics receipt is stale")
    packet_digests = tuple(audited.packet_sha256(item) for item in value.packets)
    if len(packet_digests) != len(set(packet_digests)):
        raise ProceduralCorpusError("corpus contains exact duplicate packets")
    packet_map = dict(zip(packet_digests, value.packets, strict=True))
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
    expected_map = {
        key: item
        for key, item in expected_raw.items()
        if isinstance(item, audited.ExpectedTransitionRecord)
    }
    metadata_map = {
        key: item
        for key, item in metadata_raw.items()
        if isinstance(item, CorpusPacketMetadata)
    }
    fingerprint_map = {
        key: item
        for key, item in fingerprints_raw.items()
        if isinstance(item, audited.PacketFingerprints)
    }
    if not (
        len(expected_map)
        == len(metadata_map)
        == len(fingerprint_map)
        == len(packet_digests)
    ):
        raise ProceduralCorpusError("typed ledger conversion failed")

    for digest, packet in packet_map.items():
        audited.validate_source_deleted_packet(packet)
        metadata = metadata_map[digest]
        if metadata.max_occurrence_depth > D:
            raise ProceduralCorpusError("packet exceeds path-depth geometry")
        if metadata.legal_action_count > MAX_LEGAL_ACTIONS:
            raise ProceduralCorpusError("packet contains truncated action labels")
        if metadata.legal_action_count != len(expected_map[digest].transitions):
            raise ProceduralCorpusError("metadata action count is stale")
        if fingerprint_map[digest] != audited.packet_fingerprints(packet):
            raise ProceduralCorpusError("fingerprint ledger is stale")
        if recompute_oracles:
            recomputed = assess_corpus_packet(packet)
            if expected_map[digest] != recomputed.expected:
                raise ProceduralCorpusError("expected transition ledger is stale")
            if oracles_raw[digest] != recomputed.oracle_agreement:
                raise ProceduralCorpusError("oracle agreement ledger is stale")
            if labels_raw[digest] != recomputed.label_agreement:
                raise ProceduralCorpusError("exported successor-edge bridge is stale")
        else:
            oracle = oracles_raw[digest]
            label = labels_raw[digest]
            if not getattr(oracle, "exact_agreement", False):
                raise ProceduralCorpusError("oracle agreement is not exact")
            if not getattr(label, "exact_agreement", False):
                raise ProceduralCorpusError(
                    "successor-edge bridge agreement is not exact"
                )

    assignments = tuple(
        audited.SplitAssignment(item.packet_sha256, item.partition)
        for item in value.metadata
    )
    audited.validate_split_isolation(assignments, value.fingerprints)
    _validate_family_and_depth_holds(value.metadata)
    if any(
        item.count < value.manifest.minimum_cell_count for item in value.manifest.cells
    ):
        raise ProceduralCorpusError("one generated cell is below its minimum")
    required_cells = {
        *(
            ("local_transition_train", family, behavior)
            for family, behavior in _TRAIN_REQUIRED
        ),
        *(
            ("local_transition_development", family, behavior)
            for family, behavior in _DEVELOPMENT_REQUIRED
        ),
    }
    observed_cells = {
        (item.partition, item.family, item.behavior) for item in value.metadata
    }
    if not required_cells <= observed_cells:
        raise ProceduralCorpusError("required behavior cell is absent")
    _validate_orbits(
        value,
        packet_map=packet_map,
        expected_map=expected_map,
        fingerprint_map=fingerprint_map,
    )

    recomputed_manifest = _manifest(
        seed=value.manifest.master_seed,
        minimum_cell_count=value.manifest.minimum_cell_count,
        packets=value.packets,
        expected=value.expected_records,
        metadata=value.metadata,
        fingerprints=value.fingerprints,
        oracles=value.oracle_agreements,
        labels=value.label_agreements,
        orbits=value.orbit_receipts,
        sampling=value.sampling_receipts,
    )
    if value.manifest != recomputed_manifest:
        raise ProceduralCorpusError("corpus manifest receipt is stale")


__all__ = [
    "A",
    "C",
    "D",
    "DEFAULT_DEVELOPMENT_PACKETS",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_SEED",
    "DEFAULT_TRAIN_PACKETS",
    "GENERATOR_VERSION",
    "GRAMMAR_SEMANTICS_VERSION",
    "MAX_LEGAL_ACTIONS",
    "N",
    "P",
    "PREREGISTERED_DEVELOPMENT_PACKETS",
    "PREREGISTERED_TRAIN_PACKETS",
    "R",
    "Y",
    "CorpusCellReceipt",
    "CorpusDiversityReceipt",
    "CorpusManifestReceipt",
    "CorpusOrbitReceipt",
    "CorpusPacketMetadata",
    "CorpusSamplingReceipt",
    "CorpusScalePlan",
    "GeometryReceipt",
    "ProceduralCorpusCandidateRejected",
    "ProceduralCorpusError",
    "ProceduralNeuralTcrrCorpus",
    "assess_corpus_packet",
    "generate_neural_tcrr_corpus",
    "plan_neural_tcrr_corpus",
    "validate_neural_tcrr_corpus",
]
