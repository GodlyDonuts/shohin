"""Adversarially repaired procedural board for the N-TCRR local motor.

This module is an offline generator and custody boundary. It emits opaque,
source-deleted model packets, while labels, split assignments, fingerprints,
twin receipts, and independent-oracle receipts remain separate. The committed
typed rewrite mechanics are used only to generate and cross-check offline
records. They are never serialized into a model packet.

The board is still a bounded local-transition slice. It is not a neural model,
an autonomous rewrite reactor, or evidence of general reasoning.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import random
import stat
from typing import Literal

import typed_critical_pair_rewrite_board as mechanics


MAX_CAPACITY = 16
MAX_CONSTRUCTORS = 16
MAX_TYPES = 8
MAX_RULES = 8
MAX_RULE_SIDE_NODES = 12
MAX_ARITY = 3
MAX_PATH_DEPTH = 8
MAX_CANONICAL_BACKTRACK_STATES = 50_000
OPAQUE_ID_LENGTH = 24

Partition = Literal["local_transition_train", "local_transition_development"]
TwinKind = Literal[
    "rhs_pointer",
    "shared_occurrence",
    "capacity",
    "constructor_reindex",
    "type_reindex",
    "rule_reindex",
    "storage_reindex",
    "repeated_variable_equality",
    "partial_nested_match",
    "type_mismatch",
]
ReindexNamespace = Literal["constructor", "type", "rule", "storage"]


class NeuralTcrrBoardError(ValueError):
    """Raised when a board, packet, export, or receipt fails closed."""


@dataclass(frozen=True)
class ConstructorRecord:
    """One model-visible episode-local constructor declaration."""

    identifier: str
    result_type: str
    argument_types: tuple[str, ...]


@dataclass(frozen=True)
class RuleTermRecord:
    """One model-visible LHS or RHS term record."""

    kind: Literal["constructor", "variable"]
    type_id: str
    constructor_id: str | None = None
    variable_id: str | None = None
    children: tuple[RuleTermRecord, ...] = ()


@dataclass(frozen=True)
class RuleRecord:
    """One opaque rewrite card; ``rhs=None`` denotes root deletion."""

    identifier: str
    lhs: RuleTermRecord
    rhs: RuleTermRecord | None


@dataclass(frozen=True)
class GraphNodeRecord:
    """One occupied model-visible reservoir record."""

    storage_id: str
    kind: Literal["constructor", "variable"]
    type_id: str
    constructor_id: str | None = None
    variable_id: str | None = None
    children: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphRecord:
    """One rooted graph in an opaque fixed-capacity reservoir."""

    reservoir: tuple[str, ...]
    root: str | None
    nodes: tuple[GraphNodeRecord, ...]


@dataclass(frozen=True)
class SourceDeletedPacket:
    """The complete model-visible local-transition input."""

    constructors: tuple[ConstructorRecord, ...]
    rules: tuple[RuleRecord, ...]
    graph: GraphRecord


@dataclass(frozen=True)
class BindingRecord:
    """One offline metavariable-to-storage binding label."""

    variable_id: str
    storage_id: str


@dataclass(frozen=True)
class ExpectedTransition:
    """One offline exact legal action and successor."""

    rule_id: str
    occurrence_path: tuple[int, ...]
    target_storage_id: str
    bindings: tuple[BindingRecord, ...]
    successor: GraphRecord


@dataclass(frozen=True)
class ExpectedTransitionRecord:
    """Offline labels for one packet, joined only by packet digest."""

    packet_sha256: str
    transitions: tuple[ExpectedTransition, ...]


@dataclass(frozen=True)
class SplitAssignment:
    """Offline partition membership for one packet."""

    packet_sha256: str
    partition: Partition


@dataclass(frozen=True)
class PacketFingerprints:
    """All exact and normalized split-leakage fingerprints."""

    packet_sha256: str
    exact_sha256: str
    isomorphic_sha256: str
    normalized_rule_windows: tuple[str, ...]
    normalized_rule_pairs: tuple[str, ...]
    reachable_two_rule_compositions: tuple[str, ...]


@dataclass(frozen=True)
class IdentifierRemap:
    """One exact identifier substitution in a named reindex twin."""

    old: str
    new: str


@dataclass(frozen=True)
class CausalTwinRecord:
    """A mutation receipt whose predicate is recomputed during validation."""

    kind: TwinKind
    left_packet_sha256: str
    right_packet_sha256: str
    namespace: ReindexNamespace | None = None
    remap: tuple[IdentifierRemap, ...] = ()
    axis_values: tuple[str, ...] = ()
    left_transition_index: int | None = None
    right_transition_index: int | None = None


@dataclass(frozen=True)
class OracleAgreementRecord:
    """Independent full state-graph agreement for one generated packet."""

    packet_sha256: str
    production_sha256: str
    independent_reference_sha256: str
    state_count: int
    transition_count: int
    normal_form_count: int
    cyclic_component_count: int
    exact_agreement: bool


@dataclass(frozen=True)
class PrimitiveCoverageRecord:
    """Offline local-primitive inventory derived from packet and transitions."""

    packet_sha256: str
    primitives: tuple[str, ...]


@dataclass(frozen=True)
class LocalTransitionSlice:
    """Offline aggregate used only for generation, audit, and export."""

    packets: tuple[SourceDeletedPacket, ...]
    expected_records: tuple[ExpectedTransitionRecord, ...]
    split_assignments: tuple[SplitAssignment, ...]
    fingerprints: tuple[PacketFingerprints, ...]
    twins: tuple[CausalTwinRecord, ...]
    oracle_agreements: tuple[OracleAgreementRecord, ...]
    primitive_coverage: tuple[PrimitiveCoverageRecord, ...]


@dataclass(frozen=True)
class PacketExportReceipt:
    """Receipt for physically separated packet, label, and assessor roots."""

    packet_root: str
    train_label_root: str
    development_assessment_root: str
    packet_manifest_sha256: str
    train_label_manifest_sha256: str
    development_assessment_manifest_sha256: str
    sealed_development_artifact_sha256: str
    train_packet_count: int
    development_packet_count: int


_FORBIDDEN_MODEL_KEY_FRAGMENTS = frozenset(
    {
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
)

_REQUIRED_TWIN_KINDS = frozenset(
    {
        "rhs_pointer",
        "shared_occurrence",
        "capacity",
        "constructor_reindex",
        "type_reindex",
        "rule_reindex",
        "storage_reindex",
        "repeated_variable_equality",
        "partial_nested_match",
        "type_mismatch",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _opaque(seed: int, namespace: str, index: int) -> str:
    material = f"N-TCRR:{seed}:{namespace}:{index}".encode()
    return hashlib.sha256(material).hexdigest()[:OPAQUE_ID_LENGTH]


def _is_opaque(identifier: str) -> bool:
    return len(identifier) == OPAQUE_ID_LENGTH and set(identifier) <= set(
        "0123456789abcdef"
    )


def _scan_forbidden_model_keys(value: object, location: str = "packet") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in _FORBIDDEN_MODEL_KEY_FRAGMENTS):
                raise NeuralTcrrBoardError(
                    f"{location} contains forbidden model-visible field {key!r}"
                )
            _scan_forbidden_model_keys(nested, f"{location}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, nested in enumerate(value):
            _scan_forbidden_model_keys(nested, f"{location}[{index}]")


def validate_model_packet_payload(payload: object) -> None:
    """Reject label, source, split, schedule, and oracle fields recursively."""

    if not isinstance(payload, Mapping):
        raise NeuralTcrrBoardError("model packet payload must be a mapping")
    _scan_forbidden_model_keys(payload)


def _rule_term_count(term: RuleTermRecord | None) -> int:
    if term is None:
        return 0
    return 1 + sum(_rule_term_count(child) for child in term.children)


def _validate_rule_term(
    term: RuleTermRecord,
    constructors: Mapping[str, ConstructorRecord],
    *,
    bound_variables: dict[str, str],
    is_rhs: bool,
) -> str:
    if term.kind not in {"constructor", "variable"}:
        raise NeuralTcrrBoardError("rule term has an unknown kind")
    if term.kind == "variable":
        if term.constructor_id is not None or term.children:
            raise NeuralTcrrBoardError("variable rule term has constructor content")
        if term.variable_id is None or not _is_opaque(term.variable_id):
            raise NeuralTcrrBoardError("rule variable identifier is not opaque")
        if is_rhs:
            if bound_variables.get(term.variable_id) != term.type_id:
                raise NeuralTcrrBoardError("RHS variable is absent or changes type")
        else:
            previous = bound_variables.setdefault(term.variable_id, term.type_id)
            if previous != term.type_id:
                raise NeuralTcrrBoardError("repeated LHS variable changes type")
        return term.type_id
    if term.variable_id is not None or term.constructor_id is None:
        raise NeuralTcrrBoardError("constructor rule term is malformed")
    spec = constructors.get(term.constructor_id)
    if spec is None:
        raise NeuralTcrrBoardError("rule term names an unknown constructor")
    if term.type_id != spec.result_type:
        raise NeuralTcrrBoardError("rule term has the wrong constructor result type")
    child_types = tuple(
        _validate_rule_term(
            child,
            constructors,
            bound_variables=bound_variables,
            is_rhs=is_rhs,
        )
        for child in term.children
    )
    if child_types != spec.argument_types:
        raise NeuralTcrrBoardError("rule term child types violate declaration")
    return term.type_id


def validate_source_deleted_packet(packet: SourceDeletedPacket) -> None:
    """Validate geometry and syntax without computing any target."""

    validate_model_packet_payload(dataclasses.asdict(packet))
    constructor_ids = [item.identifier for item in packet.constructors]
    if not 0 < len(constructor_ids) <= MAX_CONSTRUCTORS:
        raise NeuralTcrrBoardError("constructor count lies outside frozen geometry")
    if len(constructor_ids) != len(set(constructor_ids)):
        raise NeuralTcrrBoardError("constructor identifiers must be unique")
    if not all(_is_opaque(identifier) for identifier in constructor_ids):
        raise NeuralTcrrBoardError("constructor identifiers must be opaque")
    type_ids = {
        type_id
        for item in packet.constructors
        for type_id in (item.result_type, *item.argument_types)
    }
    if not 0 < len(type_ids) <= MAX_TYPES:
        raise NeuralTcrrBoardError("type count lies outside frozen geometry")
    if not all(_is_opaque(type_id) for type_id in type_ids):
        raise NeuralTcrrBoardError("type identifiers must be opaque")
    if any(len(item.argument_types) > MAX_ARITY for item in packet.constructors):
        raise NeuralTcrrBoardError("constructor exceeds the arity bound")
    constructors = {item.identifier: item for item in packet.constructors}

    reservoir_ids = packet.graph.reservoir
    if not 0 < len(reservoir_ids) <= MAX_CAPACITY:
        raise NeuralTcrrBoardError("reservoir capacity lies outside frozen geometry")
    if len(reservoir_ids) != len(set(reservoir_ids)):
        raise NeuralTcrrBoardError("reservoir storage identifiers must be unique")
    if not all(_is_opaque(item) for item in reservoir_ids):
        raise NeuralTcrrBoardError("storage identifiers must be opaque")
    reservoir = set(reservoir_ids)
    nodes = {item.storage_id: item for item in packet.graph.nodes}
    if len(nodes) != len(packet.graph.nodes) or not set(nodes) <= reservoir:
        raise NeuralTcrrBoardError("occupied storage identifiers are invalid")
    if packet.graph.root is None:
        if nodes:
            raise NeuralTcrrBoardError("empty graph cannot retain occupied records")
    elif packet.graph.root not in nodes:
        raise NeuralTcrrBoardError("graph root is not occupied")

    graph_variable_ids = []
    for node in nodes.values():
        if node.kind not in {"constructor", "variable"}:
            raise NeuralTcrrBoardError("graph node has an unknown kind")
        if node.type_id not in type_ids:
            raise NeuralTcrrBoardError("graph node uses an unknown type")
        if node.kind == "variable":
            if (
                node.variable_id is None
                or not _is_opaque(node.variable_id)
                or node.constructor_id is not None
                or node.children
            ):
                raise NeuralTcrrBoardError("graph variable record is malformed")
            graph_variable_ids.append(node.variable_id)
            continue
        if node.variable_id is not None or node.constructor_id is None:
            raise NeuralTcrrBoardError("graph constructor record is malformed")
        spec = constructors.get(node.constructor_id)
        if spec is None or node.type_id != spec.result_type:
            raise NeuralTcrrBoardError("graph constructor declaration mismatch")
        if any(child not in nodes for child in node.children):
            raise NeuralTcrrBoardError("graph pointer names a free record")
        if tuple(nodes[child].type_id for child in node.children) != (
            spec.argument_types
        ):
            raise NeuralTcrrBoardError("graph edge types violate declaration")
    if len(graph_variable_ids) != len(set(graph_variable_ids)):
        raise NeuralTcrrBoardError("graph variable identifiers must be unique")

    visiting: set[str] = set()
    reached: set[str] = set()

    def visit(storage_id: str) -> None:
        if storage_id in visiting:
            raise NeuralTcrrBoardError("graph must be acyclic")
        if storage_id in reached:
            return
        visiting.add(storage_id)
        for child in nodes[storage_id].children:
            visit(child)
        visiting.remove(storage_id)
        reached.add(storage_id)

    if packet.graph.root is not None:
        visit(packet.graph.root)
    if reached != set(nodes):
        raise NeuralTcrrBoardError("all occupied records must be root-reachable")

    if not 0 < len(packet.rules) <= MAX_RULES:
        raise NeuralTcrrBoardError("rule count lies outside frozen geometry")
    rule_ids = [rule.identifier for rule in packet.rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise NeuralTcrrBoardError("rule identifiers must be unique")
    if not all(_is_opaque(identifier) for identifier in rule_ids):
        raise NeuralTcrrBoardError("rule identifiers must be opaque")
    all_rule_variables: set[str] = set()
    for rule in packet.rules:
        if _rule_term_count(rule.lhs) > MAX_RULE_SIDE_NODES:
            raise NeuralTcrrBoardError("LHS exceeds the rule-side node bound")
        if _rule_term_count(rule.rhs) > MAX_RULE_SIDE_NODES:
            raise NeuralTcrrBoardError("RHS exceeds the rule-side node bound")
        variables: dict[str, str] = {}
        lhs_type = _validate_rule_term(
            rule.lhs,
            constructors,
            bound_variables=variables,
            is_rhs=False,
        )
        if all_rule_variables & set(variables):
            raise NeuralTcrrBoardError("binder identifiers must be unique by rule")
        all_rule_variables.update(variables)
        if rule.rhs is not None:
            rhs_type = _validate_rule_term(
                rule.rhs,
                constructors,
                bound_variables=variables,
                is_rhs=True,
            )
            if rhs_type != lhs_type:
                raise NeuralTcrrBoardError("rewrite changes the redex root type")
    if all_rule_variables & set(graph_variable_ids):
        raise NeuralTcrrBoardError("graph variables cannot alias rule binders")


def serialize_model_packet(packet: SourceDeletedPacket) -> str:
    """Serialize only model-visible fields after strict validation."""

    validate_source_deleted_packet(packet)
    payload = dataclasses.asdict(packet)
    validate_model_packet_payload(payload)
    return _canonical_json(payload)


def packet_sha256(packet: SourceDeletedPacket) -> str:
    """Return the exact packet-content digest used by offline ledgers."""

    return _sha256(serialize_model_packet(packet))


def _pattern_to_record(
    system: mechanics.RewriteSystem,
    pattern: mechanics.Pattern,
) -> RuleTermRecord:
    if isinstance(pattern, mechanics.PatternVariable):
        return RuleTermRecord(
            kind="variable",
            type_id=pattern.type_id,
            variable_id=pattern.name,
        )
    spec = system.constructor_map()[pattern.constructor_id]
    return RuleTermRecord(
        kind="constructor",
        type_id=spec.result_type,
        constructor_id=pattern.constructor_id,
        children=tuple(_pattern_to_record(system, child) for child in pattern.children),
    )


def _collect_pattern_variables(
    pattern: mechanics.Pattern,
    output: dict[str, str],
) -> None:
    if isinstance(pattern, mechanics.PatternVariable):
        output.setdefault(pattern.name, pattern.type_id)
        return
    for child in pattern.children:
        _collect_pattern_variables(child, output)


def _rhs_to_record(
    system: mechanics.RewriteSystem,
    expression: mechanics.RhsExpression,
    variable_types: Mapping[str, str],
) -> RuleTermRecord:
    if isinstance(expression, mechanics.RhsVariable):
        return RuleTermRecord(
            kind="variable",
            type_id=variable_types[expression.name],
            variable_id=expression.name,
        )
    spec = system.constructor_map()[expression.constructor_id]
    return RuleTermRecord(
        kind="constructor",
        type_id=spec.result_type,
        constructor_id=expression.constructor_id,
        children=tuple(
            _rhs_to_record(system, child, variable_types)
            for child in expression.children
        ),
    )


def _graph_to_record(
    graph: mechanics.TermGraph,
    storage_ids: tuple[str, ...],
    *,
    shuffle_seed: int | None,
) -> GraphRecord:
    if len(storage_ids) != graph.capacity:
        raise NeuralTcrrBoardError("storage map does not cover graph capacity")
    records = []
    for node in graph.nodes:
        if node.variable is not None:
            records.append(
                GraphNodeRecord(
                    storage_id=storage_ids[node.slot],
                    kind="variable",
                    type_id=node.type_id,
                    variable_id=node.variable,
                )
            )
        else:
            records.append(
                GraphNodeRecord(
                    storage_id=storage_ids[node.slot],
                    kind="constructor",
                    type_id=node.type_id,
                    constructor_id=node.constructor_id,
                    children=tuple(storage_ids[child] for child in node.children),
                )
            )
    if shuffle_seed is None:
        records.sort(key=lambda item: item.storage_id)
    else:
        random.Random(shuffle_seed).shuffle(records)
    return GraphRecord(
        reservoir=storage_ids,
        root=None if graph.root is None else storage_ids[graph.root],
        nodes=tuple(records),
    )


def _packet_from_mechanics(
    system: mechanics.RewriteSystem,
    graph: mechanics.TermGraph,
    *,
    storage_ids: tuple[str, ...],
    packet_seed: int,
) -> SourceDeletedPacket:
    constructors = [
        ConstructorRecord(
            item.identifier,
            item.result_type,
            item.argument_types,
        )
        for item in system.constructors
    ]
    random.Random(packet_seed ^ 0xC01157).shuffle(constructors)
    rules = []
    for rule in system.rules:
        variable_types: dict[str, str] = {}
        _collect_pattern_variables(rule.lhs, variable_types)
        rules.append(
            RuleRecord(
                identifier=rule.identifier,
                lhs=_pattern_to_record(system, rule.lhs),
                rhs=(
                    None
                    if rule.rhs is None
                    else _rhs_to_record(system, rule.rhs, variable_types)
                ),
            )
        )
    packet = SourceDeletedPacket(
        constructors=tuple(constructors),
        rules=tuple(rules),
        graph=_graph_to_record(
            graph,
            storage_ids,
            shuffle_seed=packet_seed ^ 0x6A4F,
        ),
    )
    validate_source_deleted_packet(packet)
    return packet


def _record_to_pattern(term: RuleTermRecord) -> mechanics.Pattern:
    if term.kind == "variable":
        return mechanics.PatternVariable(str(term.variable_id), term.type_id)
    return mechanics.PatternConstructor(
        str(term.constructor_id),
        tuple(_record_to_pattern(child) for child in term.children),
    )


def _record_to_rhs(term: RuleTermRecord) -> mechanics.RhsExpression:
    if term.kind == "variable":
        return mechanics.RhsVariable(str(term.variable_id))
    return mechanics.RhsConstructor(
        str(term.constructor_id),
        tuple(_record_to_rhs(child) for child in term.children),
    )


def _packet_to_mechanics(
    packet: SourceDeletedPacket,
) -> tuple[mechanics.RewriteSystem, mechanics.TermGraph, tuple[str, ...]]:
    """Reconstruct mechanics only in the offline generator/assessor process."""

    validate_source_deleted_packet(packet)
    constructors = tuple(
        mechanics.ConstructorSpec(
            item.identifier,
            item.result_type,
            item.argument_types,
        )
        for item in packet.constructors
    )
    rules = tuple(
        mechanics.RewriteRule(
            item.identifier,
            _record_to_pattern(item.lhs),
            None if item.rhs is None else _record_to_rhs(item.rhs),
        )
        for item in packet.rules
    )
    system = mechanics.RewriteSystem(constructors, rules)
    storage_ids = packet.graph.reservoir
    slot_by_storage = {
        storage_id: index for index, storage_id in enumerate(storage_ids)
    }
    nodes = []
    for item in packet.graph.nodes:
        if item.kind == "variable":
            nodes.append(
                mechanics.TermNode.free_variable(
                    slot_by_storage[item.storage_id],
                    item.type_id,
                    str(item.variable_id),
                )
            )
        else:
            nodes.append(
                mechanics.TermNode.constructor(
                    slot_by_storage[item.storage_id],
                    item.type_id,
                    str(item.constructor_id),
                    tuple(slot_by_storage[child] for child in item.children),
                )
            )
    graph = mechanics.TermGraph(
        capacity=len(storage_ids),
        root=(
            None if packet.graph.root is None else slot_by_storage[packet.graph.root]
        ),
        nodes=tuple(nodes),
    )
    mechanics.validate_graph(system, graph)
    return system, graph, storage_ids


def _expected_record_from_packet(
    packet: SourceDeletedPacket,
) -> ExpectedTransitionRecord:
    system, graph, storage_ids = _packet_to_mechanics(packet)
    actions = []
    for reduction in mechanics.legal_reductions(system, graph):
        rule = system.rule_map()[reduction.rule_id]
        bindings: dict[str, int] = {}
        if not mechanics._match_pattern(  # noqa: SLF001
            system,
            graph,
            rule.lhs,
            reduction.target_slot,
            bindings,
        ):
            raise NeuralTcrrBoardError("legal mechanics action lost its binding")
        successor = mechanics.apply_reduction(system, graph, reduction)
        actions.append(
            ExpectedTransition(
                rule_id=reduction.rule_id,
                occurrence_path=reduction.target_path,
                target_storage_id=storage_ids[reduction.target_slot],
                bindings=tuple(
                    BindingRecord(variable, storage_ids[slot])
                    for variable, slot in sorted(bindings.items())
                ),
                successor=_graph_to_record(
                    successor,
                    storage_ids,
                    shuffle_seed=None,
                ),
            )
        )
    return ExpectedTransitionRecord(
        packet_sha256=packet_sha256(packet),
        transitions=tuple(actions),
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


def _oracle_agreement(packet: SourceDeletedPacket) -> OracleAgreementRecord:
    system, graph, _storage_ids = _packet_to_mechanics(packet)
    production = mechanics.ProductionRewriteStateOracle().enumerate(system, graph)
    reference = mechanics.IndependentNestedReferenceOracle().enumerate(
        system,
        graph,
    )
    production_payload = _oracle_payload(production)
    reference_payload = _oracle_payload(reference)
    exact = production_payload == reference_payload
    if not exact:
        raise NeuralTcrrBoardError(
            "production and independent reference oracles disagree"
        )
    return OracleAgreementRecord(
        packet_sha256=packet_sha256(packet),
        production_sha256=_sha256(_canonical_json(production_payload)),
        independent_reference_sha256=_sha256(_canonical_json(reference_payload)),
        state_count=production.states_explored,
        transition_count=production.transitions_explored,
        normal_form_count=len(production.normal_forms),
        cyclic_component_count=len(production.cyclic_sccs),
        exact_agreement=True,
    )


def _graph_occurrences(graph: GraphRecord) -> tuple[tuple[tuple[int, ...], str], ...]:
    if graph.root is None:
        return ()
    nodes = {node.storage_id: node for node in graph.nodes}
    output: list[tuple[tuple[int, ...], str]] = []

    def visit(storage_id: str, path: tuple[int, ...]) -> None:
        output.append((path, storage_id))
        for child_index, child in enumerate(nodes[storage_id].children):
            visit(child, (*path, child_index))

    visit(graph.root, ())
    return tuple(output)


def _graph_subterm_key(graph: GraphRecord, storage_id: str) -> tuple[object, ...]:
    nodes = {node.storage_id: node for node in graph.nodes}
    memo: dict[str, tuple[object, ...]] = {}

    def visit(active: str) -> tuple[object, ...]:
        if active in memo:
            return memo[active]
        node = nodes[active]
        if node.kind == "variable":
            result: tuple[object, ...] = (
                "variable",
                node.type_id,
                node.variable_id,
            )
        else:
            result = (
                "constructor",
                node.type_id,
                node.constructor_id,
                tuple(visit(child) for child in node.children),
            )
        memo[active] = result
        return result

    return visit(storage_id)


def _packet_pattern_matches(
    packet: SourceDeletedPacket,
    pattern: RuleTermRecord,
    storage_id: str,
    bindings: dict[str, str],
) -> bool:
    nodes = {node.storage_id: node for node in packet.graph.nodes}
    node = nodes[storage_id]
    if pattern.kind == "variable":
        if node.type_id != pattern.type_id or pattern.variable_id is None:
            return False
        previous = bindings.get(pattern.variable_id)
        if previous is None:
            bindings[pattern.variable_id] = storage_id
            return True
        return _graph_subterm_key(packet.graph, previous) == _graph_subterm_key(
            packet.graph,
            storage_id,
        )
    if node.constructor_id != pattern.constructor_id:
        return False
    if len(node.children) != len(pattern.children):
        return False
    return all(
        _packet_pattern_matches(packet, child_pattern, child_storage, bindings)
        for child_pattern, child_storage in zip(
            pattern.children,
            node.children,
            strict=True,
        )
    )


@dataclass
class _ColoredGraph:
    labels: list[str]
    edges: list[tuple[int, str, int]]

    def vertex(self, label: str) -> int:
        index = len(self.labels)
        self.labels.append(label)
        return index

    def edge(self, source: int, label: str, target: int) -> None:
        self.edges.append((source, label, target))


def _reachable_storage(graph: GraphRecord, root: str | None) -> set[str]:
    if root is None:
        return set()
    nodes = {node.storage_id: node for node in graph.nodes}
    reached: set[str] = set()

    def visit(storage_id: str) -> None:
        if storage_id in reached:
            return
        reached.add(storage_id)
        for child in nodes[storage_id].children:
            visit(child)

    visit(root)
    return reached


def _collect_term_symbols(
    term: RuleTermRecord | None,
    constructors: set[str],
    types: set[str],
) -> None:
    if term is None:
        return
    types.add(term.type_id)
    if term.constructor_id is not None:
        constructors.add(term.constructor_id)
    for child in term.children:
        _collect_term_symbols(child, constructors, types)


def _build_semantic_colored_graph(
    packet: SourceDeletedPacket,
    *,
    selected_rules: tuple[RuleRecord, ...],
    graph_views: tuple[GraphRecord, ...],
    graph_roots: tuple[str | None, ...],
    include_capacity: bool,
    ordered_rules: bool,
) -> _ColoredGraph:
    if len(graph_views) != len(graph_roots):
        raise NeuralTcrrBoardError("graph views and roots must align")
    constructor_ids: set[str] = set()
    type_ids: set[str] = set()
    for rule in selected_rules:
        _collect_term_symbols(rule.lhs, constructor_ids, type_ids)
        _collect_term_symbols(rule.rhs, constructor_ids, type_ids)
    for graph, root in zip(graph_views, graph_roots, strict=True):
        reached = _reachable_storage(graph, root)
        for node in graph.nodes:
            if node.storage_id not in reached:
                continue
            type_ids.add(node.type_id)
            if node.constructor_id is not None:
                constructor_ids.add(node.constructor_id)
    constructor_records = {
        item.identifier: item
        for item in packet.constructors
        if item.identifier in constructor_ids
    }
    if set(constructor_records) != constructor_ids:
        raise NeuralTcrrBoardError("canonical graph lost a constructor declaration")
    for item in constructor_records.values():
        type_ids.add(item.result_type)
        type_ids.update(item.argument_types)

    colored = _ColoredGraph([], [])
    type_vertices = {
        identifier: colored.vertex("type") for identifier in sorted(type_ids)
    }
    constructor_vertices = {
        identifier: colored.vertex(
            f"constructor|arity={len(constructor_records[identifier].argument_types)}"
        )
        for identifier in sorted(constructor_ids)
    }
    for identifier, vertex in constructor_vertices.items():
        declaration = constructor_records[identifier]
        colored.edge(vertex, "result_type", type_vertices[declaration.result_type])
        for child_index, type_id in enumerate(declaration.argument_types):
            colored.edge(
                vertex,
                f"argument_type:{child_index}",
                type_vertices[type_id],
            )

    for rule_index, rule in enumerate(selected_rules):
        rule_label = f"rule:{rule_index}" if ordered_rules else "rule"
        rule_vertex = colored.vertex(rule_label)
        binders: dict[str, int] = {}

        def add_term(term: RuleTermRecord, side: str) -> int:
            if term.kind == "variable":
                vertex = colored.vertex(f"{side}_variable")
                if term.variable_id is None:
                    raise NeuralTcrrBoardError("canonical binder is absent")
                binder = binders.get(term.variable_id)
                if binder is None:
                    binder = colored.vertex("binder")
                    binders[term.variable_id] = binder
                colored.edge(vertex, "binder", binder)
            else:
                vertex = colored.vertex(
                    f"{side}_constructor|arity={len(term.children)}"
                )
                if term.constructor_id is None:
                    raise NeuralTcrrBoardError("canonical constructor is absent")
                colored.edge(
                    vertex,
                    "constructor",
                    constructor_vertices[term.constructor_id],
                )
            colored.edge(vertex, "type", type_vertices[term.type_id])
            for child_index, child in enumerate(term.children):
                colored.edge(
                    vertex,
                    f"child:{child_index}",
                    add_term(child, side),
                )
            return vertex

        colored.edge(rule_vertex, "lhs", add_term(rule.lhs, "lhs"))
        if rule.rhs is None:
            colored.edge(rule_vertex, "rhs", colored.vertex("delete"))
        else:
            colored.edge(rule_vertex, "rhs", add_term(rule.rhs, "rhs"))

    for graph_index, (graph, root) in enumerate(
        zip(graph_views, graph_roots, strict=True)
    ):
        graph_marker = colored.vertex(f"graph:{graph_index}")
        if root is None:
            colored.edge(graph_marker, "root", colored.vertex("empty_root"))
            continue
        reached = _reachable_storage(graph, root)
        records = {
            item.storage_id: item for item in graph.nodes if item.storage_id in reached
        }
        graph_vertices = {}
        graph_variable_vertices: dict[str, int] = {}
        for storage_id, node in sorted(records.items()):
            if node.kind == "variable":
                vertex = colored.vertex("graph_variable")
                if node.variable_id is None:
                    raise NeuralTcrrBoardError("graph variable is absent")
                variable_vertex = graph_variable_vertices.get(node.variable_id)
                if variable_vertex is None:
                    variable_vertex = colored.vertex("graph_variable_identity")
                    graph_variable_vertices[node.variable_id] = variable_vertex
                colored.edge(vertex, "variable", variable_vertex)
            else:
                vertex = colored.vertex(f"graph_constructor|arity={len(node.children)}")
                if node.constructor_id is None:
                    raise NeuralTcrrBoardError("graph constructor is absent")
                colored.edge(
                    vertex,
                    "constructor",
                    constructor_vertices[node.constructor_id],
                )
            colored.edge(vertex, "type", type_vertices[node.type_id])
            graph_vertices[storage_id] = vertex
        for storage_id, node in records.items():
            for child_index, child in enumerate(node.children):
                colored.edge(
                    graph_vertices[storage_id],
                    f"child:{child_index}",
                    graph_vertices[child],
                )
        colored.edge(graph_marker, "root", graph_vertices[root])
        if include_capacity:
            free_count = len(graph.reservoir) - len(records)
            capacity_vertex = colored.vertex(
                f"capacity={len(graph.reservoir)}|free={free_count}"
            )
            colored.edge(graph_marker, "capacity", capacity_vertex)
    return colored


def _compress_signatures(signatures: Sequence[object]) -> tuple[int, ...]:
    serialized = [_canonical_json(item) for item in signatures]
    classes = {value: index for index, value in enumerate(sorted(set(serialized)))}
    return tuple(classes[value] for value in serialized)


def _refine_colors(
    colored: _ColoredGraph,
    colors: tuple[int, ...],
) -> tuple[int, ...]:
    incoming: dict[int, list[tuple[str, int]]] = defaultdict(list)
    outgoing: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for source, label, target in colored.edges:
        outgoing[source].append((label, target))
        incoming[target].append((label, source))
    active = colors
    while True:
        signatures = []
        for vertex in range(len(colored.labels)):
            signatures.append(
                (
                    active[vertex],
                    tuple(
                        sorted(
                            (label, active[target])
                            for label, target in outgoing[vertex]
                        )
                    ),
                    tuple(
                        sorted(
                            (label, active[source])
                            for label, source in incoming[vertex]
                        )
                    ),
                )
            )
        refined = _compress_signatures(signatures)
        if _same_partition(active, refined):
            return refined
        active = refined


def _same_partition(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> bool:
    if len(left) != len(right):
        return False
    left_to_right: dict[int, int] = {}
    right_to_left: dict[int, int] = {}
    for left_color, right_color in zip(left, right, strict=True):
        previous_right = left_to_right.setdefault(left_color, right_color)
        previous_left = right_to_left.setdefault(right_color, left_color)
        if previous_right != right_color or previous_left != left_color:
            return False
    return True


def _canonical_colored_graph(
    colored: _ColoredGraph,
    *,
    maximum_backtrack_states: int = MAX_CANONICAL_BACKTRACK_STATES,
) -> str:
    """Canonicalize by color refinement and bounded exact individualization."""

    if maximum_backtrack_states <= 0:
        raise NeuralTcrrBoardError("canonical backtrack bound must be positive")
    initial = _compress_signatures(tuple(colored.labels))
    explored = 0
    best: str | None = None

    def leaf_payload(colors: tuple[int, ...]) -> str:
        order = sorted(range(len(colors)), key=lambda vertex: colors[vertex])
        canonical_index = {vertex: index for index, vertex in enumerate(order)}
        return _canonical_json(
            {
                "vertices": [colored.labels[vertex] for vertex in order],
                "edges": sorted(
                    (
                        canonical_index[source],
                        label,
                        canonical_index[target],
                    )
                    for source, label, target in colored.edges
                ),
            }
        )

    def search(colors: tuple[int, ...]) -> None:
        nonlocal explored, best
        explored += 1
        if explored > maximum_backtrack_states:
            raise NeuralTcrrBoardError(
                "canonical colored-graph backtrack bound exceeded"
            )
        refined = _refine_colors(colored, colors)
        cells: dict[int, list[int]] = defaultdict(list)
        for vertex, color in enumerate(refined):
            cells[color].append(vertex)
        ambiguous = [
            (len(vertices), color, vertices)
            for color, vertices in cells.items()
            if len(vertices) > 1
        ]
        if not ambiguous:
            candidate = leaf_payload(refined)
            if best is None or candidate < best:
                best = candidate
            return
        _size, _color, vertices = min(ambiguous)
        individualized_color = max(refined) + 1
        for vertex in vertices:
            branch = list(refined)
            branch[vertex] = individualized_color
            search(tuple(branch))

    search(initial)
    if best is None:
        raise NeuralTcrrBoardError("canonical colored graph has no labeling")
    return best


def _canonical_semantic_view(
    packet: SourceDeletedPacket,
    *,
    selected_rules: tuple[RuleRecord, ...],
    graph_views: tuple[GraphRecord, ...],
    graph_roots: tuple[str | None, ...],
    include_capacity: bool,
    ordered_rules: bool,
) -> str:
    colored = _build_semantic_colored_graph(
        packet,
        selected_rules=selected_rules,
        graph_views=graph_views,
        graph_roots=graph_roots,
        include_capacity=include_capacity,
        ordered_rules=ordered_rules,
    )
    return _canonical_colored_graph(colored)


def _graph_record_from_mechanics(
    graph: mechanics.TermGraph,
    storage_ids: tuple[str, ...],
) -> GraphRecord:
    return _graph_to_record(graph, storage_ids, shuffle_seed=None)


def packet_fingerprints(packet: SourceDeletedPacket) -> PacketFingerprints:
    """Compute all split fingerprints, including rule pairs and compositions."""

    validate_source_deleted_packet(packet)
    exact = packet_sha256(packet)
    isomorphic = _sha256(
        _canonical_semantic_view(
            packet,
            selected_rules=packet.rules,
            graph_views=(packet.graph,),
            graph_roots=(packet.graph.root,),
            include_capacity=True,
            ordered_rules=False,
        )
    )

    rule_windows = set()
    for rule in packet.rules:
        for _path, storage_id in _graph_occurrences(packet.graph):
            if not _packet_pattern_matches(packet, rule.lhs, storage_id, {}):
                continue
            rule_windows.add(
                _sha256(
                    _canonical_semantic_view(
                        packet,
                        selected_rules=(rule,),
                        graph_views=(packet.graph,),
                        graph_roots=(storage_id,),
                        include_capacity=False,
                        ordered_rules=False,
                    )
                )
            )
    if not rule_windows:
        rule_windows.add(
            _sha256(
                _canonical_semantic_view(
                    packet,
                    selected_rules=packet.rules,
                    graph_views=(packet.graph,),
                    graph_roots=(packet.graph.root,),
                    include_capacity=False,
                    ordered_rules=False,
                )
            )
        )

    rule_pairs = []
    for left_index, left_rule in enumerate(packet.rules):
        for right_rule in packet.rules[left_index:]:
            rule_pairs.append(
                _sha256(
                    _canonical_semantic_view(
                        packet,
                        selected_rules=(left_rule, right_rule),
                        graph_views=(),
                        graph_roots=(),
                        include_capacity=False,
                        ordered_rules=False,
                    )
                )
            )

    system, graph, storage_ids = _packet_to_mechanics(packet)
    rule_map = {rule.identifier: rule for rule in packet.rules}
    compositions = []
    for first in mechanics.legal_reductions(system, graph):
        intermediate = mechanics.apply_reduction(system, graph, first)
        for second in mechanics.legal_reductions(system, intermediate):
            terminal = mechanics.apply_reduction(system, intermediate, second)
            intermediate_record = _graph_record_from_mechanics(
                intermediate,
                storage_ids,
            )
            terminal_record = _graph_record_from_mechanics(terminal, storage_ids)
            compositions.append(
                _sha256(
                    _canonical_semantic_view(
                        packet,
                        selected_rules=(
                            rule_map[first.rule_id],
                            rule_map[second.rule_id],
                        ),
                        graph_views=(
                            packet.graph,
                            intermediate_record,
                            terminal_record,
                        ),
                        graph_roots=(
                            packet.graph.root,
                            intermediate_record.root,
                            terminal_record.root,
                        ),
                        include_capacity=True,
                        ordered_rules=True,
                    )
                )
            )
    return PacketFingerprints(
        packet_sha256=exact,
        exact_sha256=exact,
        isomorphic_sha256=isomorphic,
        normalized_rule_windows=tuple(sorted(rule_windows)),
        normalized_rule_pairs=tuple(sorted(rule_pairs)),
        reachable_two_rule_compositions=tuple(sorted(compositions)),
    )


def validate_split_isolation(
    assignments: tuple[SplitAssignment, ...],
    fingerprints: tuple[PacketFingerprints, ...],
) -> None:
    """Reject every exact or normalized overlap across train/development."""

    assignment_keys = [item.packet_sha256 for item in assignments]
    fingerprint_keys = [item.packet_sha256 for item in fingerprints]
    if len(assignment_keys) != len(set(assignment_keys)):
        raise NeuralTcrrBoardError("split ledger keys must be unique")
    if len(fingerprint_keys) != len(set(fingerprint_keys)):
        raise NeuralTcrrBoardError("fingerprint ledger keys must be unique")
    split_by_packet = {item.packet_sha256: item.partition for item in assignments}
    fingerprint_by_packet = {item.packet_sha256: item for item in fingerprints}
    if set(split_by_packet) != set(fingerprint_by_packet):
        raise NeuralTcrrBoardError("split and fingerprint ledgers do not align")
    partitions: dict[Partition, dict[str, set[str]]] = {
        "local_transition_train": defaultdict(set),
        "local_transition_development": defaultdict(set),
    }
    for packet_digest, partition in split_by_packet.items():
        item = fingerprint_by_packet[packet_digest]
        partitions[partition]["exact"].add(item.exact_sha256)
        partitions[partition]["isomorphic"].add(item.isomorphic_sha256)
        partitions[partition]["rule_window"].update(item.normalized_rule_windows)
        partitions[partition]["rule_pair"].update(item.normalized_rule_pairs)
        partitions[partition]["composition"].update(
            item.reachable_two_rule_compositions
        )
    train = partitions["local_transition_train"]
    development = partitions["local_transition_development"]
    for category in (
        "exact",
        "isomorphic",
        "rule_window",
        "rule_pair",
        "composition",
    ):
        overlap = train[category] & development[category]
        if overlap:
            raise NeuralTcrrBoardError(
                f"cross-split {category} fingerprint leakage: {sorted(overlap)}"
            )


class _EpisodeFactory:
    """Internal semantic builder whose aliases never enter a packet."""

    def __init__(self, seed: int):
        self.seed = seed
        self.types: dict[str, str] = {}
        self.constructors: dict[str, mechanics.ConstructorSpec] = {}
        self._type_index = 0
        self._constructor_index = 0
        self._rule_index = 0
        self._variable_index = 0
        self._free_variable_index = 0

    def type_id(self, alias: str) -> str:
        if alias not in self.types:
            self.types[alias] = _opaque(
                self.seed,
                "type",
                self._type_index,
            )
            self._type_index += 1
        return self.types[alias]

    def constructor(
        self,
        alias: str,
        result_type: str,
        argument_types: tuple[str, ...] = (),
    ) -> mechanics.ConstructorSpec:
        spec = mechanics.ConstructorSpec(
            _opaque(self.seed, "constructor", self._constructor_index),
            self.type_id(result_type),
            tuple(self.type_id(item) for item in argument_types),
        )
        self._constructor_index += 1
        self.constructors[alias] = spec
        return spec

    def variable(self, type_alias: str) -> mechanics.PatternVariable:
        variable = mechanics.PatternVariable(
            _opaque(self.seed, "variable", self._variable_index),
            self.type_id(type_alias),
        )
        self._variable_index += 1
        return variable

    def free_variable(self, type_alias: str) -> mechanics.GroundTerm:
        variable = mechanics.GroundTerm.free_variable(
            self.type_id(type_alias),
            _opaque(
                self.seed,
                "graph_variable",
                self._free_variable_index,
            ),
        )
        self._free_variable_index += 1
        return variable

    def pattern(
        self,
        alias: str,
        *children: mechanics.Pattern,
    ) -> mechanics.PatternConstructor:
        return mechanics.PatternConstructor(
            self.constructors[alias].identifier,
            tuple(children),
        )

    def rhs(
        self,
        alias: str,
        *children: mechanics.RhsExpression,
    ) -> mechanics.RhsConstructor:
        return mechanics.RhsConstructor(
            self.constructors[alias].identifier,
            tuple(children),
        )

    def leaf(self, alias: str) -> mechanics.GroundTerm:
        return mechanics.GroundTerm.constructor(self.constructors[alias])

    def term(
        self,
        alias: str,
        *children: mechanics.GroundTerm,
    ) -> mechanics.GroundTerm:
        return mechanics.GroundTerm.constructor(
            self.constructors[alias],
            *children,
        )

    def rule(
        self,
        lhs: mechanics.Pattern,
        rhs: mechanics.RhsExpression | None,
    ) -> mechanics.RewriteRule:
        rule = mechanics.RewriteRule(
            _opaque(self.seed, "rule", self._rule_index),
            lhs,
            rhs,
        )
        self._rule_index += 1
        return rule

    def system(
        self,
        rules: tuple[mechanics.RewriteRule, ...],
    ) -> mechanics.RewriteSystem:
        constructors = list(self.constructors.values())
        random.Random(self.seed ^ 0x51A7).shuffle(constructors)
        return mechanics.RewriteSystem(tuple(constructors), rules)


@dataclass(frozen=True)
class _GeneratedExample:
    system: mechanics.RewriteSystem
    graph: mechanics.TermGraph
    storage_ids: tuple[str, ...]
    packet_seed: int


def _pack_example(
    factory: _EpisodeFactory,
    system: mechanics.RewriteSystem,
    term: mechanics.GroundTerm,
    *,
    capacity: int,
) -> _GeneratedExample:
    permutation = list(range(capacity))
    random.Random(factory.seed ^ 0x5702).shuffle(permutation)
    graph = mechanics.pack_ground_term(
        system,
        term,
        capacity,
        slot_permutation=tuple(permutation),
    )
    storage_ids = [_opaque(factory.seed, "storage", index) for index in range(capacity)]
    random.Random(factory.seed ^ 0x570A).shuffle(storage_ids)
    return _GeneratedExample(
        system=system,
        graph=graph,
        storage_ids=tuple(storage_ids),
        packet_seed=factory.seed,
    )


def _packet_from_generated(example: _GeneratedExample) -> SourceDeletedPacket:
    return _packet_from_mechanics(
        example.system,
        example.graph,
        storage_ids=example.storage_ids,
        packet_seed=example.packet_seed,
    )


def _shared_cancellation_base(seed: int) -> SourceDeletedPacket:
    factory = _EpisodeFactory(seed)
    factory.constructor("pair", "value", ("value", "value"))
    factory.constructor("drop", "value", ("value",))
    factory.constructor("payload", "value")
    variable = factory.variable("value")
    rule = factory.rule(
        factory.pattern("drop", variable),
        mechanics.RhsVariable(variable.name),
    )
    system = factory.system((rule,))
    shared = factory.term("drop", factory.leaf("payload"))
    return _packet_from_generated(
        _pack_example(
            factory,
            system,
            factory.term("pair", shared, shared),
            capacity=8,
        )
    )


def _capacity_base(seed: int) -> SourceDeletedPacket:
    factory = _EpisodeFactory(seed)
    factory.constructor("context", "value", ("value",))
    factory.constructor("reservoir_root", "value", ("value", "value", "value"))
    factory.constructor("source", "value")
    factory.constructor("left", "value")
    factory.constructor("right", "value")
    factory.constructor("pair", "value", ("value", "value"))
    rule = factory.rule(
        factory.pattern("source"),
        factory.rhs("pair", factory.rhs("left"), factory.rhs("right")),
    )
    system = factory.system((rule,))
    term = factory.term(
        "reservoir_root",
        _wrap_term(factory, factory.leaf("left"), 4),
        _wrap_term(factory, factory.leaf("right"), 3),
        _wrap_term(factory, factory.leaf("source"), 3),
    )
    return _packet_from_generated(_pack_example(factory, system, term, capacity=16))


def _wrap_term(
    factory: _EpisodeFactory,
    child: mechanics.GroundTerm,
    depth: int,
) -> mechanics.GroundTerm:
    result = child
    for _ in range(depth):
        result = factory.term("context", result)
    return result


def _repeated_variable_base(seed: int) -> SourceDeletedPacket:
    factory = _EpisodeFactory(seed)
    factory.constructor("same", "value", ("value", "value"))
    factory.constructor("hit", "value")
    factory.constructor("a", "value")
    factory.constructor("b", "value")
    variable = factory.variable("value")
    rule = factory.rule(
        factory.pattern("same", variable, variable),
        factory.rhs("hit"),
    )
    system = factory.system((rule,))
    term = factory.term("same", factory.leaf("a"), factory.leaf("a"))
    return _packet_from_generated(_pack_example(factory, system, term, capacity=7))


def _nested_match_base(seed: int) -> SourceDeletedPacket:
    factory = _EpisodeFactory(seed)
    factory.constructor("payload", "value")
    factory.constructor("inner", "value", ("value",))
    factory.constructor("other", "value", ("value",))
    factory.constructor("outer", "value", ("value",))
    factory.constructor("done", "value", ("value",))
    variable = factory.variable("value")
    rule = factory.rule(
        factory.pattern("outer", factory.pattern("inner", variable)),
        factory.rhs("done", mechanics.RhsVariable(variable.name)),
    )
    system = factory.system((rule,))
    term = factory.term(
        "outer",
        factory.term("inner", factory.leaf("payload")),
    )
    return _packet_from_generated(_pack_example(factory, system, term, capacity=8))


def _type_match_base(seed: int) -> SourceDeletedPacket:
    factory = _EpisodeFactory(seed)
    factory.constructor("box_constant", "box")
    factory.constructor("value_constant", "value")
    factory.constructor("distractor_lhs", "value")
    factory.constructor("distractor_rhs", "value")
    variable = factory.variable("box")
    rules = (
        factory.rule(variable, factory.rhs("box_constant")),
        factory.rule(
            factory.pattern("distractor_lhs"),
            factory.rhs("distractor_rhs"),
        ),
    )
    system = factory.system(rules)
    return _packet_from_generated(
        _pack_example(
            factory,
            system,
            factory.free_variable("box"),
            capacity=5,
        )
    )


def _rhs_pointer_base(seed: int) -> SourceDeletedPacket:
    factory = _EpisodeFactory(seed)
    factory.constructor("pick", "value", ("value", "value"))
    factory.constructor("output", "value", ("value", "value"))
    factory.constructor("left", "value")
    factory.constructor("right", "value")
    left = factory.variable("value")
    right = factory.variable("value")
    rule = factory.rule(
        factory.pattern("pick", left, right),
        factory.rhs(
            "output",
            mechanics.RhsVariable(left.name),
            mechanics.RhsVariable(right.name),
        ),
    )
    system = factory.system((rule,))
    term = factory.term("pick", factory.leaf("left"), factory.leaf("right"))
    return _packet_from_generated(_pack_example(factory, system, term, capacity=7))


def _multi_rule_reindex_base(seed: int) -> SourceDeletedPacket:
    factory = _EpisodeFactory(seed)
    factory.constructor("left", "value")
    factory.constructor("right", "value")
    factory.constructor("left_nf", "value")
    factory.constructor("absent", "value")
    factory.constructor("pair", "box", ("value", "value"))
    left_variable = factory.variable("value")
    right_variable = factory.variable("value")
    rules = (
        factory.rule(factory.pattern("left"), factory.rhs("left_nf")),
        factory.rule(
            factory.pattern("pair", left_variable, right_variable),
            None,
        ),
        factory.rule(factory.pattern("absent"), factory.rhs("left_nf")),
    )
    system = factory.system(rules)
    term = factory.term("pair", factory.leaf("left"), factory.leaf("right"))
    return _packet_from_generated(_pack_example(factory, system, term, capacity=8))


def _development_nested_composition(seed: int) -> SourceDeletedPacket:
    factory = _EpisodeFactory(seed)
    factory.constructor("payload", "value")
    factory.constructor("side", "value")
    factory.constructor("inner", "value", ("value",))
    factory.constructor("branch", "value", ("value", "value"))
    factory.constructor("outer", "value", ("value",))
    factory.constructor("done", "value", ("value", "value"))
    payload = factory.variable("value")
    side = factory.variable("value")
    rule = factory.rule(
        factory.pattern(
            "outer",
            factory.pattern(
                "branch",
                factory.pattern("inner", payload),
                side,
            ),
        ),
        factory.rhs(
            "done",
            mechanics.RhsVariable(side.name),
            mechanics.RhsVariable(payload.name),
        ),
    )
    system = factory.system((rule,))
    term = factory.term(
        "outer",
        factory.term(
            "branch",
            factory.term("inner", factory.leaf("payload")),
            factory.leaf("side"),
        ),
    )
    return _packet_from_generated(_pack_example(factory, system, term, capacity=10))


def _development_rotation(seed: int) -> SourceDeletedPacket:
    factory = _EpisodeFactory(seed)
    factory.constructor("pair", "value", ("value", "value"))
    factory.constructor("left", "value")
    factory.constructor("right", "value")
    left = factory.variable("value")
    right = factory.variable("value")
    rule = factory.rule(
        factory.pattern("pair", left, right),
        factory.rhs(
            "pair",
            mechanics.RhsVariable(right.name),
            mechanics.RhsVariable(left.name),
        ),
    )
    system = factory.system((rule,))
    term = factory.term("pair", factory.leaf("left"), factory.leaf("right"))
    return _packet_from_generated(_pack_example(factory, system, term, capacity=7))


def _development_nested_rhs(seed: int) -> SourceDeletedPacket:
    factory = _EpisodeFactory(seed)
    factory.constructor("source", "value", ("value",))
    factory.constructor("inner", "value", ("value",))
    factory.constructor("outer", "value", ("value",))
    factory.constructor("payload", "value")
    variable = factory.variable("value")
    rule = factory.rule(
        factory.pattern("source", variable),
        factory.rhs(
            "outer",
            factory.rhs(
                "inner",
                mechanics.RhsVariable(variable.name),
            ),
        ),
    )
    system = factory.system((rule,))
    term = factory.term("source", factory.leaf("payload"))
    return _packet_from_generated(_pack_example(factory, system, term, capacity=8))


def _development_repeated_ternary(seed: int) -> SourceDeletedPacket:
    factory = _EpisodeFactory(seed)
    factory.constructor("same", "value", ("value", "value", "value"))
    factory.constructor("output", "value", ("value", "value"))
    factory.constructor("a", "value")
    factory.constructor("b", "value")
    left = factory.variable("value")
    right = factory.variable("value")
    rule = factory.rule(
        factory.pattern("same", left, left, right),
        factory.rhs(
            "output",
            mechanics.RhsVariable(right.name),
            mechanics.RhsVariable(left.name),
        ),
    )
    system = factory.system((rule,))
    term = factory.term(
        "same",
        factory.leaf("a"),
        factory.leaf("a"),
        factory.leaf("b"),
    )
    return _packet_from_generated(_pack_example(factory, system, term, capacity=9))


def _development_unary_deletion(seed: int) -> SourceDeletedPacket:
    factory = _EpisodeFactory(seed)
    factory.constructor("erase", "value", ("value",))
    factory.constructor("payload", "value")
    variable = factory.variable("value")
    rule = factory.rule(factory.pattern("erase", variable), None)
    system = factory.system((rule,))
    term = factory.term("erase", factory.leaf("payload"))
    return _packet_from_generated(_pack_example(factory, system, term, capacity=5))


def _development_heterogeneous(seed: int) -> SourceDeletedPacket:
    factory = _EpisodeFactory(seed)
    factory.constructor("atom", "value")
    factory.constructor("box", "box", ("value",))
    factory.constructor("sealed", "box", ("value",))
    factory.constructor("outer", "box", ("box",))
    variable = factory.variable("value")
    rule = factory.rule(
        factory.pattern("outer", factory.pattern("box", variable)),
        factory.rhs(
            "outer",
            factory.rhs(
                "sealed",
                mechanics.RhsVariable(variable.name),
            ),
        ),
    )
    system = factory.system((rule,))
    term = factory.term(
        "outer",
        factory.term("box", factory.leaf("atom")),
    )
    return _packet_from_generated(_pack_example(factory, system, term, capacity=8))


def _map_rule_term(
    term: RuleTermRecord | None,
    *,
    constructor_map: Mapping[str, str],
    type_map: Mapping[str, str],
) -> RuleTermRecord | None:
    if term is None:
        return None
    return RuleTermRecord(
        kind=term.kind,
        type_id=type_map.get(term.type_id, term.type_id),
        constructor_id=(
            None
            if term.constructor_id is None
            else constructor_map.get(term.constructor_id, term.constructor_id)
        ),
        variable_id=term.variable_id,
        children=tuple(
            child
            for child in (
                _map_rule_term(
                    item,
                    constructor_map=constructor_map,
                    type_map=type_map,
                )
                for item in term.children
            )
            if child is not None
        ),
    )


def _reindex_packet(
    packet: SourceDeletedPacket,
    namespace: ReindexNamespace,
    remap: tuple[IdentifierRemap, ...],
) -> SourceDeletedPacket:
    mapping = {item.old: item.new for item in remap}
    if len(mapping) != len(remap) or len(set(mapping.values())) != len(mapping):
        raise NeuralTcrrBoardError("reindex mapping must be bijective")
    constructor_map = mapping if namespace == "constructor" else {}
    type_map = mapping if namespace == "type" else {}
    rule_map = mapping if namespace == "rule" else {}
    storage_map = mapping if namespace == "storage" else {}
    result = SourceDeletedPacket(
        constructors=tuple(
            ConstructorRecord(
                identifier=constructor_map.get(item.identifier, item.identifier),
                result_type=type_map.get(item.result_type, item.result_type),
                argument_types=tuple(
                    type_map.get(type_id, type_id) for type_id in item.argument_types
                ),
            )
            for item in packet.constructors
        ),
        rules=tuple(
            RuleRecord(
                identifier=rule_map.get(item.identifier, item.identifier),
                lhs=(
                    mapped
                    if (
                        mapped := _map_rule_term(
                            item.lhs,
                            constructor_map=constructor_map,
                            type_map=type_map,
                        )
                    )
                    is not None
                    else item.lhs
                ),
                rhs=_map_rule_term(
                    item.rhs,
                    constructor_map=constructor_map,
                    type_map=type_map,
                ),
            )
            for item in packet.rules
        ),
        graph=GraphRecord(
            reservoir=tuple(
                storage_map.get(item, item) for item in packet.graph.reservoir
            ),
            root=(
                None
                if packet.graph.root is None
                else storage_map.get(packet.graph.root, packet.graph.root)
            ),
            nodes=tuple(
                GraphNodeRecord(
                    storage_id=storage_map.get(item.storage_id, item.storage_id),
                    kind=item.kind,
                    type_id=type_map.get(item.type_id, item.type_id),
                    constructor_id=(
                        None
                        if item.constructor_id is None
                        else constructor_map.get(
                            item.constructor_id,
                            item.constructor_id,
                        )
                    ),
                    variable_id=item.variable_id,
                    children=tuple(
                        storage_map.get(child, child) for child in item.children
                    ),
                )
                for item in packet.graph.nodes
            ),
        ),
    )
    validate_source_deleted_packet(result)
    return result


def _reverse_namespace_remap(
    packet: SourceDeletedPacket,
    namespace: ReindexNamespace,
) -> tuple[IdentifierRemap, ...]:
    if namespace == "constructor":
        identifiers = sorted(item.identifier for item in packet.constructors)
    elif namespace == "type":
        identifiers = sorted(
            {
                type_id
                for item in packet.constructors
                for type_id in (item.result_type, *item.argument_types)
            }
        )
    elif namespace == "rule":
        identifiers = sorted(item.identifier for item in packet.rules)
    else:
        identifiers = sorted(packet.graph.reservoir)
    if len(identifiers) < 2:
        raise NeuralTcrrBoardError("reindex twin needs at least two identifiers")
    return tuple(
        IdentifierRemap(old, new)
        for old, new in zip(identifiers, reversed(identifiers), strict=True)
    )


def _mutate_rhs_pointer(
    packet: SourceDeletedPacket,
    rule_id: str,
) -> SourceDeletedPacket:
    changed = 0
    rules = []
    for rule in packet.rules:
        if rule.identifier != rule_id:
            rules.append(rule)
            continue
        if rule.rhs is None or len(rule.rhs.children) != 2:
            raise NeuralTcrrBoardError("RHS pointer twin requires binary RHS")
        rules.append(
            dataclasses.replace(
                rule,
                rhs=dataclasses.replace(
                    rule.rhs,
                    children=tuple(reversed(rule.rhs.children)),
                ),
            )
        )
        changed += 1
    if changed != 1:
        raise NeuralTcrrBoardError("RHS pointer twin must change exactly one card")
    result = dataclasses.replace(packet, rules=tuple(rules))
    validate_source_deleted_packet(result)
    return result


def _mutate_graph_constructor(
    packet: SourceDeletedPacket,
    storage_id: str,
    constructor_id: str,
) -> SourceDeletedPacket:
    constructors = {item.identifier: item for item in packet.constructors}
    changed = 0
    nodes = []
    for node in packet.graph.nodes:
        if node.storage_id != storage_id:
            nodes.append(node)
            continue
        declaration = constructors[constructor_id]
        if (
            node.kind != "constructor"
            or node.type_id != declaration.result_type
            or len(node.children) != len(declaration.argument_types)
        ):
            raise NeuralTcrrBoardError(
                "constructor mutation violates graph shape or type"
            )
        nodes.append(dataclasses.replace(node, constructor_id=constructor_id))
        changed += 1
    if changed != 1:
        raise NeuralTcrrBoardError("constructor mutation must name one graph node")
    result = dataclasses.replace(
        packet,
        graph=dataclasses.replace(packet.graph, nodes=tuple(nodes)),
    )
    validate_source_deleted_packet(result)
    return result


def _mutate_graph_variable_type(
    packet: SourceDeletedPacket,
    storage_id: str,
    type_id: str,
) -> SourceDeletedPacket:
    changed = 0
    nodes = []
    for node in packet.graph.nodes:
        if node.storage_id != storage_id:
            nodes.append(node)
            continue
        if node.kind != "variable":
            raise NeuralTcrrBoardError("type mismatch twin requires graph variable")
        nodes.append(dataclasses.replace(node, type_id=type_id))
        changed += 1
    if changed != 1:
        raise NeuralTcrrBoardError("type mutation must name one graph variable")
    result = dataclasses.replace(
        packet,
        graph=dataclasses.replace(packet.graph, nodes=tuple(nodes)),
    )
    validate_source_deleted_packet(result)
    return result


def _mutate_capacity(
    packet: SourceDeletedPacket,
    removed_storage_id: str,
) -> SourceDeletedPacket:
    if removed_storage_id not in packet.graph.reservoir:
        raise NeuralTcrrBoardError("capacity mutation names an absent record")
    if removed_storage_id in {node.storage_id for node in packet.graph.nodes}:
        raise NeuralTcrrBoardError("capacity mutation cannot remove occupied storage")
    result = dataclasses.replace(
        packet,
        graph=dataclasses.replace(
            packet.graph,
            reservoir=tuple(
                item for item in packet.graph.reservoir if item != removed_storage_id
            ),
        ),
    )
    validate_source_deleted_packet(result)
    return result


def _apply_twin_mutation(
    packet: SourceDeletedPacket,
    twin: CausalTwinRecord,
) -> SourceDeletedPacket:
    if twin.kind in {
        "constructor_reindex",
        "type_reindex",
        "rule_reindex",
        "storage_reindex",
    }:
        if twin.namespace is None:
            raise NeuralTcrrBoardError("reindex twin lacks namespace")
        return _reindex_packet(packet, twin.namespace, twin.remap)
    if twin.kind == "rhs_pointer":
        return _mutate_rhs_pointer(packet, twin.axis_values[0])
    if twin.kind in {"repeated_variable_equality", "partial_nested_match"}:
        return _mutate_graph_constructor(
            packet,
            twin.axis_values[0],
            twin.axis_values[1],
        )
    if twin.kind == "type_mismatch":
        return _mutate_graph_variable_type(
            packet,
            twin.axis_values[0],
            twin.axis_values[1],
        )
    if twin.kind == "capacity":
        return _mutate_capacity(packet, twin.axis_values[0])
    if twin.kind == "shared_occurrence":
        return packet
    raise NeuralTcrrBoardError(f"unknown twin kind {twin.kind!r}")


def _rule_term_depth(term: RuleTermRecord | None) -> int:
    if term is None:
        return 0
    return 1 + max((_rule_term_depth(child) for child in term.children), default=0)


def _rule_variable_counts(term: RuleTermRecord | None) -> Counter[str]:
    output: Counter[str] = Counter()

    def visit(active: RuleTermRecord | None) -> None:
        if active is None:
            return
        if active.variable_id is not None:
            output[active.variable_id] += 1
        for child in active.children:
            visit(child)

    visit(term)
    return output


def _infer_local_primitives(
    packet: SourceDeletedPacket,
    expected: ExpectedTransitionRecord,
) -> tuple[str, ...]:
    primitives = {"constructor_match"}
    lhs_variables = Counter()
    rhs_variables = Counter()
    rhs_constructor_nodes = 0
    for rule in packet.rules:
        lhs_variables.update(_rule_variable_counts(rule.lhs))
        rhs_variables.update(_rule_variable_counts(rule.rhs))
        if _rule_term_depth(rule.lhs) >= 3:
            primitives.add("nested_match")
        if rule.rhs is None:
            primitives.add("root_deletion")
        if rule.rhs is not None:
            rhs_constructor_nodes += sum(
                1 for item in _walk_rule_terms(rule.rhs) if item.kind == "constructor"
            )
    if lhs_variables:
        primitives.add("typed_variable_binding")
    if any(count > 1 for count in lhs_variables.values()):
        primitives.add("repeated_variable_equality")
    if rhs_variables:
        primitives.add("rhs_pointer_reuse")
    if rhs_constructor_nodes:
        primitives.add("rhs_construction")
    if (
        len(
            {
                type_id
                for item in packet.constructors
                for type_id in (item.result_type, *item.argument_types)
            }
        )
        > 1
    ):
        primitives.add("heterogeneous_typing")
    if any(action.occurrence_path for action in expected.transitions):
        primitives.add("occurrence_specific_update")
    if not expected.transitions:
        primitives.add("no_redex")
    if len(packet.rules) > 1:
        primitives.add("multi_rule_selection")
        legal_rule_ids = {action.rule_id for action in expected.transitions}
        if len(legal_rule_ids) < len(packet.rules):
            primitives.add("distractor_rule_rejection")
    child_pointers = [child for node in packet.graph.nodes for child in node.children]
    if len(child_pointers) != len(set(child_pointers)):
        primitives.add("shared_occurrence")
    return tuple(sorted(primitives))


def _walk_rule_terms(term: RuleTermRecord) -> tuple[RuleTermRecord, ...]:
    output = [term]
    for child in term.children:
        output.extend(_walk_rule_terms(child))
    return tuple(output)


def _validate_primitive_coverage(
    assignments: tuple[SplitAssignment, ...],
    coverage: tuple[PrimitiveCoverageRecord, ...],
) -> None:
    split = {item.packet_sha256: item.partition for item in assignments}
    train_primitives = {
        primitive
        for item in coverage
        if split[item.packet_sha256] == "local_transition_train"
        for primitive in item.primitives
    }
    development_primitives = {
        primitive
        for item in coverage
        if split[item.packet_sha256] == "local_transition_development"
        for primitive in item.primitives
    }
    exempt = {"no_redex"}
    missing = development_primitives - train_primitives - exempt
    if missing:
        raise NeuralTcrrBoardError(
            f"development primitives lack isolated train coverage: {sorted(missing)}"
        )


def _successor_isomorphic_digest(
    packet: SourceDeletedPacket,
    action: ExpectedTransition,
) -> str:
    successor_packet = dataclasses.replace(packet, graph=action.successor)
    return packet_fingerprints(successor_packet).isomorphic_sha256


def _validate_twin_predicate(
    twin: CausalTwinRecord,
    packets: Mapping[str, SourceDeletedPacket],
    expected: Mapping[str, ExpectedTransitionRecord],
    fingerprints: Mapping[str, PacketFingerprints],
) -> None:
    left = packets[twin.left_packet_sha256]
    right = packets[twin.right_packet_sha256]
    reconstructed = _apply_twin_mutation(left, twin)
    if reconstructed != right:
        raise NeuralTcrrBoardError(f"{twin.kind} twin changes more than its named axis")
    left_expected = _expected_record_from_packet(left)
    right_expected = _expected_record_from_packet(right)
    if left_expected != expected[twin.left_packet_sha256]:
        raise NeuralTcrrBoardError("left twin label ledger is stale")
    if right_expected != expected[twin.right_packet_sha256]:
        raise NeuralTcrrBoardError("right twin label ledger is stale")

    if twin.kind in {
        "constructor_reindex",
        "type_reindex",
        "rule_reindex",
        "storage_reindex",
    }:
        left_fingerprint = fingerprints[twin.left_packet_sha256]
        right_fingerprint = fingerprints[twin.right_packet_sha256]
        if left_fingerprint.isomorphic_sha256 != (right_fingerprint.isomorphic_sha256):
            raise NeuralTcrrBoardError(f"{twin.kind} changed packet semantics")
        if left_fingerprint.exact_sha256 == right_fingerprint.exact_sha256:
            raise NeuralTcrrBoardError(f"{twin.kind} did not change packet bytes")
        return

    if twin.kind == "shared_occurrence":
        if twin.left_packet_sha256 != twin.right_packet_sha256:
            raise NeuralTcrrBoardError("shared occurrence must use one base packet")
        if twin.left_transition_index is None or twin.right_transition_index is None:
            raise NeuralTcrrBoardError("shared occurrence lacks action selectors")
        left_action = left_expected.transitions[twin.left_transition_index]
        right_action = left_expected.transitions[twin.right_transition_index]
        if left_action.target_storage_id != right_action.target_storage_id:
            raise NeuralTcrrBoardError("shared occurrence lost its shared target")
        if left_action.occurrence_path == right_action.occurrence_path:
            raise NeuralTcrrBoardError("shared occurrence paths collapsed")
        if _successor_isomorphic_digest(left, left_action) == (
            _successor_isomorphic_digest(left, right_action)
        ):
            raise NeuralTcrrBoardError("shared occurrence successors collapsed")
        return

    if twin.kind in {
        "repeated_variable_equality",
        "partial_nested_match",
        "type_mismatch",
        "capacity",
    }:
        if not left_expected.transitions or right_expected.transitions:
            raise NeuralTcrrBoardError(
                f"{twin.kind} does not isolate a positive/no-redex contrast"
            )
        if twin.kind == "capacity":
            left_windows = fingerprints[twin.left_packet_sha256].normalized_rule_windows
            right_windows = fingerprints[
                twin.right_packet_sha256
            ].normalized_rule_windows
            if left_windows != right_windows:
                raise NeuralTcrrBoardError("capacity twin changed its rule window")
        return

    if twin.kind == "rhs_pointer":
        if len(left_expected.transitions) != 1 or len(right_expected.transitions) != 1:
            raise NeuralTcrrBoardError("RHS pointer twin must retain one action")
        if _successor_isomorphic_digest(
            left,
            left_expected.transitions[0],
        ) == _successor_isomorphic_digest(
            right,
            right_expected.transitions[0],
        ):
            raise NeuralTcrrBoardError("RHS pointer twin did not change successor")
        return
    raise NeuralTcrrBoardError(f"unvalidated twin kind {twin.kind!r}")


def validate_local_transition_slice(value: LocalTransitionSlice) -> None:
    """Recompute every ledger, oracle receipt, split gate, and twin predicate."""

    packet_digests = tuple(packet_sha256(packet) for packet in value.packets)
    if len(packet_digests) != len(set(packet_digests)):
        raise NeuralTcrrBoardError("packet corpus contains exact duplicates")
    packet_set = set(packet_digests)

    def unique_ledger(
        records: Sequence[object],
        label: str,
    ) -> dict[str, object]:
        keys = [str(getattr(item, "packet_sha256")) for item in records]
        if len(keys) != len(set(keys)):
            raise NeuralTcrrBoardError(f"{label} ledger keys must be unique")
        if len(keys) != len(packet_digests) or set(keys) != packet_set:
            raise NeuralTcrrBoardError(
                f"{label} ledger cardinality does not match packets"
            )
        return dict(zip(keys, records, strict=True))

    expected_raw = unique_ledger(value.expected_records, "expected")
    assignments_raw = unique_ledger(value.split_assignments, "split")
    fingerprints_raw = unique_ledger(value.fingerprints, "fingerprint")
    agreements_raw = unique_ledger(value.oracle_agreements, "oracle")
    coverage_raw = unique_ledger(value.primitive_coverage, "primitive")
    packets = dict(zip(packet_digests, value.packets, strict=True))
    expected = {
        key: item
        for key, item in expected_raw.items()
        if isinstance(item, ExpectedTransitionRecord)
    }
    fingerprints = {
        key: item
        for key, item in fingerprints_raw.items()
        if isinstance(item, PacketFingerprints)
    }
    if len(expected) != len(packet_digests) or len(fingerprints) != len(packet_digests):
        raise NeuralTcrrBoardError("typed ledger conversion failed")

    for digest, packet in packets.items():
        validate_source_deleted_packet(packet)
        recomputed_expected = _expected_record_from_packet(packet)
        if expected[digest] != recomputed_expected:
            raise NeuralTcrrBoardError("expected transition ledger is stale")
        recomputed_fingerprint = packet_fingerprints(packet)
        if fingerprints[digest] != recomputed_fingerprint:
            raise NeuralTcrrBoardError("fingerprint ledger is stale")
        recomputed_agreement = _oracle_agreement(packet)
        if agreements_raw[digest] != recomputed_agreement:
            raise NeuralTcrrBoardError("independent oracle ledger is stale")
        recomputed_coverage = PrimitiveCoverageRecord(
            digest,
            _infer_local_primitives(packet, recomputed_expected),
        )
        if coverage_raw[digest] != recomputed_coverage:
            raise NeuralTcrrBoardError("primitive coverage ledger is stale")

    validate_split_isolation(value.split_assignments, value.fingerprints)
    _validate_primitive_coverage(
        value.split_assignments,
        value.primitive_coverage,
    )

    twin_keys = [
        (
            item.kind,
            item.left_packet_sha256,
            item.right_packet_sha256,
            item.left_transition_index,
            item.right_transition_index,
        )
        for item in value.twins
    ]
    if len(twin_keys) != len(set(twin_keys)):
        raise NeuralTcrrBoardError("twin ledger keys must be unique")
    twin_kinds = [item.kind for item in value.twins]
    if set(twin_kinds) != _REQUIRED_TWIN_KINDS or len(twin_kinds) != len(
        _REQUIRED_TWIN_KINDS
    ):
        raise NeuralTcrrBoardError("twin ledger lacks one exact required predicate")
    for twin in value.twins:
        if (
            twin.left_packet_sha256 not in packets
            or twin.right_packet_sha256 not in packets
        ):
            raise NeuralTcrrBoardError("twin names an absent packet")
        _validate_twin_predicate(twin, packets, expected, fingerprints)

    split_by_digest = {
        key: item.partition
        for key, item in assignments_raw.items()
        if isinstance(item, SplitAssignment)
    }
    train_packets = [
        digest
        for digest, partition in split_by_digest.items()
        if partition == "local_transition_train"
    ]
    controlled_no_redex = [
        twin
        for twin in value.twins
        if twin.kind
        in {
            "repeated_variable_equality",
            "partial_nested_match",
            "type_mismatch",
            "capacity",
        }
        and not expected[twin.right_packet_sha256].transitions
    ]
    if len(controlled_no_redex) != 4:
        raise NeuralTcrrBoardError("controlled no-redex twin coverage is incomplete")
    if not any(
        len(packets[digest].rules) > 1 and not expected[digest].transitions
        for digest in train_packets
    ):
        raise NeuralTcrrBoardError("all-distractor no-redex packet is absent")
    if not any(
        len(packets[digest].rules) > 1
        and 0
        < len({item.rule_id for item in expected[digest].transitions})
        < len(packets[digest].rules)
        for digest in train_packets
    ):
        raise NeuralTcrrBoardError("subset-legal multi-rule packet is absent")


def _unused_constructor_for_node(
    packet: SourceDeletedPacket,
    storage_id: str,
) -> str:
    nodes = {item.storage_id: item for item in packet.graph.nodes}
    active = nodes[storage_id]
    used = {
        item.constructor_id
        for item in packet.graph.nodes
        if item.constructor_id is not None
    }
    for rule in packet.rules:
        used.update(
            item.constructor_id
            for term in (rule.lhs, rule.rhs)
            if term is not None
            for item in _walk_rule_terms(term)
            if item.constructor_id is not None
        )
    candidates = [
        item.identifier
        for item in packet.constructors
        if item.identifier not in used
        and item.result_type == active.type_id
        and len(item.argument_types) == len(active.children)
    ]
    if len(candidates) != 1:
        raise NeuralTcrrBoardError("controlled constructor mutant is ambiguous")
    return candidates[0]


def build_local_transition_slice(seed: int = 2026072301) -> LocalTransitionSlice:
    """Build the repaired deterministic N-TCRR local-transition slice."""

    entries: list[tuple[Partition, SourceDeletedPacket]] = []
    twins_pending: list[
        tuple[
            TwinKind,
            SourceDeletedPacket,
            SourceDeletedPacket,
            ReindexNamespace | None,
            tuple[IdentifierRemap, ...],
            tuple[str, ...],
            int | None,
            int | None,
        ]
    ] = []

    def add(partition: Partition, packet: SourceDeletedPacket) -> None:
        entries.append((partition, packet))

    reindex_base = _multi_rule_reindex_base(seed + 1)
    add("local_transition_train", reindex_base)
    for kind, namespace in (
        ("constructor_reindex", "constructor"),
        ("type_reindex", "type"),
        ("rule_reindex", "rule"),
        ("storage_reindex", "storage"),
    ):
        typed_kind: TwinKind = kind
        typed_namespace: ReindexNamespace = namespace
        remap = _reverse_namespace_remap(reindex_base, typed_namespace)
        variant = _reindex_packet(reindex_base, typed_namespace, remap)
        add("local_transition_train", variant)
        twins_pending.append(
            (
                typed_kind,
                reindex_base,
                variant,
                typed_namespace,
                remap,
                (),
                None,
                None,
            )
        )

    repeated_equal = _repeated_variable_base(seed + 2)
    repeated_root = repeated_equal.graph.root
    if repeated_root is None:
        raise NeuralTcrrBoardError("repeated-variable base lost its root")
    repeated_root_node = next(
        item for item in repeated_equal.graph.nodes if item.storage_id == repeated_root
    )
    repeated_storage = repeated_root_node.children[1]
    repeated_constructor = _unused_constructor_for_node(
        repeated_equal,
        repeated_storage,
    )
    repeated_unequal = _mutate_graph_constructor(
        repeated_equal,
        repeated_storage,
        repeated_constructor,
    )
    add("local_transition_train", repeated_equal)
    add("local_transition_train", repeated_unequal)
    twins_pending.append(
        (
            "repeated_variable_equality",
            repeated_equal,
            repeated_unequal,
            None,
            (),
            (repeated_storage, repeated_constructor),
            0,
            None,
        )
    )

    nested_positive = _nested_match_base(seed + 3)
    nested_root = nested_positive.graph.root
    if nested_root is None:
        raise NeuralTcrrBoardError("nested base lost its root")
    nested_root_node = next(
        item for item in nested_positive.graph.nodes if item.storage_id == nested_root
    )
    nested_storage = nested_root_node.children[0]
    nested_constructor = _unused_constructor_for_node(
        nested_positive,
        nested_storage,
    )
    nested_partial = _mutate_graph_constructor(
        nested_positive,
        nested_storage,
        nested_constructor,
    )
    add("local_transition_train", nested_positive)
    add("local_transition_train", nested_partial)
    twins_pending.append(
        (
            "partial_nested_match",
            nested_positive,
            nested_partial,
            None,
            (),
            (nested_storage, nested_constructor),
            0,
            None,
        )
    )

    type_positive = _type_match_base(seed + 4)
    type_storage = str(type_positive.graph.root)
    type_node = next(
        item for item in type_positive.graph.nodes if item.storage_id == type_storage
    )
    available_types = sorted(
        {
            type_id
            for item in type_positive.constructors
            for type_id in (item.result_type, *item.argument_types)
        }
        - {type_node.type_id}
    )
    if len(available_types) != 1:
        raise NeuralTcrrBoardError("type mismatch axis is ambiguous")
    mismatch_type = available_types[0]
    type_negative = _mutate_graph_variable_type(
        type_positive,
        type_storage,
        mismatch_type,
    )
    add("local_transition_train", type_positive)
    add("local_transition_train", type_negative)
    twins_pending.append(
        (
            "type_mismatch",
            type_positive,
            type_negative,
            None,
            (),
            (type_storage, mismatch_type),
            0,
            None,
        )
    )

    rhs_forward = _rhs_pointer_base(seed + 5)
    rhs_rule_id = rhs_forward.rules[0].identifier
    rhs_reverse = _mutate_rhs_pointer(rhs_forward, rhs_rule_id)
    add("local_transition_train", rhs_forward)
    add("local_transition_train", rhs_reverse)
    twins_pending.append(
        (
            "rhs_pointer",
            rhs_forward,
            rhs_reverse,
            None,
            (),
            (rhs_rule_id,),
            0,
            0,
        )
    )

    shared = _shared_cancellation_base(seed + 6)
    add("local_transition_train", shared)
    twins_pending.append(
        (
            "shared_occurrence",
            shared,
            shared,
            None,
            (),
            (),
            0,
            1,
        )
    )

    capacity_16 = _capacity_base(seed + 7)
    free_storage = sorted(
        set(capacity_16.graph.reservoir)
        - {item.storage_id for item in capacity_16.graph.nodes}
    )
    if len(free_storage) != 2:
        raise NeuralTcrrBoardError("capacity base must expose two free records")
    removed_storage = free_storage[0]
    capacity_15 = _mutate_capacity(capacity_16, removed_storage)
    add("local_transition_train", capacity_16)
    add("local_transition_train", capacity_15)
    twins_pending.append(
        (
            "capacity",
            capacity_16,
            capacity_15,
            None,
            (),
            (removed_storage,),
            0,
            None,
        )
    )

    for packet in (
        _development_nested_composition(seed + 101),
        _development_rotation(seed + 102),
        _development_nested_rhs(seed + 103),
        _development_repeated_ternary(seed + 104),
        _development_unary_deletion(seed + 105),
        _development_heterogeneous(seed + 106),
    ):
        add("local_transition_development", packet)

    packets = tuple(packet for _partition, packet in entries)
    expected_records = tuple(_expected_record_from_packet(packet) for packet in packets)
    split_assignments = tuple(
        SplitAssignment(packet_sha256(packet), partition)
        for partition, packet in entries
    )
    fingerprints = tuple(packet_fingerprints(packet) for packet in packets)
    oracle_agreements = tuple(_oracle_agreement(packet) for packet in packets)
    primitive_coverage = tuple(
        PrimitiveCoverageRecord(
            packet_sha256(packet),
            _infer_local_primitives(packet, expected),
        )
        for packet, expected in zip(packets, expected_records, strict=True)
    )
    twins = tuple(
        CausalTwinRecord(
            kind=kind,
            left_packet_sha256=packet_sha256(left),
            right_packet_sha256=packet_sha256(right),
            namespace=namespace,
            remap=remap,
            axis_values=axis_values,
            left_transition_index=left_index,
            right_transition_index=right_index,
        )
        for (
            kind,
            left,
            right,
            namespace,
            remap,
            axis_values,
            left_index,
            right_index,
        ) in twins_pending
    )
    value = LocalTransitionSlice(
        packets=packets,
        expected_records=expected_records,
        split_assignments=split_assignments,
        fingerprints=fingerprints,
        twins=twins,
        oracle_agreements=oracle_agreements,
        primitive_coverage=primitive_coverage,
    )
    validate_local_transition_slice(value)
    return value


def _rule_term_from_payload(payload: Mapping[str, object]) -> RuleTermRecord:
    raw_children = payload.get("children", [])
    if not isinstance(raw_children, list):
        raise NeuralTcrrBoardError("rule-term children must be a list")
    return RuleTermRecord(
        kind=str(payload["kind"]),  # type: ignore[arg-type]
        type_id=str(payload["type_id"]),
        constructor_id=(
            None
            if payload.get("constructor_id") is None
            else str(payload["constructor_id"])
        ),
        variable_id=(
            None if payload.get("variable_id") is None else str(payload["variable_id"])
        ),
        children=tuple(
            _rule_term_from_payload(item)
            for item in raw_children
            if isinstance(item, Mapping)
        ),
    )


def deserialize_model_packet(serialized: str) -> SourceDeletedPacket:
    """Load one packet-only JSON file without any offline ledger."""

    payload = json.loads(serialized)
    validate_model_packet_payload(payload)
    if not isinstance(payload, Mapping):
        raise NeuralTcrrBoardError("packet JSON must contain an object")
    raw_constructors = payload.get("constructors")
    raw_rules = payload.get("rules")
    raw_graph = payload.get("graph")
    if (
        not isinstance(raw_constructors, list)
        or not isinstance(raw_rules, list)
        or not isinstance(raw_graph, Mapping)
    ):
        raise NeuralTcrrBoardError("packet JSON has an invalid top-level schema")
    constructors = tuple(
        ConstructorRecord(
            identifier=str(item["identifier"]),
            result_type=str(item["result_type"]),
            argument_types=tuple(str(value) for value in item["argument_types"]),
        )
        for item in raw_constructors
        if isinstance(item, Mapping) and isinstance(item.get("argument_types"), list)
    )
    rules = []
    for item in raw_rules:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("lhs"),
            Mapping,
        ):
            raise NeuralTcrrBoardError("packet rule record is malformed")
        raw_rhs = item.get("rhs")
        if raw_rhs is not None and not isinstance(raw_rhs, Mapping):
            raise NeuralTcrrBoardError("packet RHS record is malformed")
        rules.append(
            RuleRecord(
                identifier=str(item["identifier"]),
                lhs=_rule_term_from_payload(item["lhs"]),
                rhs=(None if raw_rhs is None else _rule_term_from_payload(raw_rhs)),
            )
        )
    raw_reservoir = raw_graph.get("reservoir")
    raw_nodes = raw_graph.get("nodes")
    if not isinstance(raw_reservoir, list) or not isinstance(raw_nodes, list):
        raise NeuralTcrrBoardError("packet graph record is malformed")
    nodes = tuple(
        GraphNodeRecord(
            storage_id=str(item["storage_id"]),
            kind=str(item["kind"]),  # type: ignore[arg-type]
            type_id=str(item["type_id"]),
            constructor_id=(
                None
                if item.get("constructor_id") is None
                else str(item["constructor_id"])
            ),
            variable_id=(
                None if item.get("variable_id") is None else str(item["variable_id"])
            ),
            children=tuple(str(value) for value in item["children"]),
        )
        for item in raw_nodes
        if isinstance(item, Mapping) and isinstance(item.get("children"), list)
    )
    packet = SourceDeletedPacket(
        constructors=constructors,
        rules=tuple(rules),
        graph=GraphRecord(
            reservoir=tuple(str(item) for item in raw_reservoir),
            root=(None if raw_graph.get("root") is None else str(raw_graph["root"])),
            nodes=nodes,
        ),
    )
    validate_source_deleted_packet(packet)
    return packet


def _path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_export_roots(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    resolved = tuple(item.expanduser().resolve() for item in roots)
    for left_index, left in enumerate(resolved):
        for right in resolved[left_index + 1 :]:
            if (
                left == right
                or _path_is_within(left, right)
                or _path_is_within(right, left)
            ):
                raise NeuralTcrrBoardError(
                    "packet, train-label, and assessor roots must be disjoint"
                )
    return resolved


def _write_canonical_json(path: Path, payload: object) -> str:
    material = (_canonical_json(payload) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(material)
    return hashlib.sha256(material).hexdigest()


def _manifest_entries(
    packets: Sequence[SourceDeletedPacket],
    directory: Path,
) -> tuple[dict[str, str], ...]:
    entries = []
    for packet in sorted(packets, key=packet_sha256):
        digest = packet_sha256(packet)
        relative = f"{digest}.json"
        material = (serialize_model_packet(packet) + "\n").encode()
        (directory / relative).write_bytes(material)
        entries.append(
            {
                "file": relative,
                "packet_sha256": digest,
                "file_sha256": hashlib.sha256(material).hexdigest(),
            }
        )
    return tuple(entries)


def export_packet_only_corpus(
    value: LocalTransitionSlice,
    *,
    packet_root: Path,
    train_label_root: Path,
    development_assessment_root: Path,
) -> PacketExportReceipt:
    """Offline export into three disjoint roots.

    Packet roots contain only source-deleted packets and packet manifests.
    Train labels never include development records. Development labels exist
    only in one read-only sealed assessor artifact.
    """

    validate_local_transition_slice(value)
    packet_root, train_label_root, development_assessment_root = _validate_export_roots(
        (packet_root, train_label_root, development_assessment_root)
    )
    for root in (
        packet_root,
        train_label_root,
        development_assessment_root,
    ):
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise NeuralTcrrBoardError(f"export root is not empty: {root}")

    split = {item.packet_sha256: item.partition for item in value.split_assignments}
    packets = {packet_sha256(packet): packet for packet in value.packets}
    expected = {item.packet_sha256: item for item in value.expected_records}
    agreements = {item.packet_sha256: item for item in value.oracle_agreements}
    train_digests = sorted(
        digest
        for digest, partition in split.items()
        if partition == "local_transition_train"
    )
    development_digests = sorted(
        digest
        for digest, partition in split.items()
        if partition == "local_transition_development"
    )

    train_packet_dir = packet_root / "train"
    development_packet_dir = packet_root / "development"
    train_packet_dir.mkdir()
    development_packet_dir.mkdir()
    train_entries = _manifest_entries(
        [packets[digest] for digest in train_digests],
        train_packet_dir,
    )
    development_entries = _manifest_entries(
        [packets[digest] for digest in development_digests],
        development_packet_dir,
    )
    _write_canonical_json(
        train_packet_dir / "manifest.json",
        {
            "protocol": "neural_tcrr_packet_manifest_v2",
            "packet_count": len(train_entries),
            "packets": train_entries,
        },
    )
    _write_canonical_json(
        development_packet_dir / "manifest.json",
        {
            "protocol": "neural_tcrr_packet_manifest_v2",
            "packet_count": len(development_entries),
            "packets": development_entries,
        },
    )
    packet_manifest_sha256 = _write_canonical_json(
        packet_root / "manifest.json",
        {
            "protocol": "neural_tcrr_packet_root_manifest_v2",
            "train_manifest": "train/manifest.json",
            "development_manifest": "development/manifest.json",
            "train_packet_count": len(train_entries),
            "development_packet_count": len(development_entries),
        },
    )

    train_artifact = train_label_root / "train_labels.json"
    train_artifact_sha256 = _write_canonical_json(
        train_artifact,
        {
            "protocol": "neural_tcrr_train_labels_v2",
            "records": [
                dataclasses.asdict(expected[digest]) for digest in train_digests
            ],
            "oracle_agreements": [
                dataclasses.asdict(agreements[digest]) for digest in train_digests
            ],
        },
    )
    train_label_manifest_sha256 = _write_canonical_json(
        train_label_root / "manifest.json",
        {
            "protocol": "neural_tcrr_train_label_manifest_v2",
            "artifact": train_artifact.name,
            "artifact_sha256": train_artifact_sha256,
            "record_count": len(train_digests),
        },
    )

    assessor_artifact = (
        development_assessment_root / "sealed_development_assessment.json"
    )
    sealed_sha256 = _write_canonical_json(
        assessor_artifact,
        {
            "protocol": "neural_tcrr_sealed_development_assessment_v2",
            "records": [
                dataclasses.asdict(expected[digest]) for digest in development_digests
            ],
            "oracle_agreements": [
                dataclasses.asdict(agreements[digest]) for digest in development_digests
            ],
        },
    )
    assessor_artifact.chmod(0o400)
    development_manifest = development_assessment_root / "manifest.json"
    development_assessment_manifest_sha256 = _write_canonical_json(
        development_manifest,
        {
            "protocol": "neural_tcrr_development_assessor_manifest_v2",
            "sealed_artifact": assessor_artifact.name,
            "sealed_artifact_sha256": sealed_sha256,
            "record_count": len(development_digests),
        },
    )
    development_manifest.chmod(0o400)
    return PacketExportReceipt(
        packet_root=str(packet_root),
        train_label_root=str(train_label_root),
        development_assessment_root=str(development_assessment_root),
        packet_manifest_sha256=packet_manifest_sha256,
        train_label_manifest_sha256=train_label_manifest_sha256,
        development_assessment_manifest_sha256=(development_assessment_manifest_sha256),
        sealed_development_artifact_sha256=sealed_sha256,
        train_packet_count=len(train_digests),
        development_packet_count=len(development_digests),
    )


def load_packet_only_partition(
    packet_root: Path,
    partition: Literal["train", "development"],
) -> tuple[SourceDeletedPacket, ...]:
    """Evaluation-side loader whose only input is a packet filesystem root."""

    directory = packet_root.expanduser().resolve() / partition
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "neural_tcrr_packet_manifest_v2":
        raise NeuralTcrrBoardError("packet manifest protocol mismatch")
    raw_entries = manifest.get("packets")
    if not isinstance(raw_entries, list):
        raise NeuralTcrrBoardError("packet manifest entries are absent")
    packets = []
    seen = set()
    for item in raw_entries:
        if not isinstance(item, Mapping):
            raise NeuralTcrrBoardError("packet manifest entry is malformed")
        digest = str(item["packet_sha256"])
        if digest in seen:
            raise NeuralTcrrBoardError("packet manifest key is duplicated")
        seen.add(digest)
        path = directory / str(item["file"])
        material = path.read_bytes()
        if hashlib.sha256(material).hexdigest() != item["file_sha256"]:
            raise NeuralTcrrBoardError("packet file receipt mismatch")
        packet = deserialize_model_packet(material.decode())
        if packet_sha256(packet) != digest:
            raise NeuralTcrrBoardError("packet content digest mismatch")
        packets.append(packet)
    if len(packets) != manifest.get("packet_count"):
        raise NeuralTcrrBoardError("packet manifest cardinality mismatch")
    return tuple(packets)


def load_sealed_development_assessment(
    artifact: Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    """Assessor-only loader for the read-only development target artifact."""

    artifact = artifact.expanduser().resolve()
    mode = stat.S_IMODE(artifact.stat().st_mode)
    if mode & 0o222:
        raise NeuralTcrrBoardError("development assessor artifact is not sealed")
    material = artifact.read_bytes()
    if hashlib.sha256(material).hexdigest() != expected_sha256:
        raise NeuralTcrrBoardError("development assessor artifact hash mismatch")
    payload = json.loads(material)
    if payload.get("protocol") != "neural_tcrr_sealed_development_assessment_v2":
        raise NeuralTcrrBoardError("development assessor protocol mismatch")
    return payload


__all__ = [
    "BindingRecord",
    "CausalTwinRecord",
    "ConstructorRecord",
    "ExpectedTransition",
    "ExpectedTransitionRecord",
    "GraphNodeRecord",
    "GraphRecord",
    "IdentifierRemap",
    "LocalTransitionSlice",
    "NeuralTcrrBoardError",
    "OracleAgreementRecord",
    "PacketExportReceipt",
    "PacketFingerprints",
    "PrimitiveCoverageRecord",
    "RuleRecord",
    "RuleTermRecord",
    "SourceDeletedPacket",
    "SplitAssignment",
    "build_local_transition_slice",
    "deserialize_model_packet",
    "export_packet_only_corpus",
    "load_packet_only_partition",
    "load_sealed_development_assessment",
    "packet_fingerprints",
    "packet_sha256",
    "serialize_model_packet",
    "validate_local_transition_slice",
    "validate_model_packet_payload",
    "validate_source_deleted_packet",
    "validate_split_isolation",
]
