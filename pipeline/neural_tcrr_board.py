"""Bounded procedural board slice for the Neural TCRR local motor.

This module is an offline data generator. It converts procedurally generated
typed rewrite systems into source-deleted packets and keeps all expected
transitions in a separate ledger. The committed symbolic mechanics are used
only while generating that ledger; they are not part of a model packet.

The slice is intentionally smaller than the preregistered training board. It
exercises the local one-step operations needed before an autonomous neural
reactor can be trained: occurrence-specific replacement, pointer reuse,
destructive cancellation, deletion, construction, repeated-variable binding,
heterogeneous typing, shared DAG occurrences, and capacity blocking.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import typed_critical_pair_rewrite_board as mechanics


MAX_CAPACITY = 16
MAX_RULES = 8
MAX_RULE_SIDE_NODES = 12
MAX_ARITY = 3
MAX_PATH_DEPTH = 8
OPAQUE_ID_LENGTH = 24

Partition = Literal["local_transition_train", "local_transition_development"]


class NeuralTcrrBoardError(ValueError):
    """Raised when a procedural board or packet violates its contract."""


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
    """One opaque rewrite card.

    ``rhs=None`` denotes explicit root deletion. Rule identifiers and binder
    identifiers are local to one packet.
    """

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
    """Leakage fingerprints derived from packet content."""

    packet_sha256: str
    exact_sha256: str
    isomorphic_sha256: str
    normalized_rule_windows: tuple[str, ...]


@dataclass(frozen=True)
class CausalTwinRecord:
    """Offline membership of one mandatory causal twin."""

    kind: Literal[
        "rhs_pointer",
        "shared_occurrence",
        "capacity",
        "storage_reindex",
        "rule_reindex",
    ]
    left_packet_sha256: str
    right_packet_sha256: str
    left_transition_index: int | None = None
    right_transition_index: int | None = None


@dataclass(frozen=True)
class LocalTransitionSlice:
    """A packet corpus plus physically separate offline ledgers."""

    packets: tuple[SourceDeletedPacket, ...]
    expected_records: tuple[ExpectedTransitionRecord, ...]
    split_assignments: tuple[SplitAssignment, ...]
    fingerprints: tuple[PacketFingerprints, ...]
    twins: tuple[CausalTwinRecord, ...]


_FORBIDDEN_MODEL_KEY_FRAGMENTS = frozenset(
    {
        "answer",
        "class",
        "expected",
        "family",
        "label",
        "legal",
        "mask",
        "oracle",
        "schedule",
        "seed",
        "source",
        "split",
        "target",
        "trace",
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
    return (
        len(identifier) == OPAQUE_ID_LENGTH
        and set(identifier) <= set("0123456789abcdef")
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
    """Validate the complete source-deleted packet without computing labels."""

    payload = dataclasses.asdict(packet)
    validate_model_packet_payload(payload)
    if not packet.constructors:
        raise NeuralTcrrBoardError("packet must declare at least one constructor")
    constructor_ids = [item.identifier for item in packet.constructors]
    if len(constructor_ids) != len(set(constructor_ids)):
        raise NeuralTcrrBoardError("constructor identifiers must be unique")
    if not all(_is_opaque(identifier) for identifier in constructor_ids):
        raise NeuralTcrrBoardError("constructor identifiers must be opaque")
    type_ids = {
        type_id
        for item in packet.constructors
        for type_id in (item.result_type, *item.argument_types)
    }
    if not type_ids or not all(_is_opaque(type_id) for type_id in type_ids):
        raise NeuralTcrrBoardError("type identifiers must be opaque")
    if any(len(item.argument_types) > MAX_ARITY for item in packet.constructors):
        raise NeuralTcrrBoardError("constructor exceeds the arity bound")
    constructors = {item.identifier: item for item in packet.constructors}

    if not 0 < len(packet.graph.reservoir) <= MAX_CAPACITY:
        raise NeuralTcrrBoardError("reservoir capacity lies outside the geometry")
    if len(packet.graph.reservoir) != len(set(packet.graph.reservoir)):
        raise NeuralTcrrBoardError("reservoir storage identifiers must be unique")
    if not all(_is_opaque(item) for item in packet.graph.reservoir):
        raise NeuralTcrrBoardError("storage identifiers must be opaque")
    reservoir = set(packet.graph.reservoir)
    nodes = {item.storage_id: item for item in packet.graph.nodes}
    if len(nodes) != len(packet.graph.nodes) or not set(nodes) <= reservoir:
        raise NeuralTcrrBoardError("occupied storage identifiers are invalid")
    if packet.graph.root is None:
        if nodes:
            raise NeuralTcrrBoardError("empty graph cannot retain occupied records")
    elif packet.graph.root not in nodes:
        raise NeuralTcrrBoardError("graph root is not occupied")

    for node in nodes.values():
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

    visiting: set[str] = set()
    reached: set[str] = set()

    def visit(storage_id: str) -> None:
        if storage_id in visiting:
            raise NeuralTcrrBoardError("graph must be acyclic")
        if storage_id in reached:
            return
        visiting.add(storage_id)
        for child in nodes[storage_id].children:
            if child not in nodes:
                raise NeuralTcrrBoardError("graph pointer names a free record")
            visit(child)
        visiting.remove(storage_id)
        reached.add(storage_id)

    if packet.graph.root is not None:
        visit(packet.graph.root)
    if reached != set(nodes):
        raise NeuralTcrrBoardError("all occupied records must be root-reachable")

    if not 0 < len(packet.rules) <= MAX_RULES:
        raise NeuralTcrrBoardError("rule count lies outside the frozen geometry")
    rule_ids = [rule.identifier for rule in packet.rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise NeuralTcrrBoardError("rule identifiers must be unique")
    if not all(_is_opaque(identifier) for identifier in rule_ids):
        raise NeuralTcrrBoardError("rule identifiers must be opaque")
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
        if rule.rhs is not None:
            rhs_type = _validate_rule_term(
                rule.rhs,
                constructors,
                bound_variables=variables,
                is_rhs=True,
            )
            if rhs_type != lhs_type:
                raise NeuralTcrrBoardError("rewrite changes the redex root type")


def serialize_model_packet(packet: SourceDeletedPacket) -> str:
    """Serialize only the model-visible packet after strict validation."""

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
        children=tuple(
            _pattern_to_record(system, child) for child in pattern.children
        ),
    )


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


def _collect_pattern_variables(
    pattern: mechanics.Pattern,
    output: dict[str, str],
) -> None:
    if isinstance(pattern, mechanics.PatternVariable):
        output.setdefault(pattern.name, pattern.type_id)
        return
    for child in pattern.children:
        _collect_pattern_variables(child, output)


def _graph_to_record(
    graph: mechanics.TermGraph,
    storage_ids: tuple[str, ...],
    *,
    node_order_seed: int,
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
    random.Random(node_order_seed).shuffle(records)
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
            node_order_seed=packet_seed ^ 0x6A4F,
        ),
    )
    validate_source_deleted_packet(packet)
    return packet


def _expected_record(
    system: mechanics.RewriteSystem,
    graph: mechanics.TermGraph,
    packet: SourceDeletedPacket,
    storage_ids: tuple[str, ...],
    *,
    packet_seed: int,
) -> ExpectedTransitionRecord:
    actions = []
    for action_index, reduction in enumerate(
        mechanics.legal_reductions(system, graph)
    ):
        rule = system.rule_map()[reduction.rule_id]
        bindings: dict[str, int] = {}
        matched = mechanics._match_pattern(  # noqa: SLF001
            system,
            graph,
            rule.lhs,
            reduction.target_slot,
            bindings,
        )
        if not matched:
            raise NeuralTcrrBoardError("legal mechanics reduction lost its binding")
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
                    node_order_seed=packet_seed ^ (0xA110 + action_index),
                ),
            )
        )
    return ExpectedTransitionRecord(
        packet_sha256=packet_sha256(packet),
        transitions=tuple(actions),
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


def _collect_rule_constructors(term: RuleTermRecord | None, output: set[str]) -> None:
    if term is None:
        return
    if term.constructor_id is not None:
        output.add(term.constructor_id)
    for child in term.children:
        _collect_rule_constructors(child, output)


def _collect_graph_constructors(
    graph: GraphRecord,
    root: str | None,
    output: set[str],
) -> None:
    if root is None:
        return
    nodes = {node.storage_id: node for node in graph.nodes}
    seen: set[str] = set()

    def visit(storage_id: str) -> None:
        if storage_id in seen:
            return
        seen.add(storage_id)
        node = nodes[storage_id]
        if node.constructor_id is not None:
            output.add(node.constructor_id)
        for child in node.children:
            visit(child)

    visit(root)


def _canonical_semantic_view(
    packet: SourceDeletedPacket,
    *,
    selected_rules: tuple[RuleRecord, ...],
    graph_root: str | None,
    include_capacity: bool,
) -> str:
    constructor_ids: set[str] = set()
    for rule in selected_rules:
        _collect_rule_constructors(rule.lhs, constructor_ids)
        _collect_rule_constructors(rule.rhs, constructor_ids)
    _collect_graph_constructors(packet.graph, graph_root, constructor_ids)
    constructor_records = {
        item.identifier: item
        for item in packet.constructors
        if item.identifier in constructor_ids
    }
    type_ids = sorted(
        {
            type_id
            for item in constructor_records.values()
            for type_id in (item.result_type, *item.argument_types)
        }
    )
    ordered_constructors = sorted(constructor_records)
    if not ordered_constructors:
        raise NeuralTcrrBoardError("canonical view has no constructors")

    best: str | None = None
    for type_labels in itertools.permutations(range(len(type_ids))):
        type_map = dict(zip(type_ids, type_labels, strict=True))
        for constructor_labels in itertools.permutations(
            range(len(ordered_constructors))
        ):
            constructor_map = dict(
                zip(
                    ordered_constructors,
                    constructor_labels,
                    strict=True,
                )
            )
            declarations = []
            for identifier in ordered_constructors:
                declaration = constructor_records[identifier]
                declarations.append(
                    {
                        "constructor": constructor_map[identifier],
                        "result_type": type_map[declaration.result_type],
                        "argument_types": [
                            type_map[type_id]
                            for type_id in declaration.argument_types
                        ],
                    }
                )
            declarations.sort(key=lambda item: int(item["constructor"]))

            def encode_rule_term(
                term: RuleTermRecord,
                variables: dict[str, int],
            ) -> dict[str, object]:
                if term.kind == "variable":
                    if term.variable_id is None:
                        raise NeuralTcrrBoardError("canonical variable is missing")
                    variable_index = variables.setdefault(
                        term.variable_id,
                        len(variables),
                    )
                    return {
                        "kind": "variable",
                        "type": type_map[term.type_id],
                        "binder": variable_index,
                    }
                if term.constructor_id is None:
                    raise NeuralTcrrBoardError("canonical constructor is missing")
                return {
                    "kind": "constructor",
                    "type": type_map[term.type_id],
                    "constructor": constructor_map[term.constructor_id],
                    "children": [
                        encode_rule_term(child, variables)
                        for child in term.children
                    ],
                }

            encoded_rules = []
            for rule in selected_rules:
                variables: dict[str, int] = {}
                encoded_rules.append(
                    {
                        "lhs": encode_rule_term(rule.lhs, variables),
                        "rhs": (
                            None
                            if rule.rhs is None
                            else encode_rule_term(rule.rhs, variables)
                        ),
                    }
                )
            encoded_rules.sort(key=_canonical_json)

            nodes = {node.storage_id: node for node in packet.graph.nodes}
            graph_indices: dict[str, int] = {}
            graph_records: list[dict[str, object] | None] = []
            graph_variables: dict[str, int] = {}

            def encode_graph(storage_id: str) -> int:
                previous = graph_indices.get(storage_id)
                if previous is not None:
                    return previous
                index = len(graph_records)
                graph_indices[storage_id] = index
                graph_records.append(None)
                node = nodes[storage_id]
                if node.kind == "variable":
                    if node.variable_id is None:
                        raise NeuralTcrrBoardError("graph variable is missing")
                    record: dict[str, object] = {
                        "kind": "variable",
                        "type": type_map[node.type_id],
                        "alpha": graph_variables.setdefault(
                            node.variable_id,
                            len(graph_variables),
                        ),
                    }
                else:
                    if node.constructor_id is None:
                        raise NeuralTcrrBoardError("graph constructor is missing")
                    record = {
                        "kind": "constructor",
                        "type": type_map[node.type_id],
                        "constructor": constructor_map[node.constructor_id],
                        "children": [
                            encode_graph(child) for child in node.children
                        ],
                    }
                graph_records[index] = record
                return index

            encoded_root = None if graph_root is None else encode_graph(graph_root)
            payload: dict[str, object] = {
                "constructors": declarations,
                "rules": encoded_rules,
                "graph": {
                    "root": encoded_root,
                    "nodes": graph_records,
                },
            }
            if include_capacity:
                payload["capacity"] = len(packet.graph.reservoir)
            candidate = _canonical_json(payload)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise NeuralTcrrBoardError("canonicalization produced no candidate")
    return best


def packet_fingerprints(packet: SourceDeletedPacket) -> PacketFingerprints:
    """Compute exact, full-isomorphism, and matched rule-window digests."""

    validate_source_deleted_packet(packet)
    exact = packet_sha256(packet)
    isomorphic = _sha256(
        _canonical_semantic_view(
            packet,
            selected_rules=packet.rules,
            graph_root=packet.graph.root,
            include_capacity=True,
        )
    )
    windows = set()
    for rule in packet.rules:
        for _path, storage_id in _graph_occurrences(packet.graph):
            if not _packet_pattern_matches(packet, rule.lhs, storage_id, {}):
                continue
            windows.add(
                _sha256(
                    _canonical_semantic_view(
                        packet,
                        selected_rules=(rule,),
                        graph_root=storage_id,
                        include_capacity=False,
                    )
                )
            )
    if not windows:
        windows.add(
            _sha256(
                _canonical_semantic_view(
                    packet,
                    selected_rules=packet.rules,
                    graph_root=packet.graph.root,
                    include_capacity=False,
                )
            )
        )
    return PacketFingerprints(
        packet_sha256=exact,
        exact_sha256=exact,
        isomorphic_sha256=isomorphic,
        normalized_rule_windows=tuple(sorted(windows)),
    )


def validate_split_isolation(
    assignments: tuple[SplitAssignment, ...],
    fingerprints: tuple[PacketFingerprints, ...],
) -> None:
    """Reject exact, isomorphic, or normalized-window overlap across splits."""

    split_by_packet: dict[str, Partition] = {}
    for item in assignments:
        if item.packet_sha256 in split_by_packet:
            raise NeuralTcrrBoardError("packet has more than one split assignment")
        split_by_packet[item.packet_sha256] = item.partition
    fingerprint_by_packet = {item.packet_sha256: item for item in fingerprints}
    if set(split_by_packet) != set(fingerprint_by_packet):
        raise NeuralTcrrBoardError("split and fingerprint ledgers do not align")
    partitions: dict[Partition, dict[str, set[str]]] = {
        "local_transition_train": {
            "exact": set(),
            "isomorphic": set(),
            "window": set(),
        },
        "local_transition_development": {
            "exact": set(),
            "isomorphic": set(),
            "window": set(),
        },
    }
    for packet_digest, partition in split_by_packet.items():
        item = fingerprint_by_packet[packet_digest]
        partitions[partition]["exact"].add(item.exact_sha256)
        partitions[partition]["isomorphic"].add(item.isomorphic_sha256)
        partitions[partition]["window"].update(item.normalized_rule_windows)
    train = partitions["local_transition_train"]
    development = partitions["local_transition_development"]
    for category in ("exact", "isomorphic", "window"):
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
        *,
        reverse_rules: bool = False,
    ) -> mechanics.RewriteSystem:
        ordered = tuple(reversed(rules)) if reverse_rules else rules
        constructors = list(self.constructors.values())
        random.Random(self.seed ^ 0x51A7).shuffle(constructors)
        return mechanics.RewriteSystem(tuple(constructors), ordered)


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
    reverse_storage: bool = False,
) -> _GeneratedExample:
    permutation = list(range(capacity))
    random.Random(factory.seed ^ 0x5702).shuffle(permutation)
    if reverse_storage:
        permutation.reverse()
    graph = mechanics.pack_ground_term(
        system,
        term,
        capacity,
        slot_permutation=tuple(permutation),
    )
    storage_ids = [
        _opaque(factory.seed, "storage", index) for index in range(capacity)
    ]
    random.Random(factory.seed ^ 0x570A).shuffle(storage_ids)
    return _GeneratedExample(
        system=system,
        graph=graph,
        storage_ids=tuple(storage_ids),
        packet_seed=factory.seed,
    )


def _wrap_term(
    factory: _EpisodeFactory,
    child: mechanics.GroundTerm,
    depth: int,
) -> mechanics.GroundTerm:
    result = child
    for _ in range(depth):
        result = factory.term("context", result)
    return result


def _replace_example(seed: int, *, depth: int) -> _GeneratedExample:
    factory = _EpisodeFactory(seed)
    factory.constructor("redex", "value")
    factory.constructor("normal", "value")
    factory.constructor("context", "value", ("value",))
    rule = factory.rule(factory.pattern("redex"), factory.rhs("normal"))
    system = factory.system((rule,))
    return _pack_example(
        factory,
        system,
        _wrap_term(factory, factory.leaf("redex"), depth),
        capacity=8,
    )


def _cancel_example(seed: int, *, depth: int) -> _GeneratedExample:
    factory = _EpisodeFactory(seed)
    factory.constructor("payload", "value")
    factory.constructor("drop", "value", ("value",))
    factory.constructor("context", "value", ("value",))
    variable = factory.variable("value")
    rule = factory.rule(
        factory.pattern("drop", variable),
        mechanics.RhsVariable(variable.name),
    )
    system = factory.system((rule,))
    term = factory.term("drop", factory.leaf("payload"))
    return _pack_example(
        factory,
        system,
        _wrap_term(factory, term, depth),
        capacity=9,
    )


def _growth_example(
    seed: int,
    *,
    rhs_arity: int,
    depth: int,
) -> _GeneratedExample:
    factory = _EpisodeFactory(seed)
    factory.constructor("source", "value")
    factory.constructor("context", "value", ("value",))
    child_aliases = []
    for index in range(rhs_arity):
        alias = f"payload_{index}"
        child_aliases.append(alias)
        factory.constructor(alias, "value")
    factory.constructor(
        "result",
        "value",
        ("value",) * rhs_arity,
    )
    rule = factory.rule(
        factory.pattern("source"),
        factory.rhs(
            "result",
            *(factory.rhs(alias) for alias in child_aliases),
        ),
    )
    system = factory.system((rule,))
    return _pack_example(
        factory,
        system,
        _wrap_term(factory, factory.leaf("source"), depth),
        capacity=12,
    )


def _deletion_example(seed: int, *, unary: bool) -> _GeneratedExample:
    factory = _EpisodeFactory(seed)
    if unary:
        factory.constructor("payload", "value")
        factory.constructor("erase", "value", ("value",))
        variable = factory.variable("value")
        lhs: mechanics.Pattern = factory.pattern("erase", variable)
        term = factory.term("erase", factory.leaf("payload"))
    else:
        factory.constructor("erase", "value")
        lhs = factory.pattern("erase")
        term = factory.leaf("erase")
    rule = factory.rule(lhs, None)
    system = factory.system((rule,))
    return _pack_example(factory, system, term, capacity=5)


def _repeated_variable_example(
    seed: int,
    *,
    arity: int,
) -> _GeneratedExample:
    factory = _EpisodeFactory(seed)
    factory.constructor("same", "value", ("value",) * arity)
    factory.constructor("hit", "value")
    factory.constructor("a", "value")
    factory.constructor("b", "value")
    left = factory.variable("value")
    right = factory.variable("value")
    if arity == 2:
        lhs = factory.pattern("same", left, left)
        rhs = factory.rhs("hit")
        term = factory.term("same", factory.leaf("a"), factory.leaf("a"))
    elif arity == 3:
        lhs = factory.pattern("same", left, left, right)
        factory.constructor("output", "value", ("value", "value"))
        rhs = factory.rhs(
            "output",
            mechanics.RhsVariable(right.name),
            mechanics.RhsVariable(left.name),
        )
        term = factory.term(
            "same",
            factory.leaf("a"),
            factory.leaf("a"),
            factory.leaf("b"),
        )
    else:
        raise NeuralTcrrBoardError("repeated-variable arity must be two or three")
    rule = factory.rule(lhs, rhs)
    system = factory.system((rule,))
    return _pack_example(factory, system, term, capacity=10)


def _rhs_pointer_example(seed: int, *, reverse: bool) -> _GeneratedExample:
    factory = _EpisodeFactory(seed)
    factory.constructor("pick", "value", ("value", "value"))
    factory.constructor("output", "value", ("value", "value"))
    factory.constructor("left", "value")
    factory.constructor("right", "value")
    left = factory.variable("value")
    right = factory.variable("value")
    rhs_children: tuple[mechanics.RhsExpression, ...] = (
        mechanics.RhsVariable(left.name),
        mechanics.RhsVariable(right.name),
    )
    if reverse:
        rhs_children = tuple(reversed(rhs_children))
    rule = factory.rule(
        factory.pattern("pick", left, right),
        factory.rhs("output", *rhs_children),
    )
    system = factory.system((rule,))
    term = factory.term("pick", factory.leaf("left"), factory.leaf("right"))
    return _pack_example(factory, system, term, capacity=7)


def _shared_occurrence_example(seed: int) -> _GeneratedExample:
    factory = _EpisodeFactory(seed)
    factory.constructor("pair", "value", ("value", "value"))
    factory.constructor("redex", "value")
    factory.constructor("normal", "value")
    rule = factory.rule(factory.pattern("redex"), factory.rhs("normal"))
    system = factory.system((rule,))
    shared = factory.leaf("redex")
    return _pack_example(
        factory,
        system,
        factory.term("pair", shared, shared),
        capacity=7,
    )


def _capacity_example(seed: int, *, capacity: int) -> _GeneratedExample:
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
    return _pack_example(factory, system, term, capacity=capacity)


def _independent_example(
    seed: int,
    *,
    reverse_storage: bool = False,
    reverse_rules: bool = False,
) -> _GeneratedExample:
    factory = _EpisodeFactory(seed)
    for alias in ("left", "right", "left_nf", "right_nf"):
        factory.constructor(alias, "value")
    factory.constructor("pair", "value", ("value", "value"))
    rules = (
        factory.rule(factory.pattern("left"), factory.rhs("left_nf")),
        factory.rule(factory.pattern("right"), factory.rhs("right_nf")),
    )
    system = factory.system(rules, reverse_rules=reverse_rules)
    term = factory.term("pair", factory.leaf("left"), factory.leaf("right"))
    return _pack_example(
        factory,
        system,
        term,
        capacity=8,
        reverse_storage=reverse_storage,
    )


def _heterogeneous_example(seed: int, *, nested: bool) -> _GeneratedExample:
    factory = _EpisodeFactory(seed)
    factory.constructor("atom", "value")
    factory.constructor("box", "box", ("value",))
    factory.constructor("sealed", "box", ("value",))
    factory.constructor("outer", "box", ("box",))
    variable = factory.variable("value")
    if nested:
        lhs = factory.pattern("outer", factory.pattern("box", variable))
        rhs = factory.rhs(
            "outer",
            factory.rhs(
                "sealed",
                mechanics.RhsVariable(variable.name),
            ),
        )
        term = factory.term(
            "outer",
            factory.term("box", factory.leaf("atom")),
        )
    else:
        lhs = factory.pattern("box", variable)
        rhs = factory.rhs(
            "sealed",
            mechanics.RhsVariable(variable.name),
        )
        term = factory.term("box", factory.leaf("atom"))
    rule = factory.rule(lhs, rhs)
    system = factory.system((rule,))
    return _pack_example(factory, system, term, capacity=8)


def _nested_pattern_example(seed: int) -> _GeneratedExample:
    factory = _EpisodeFactory(seed)
    factory.constructor("payload", "value")
    factory.constructor("inner", "value", ("value",))
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
    return _pack_example(factory, system, term, capacity=8)


def _rotation_example(seed: int) -> _GeneratedExample:
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
    return _pack_example(factory, system, term, capacity=7)


def _nested_rhs_example(seed: int) -> _GeneratedExample:
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
    return _pack_example(factory, system, term, capacity=8)


def _materialize(
    generated: _GeneratedExample,
) -> tuple[SourceDeletedPacket, ExpectedTransitionRecord, PacketFingerprints]:
    packet = _packet_from_mechanics(
        generated.system,
        generated.graph,
        storage_ids=generated.storage_ids,
        packet_seed=generated.packet_seed,
    )
    expected = _expected_record(
        generated.system,
        generated.graph,
        packet,
        generated.storage_ids,
        packet_seed=generated.packet_seed,
    )
    return packet, expected, packet_fingerprints(packet)


def validate_local_transition_slice(board: LocalTransitionSlice) -> None:
    """Validate ledger separation, joins, twins, and split isolation."""

    packet_digests = tuple(packet_sha256(packet) for packet in board.packets)
    if len(packet_digests) != len(set(packet_digests)):
        raise NeuralTcrrBoardError("packet corpus contains exact duplicates")
    expected = {item.packet_sha256: item for item in board.expected_records}
    assignments = {
        item.packet_sha256: item for item in board.split_assignments
    }
    fingerprints = {
        item.packet_sha256: item for item in board.fingerprints
    }
    packet_set = set(packet_digests)
    if (
        packet_set != set(expected)
        or packet_set != set(assignments)
        or packet_set != set(fingerprints)
    ):
        raise NeuralTcrrBoardError("offline ledgers do not join one-to-one")
    for packet in board.packets:
        validate_source_deleted_packet(packet)
    validate_split_isolation(board.split_assignments, board.fingerprints)
    for twin in board.twins:
        if (
            twin.left_packet_sha256 not in packet_set
            or twin.right_packet_sha256 not in packet_set
        ):
            raise NeuralTcrrBoardError("twin names an absent packet")
        if twin.left_transition_index is not None:
            left_count = len(expected[twin.left_packet_sha256].transitions)
            if twin.left_transition_index not in range(left_count):
                raise NeuralTcrrBoardError("left twin transition is absent")
        if twin.right_transition_index is not None:
            right_count = len(expected[twin.right_packet_sha256].transitions)
            if twin.right_transition_index not in range(right_count):
                raise NeuralTcrrBoardError("right twin transition is absent")


def build_local_transition_slice(seed: int = 2026072301) -> LocalTransitionSlice:
    """Build the first deterministic, bounded N-TCRR procedural board slice."""

    generated: list[tuple[Partition, _GeneratedExample]] = []

    def add(partition: Partition, example: _GeneratedExample) -> int:
        generated.append((partition, example))
        return len(generated) - 1

    add("local_transition_train", _replace_example(seed + 1, depth=1))
    add("local_transition_train", _cancel_example(seed + 2, depth=1))
    add(
        "local_transition_train",
        _growth_example(seed + 3, rhs_arity=2, depth=0),
    )
    add("local_transition_train", _deletion_example(seed + 4, unary=False))
    add(
        "local_transition_train",
        _repeated_variable_example(seed + 5, arity=2),
    )
    rhs_forward = add(
        "local_transition_train",
        _rhs_pointer_example(seed + 6, reverse=False),
    )
    rhs_reverse = add(
        "local_transition_train",
        _rhs_pointer_example(seed + 7, reverse=True),
    )
    shared = add(
        "local_transition_train",
        _shared_occurrence_example(seed + 8),
    )
    capacity_16 = add(
        "local_transition_train",
        _capacity_example(seed + 9, capacity=16),
    )
    capacity_15 = add(
        "local_transition_train",
        _capacity_example(seed + 10, capacity=15),
    )
    storage_plain = add(
        "local_transition_train",
        _independent_example(seed + 11),
    )
    storage_reindexed = add(
        "local_transition_train",
        _independent_example(seed + 12, reverse_storage=True),
    )
    rule_plain = add(
        "local_transition_train",
        _independent_example(seed + 13),
    )
    rule_reindexed = add(
        "local_transition_train",
        _independent_example(seed + 14, reverse_rules=True),
    )
    add(
        "local_transition_train",
        _heterogeneous_example(seed + 15, nested=False),
    )

    add(
        "local_transition_development",
        _nested_pattern_example(seed + 101),
    )
    add(
        "local_transition_development",
        _rotation_example(seed + 102),
    )
    add(
        "local_transition_development",
        _nested_rhs_example(seed + 103),
    )
    add(
        "local_transition_development",
        _repeated_variable_example(seed + 104, arity=3),
    )
    add(
        "local_transition_development",
        _deletion_example(seed + 105, unary=True),
    )
    add(
        "local_transition_development",
        _heterogeneous_example(seed + 106, nested=True),
    )

    packets = []
    expected_records = []
    split_assignments = []
    fingerprints = []
    digests = []
    for partition, example in generated:
        packet, expected, fingerprint = _materialize(example)
        digest = packet_sha256(packet)
        packets.append(packet)
        expected_records.append(expected)
        split_assignments.append(SplitAssignment(digest, partition))
        fingerprints.append(fingerprint)
        digests.append(digest)

    shared_actions = expected_records[shared].transitions
    if len(shared_actions) != 2:
        raise NeuralTcrrBoardError("shared occurrence packet lost its action twins")
    twins = (
        CausalTwinRecord(
            "rhs_pointer",
            digests[rhs_forward],
            digests[rhs_reverse],
            0,
            0,
        ),
        CausalTwinRecord(
            "shared_occurrence",
            digests[shared],
            digests[shared],
            0,
            1,
        ),
        CausalTwinRecord(
            "capacity",
            digests[capacity_16],
            digests[capacity_15],
            0,
            None,
        ),
        CausalTwinRecord(
            "storage_reindex",
            digests[storage_plain],
            digests[storage_reindexed],
        ),
        CausalTwinRecord(
            "rule_reindex",
            digests[rule_plain],
            digests[rule_reindexed],
        ),
    )
    board = LocalTransitionSlice(
        packets=tuple(packets),
        expected_records=tuple(expected_records),
        split_assignments=tuple(split_assignments),
        fingerprints=tuple(fingerprints),
        twins=twins,
    )
    validate_local_transition_slice(board)
    return board


__all__ = [
    "BindingRecord",
    "CausalTwinRecord",
    "ConstructorRecord",
    "ExpectedTransition",
    "ExpectedTransitionRecord",
    "GraphNodeRecord",
    "GraphRecord",
    "LocalTransitionSlice",
    "NeuralTcrrBoardError",
    "PacketFingerprints",
    "RuleRecord",
    "RuleTermRecord",
    "SourceDeletedPacket",
    "SplitAssignment",
    "build_local_transition_slice",
    "packet_fingerprints",
    "packet_sha256",
    "serialize_model_packet",
    "validate_local_transition_slice",
    "validate_model_packet_payload",
    "validate_source_deleted_packet",
    "validate_split_isolation",
]
