"""CPU mechanics for a typed critical-pair rewrite-reactor falsifier.

This module is deliberately limited to symbolic board mechanics.  It contains
no neural model, score claim, source-deleted tensorizer, or host runtime
candidate.  The independent oracle exhaustively follows every legal rewrite
order and canonicalizes terminal term graphs modulo variable alpha-renaming
and physical reservoir-slot permutation.

The production mechanics use **occurrence/path rewriting**.  Rewriting one
root-to-redex path copies only its ancestors; aliases of a shared DAG child are
not rewritten.  A separate nested-reference implementation below independently
checks the production state graph.

The mechanics support:

* typed finite term trees and DAGs in a fixed slot reservoir;
* opaque constructor and rule identifiers;
* alpha-renamable linear and repeated-variable patterns;
* typed RHS construction and bound-subterm pointer reuse;
* destructive cancellation, root deletion, and pointer replacement;
* exhaustive confluent and nonconfluent normal-form enumeration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TypeAlias


class RewriteMechanicsError(ValueError):
    """Raised when a typed rewrite packet violates its mechanics contract."""


class ReservoirExhausted(RewriteMechanicsError):
    """Raised when a rewrite needs more occupied slots than the reservoir."""


class NonTerminatingRewriteSystem(RewriteMechanicsError):
    """Raised when exhaustive enumeration encounters a rewrite cycle."""


@dataclass(frozen=True)
class ConstructorSpec:
    """One opaque typed constructor."""

    identifier: str
    result_type: str
    argument_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class TermNode:
    """One occupied physical slot in a typed term graph."""

    slot: int
    type_id: str
    constructor_id: str | None
    children: tuple[int, ...] = ()
    variable: str | None = None

    @classmethod
    def constructor(
        cls,
        slot: int,
        type_id: str,
        constructor_id: str,
        children: tuple[int, ...] = (),
    ) -> TermNode:
        return cls(
            slot=slot,
            type_id=type_id,
            constructor_id=constructor_id,
            children=children,
        )

    @classmethod
    def free_variable(cls, slot: int, type_id: str, variable: str) -> TermNode:
        return cls(
            slot=slot,
            type_id=type_id,
            constructor_id=None,
            variable=variable,
        )

    def __post_init__(self) -> None:
        if self.slot < 0 or not self.type_id:
            raise RewriteMechanicsError("term node has an invalid slot or type")
        is_constructor = self.constructor_id is not None
        is_variable = self.variable is not None
        if is_constructor == is_variable:
            raise RewriteMechanicsError(
                "term node must be exactly one constructor or free variable"
            )
        if is_variable and self.children:
            raise RewriteMechanicsError("free-variable nodes cannot have children")


@dataclass(frozen=True)
class TermGraph:
    """A rooted typed term DAG backed by a fixed physical slot reservoir."""

    capacity: int
    root: int | None
    nodes: tuple[TermNode, ...]

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise RewriteMechanicsError("reservoir capacity must be positive")
        slots = [node.slot for node in self.nodes]
        if len(slots) != len(set(slots)):
            raise RewriteMechanicsError("physical slots must be unique")
        if any(slot >= self.capacity for slot in slots):
            raise RewriteMechanicsError("occupied slot lies outside the reservoir")
        by_slot = {node.slot: node for node in self.nodes}
        if self.root is None:
            if self.nodes:
                raise RewriteMechanicsError("an empty root cannot retain occupied slots")
            return
        if self.root not in by_slot:
            raise RewriteMechanicsError("root does not name an occupied slot")
        if any(
            child not in by_slot
            for node in self.nodes
            for child in node.children
        ):
            raise RewriteMechanicsError("term pointer names a free slot")

        visiting: set[int] = set()
        reached: set[int] = set()

        def visit(slot: int) -> None:
            if slot in visiting:
                raise RewriteMechanicsError("term graph must be acyclic")
            if slot in reached:
                return
            visiting.add(slot)
            for child in by_slot[slot].children:
                visit(child)
            visiting.remove(slot)
            reached.add(slot)

        visit(self.root)
        if reached != set(slots):
            raise RewriteMechanicsError(
                "all occupied slots must be reachable from the root"
            )
        free_variables = [
            node.variable for node in self.nodes if node.variable is not None
        ]
        if len(free_variables) != len(set(free_variables)):
            raise RewriteMechanicsError(
                "one free variable must occupy one shared DAG slot"
            )

    @property
    def occupied_count(self) -> int:
        return len(self.nodes)

    @property
    def free_count(self) -> int:
        return self.capacity - self.occupied_count

    def node_map(self) -> dict[int, TermNode]:
        return {node.slot: node for node in self.nodes}

    def conservation_receipt(self) -> dict[str, int | bool]:
        return {
            "capacity": self.capacity,
            "occupied": self.occupied_count,
            "free": self.free_count,
            "conserved": self.occupied_count + self.free_count == self.capacity,
        }


@dataclass(frozen=True)
class PatternVariable:
    """A typed alpha-renamable metavariable in a rule LHS."""

    name: str
    type_id: str


@dataclass(frozen=True)
class PatternConstructor:
    """A constructor application in a rule LHS."""

    constructor_id: str
    children: tuple[Pattern, ...] = ()


Pattern: TypeAlias = PatternVariable | PatternConstructor


@dataclass(frozen=True)
class RhsVariable:
    """A pointer to a subterm bound by the LHS."""

    name: str


@dataclass(frozen=True)
class RhsConstructor:
    """A freshly constructed RHS node."""

    constructor_id: str
    children: tuple[RhsExpression, ...] = ()


RhsExpression: TypeAlias = RhsVariable | RhsConstructor


@dataclass(frozen=True)
class RewriteRule:
    """One opaque typed rewrite declaration.

    ``rhs=None`` is an explicit deletion.  It is legal only when the matched
    redex is the graph root, because fixed-arity parent constructors cannot
    retain a missing child.
    """

    identifier: str
    lhs: Pattern
    rhs: RhsExpression | None


@dataclass(frozen=True)
class RewriteSystem:
    """A finite typed constructor catalog and opaque rewrite-card bank."""

    constructors: tuple[ConstructorSpec, ...]
    rules: tuple[RewriteRule, ...]

    def __post_init__(self) -> None:
        constructor_ids = [item.identifier for item in self.constructors]
        rule_ids = [item.identifier for item in self.rules]
        if len(constructor_ids) != len(set(constructor_ids)):
            raise RewriteMechanicsError("constructor identifiers must be unique")
        if len(rule_ids) != len(set(rule_ids)):
            raise RewriteMechanicsError("rule identifiers must be unique")
        if any(not identifier for identifier in (*constructor_ids, *rule_ids)):
            raise RewriteMechanicsError("opaque identifiers cannot be empty")
        for rule in self.rules:
            _validate_rule(self, rule)

    def constructor_map(self) -> dict[str, ConstructorSpec]:
        return {item.identifier: item for item in self.constructors}

    def rule_map(self) -> dict[str, RewriteRule]:
        return {item.identifier: item for item in self.rules}


@dataclass(frozen=True)
class GroundTerm:
    """Storage-independent source term used only to construct board packets."""

    type_id: str
    constructor_id: str | None = None
    children: tuple[GroundTerm, ...] = ()
    variable: str | None = None

    @classmethod
    def constructor(
        cls,
        spec: ConstructorSpec,
        *children: GroundTerm,
    ) -> GroundTerm:
        return cls(
            type_id=spec.result_type,
            constructor_id=spec.identifier,
            children=tuple(children),
        )

    @classmethod
    def free_variable(cls, type_id: str, variable: str) -> GroundTerm:
        return cls(type_id=type_id, variable=variable)

    def __post_init__(self) -> None:
        if (self.constructor_id is None) == (self.variable is None):
            raise RewriteMechanicsError(
                "ground term must be exactly one constructor or free variable"
            )
        if self.variable is not None and self.children:
            raise RewriteMechanicsError("free-variable terms cannot have children")


@dataclass(frozen=True)
class Reduction:
    """One legal opaque rule application at a storage-invariant root path."""

    rule_id: str
    target_slot: int
    target_path: tuple[int, ...]

    @property
    def trace_token(self) -> str:
        path = ".".join(str(index) for index in self.target_path) or "root"
        return f"{self.rule_id}@{path}"


@dataclass(frozen=True)
class OracleTrace:
    """One complete legal reduction order ending at one normal form."""

    normal_form: str
    steps: tuple[str, ...]


@dataclass(frozen=True)
class OracleTransition:
    """One canonical finite-state edge retained by an exhaustive oracle."""

    source: str
    reduction: str
    target: str


@dataclass(frozen=True)
class OracleResult:
    """Deterministic finite canonical state-graph result."""

    normal_forms: tuple[str, ...]
    normal_form_graphs: tuple[TermGraph, ...]
    traces: tuple[OracleTrace, ...]
    states_explored: int
    transitions_explored: int
    transitions: tuple[OracleTransition, ...]
    cyclic_sccs: tuple[tuple[str, ...], ...]
    cyclic_states: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceOracleResult:
    """Independent nested-reference state-graph result."""

    normal_forms: tuple[str, ...]
    traces: tuple[OracleTrace, ...]
    states_explored: int
    transitions_explored: int
    transitions: tuple[OracleTransition, ...]
    cyclic_sccs: tuple[tuple[str, ...], ...]
    cyclic_states: tuple[str, ...]


@dataclass(frozen=True)
class RewriteEpisode:
    """One mechanics-only synthetic episode."""

    name: str
    episode_class: str
    system: RewriteSystem
    initial_graph: TermGraph


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _opaque(namespace: str, seed: int, index: int) -> str:
    payload = f"TCRR:{seed}:{namespace}:{index}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _pattern_type(
    system: RewriteSystem,
    pattern: Pattern,
    variables: dict[str, str],
) -> str:
    if isinstance(pattern, PatternVariable):
        previous = variables.setdefault(pattern.name, pattern.type_id)
        if previous != pattern.type_id:
            raise RewriteMechanicsError(
                "repeated LHS variable declarations must preserve type"
            )
        return pattern.type_id
    constructors = system.constructor_map()
    spec = constructors.get(pattern.constructor_id)
    if spec is None or len(pattern.children) != len(spec.argument_types):
        raise RewriteMechanicsError("LHS constructor is absent or has wrong arity")
    child_types = tuple(
        _pattern_type(system, child, variables) for child in pattern.children
    )
    if child_types != spec.argument_types:
        raise RewriteMechanicsError("LHS child type differs from constructor type")
    return spec.result_type


def _rhs_type(
    system: RewriteSystem,
    expression: RhsExpression,
    variables: dict[str, str],
) -> str:
    if isinstance(expression, RhsVariable):
        if expression.name not in variables:
            raise RewriteMechanicsError("RHS refers to an unbound variable")
        return variables[expression.name]
    constructors = system.constructor_map()
    spec = constructors.get(expression.constructor_id)
    if spec is None or len(expression.children) != len(spec.argument_types):
        raise RewriteMechanicsError("RHS constructor is absent or has wrong arity")
    child_types = tuple(
        _rhs_type(system, child, variables) for child in expression.children
    )
    if child_types != spec.argument_types:
        raise RewriteMechanicsError("RHS child type differs from constructor type")
    return spec.result_type


def _validate_rule(system: RewriteSystem, rule: RewriteRule) -> None:
    variables: dict[str, str] = {}
    lhs_type = _pattern_type(system, rule.lhs, variables)
    if rule.rhs is not None and _rhs_type(system, rule.rhs, variables) != lhs_type:
        raise RewriteMechanicsError("rewrite must preserve the redex root type")


def validate_graph(system: RewriteSystem, graph: TermGraph) -> None:
    """Validate constructor arity and every edge type in a term graph."""

    constructors = system.constructor_map()
    by_slot = graph.node_map()
    for node in graph.nodes:
        if node.constructor_id is None:
            continue
        spec = constructors.get(node.constructor_id)
        if spec is None:
            raise RewriteMechanicsError("graph uses an unknown constructor")
        if node.type_id != spec.result_type:
            raise RewriteMechanicsError("constructor result type differs")
        if len(node.children) != len(spec.argument_types):
            raise RewriteMechanicsError("constructor arity differs")
        child_types = tuple(by_slot[slot].type_id for slot in node.children)
        if child_types != spec.argument_types:
            raise RewriteMechanicsError("constructor child type differs")


def pack_ground_term(
    system: RewriteSystem,
    term: GroundTerm,
    capacity: int,
    *,
    slot_permutation: tuple[int, ...] | None = None,
) -> TermGraph:
    """Pack a source term into a fixed reservoir.

    Reusing the same ``GroundTerm`` object creates a shared DAG node.  Equal
    but separately constructed objects remain separate nodes, which is
    important for testing structural repeated-variable equality.
    """

    if slot_permutation is None:
        slot_permutation = tuple(range(capacity))
    if sorted(slot_permutation) != list(range(capacity)):
        raise RewriteMechanicsError("slot permutation must cover the reservoir")
    constructors = system.constructor_map()
    semantic_nodes: list[GroundTerm] = []
    by_identity: dict[int, int] = {}

    def visit(item: GroundTerm) -> int:
        identity = id(item)
        if identity in by_identity:
            return by_identity[identity]
        semantic_index = len(semantic_nodes)
        by_identity[identity] = semantic_index
        semantic_nodes.append(item)
        for child in item.children:
            visit(child)
        return semantic_index

    root_index = visit(term)
    if len(semantic_nodes) > capacity:
        raise ReservoirExhausted("source term does not fit the fixed reservoir")
    semantic_to_slot = {
        index: slot_permutation[index] for index in range(len(semantic_nodes))
    }
    index_by_identity = {
        id(item): index for index, item in enumerate(semantic_nodes)
    }
    nodes = []
    for index, item in enumerate(semantic_nodes):
        if item.constructor_id is None:
            nodes.append(
                TermNode.free_variable(
                    semantic_to_slot[index],
                    item.type_id,
                    str(item.variable),
                )
            )
            continue
        spec = constructors.get(item.constructor_id)
        if spec is None:
            raise RewriteMechanicsError("source term uses an unknown constructor")
        child_slots = tuple(
            semantic_to_slot[index_by_identity[id(child)]]
            for child in item.children
        )
        nodes.append(
            TermNode.constructor(
                semantic_to_slot[index],
                item.type_id,
                item.constructor_id,
                child_slots,
            )
        )
    graph = TermGraph(
        capacity=capacity,
        root=semantic_to_slot[root_index],
        nodes=tuple(nodes),
    )
    validate_graph(system, graph)
    return graph


def canonical_graph_payload(graph: TermGraph) -> dict[str, object]:
    """Serialize modulo physical slots and free-variable alpha names."""

    if graph.root is None:
        return {"capacity": graph.capacity, "root": None, "nodes": []}
    by_slot = graph.node_map()
    canonical_indices: dict[int, int] = {}
    records: list[dict[str, object] | None] = []
    variable_indices: dict[str, int] = {}

    def visit(slot: int) -> int:
        if slot in canonical_indices:
            return canonical_indices[slot]
        index = len(records)
        canonical_indices[slot] = index
        records.append(None)
        node = by_slot[slot]
        if node.variable is not None:
            variable_index = variable_indices.setdefault(
                node.variable,
                len(variable_indices),
            )
            record: dict[str, object] = {
                "kind": "variable",
                "type": node.type_id,
                "alpha": variable_index,
            }
        else:
            record = {
                "kind": "constructor",
                "type": node.type_id,
                "constructor": node.constructor_id,
                "children": [visit(child) for child in node.children],
            }
        records[index] = record
        return index

    root_index = visit(graph.root)
    return {
        "capacity": graph.capacity,
        "root": root_index,
        "nodes": records,
    }


def canonical_graph_serialization(graph: TermGraph) -> str:
    return _canonical_json(canonical_graph_payload(graph))


def _subterm_key(graph: TermGraph, slot: int) -> tuple[object, ...]:
    """Return structural term equality, intentionally ignoring slot identity."""

    by_slot = graph.node_map()
    memo: dict[int, tuple[object, ...]] = {}

    def visit(current: int) -> tuple[object, ...]:
        if current in memo:
            return memo[current]
        node = by_slot[current]
        if node.variable is not None:
            value: tuple[object, ...] = (
                "variable",
                node.type_id,
                node.variable,
            )
        else:
            value = (
                "constructor",
                node.type_id,
                node.constructor_id,
                tuple(visit(child) for child in node.children),
            )
        memo[current] = value
        return value

    return visit(slot)


def _binding_equal(graph: TermGraph, left: int, right: int) -> bool:
    """Correct repeated-variable equality is structural, not physical identity."""

    return _subterm_key(graph, left) == _subterm_key(graph, right)


def _match_pattern(
    system: RewriteSystem,
    graph: TermGraph,
    pattern: Pattern,
    slot: int,
    bindings: dict[str, int],
) -> bool:
    node = graph.node_map()[slot]
    if isinstance(pattern, PatternVariable):
        if node.type_id != pattern.type_id:
            return False
        previous = bindings.get(pattern.name)
        if previous is None:
            bindings[pattern.name] = slot
            return True
        return _binding_equal(graph, previous, slot)
    if node.constructor_id != pattern.constructor_id:
        return False
    spec = system.constructor_map()[pattern.constructor_id]
    if len(node.children) != len(pattern.children) != len(spec.argument_types):
        return False
    return all(
        _match_pattern(system, graph, child_pattern, child_slot, bindings)
        for child_pattern, child_slot in zip(
            pattern.children,
            node.children,
            strict=True,
        )
    )


def _reachable_occurrences(
    graph: TermGraph,
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Enumerate every root occurrence, including aliases of shared DAG nodes."""

    if graph.root is None:
        return ()
    by_slot = graph.node_map()
    occurrences: list[tuple[tuple[int, ...], int]] = []

    def visit(slot: int, path: tuple[int, ...]) -> None:
        occurrences.append((path, slot))
        for index, child in enumerate(by_slot[slot].children):
            visit(child, (*path, index))

    visit(graph.root, ())
    return tuple(occurrences)


def _resolve_path(graph: TermGraph, path: tuple[int, ...]) -> int:
    if graph.root is None:
        raise RewriteMechanicsError("an empty graph has no occurrences")
    by_slot = graph.node_map()
    slot = graph.root
    for child_index in path:
        children = by_slot[slot].children
        if child_index < 0 or child_index >= len(children):
            raise RewriteMechanicsError("reduction path leaves the term graph")
        slot = children[child_index]
    return slot


@dataclass(frozen=True)
class _SemanticNode:
    type_id: str
    constructor_id: str | None
    children: tuple[_Handle, ...] = ()
    variable: str | None = None


_Handle: TypeAlias = tuple[str, int]


def _production_rhs_children(
    expression: RhsConstructor,
) -> tuple[RhsExpression, ...]:
    """Production RHS pointer order, isolated for hostile mutation tests."""

    return expression.children


def _rewrite_occurrence(
    system: RewriteSystem,
    graph: TermGraph,
    target_slot: int,
    target_path: tuple[int, ...],
    rule: RewriteRule,
    bindings: dict[str, int],
) -> TermGraph:
    """Rewrite exactly one root occurrence and path-copy its ancestors."""

    by_slot = graph.node_map()
    semantic_nodes: dict[_Handle, _SemanticNode] = {}
    for slot, node in by_slot.items():
        semantic_nodes[("old", slot)] = _SemanticNode(
            type_id=node.type_id,
            constructor_id=node.constructor_id,
            children=tuple(("old", child) for child in node.children),
            variable=node.variable,
        )

    next_new = 0

    def build_rhs(expression: RhsExpression) -> _Handle:
        nonlocal next_new
        if isinstance(expression, RhsVariable):
            return ("old", bindings[expression.name])
        spec = system.constructor_map()[expression.constructor_id]
        handle = ("new", next_new)
        next_new += 1
        semantic_nodes[handle] = _SemanticNode(
            type_id=spec.result_type,
            constructor_id=expression.constructor_id,
            children=tuple(
                build_rhs(child)
                for child in _production_rhs_children(expression)
            ),
        )
        return handle

    replacement = None if rule.rhs is None else build_rhs(rule.rhs)

    def transform(slot: int, depth: int) -> _Handle | None:
        if depth == len(target_path):
            if slot != target_slot:
                raise RewriteMechanicsError(
                    "target path and target slot identify different occurrences"
                )
            return replacement
        original = semantic_nodes[("old", slot)]
        if original.variable is not None:
            raise RewriteMechanicsError("target path descends through a variable")
        child_index = target_path[depth]
        if child_index < 0 or child_index >= len(original.children):
            raise RewriteMechanicsError("target path leaves an ancestor")
        copied_children = list(original.children)
        selected_slot = by_slot[slot].children[child_index]
        selected = transform(selected_slot, depth + 1)
        if selected is None:
            raise RewriteMechanicsError(
                "explicit deletion is legal only at the graph root"
            )
        copied_children[child_index] = selected
        handle = ("copy", slot)
        semantic_nodes[handle] = _SemanticNode(
            type_id=original.type_id,
            constructor_id=original.constructor_id,
            children=tuple(copied_children),
        )
        return handle

    if graph.root is None:
        raise RewriteMechanicsError("cannot rewrite an empty graph")
    new_root = transform(graph.root, 0)
    if new_root is None:
        return TermGraph(capacity=graph.capacity, root=None, nodes=())
    return _pack_semantic_graph(system, graph.capacity, new_root, semantic_nodes)


def _pack_semantic_graph(
    system: RewriteSystem,
    capacity: int,
    root: _Handle,
    semantic_nodes: dict[_Handle, _SemanticNode],
) -> TermGraph:
    ordered: list[_Handle] = []
    seen: set[_Handle] = set()

    def visit(handle: _Handle) -> None:
        if handle in seen:
            return
        seen.add(handle)
        ordered.append(handle)
        for child in semantic_nodes[handle].children:
            visit(child)

    visit(root)
    if len(ordered) > capacity:
        raise ReservoirExhausted("rewrite exceeds the fixed slot reservoir")
    slots = {handle: index for index, handle in enumerate(ordered)}
    nodes = []
    for handle in ordered:
        semantic = semantic_nodes[handle]
        slot = slots[handle]
        if semantic.variable is not None:
            nodes.append(
                TermNode.free_variable(
                    slot,
                    semantic.type_id,
                    semantic.variable,
                )
            )
        else:
            nodes.append(
                TermNode.constructor(
                    slot,
                    semantic.type_id,
                    str(semantic.constructor_id),
                    tuple(slots[child] for child in semantic.children),
                )
            )
    graph = TermGraph(capacity=capacity, root=slots[root], nodes=tuple(nodes))
    validate_graph(system, graph)
    return graph


def apply_reduction(
    system: RewriteSystem,
    graph: TermGraph,
    reduction: Reduction,
) -> TermGraph:
    """Apply one previously identified legal reduction transactionally."""

    validate_graph(system, graph)
    rule = system.rule_map().get(reduction.rule_id)
    if rule is None or reduction.target_slot not in graph.node_map():
        raise RewriteMechanicsError("reduction no longer names a rule or redex")
    resolved_slot = _resolve_path(graph, reduction.target_path)
    if resolved_slot != reduction.target_slot:
        raise RewriteMechanicsError(
            "reduction path no longer identifies its target occurrence"
        )
    bindings: dict[str, int] = {}
    if not _match_pattern(
        system,
        graph,
        rule.lhs,
        reduction.target_slot,
        bindings,
    ):
        raise RewriteMechanicsError("reduction target no longer matches its LHS")
    result = _rewrite_occurrence(
        system,
        graph,
        reduction.target_slot,
        reduction.target_path,
        rule,
        bindings,
    )
    if not result.conservation_receipt()["conserved"]:
        raise RewriteMechanicsError("reservoir conservation failed")
    return result


def legal_reductions(
    system: RewriteSystem,
    graph: TermGraph,
) -> tuple[Reduction, ...]:
    """Return every legal rule/redex pair in deterministic semantic order."""

    validate_graph(system, graph)
    candidates = []
    for rule in system.rules:
        for target_path, target_slot in _reachable_occurrences(graph):
            bindings: dict[str, int] = {}
            if not _match_pattern(
                system,
                graph,
                rule.lhs,
                target_slot,
                bindings,
            ):
                continue
            reduction = Reduction(rule.identifier, target_slot, target_path)
            try:
                result = apply_reduction(system, graph, reduction)
            except (ReservoirExhausted, RewriteMechanicsError):
                continue
            if canonical_graph_serialization(result) == canonical_graph_serialization(
                graph
            ):
                continue
            candidates.append(reduction)
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.rule_id, item.target_path),
        )
    )


def _cyclic_components(
    state_keys: tuple[str, ...],
    transitions: tuple[OracleTransition, ...],
) -> tuple[tuple[str, ...], ...]:
    """Compute cyclic SCCs of a finite canonical state graph."""

    adjacency = {key: set() for key in state_keys}
    for edge in transitions:
        adjacency[edge.source].add(edge.target)
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def strong_connect(vertex: str) -> None:
        nonlocal index
        indices[vertex] = index
        lowlinks[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack.add(vertex)
        for successor in sorted(adjacency[vertex]):
            if successor not in indices:
                strong_connect(successor)
                lowlinks[vertex] = min(lowlinks[vertex], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[vertex] = min(lowlinks[vertex], indices[successor])
        if lowlinks[vertex] != indices[vertex]:
            return
        component = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == vertex:
                break
        ordered = tuple(sorted(component))
        if len(ordered) > 1 or ordered[0] in adjacency[ordered[0]]:
            components.append(ordered)

    for state_key in sorted(state_keys):
        if state_key not in indices:
            strong_connect(state_key)
    return tuple(sorted(components))


def _trace_from_parents(
    terminal: str,
    parents: dict[str, tuple[str, str] | None],
) -> tuple[str, ...]:
    trace = []
    cursor = terminal
    while parents[cursor] is not None:
        parent, token = parents[cursor]  # type: ignore[misc]
        trace.append(token)
        cursor = parent
    return tuple(reversed(trace))


class ProductionRewriteStateOracle:
    """Explore the complete reachable finite canonical production state graph."""

    def __init__(self, *, maximum_states: int = 100_000):
        if maximum_states <= 0:
            raise RewriteMechanicsError("maximum_states must be positive")
        self.maximum_states = maximum_states

    def enumerate(
        self,
        system: RewriteSystem,
        initial_graph: TermGraph,
    ) -> OracleResult:
        validate_graph(system, initial_graph)
        initial_key = canonical_graph_serialization(initial_graph)
        states = {initial_key: initial_graph}
        parents: dict[str, tuple[str, str] | None] = {initial_key: None}
        queue = [initial_key]
        terminal_keys: set[str] = set()
        transition_set: set[OracleTransition] = set()
        while queue:
            source_key = queue.pop(0)
            source_graph = states[source_key]
            reductions = legal_reductions(system, source_graph)
            if not reductions:
                terminal_keys.add(source_key)
            for reduction in reductions:
                successor = apply_reduction(system, source_graph, reduction)
                target_key = canonical_graph_serialization(successor)
                edge = OracleTransition(
                    source_key,
                    reduction.trace_token,
                    target_key,
                )
                transition_set.add(edge)
                if target_key in states:
                    continue
                if len(states) >= self.maximum_states:
                    raise RewriteMechanicsError(
                        "canonical production state budget exceeded"
                    )
                states[target_key] = successor
                parents[target_key] = (source_key, reduction.trace_token)
                queue.append(target_key)

        transitions = tuple(
            sorted(
                transition_set,
                key=lambda edge: (edge.source, edge.reduction, edge.target),
            )
        )
        cyclic_sccs = _cyclic_components(tuple(states), transitions)
        cyclic_states = tuple(
            sorted({state for component in cyclic_sccs for state in component})
        )
        normal_forms = tuple(sorted(terminal_keys))
        return OracleResult(
            normal_forms=normal_forms,
            normal_form_graphs=tuple(states[key] for key in normal_forms),
            traces=tuple(
                OracleTrace(key, _trace_from_parents(key, parents))
                for key in normal_forms
            ),
            states_explored=len(states),
            transitions_explored=len(transitions),
            transitions=transitions,
            cyclic_sccs=cyclic_sccs,
            cyclic_states=cyclic_states,
        )


@dataclass(frozen=True, eq=False)
class _ReferenceTerm:
    """Immutable nested reference term; object identity represents DAG sharing."""

    type_id: str
    constructor_id: str | None
    children: tuple[_ReferenceTerm, ...] = ()
    variable: str | None = None


@dataclass(frozen=True)
class _ReferenceState:
    capacity: int
    root: _ReferenceTerm | None


def _reference_from_graph(graph: TermGraph) -> _ReferenceState:
    """Convert physical slots to nested immutable terms without production helpers."""

    if graph.root is None:
        return _ReferenceState(graph.capacity, None)
    nodes = {node.slot: node for node in graph.nodes}
    memo: dict[int, _ReferenceTerm] = {}

    def build(slot: int) -> _ReferenceTerm:
        if slot in memo:
            return memo[slot]
        node = nodes[slot]
        term = _ReferenceTerm(
            type_id=node.type_id,
            constructor_id=node.constructor_id,
            children=tuple(build(child) for child in node.children),
            variable=node.variable,
        )
        memo[slot] = term
        return term

    return _ReferenceState(graph.capacity, build(graph.root))


def _reference_canonical_serialization(state: _ReferenceState) -> str:
    """Independent alpha/storage-invariant nested-reference canonicalizer."""

    if state.root is None:
        payload: dict[str, object] = {
            "capacity": state.capacity,
            "root": None,
            "nodes": [],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    indices: dict[int, int] = {}
    records: list[dict[str, object] | None] = []
    variables: dict[str, int] = {}

    def visit(term: _ReferenceTerm) -> int:
        identity = id(term)
        if identity in indices:
            return indices[identity]
        index = len(records)
        indices[identity] = index
        records.append(None)
        if term.variable is not None:
            record: dict[str, object] = {
                "kind": "variable",
                "type": term.type_id,
                "alpha": variables.setdefault(
                    term.variable,
                    len(variables),
                ),
            }
        else:
            record = {
                "kind": "constructor",
                "type": term.type_id,
                "constructor": term.constructor_id,
                "children": [visit(child) for child in term.children],
            }
        records[index] = record
        return index

    root_index = visit(state.root)
    payload = {
        "capacity": state.capacity,
        "root": root_index,
        "nodes": records,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _reference_structural_key(term: _ReferenceTerm) -> tuple[object, ...]:
    memo: dict[int, tuple[object, ...]] = {}

    def visit(item: _ReferenceTerm) -> tuple[object, ...]:
        identity = id(item)
        if identity in memo:
            return memo[identity]
        if item.variable is not None:
            key: tuple[object, ...] = (
                "variable",
                item.type_id,
                item.variable,
            )
        else:
            key = (
                "constructor",
                item.type_id,
                item.constructor_id,
                tuple(visit(child) for child in item.children),
            )
        memo[identity] = key
        return key

    return visit(term)


def _reference_match(
    pattern: Pattern,
    term: _ReferenceTerm,
    bindings: dict[str, _ReferenceTerm],
) -> bool:
    """Independent recursive matcher for the nested reference representation."""

    if isinstance(pattern, PatternVariable):
        if term.type_id != pattern.type_id:
            return False
        previous = bindings.get(pattern.name)
        if previous is None:
            bindings[pattern.name] = term
            return True
        return _reference_structural_key(previous) == (
            _reference_structural_key(term)
        )
    if term.constructor_id != pattern.constructor_id:
        return False
    if len(term.children) != len(pattern.children):
        return False
    return all(
        _reference_match(child_pattern, child_term, bindings)
        for child_pattern, child_term in zip(
            pattern.children,
            term.children,
            strict=True,
        )
    )


def _reference_occurrences(
    root: _ReferenceTerm | None,
) -> tuple[tuple[tuple[int, ...], _ReferenceTerm], ...]:
    output: list[tuple[tuple[int, ...], _ReferenceTerm]] = []

    def visit(term: _ReferenceTerm, path: tuple[int, ...]) -> None:
        output.append((path, term))
        for child_index, child in enumerate(term.children):
            visit(child, (*path, child_index))

    if root is not None:
        visit(root, ())
    return tuple(output)


def _reference_occupied_count(root: _ReferenceTerm | None) -> int:
    seen: set[int] = set()

    def visit(term: _ReferenceTerm) -> None:
        identity = id(term)
        if identity in seen:
            return
        seen.add(identity)
        for child in term.children:
            visit(child)

    if root is not None:
        visit(root)
    return len(seen)


def _reference_rewrite(
    system: RewriteSystem,
    state: _ReferenceState,
    rule: RewriteRule,
    path: tuple[int, ...],
    target: _ReferenceTerm,
    bindings: dict[str, _ReferenceTerm],
) -> _ReferenceState:
    """Independent occurrence rewrite over immutable nested reference terms."""

    constructors = {
        constructor.identifier: constructor
        for constructor in system.constructors
    }

    def build_rhs(expression: RhsExpression) -> _ReferenceTerm:
        if isinstance(expression, RhsVariable):
            return bindings[expression.name]
        spec = constructors[expression.constructor_id]
        return _ReferenceTerm(
            type_id=spec.result_type,
            constructor_id=expression.constructor_id,
            children=tuple(build_rhs(child) for child in expression.children),
        )

    replacement = None if rule.rhs is None else build_rhs(rule.rhs)

    def replace(term: _ReferenceTerm, depth: int) -> _ReferenceTerm | None:
        if depth == len(path):
            if term is not target:
                raise RewriteMechanicsError(
                    "reference path and occurrence identity differ"
                )
            return replacement
        child_index = path[depth]
        if child_index < 0 or child_index >= len(term.children):
            raise RewriteMechanicsError("reference path leaves an ancestor")
        children = list(term.children)
        selected = replace(children[child_index], depth + 1)
        if selected is None:
            raise RewriteMechanicsError(
                "reference deletion is legal only at the root"
            )
        children[child_index] = selected
        return _ReferenceTerm(
            type_id=term.type_id,
            constructor_id=term.constructor_id,
            children=tuple(children),
            variable=term.variable,
        )

    if state.root is None:
        raise RewriteMechanicsError("cannot rewrite an empty reference state")
    root = replace(state.root, 0)
    if _reference_occupied_count(root) > state.capacity:
        raise ReservoirExhausted("reference rewrite exhausts its reservoir")
    return _ReferenceState(state.capacity, root)


def _reference_legal_transitions(
    system: RewriteSystem,
    state: _ReferenceState,
) -> tuple[tuple[str, _ReferenceState], ...]:
    transitions = []
    source_key = _reference_canonical_serialization(state)
    for rule in system.rules:
        for path, target in _reference_occurrences(state.root):
            bindings: dict[str, _ReferenceTerm] = {}
            if not _reference_match(rule.lhs, target, bindings):
                continue
            try:
                successor = _reference_rewrite(
                    system,
                    state,
                    rule,
                    path,
                    target,
                    bindings,
                )
            except (ReservoirExhausted, RewriteMechanicsError):
                continue
            target_key = _reference_canonical_serialization(successor)
            if target_key == source_key:
                continue
            path_text = ".".join(str(index) for index in path) or "root"
            transitions.append((f"{rule.identifier}@{path_text}", successor))
    return tuple(
        sorted(
            transitions,
            key=lambda item: (
                item[0],
                _reference_canonical_serialization(item[1]),
            ),
        )
    )


def _reference_cyclic_components(
    state_keys: tuple[str, ...],
    transitions: tuple[OracleTransition, ...],
) -> tuple[tuple[str, ...], ...]:
    """Independent Tarjan pass for the nested-reference state graph."""

    outgoing = {key: set() for key in state_keys}
    for transition in transitions:
        outgoing[transition.source].add(transition.target)
    next_index = 0
    index_by_state: dict[str, int] = {}
    low_by_state: dict[str, int] = {}
    stack: list[str] = []
    stacked: set[str] = set()
    cyclic: list[tuple[str, ...]] = []

    def visit(state_key: str) -> None:
        nonlocal next_index
        index_by_state[state_key] = next_index
        low_by_state[state_key] = next_index
        next_index += 1
        stack.append(state_key)
        stacked.add(state_key)
        for successor in sorted(outgoing[state_key]):
            if successor not in index_by_state:
                visit(successor)
                low_by_state[state_key] = min(
                    low_by_state[state_key],
                    low_by_state[successor],
                )
            elif successor in stacked:
                low_by_state[state_key] = min(
                    low_by_state[state_key],
                    index_by_state[successor],
                )
        if low_by_state[state_key] != index_by_state[state_key]:
            return
        component = []
        while True:
            member = stack.pop()
            stacked.remove(member)
            component.append(member)
            if member == state_key:
                break
        ordered = tuple(sorted(component))
        if len(ordered) > 1 or ordered[0] in outgoing[ordered[0]]:
            cyclic.append(ordered)

    for state_key in sorted(state_keys):
        if state_key not in index_by_state:
            visit(state_key)
    return tuple(sorted(cyclic))


class IndependentNestedReferenceOracle:
    """Independent immutable nested-term oracle for production cross-checking."""

    def __init__(self, *, maximum_states: int = 100_000):
        if maximum_states <= 0:
            raise RewriteMechanicsError("maximum_states must be positive")
        self.maximum_states = maximum_states

    def enumerate(
        self,
        system: RewriteSystem,
        initial_graph: TermGraph,
    ) -> ReferenceOracleResult:
        initial = _reference_from_graph(initial_graph)
        initial_key = _reference_canonical_serialization(initial)
        states = {initial_key: initial}
        parents: dict[str, tuple[str, str] | None] = {initial_key: None}
        queue = [initial_key]
        terminal_keys: set[str] = set()
        edge_set: set[OracleTransition] = set()
        while queue:
            source_key = queue.pop(0)
            candidates = _reference_legal_transitions(
                system,
                states[source_key],
            )
            if not candidates:
                terminal_keys.add(source_key)
            for token, successor in candidates:
                target_key = _reference_canonical_serialization(successor)
                edge_set.add(OracleTransition(source_key, token, target_key))
                if target_key in states:
                    continue
                if len(states) >= self.maximum_states:
                    raise RewriteMechanicsError(
                        "canonical reference state budget exceeded"
                    )
                states[target_key] = successor
                parents[target_key] = (source_key, token)
                queue.append(target_key)
        transitions = tuple(
            sorted(
                edge_set,
                key=lambda edge: (edge.source, edge.reduction, edge.target),
            )
        )
        cyclic_sccs = _reference_cyclic_components(
            tuple(states),
            transitions,
        )
        cyclic_states = tuple(
            sorted({state for component in cyclic_sccs for state in component})
        )
        normal_forms = tuple(sorted(terminal_keys))

        def reference_trace(terminal: str) -> tuple[str, ...]:
            steps = []
            cursor = terminal
            while parents[cursor] is not None:
                parent, token = parents[cursor]  # type: ignore[misc]
                steps.append(token)
                cursor = parent
            return tuple(reversed(steps))

        return ReferenceOracleResult(
            normal_forms=normal_forms,
            traces=tuple(
                OracleTrace(key, reference_trace(key)) for key in normal_forms
            ),
            states_explored=len(states),
            transitions_explored=len(transitions),
            transitions=transitions,
            cyclic_sccs=cyclic_sccs,
            cyclic_states=cyclic_states,
        )


def greedy_normal_form(
    system: RewriteSystem,
    graph: TermGraph,
    *,
    maximum_steps: int = 1_000,
) -> TermGraph:
    """Diagnostic greedy reducer used only as a hostile comparison."""

    active = graph
    seen = set()
    for _ in range(maximum_steps):
        key = canonical_graph_serialization(active)
        if key in seen:
            raise NonTerminatingRewriteSystem("greedy reducer entered a cycle")
        seen.add(key)
        reductions = legal_reductions(system, active)
        if not reductions:
            return active
        active = apply_reduction(system, active, reductions[0])
    raise RewriteMechanicsError("greedy reducer exceeded its step limit")


def reindex_graph(graph: TermGraph, permutation: tuple[int, ...]) -> TermGraph:
    """Apply an arbitrary physical reservoir-slot permutation."""

    if sorted(permutation) != list(range(graph.capacity)):
        raise RewriteMechanicsError("slot permutation must cover the reservoir")
    remap = {old: permutation[old] for old in range(graph.capacity)}
    nodes = tuple(
        TermNode(
            slot=remap[node.slot],
            type_id=node.type_id,
            constructor_id=node.constructor_id,
            children=tuple(remap[child] for child in node.children),
            variable=node.variable,
        )
        for node in reversed(graph.nodes)
    )
    return TermGraph(
        capacity=graph.capacity,
        root=None if graph.root is None else remap[graph.root],
        nodes=nodes,
    )


def canonical_rule_payload(rule: RewriteRule) -> dict[str, object]:
    """Serialize a rule modulo its LHS metavariable names."""

    variables: dict[str, int] = {}

    def lhs_payload(pattern: Pattern) -> dict[str, object]:
        if isinstance(pattern, PatternVariable):
            index = variables.setdefault(pattern.name, len(variables))
            return {"kind": "variable", "alpha": index, "type": pattern.type_id}
        return {
            "kind": "constructor",
            "constructor": pattern.constructor_id,
            "children": [lhs_payload(child) for child in pattern.children],
        }

    def rhs_payload(expression: RhsExpression) -> dict[str, object]:
        if isinstance(expression, RhsVariable):
            return {"kind": "variable", "alpha": variables[expression.name]}
        return {
            "kind": "constructor",
            "constructor": expression.constructor_id,
            "children": [rhs_payload(child) for child in expression.children],
        }

    return {
        "rule": rule.identifier,
        "lhs": lhs_payload(rule.lhs),
        "rhs": None if rule.rhs is None else rhs_payload(rule.rhs),
    }


def rule_shape_payload(rule: RewriteRule) -> dict[str, object]:
    """Mask binder pointers while retaining exact constructor/arity statistics."""

    payload = canonical_rule_payload(rule)

    def mask(value: object) -> object:
        if isinstance(value, list):
            return [mask(item) for item in value]
        if isinstance(value, dict):
            if value.get("kind") == "variable":
                return {"kind": "variable", "type": value.get("type")}
            return {key: mask(item) for key, item in value.items()}
        return value

    return mask(payload)  # type: ignore[return-value]


def alpha_rename_rule(
    rule: RewriteRule,
    renaming: dict[str, str],
) -> RewriteRule:
    """Rename every metavariable consistently without changing rule semantics."""

    bound_names: set[str] = set()

    def collect(pattern: Pattern) -> None:
        if isinstance(pattern, PatternVariable):
            bound_names.add(pattern.name)
            return
        for child in pattern.children:
            collect(child)

    collect(rule.lhs)
    if set(renaming) - bound_names:
        raise RewriteMechanicsError("alpha renaming names an unbound variable")
    renamed_names = [
        renaming.get(name, name) for name in sorted(bound_names)
    ]
    if any(not name for name in renamed_names):
        raise RewriteMechanicsError("alpha-renamed variables cannot be empty")
    if len(renamed_names) != len(set(renamed_names)):
        raise RewriteMechanicsError(
            "alpha renaming must be injective and cannot capture a binder"
        )

    def rename_lhs(pattern: Pattern) -> Pattern:
        if isinstance(pattern, PatternVariable):
            return PatternVariable(
                renaming.get(pattern.name, pattern.name),
                pattern.type_id,
            )
        return PatternConstructor(
            pattern.constructor_id,
            tuple(rename_lhs(child) for child in pattern.children),
        )

    def rename_rhs(expression: RhsExpression) -> RhsExpression:
        if isinstance(expression, RhsVariable):
            return RhsVariable(renaming.get(expression.name, expression.name))
        return RhsConstructor(
            expression.constructor_id,
            tuple(rename_rhs(child) for child in expression.children),
        )

    return RewriteRule(
        identifier=rule.identifier,
        lhs=rename_lhs(rule.lhs),
        rhs=None if rule.rhs is None else rename_rhs(rule.rhs),
    )


class _EpisodeFactory:
    def __init__(self, seed: int):
        self.seed = seed
        self.type_id = _opaque("type", seed, 0)
        self._constructor_index = 0
        self._rule_index = 0
        self.constructors: dict[str, ConstructorSpec] = {}

    def constructor(self, alias: str, arity: int) -> ConstructorSpec:
        identifier = _opaque("constructor", self.seed, self._constructor_index)
        self._constructor_index += 1
        spec = ConstructorSpec(
            identifier,
            self.type_id,
            (self.type_id,) * arity,
        )
        self.constructors[alias] = spec
        return spec

    def rule_id(self) -> str:
        identifier = _opaque("rule", self.seed, self._rule_index)
        self._rule_index += 1
        return identifier

    def system(self, *rules: RewriteRule) -> RewriteSystem:
        return RewriteSystem(tuple(self.constructors.values()), tuple(rules))

    def leaf(self, alias: str) -> GroundTerm:
        return GroundTerm.constructor(self.constructors[alias])

    def term(self, alias: str, *children: GroundTerm) -> GroundTerm:
        return GroundTerm.constructor(self.constructors[alias], *children)

    def variable(self, name: str) -> PatternVariable:
        return PatternVariable(name, self.type_id)

    def pattern(self, alias: str, *children: Pattern) -> PatternConstructor:
        return PatternConstructor(
            self.constructors[alias].identifier,
            tuple(children),
        )

    def rhs(self, alias: str, *children: RhsExpression) -> RhsConstructor:
        return RhsConstructor(
            self.constructors[alias].identifier,
            tuple(children),
        )


def _independent_episode(seed: int) -> RewriteEpisode:
    factory = _EpisodeFactory(seed)
    for alias, arity in (
        ("left", 0),
        ("right", 0),
        ("left_nf", 0),
        ("right_nf", 0),
        ("pair", 2),
    ):
        factory.constructor(alias, arity)
    rules = (
        RewriteRule(
            factory.rule_id(),
            factory.pattern("left"),
            factory.rhs("left_nf"),
        ),
        RewriteRule(
            factory.rule_id(),
            factory.pattern("right"),
            factory.rhs("right_nf"),
        ),
    )
    system = factory.system(*rules)
    graph = pack_ground_term(
        system,
        factory.term("pair", factory.leaf("left"), factory.leaf("right")),
        8,
    )
    return RewriteEpisode("independent", "independent_redexes", system, graph)


def _diamond_episode(seed: int) -> RewriteEpisode:
    factory = _EpisodeFactory(seed)
    for alias in ("source", "left", "right", "normal"):
        factory.constructor(alias, 0)
    rules = (
        RewriteRule(
            factory.rule_id(),
            factory.pattern("source"),
            factory.rhs("left"),
        ),
        RewriteRule(
            factory.rule_id(),
            factory.pattern("source"),
            factory.rhs("right"),
        ),
        RewriteRule(
            factory.rule_id(),
            factory.pattern("left"),
            factory.rhs("normal"),
        ),
        RewriteRule(
            factory.rule_id(),
            factory.pattern("right"),
            factory.rhs("normal"),
        ),
    )
    system = factory.system(*rules)
    graph = pack_ground_term(system, factory.leaf("source"), 6)
    return RewriteEpisode("diamond", "confluent_diamond", system, graph)


def _fork_episode(seed: int) -> RewriteEpisode:
    factory = _EpisodeFactory(seed)
    for alias in ("source", "left_nf", "right_nf"):
        factory.constructor(alias, 0)
    rules = (
        RewriteRule(
            factory.rule_id(),
            factory.pattern("source"),
            factory.rhs("left_nf"),
        ),
        RewriteRule(
            factory.rule_id(),
            factory.pattern("source"),
            factory.rhs("right_nf"),
        ),
    )
    system = factory.system(*rules)
    graph = pack_ground_term(system, factory.leaf("source"), 4)
    return RewriteEpisode("fork", "nonconfluent_fork", system, graph)


def _nested_episode(seed: int) -> RewriteEpisode:
    factory = _EpisodeFactory(seed)
    for alias, arity in (
        ("outer", 1),
        ("seed", 0),
        ("ready", 0),
        ("created_nf", 0),
        ("removed_nf", 0),
    ):
        factory.constructor(alias, arity)
    rules = (
        RewriteRule(
            factory.rule_id(),
            factory.pattern("seed"),
            factory.rhs("ready"),
        ),
        RewriteRule(
            factory.rule_id(),
            factory.pattern("outer", factory.pattern("ready")),
            factory.rhs("created_nf"),
        ),
        RewriteRule(
            factory.rule_id(),
            factory.pattern("outer", factory.pattern("seed")),
            factory.rhs("removed_nf"),
        ),
    )
    system = factory.system(*rules)
    graph = pack_ground_term(
        system,
        factory.term("outer", factory.leaf("seed")),
        6,
    )
    return RewriteEpisode(
        "nested",
        "nested_redex_creation_removal",
        system,
        graph,
    )


def _repeated_variable_episode(seed: int) -> RewriteEpisode:
    factory = _EpisodeFactory(seed)
    for alias, arity in (
        ("pair", 2),
        ("equal", 2),
        ("a", 0),
        ("b", 0),
        ("hit", 0),
    ):
        factory.constructor(alias, arity)
    variable = factory.variable("bound")
    rule = RewriteRule(
        factory.rule_id(),
        factory.pattern("equal", variable, variable),
        factory.rhs("hit"),
    )
    system = factory.system(rule)
    graph = pack_ground_term(
        system,
        factory.term(
            "pair",
            factory.term("equal", factory.leaf("a"), factory.leaf("a")),
            factory.term("equal", factory.leaf("a"), factory.leaf("b")),
        ),
        10,
    )
    return RewriteEpisode(
        "repeated_variable",
        "repeated_variable_binding",
        system,
        graph,
    )


def _destructive_episode(seed: int) -> RewriteEpisode:
    factory = _EpisodeFactory(seed)
    for alias, arity in (("wrap", 1), ("cancel", 1), ("payload", 0)):
        factory.constructor(alias, arity)
    variable = factory.variable("payload")
    rule = RewriteRule(
        factory.rule_id(),
        factory.pattern("cancel", variable),
        RhsVariable(variable.name),
    )
    system = factory.system(rule)
    graph = pack_ground_term(
        system,
        factory.term(
            "wrap",
            factory.term("cancel", factory.leaf("payload")),
        ),
        6,
    )
    return RewriteEpisode(
        "destructive",
        "destructive_cancellation",
        system,
        graph,
    )


def _deletion_episode(seed: int) -> RewriteEpisode:
    factory = _EpisodeFactory(seed)
    factory.constructor("erase", 0)
    rule = RewriteRule(factory.rule_id(), factory.pattern("erase"), None)
    system = factory.system(rule)
    graph = pack_ground_term(system, factory.leaf("erase"), 3)
    return RewriteEpisode("deletion", "root_deletion", system, graph)


def _shared_occurrence_episode(seed: int) -> RewriteEpisode:
    factory = _EpisodeFactory(seed)
    for alias, arity in (("pair", 2), ("redex", 0), ("normal", 0)):
        factory.constructor(alias, arity)
    rule = RewriteRule(
        factory.rule_id(),
        factory.pattern("redex"),
        factory.rhs("normal"),
    )
    system = factory.system(rule)
    shared = factory.leaf("redex")
    graph = pack_ground_term(
        system,
        factory.term("pair", shared, shared),
        6,
    )
    return RewriteEpisode(
        "shared_occurrence",
        "shared_occurrence_redexes",
        system,
        graph,
    )


def _repeated_rhs_pointer_episode(seed: int) -> RewriteEpisode:
    factory = _EpisodeFactory(seed)
    for alias, arity in (("source", 1), ("pair", 2), ("payload", 0)):
        factory.constructor(alias, arity)
    variable = factory.variable("shared")
    rule = RewriteRule(
        factory.rule_id(),
        factory.pattern("source", variable),
        factory.rhs(
            "pair",
            RhsVariable(variable.name),
            RhsVariable(variable.name),
        ),
    )
    system = factory.system(rule)
    graph = pack_ground_term(
        system,
        factory.term("source", factory.leaf("payload")),
        4,
    )
    return RewriteEpisode(
        "repeated_rhs_pointer",
        "repeated_rhs_pointer_sharing",
        system,
        graph,
    )


def _heterogeneous_typing_episode(seed: int) -> RewriteEpisode:
    value_type = _opaque("type", seed, 0)
    box_type = _opaque("type", seed, 1)
    atom = ConstructorSpec(
        _opaque("constructor", seed, 0),
        value_type,
    )
    box = ConstructorSpec(
        _opaque("constructor", seed, 1),
        box_type,
        (value_type,),
    )
    sealed = ConstructorSpec(
        _opaque("constructor", seed, 2),
        box_type,
        (value_type,),
    )
    variable = PatternVariable("value", value_type)
    rule = RewriteRule(
        _opaque("rule", seed, 0),
        PatternConstructor(box.identifier, (variable,)),
        RhsConstructor(sealed.identifier, (RhsVariable(variable.name),)),
    )
    system = RewriteSystem((atom, box, sealed), (rule,))
    graph = pack_ground_term(
        system,
        GroundTerm.constructor(box, GroundTerm.constructor(atom)),
        5,
    )
    return RewriteEpisode(
        "heterogeneous_typing",
        "heterogeneous_valid_typing",
        system,
        graph,
    )


def _capacity_unblocking_episode(seed: int) -> RewriteEpisode:
    factory = _EpisodeFactory(seed)
    for alias, arity in (
        ("pair", 2),
        ("drop", 1),
        ("kept", 0),
        ("seed", 0),
        ("wrap", 1),
        ("new_payload", 0),
    ):
        factory.constructor(alias, arity)
    variable = factory.variable("kept")
    rules = (
        RewriteRule(
            factory.rule_id(),
            factory.pattern("drop", variable),
            RhsVariable(variable.name),
        ),
        RewriteRule(
            factory.rule_id(),
            factory.pattern("seed"),
            factory.rhs("wrap", factory.rhs("new_payload")),
        ),
    )
    system = factory.system(*rules)
    graph = pack_ground_term(
        system,
        factory.term(
            "pair",
            factory.term("drop", factory.leaf("kept")),
            factory.leaf("seed"),
        ),
        4,
    )
    return RewriteEpisode(
        "capacity_unblocking",
        "capacity_unblocking_order",
        system,
        graph,
    )


def _mixed_cyclic_episode(seed: int) -> RewriteEpisode:
    factory = _EpisodeFactory(seed)
    for alias in ("source", "loop", "normal"):
        factory.constructor(alias, 0)
    rules = (
        RewriteRule(
            factory.rule_id(),
            factory.pattern("source"),
            factory.rhs("loop"),
        ),
        RewriteRule(
            factory.rule_id(),
            factory.pattern("loop"),
            factory.rhs("source"),
        ),
        RewriteRule(
            factory.rule_id(),
            factory.pattern("source"),
            factory.rhs("normal"),
        ),
    )
    system = factory.system(*rules)
    graph = pack_ground_term(system, factory.leaf("source"), 4)
    return RewriteEpisode(
        "mixed_cyclic",
        "mixed_cyclic_terminating",
        system,
        graph,
    )


def _counterfactual_twins(seed: int) -> tuple[RewriteEpisode, RewriteEpisode]:
    factory = _EpisodeFactory(seed)
    for alias, arity in (
        ("pick", 2),
        ("output", 2),
        ("left", 0),
        ("right", 0),
    ):
        factory.constructor(alias, arity)
    left = factory.variable("left_pointer")
    right = factory.variable("right_pointer")
    rule_id = factory.rule_id()
    lhs = factory.pattern("pick", left, right)
    forward = RewriteRule(
        rule_id,
        lhs,
        factory.rhs(
            "output",
            RhsVariable(left.name),
            RhsVariable(right.name),
        ),
    )
    reversed_rule = RewriteRule(
        rule_id,
        lhs,
        factory.rhs(
            "output",
            RhsVariable(right.name),
            RhsVariable(left.name),
        ),
    )
    graph_term = factory.term("pick", factory.leaf("left"), factory.leaf("right"))
    forward_system = factory.system(forward)
    reverse_system = factory.system(reversed_rule)
    forward_graph = pack_ground_term(forward_system, graph_term, 7)
    reverse_graph = pack_ground_term(reverse_system, graph_term, 7)
    return (
        RewriteEpisode(
            "counterfactual_forward",
            "counterfactual_rhs_pointer",
            forward_system,
            forward_graph,
        ),
        RewriteEpisode(
            "counterfactual_reverse",
            "counterfactual_rhs_pointer",
            reverse_system,
            reverse_graph,
        ),
    )


def build_mechanics_board(seed: int = 20260723) -> tuple[RewriteEpisode, ...]:
    """Build all required deterministic mechanics episode classes."""

    episodes = [
        _independent_episode(seed + 1),
        _diamond_episode(seed + 2),
        _fork_episode(seed + 3),
        _nested_episode(seed + 4),
        _repeated_variable_episode(seed + 5),
        _destructive_episode(seed + 6),
        _deletion_episode(seed + 7),
        _shared_occurrence_episode(seed + 8),
        _repeated_rhs_pointer_episode(seed + 9),
        _heterogeneous_typing_episode(seed + 10),
        _capacity_unblocking_episode(seed + 11),
        _mixed_cyclic_episode(seed + 12),
    ]
    episodes.extend(_counterfactual_twins(seed + 13))
    return tuple(episodes)
