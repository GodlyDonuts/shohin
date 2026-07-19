"""Exact mechanics for S8 model-owned nil-linked law graphs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from s6_contextual_affine_law import pop_insert, validate_state
from s7_learned_cayley_law import compile_destination, validate_successor


NIL = -1


@dataclass(frozen=True)
class LawCardNode:
    """Two witnessed outputs for one source-named operation."""

    operation: str
    y0: int
    y1: int


@dataclass(frozen=True)
class EventNode:
    """One event and the model-owned pointer to its successor event."""

    identity: int
    operation: str
    next_node: int


@dataclass(frozen=True)
class NilLinkedLawGraph:
    """A discrete executable graph emitted from a whole source."""

    modulus: int
    initial_state: tuple[int, ...]
    cards: tuple[LawCardNode, ...]
    nodes: tuple[EventNode, ...]
    entry_node: int
    query_position: int


def card_map(cards: Sequence[LawCardNode], modulus: int) -> dict[str, LawCardNode]:
    result: dict[str, LawCardNode] = {}
    for card in cards:
        if not card.operation or card.operation in result:
            raise ValueError("S8 cards require unique nonempty operation names")
        if not 0 <= card.y0 < modulus or not 0 <= card.y1 < modulus:
            raise ValueError("S8 card symbol outside modulus")
        if card.y0 == card.y1:
            raise ValueError("S8 bijective law card requires distinct witnesses")
        result[card.operation] = card
    if not result:
        raise ValueError("S8 graph requires at least one law card")
    return result


def linked_path(graph: NilLinkedLawGraph) -> tuple[int, ...]:
    """Validate and return the unique nil-terminated path through all nodes."""

    modulus = int(graph.modulus)
    if modulus < 3:
        raise ValueError("S8 modulus is too small")
    validate_state(graph.initial_state, modulus)
    cards = card_map(graph.cards, modulus)
    if not graph.nodes:
        raise ValueError("S8 graph requires at least one event")
    if not 0 <= graph.entry_node < len(graph.nodes):
        raise ValueError("S8 entry pointer outside node table")
    if not 0 <= graph.query_position < modulus:
        raise ValueError("S8 query position outside state")
    for node in graph.nodes:
        if not 0 <= node.identity < modulus:
            raise ValueError("S8 event identity outside roster")
        if node.operation not in cards:
            raise ValueError("S8 event references an unknown law card")
        if node.next_node != NIL and not 0 <= node.next_node < len(graph.nodes):
            raise ValueError("S8 next pointer outside node table")

    path: list[int] = []
    seen: set[int] = set()
    cursor = graph.entry_node
    for _ in range(len(graph.nodes) + 1):
        if cursor == NIL:
            break
        if cursor in seen:
            raise ValueError("S8 event graph contains a cycle")
        seen.add(cursor)
        path.append(cursor)
        cursor = graph.nodes[cursor].next_node
    else:
        raise ValueError("S8 event graph did not terminate")
    if cursor != NIL:
        raise ValueError("S8 event graph did not reach nil")
    if len(path) != len(graph.nodes):
        raise ValueError("S8 event graph omits or strands nodes")
    return tuple(path)


def rewire_path(
    graph: NilLinkedLawGraph, path: Sequence[int]
) -> NilLinkedLawGraph:
    """Return the same event records linked in a new complete order."""

    order = tuple(int(value) for value in path)
    if set(order) != set(range(len(graph.nodes))) or len(order) != len(graph.nodes):
        raise ValueError("S8 rewiring path must be a complete node permutation")
    nodes = list(graph.nodes)
    for index, node_id in enumerate(order):
        next_node = order[index + 1] if index + 1 < len(order) else NIL
        nodes[node_id] = replace(nodes[node_id], next_node=next_node)
    result = replace(graph, nodes=tuple(nodes), entry_node=order[0])
    linked_path(result)
    return result


def derange_cards(graph: NilLinkedLawGraph) -> NilLinkedLawGraph:
    """Rotate witnessed outputs among operation labels without changing events."""

    if len(graph.cards) < 2:
        raise ValueError("S8 card derangement requires at least two laws")
    values = [(card.y0, card.y1) for card in graph.cards]
    rotated = values[1:] + values[:1]
    cards = tuple(
        LawCardNode(card.operation, value[0], value[1])
        for card, value in zip(graph.cards, rotated, strict=True)
    )
    return replace(graph, cards=cards)


def one_witness_unit_completion(
    graph: NilLinkedLawGraph,
    successor: Sequence[int],
    zero_symbol: int,
) -> NilLinkedLawGraph:
    """Replace the unavailable second witness with a unit-slope default."""

    cycle = validate_successor(successor, zero_symbol)
    if len(cycle) != graph.modulus:
        raise ValueError("S8 successor/graph modulus mismatch")
    cards = tuple(
        LawCardNode(card.operation, card.y0, cycle[card.y0])
        for card in graph.cards
    )
    return replace(graph, cards=cards)


def execute_graph(
    graph: NilLinkedLawGraph,
    successor: Sequence[int],
    zero_symbol: int,
    *,
    reset_state: bool = False,
    halt_after: int | None = None,
    storage_order: bool = False,
) -> tuple[tuple[int, ...], int, tuple[int, ...]]:
    """Execute only the graph's predicted cards, links, identities, and query."""

    cycle = validate_successor(successor, zero_symbol)
    if len(cycle) != graph.modulus:
        raise ValueError("S8 successor/graph modulus mismatch")
    path = tuple(range(len(graph.nodes))) if storage_order else linked_path(graph)
    if halt_after is not None:
        if halt_after < 0:
            raise ValueError("S8 halt_after must be nonnegative")
        path = path[:halt_after]
    cards = card_map(graph.cards, graph.modulus)
    initial = tuple(graph.initial_state)
    state = initial
    transitions: list[tuple[int, ...]] = []
    for node_id in path:
        if reset_state:
            state = initial
        node = graph.nodes[node_id]
        card = cards[node.operation]
        source = state.index(node.identity)
        destination = compile_destination(
            cycle,
            zero_symbol,
            card.y0,
            card.y1,
            source,
        )
        state = pop_insert(state, node.identity, destination)
        transitions.append(tuple(state))
    return tuple(state), int(state[graph.query_position]), tuple(transitions)


def graph_from_ordered_events(
    *,
    modulus: int,
    initial_state: Sequence[int],
    cards: Mapping[str, tuple[int, int]],
    events: Sequence[tuple[int, str]],
    storage_ids: Sequence[int],
    query_position: int,
) -> NilLinkedLawGraph:
    """Build a graph whose storage order is independent of execution order."""

    if len(events) != len(storage_ids):
        raise ValueError("S8 events/storage IDs length mismatch")
    if set(storage_ids) != set(range(len(events))):
        raise ValueError("S8 storage IDs must be a complete permutation")
    nodes: list[EventNode | None] = [None] * len(events)
    for index, ((identity, operation), node_id) in enumerate(
        zip(events, storage_ids, strict=True)
    ):
        next_node = storage_ids[index + 1] if index + 1 < len(events) else NIL
        nodes[node_id] = EventNode(int(identity), str(operation), int(next_node))
    graph = NilLinkedLawGraph(
        modulus=int(modulus),
        initial_state=tuple(int(value) for value in initial_state),
        cards=tuple(
            LawCardNode(str(name), int(value[0]), int(value[1]))
            for name, value in sorted(cards.items())
        ),
        nodes=tuple(node for node in nodes if node is not None),
        entry_node=int(storage_ids[0]),
        query_position=int(query_position),
    )
    linked_path(graph)
    return graph
