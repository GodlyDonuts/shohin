"""Audited UROM-3 multi-family board generation and independent CPU oracle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import string
from typing import Iterable, Literal, Sequence


MIN_OBJECTS = 2
MAX_OBJECTS = 8
MAX_RULES = 8
MAX_EVENTS = 32
MAX_RELATION_EDGES = 24
APPLY_KIND = 0
STOP_KIND = 1
NOOP_KIND = 2
Family = Literal["transport", "graph", "constraint", "hybrid"]
Renderer = Literal["ledger", "prose", "symbolic", "records"]
AxisCell = Literal["in_domain", "renderer", "scale", "joint", "hybrid"]
Relation = tuple[tuple[int, ...], ...]


class UROMBoardError(ValueError):
    """Raised when a generated world violates the frozen board contract."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def identity_relation(cardinality: int) -> Relation:
    if not MIN_OBJECTS <= cardinality <= MAX_OBJECTS:
        raise UROMBoardError("identity cardinality differs")
    return tuple(
        tuple(int(row == column) for column in range(cardinality))
        for row in range(cardinality)
    )


def compose_relations(left: Relation, right: Relation) -> Relation:
    """Pure-Python Boolean relation composition; independent of torch code."""

    cardinality = len(left)
    if (
        cardinality < MIN_OBJECTS
        or len(right) != cardinality
        or any(len(row) != cardinality for row in (*left, *right))
    ):
        raise UROMBoardError("relation composition geometry differs")
    return tuple(
        tuple(
            int(
                any(
                    bool(left[output][middle] and right[middle][source])
                    for middle in range(cardinality)
                )
            )
            for source in range(cardinality)
        )
        for output in range(cardinality)
    )


def execute_relations(
    initial: Relation,
    rules: Sequence[Relation],
    event_rules: Sequence[int],
) -> tuple[Relation, tuple[Relation, ...]]:
    state = initial
    trajectory = [state]
    for rule in event_rules:
        if not 0 <= int(rule) < len(rules):
            raise UROMBoardError("event rule leaves the episode")
        state = compose_relations(rules[int(rule)], state)
        trajectory.append(state)
    return state, tuple(trajectory)


def relation_edges(relation: Relation) -> tuple[tuple[int, int], ...]:
    return tuple(
        (destination, source)
        for destination, row in enumerate(relation)
        for source, value in enumerate(row)
        if value
    )


def _nonempty_relation(
    cardinality: int,
    rows: Iterable[Iterable[int]],
) -> Relation:
    relation = tuple(tuple(int(value) for value in row) for row in rows)
    if (
        len(relation) != cardinality
        or any(len(row) != cardinality for row in relation)
        or any(value not in {0, 1} for row in relation for value in row)
        or any(not any(row) for row in relation)
        or len(relation_edges(relation)) > MAX_RELATION_EDGES
    ):
        raise UROMBoardError("generated relation leaves the board contract")
    return relation


def _transport_relation(cardinality: int, rng: random.Random) -> Relation:
    if rng.random() < 0.6:
        sources = list(range(cardinality))
        rng.shuffle(sources)
    else:
        sources = [rng.randrange(cardinality) for _ in range(cardinality)]
    return _nonempty_relation(
        cardinality,
        (
            tuple(int(source == selected) for source in range(cardinality))
            for selected in sources
        ),
    )


def _graph_relation(cardinality: int, rng: random.Random) -> Relation:
    rows = []
    edge_budget = 0
    for destination in range(cardinality):
        maximum = min(2, cardinality, MAX_RELATION_EDGES - edge_budget)
        count = rng.randint(1, max(1, maximum))
        selected = set(rng.sample(range(cardinality), count))
        edge_budget += count
        rows.append(
            tuple(int(source in selected) for source in range(cardinality))
        )
    return _nonempty_relation(cardinality, rows)


def _constraint_relation(cardinality: int, rng: random.Random) -> Relation:
    colors = [rng.randrange(3) for _ in range(cardinality)]
    rows = []
    edge_budget = 0
    for destination in range(cardinality):
        allowed = [
            source
            for source in range(cardinality)
            if colors[source] != colors[destination]
        ]
        if not allowed:
            allowed = list(range(cardinality))
        maximum = min(2, len(allowed), MAX_RELATION_EDGES - edge_budget)
        count = rng.randint(1, max(1, maximum))
        selected = set(rng.sample(allowed, count))
        edge_budget += count
        rows.append(
            tuple(int(source in selected) for source in range(cardinality))
        )
    return _nonempty_relation(cardinality, rows)


def generate_relation(
    family: Family,
    cardinality: int,
    rng: random.Random,
    rule_index: int,
) -> Relation:
    if family == "transport":
        return _transport_relation(cardinality, rng)
    if family == "graph":
        return _graph_relation(cardinality, rng)
    if family == "constraint":
        return _constraint_relation(cardinality, rng)
    if family == "hybrid":
        generators = (
            _transport_relation,
            _graph_relation,
            _constraint_relation,
        )
        return generators[rule_index % len(generators)](cardinality, rng)
    raise UROMBoardError("unknown UROM family")


def opaque_name(rng: random.Random, used: set[str]) -> str:
    alphabet = string.ascii_lowercase + string.digits
    while True:
        value = "".join(rng.choice(alphabet) for _ in range(6))
        if value not in used:
            used.add(value)
            return value


class TrackedText:
    """ASCII renderer that records semantic byte spans while appending."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.spans: dict[str, list[tuple[int, int]]] = {}

    def text(self, value: str) -> None:
        if not value.isascii():
            raise UROMBoardError("UROM renderer must remain ASCII")
        self.parts.append(value)
        self.length += len(value.encode("ascii"))

    def symbol(self, key: str, value: str) -> None:
        start = self.length
        self.text(value)
        self.spans.setdefault(key, []).append((start, self.length))

    def finish(self) -> tuple[str, dict[str, tuple[tuple[int, int], ...]]]:
        return "".join(self.parts), {
            key: tuple(values) for key, values in sorted(self.spans.items())
        }


@dataclass(frozen=True, slots=True)
class UROMWorld:
    family: Family
    cardinality: int
    objects: tuple[str, ...]
    rule_names: tuple[str, ...]
    rules: tuple[Relation, ...]
    event_rules: tuple[int, ...]
    suffix_rules: tuple[int, ...]
    query_position: int

    def semantic_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "cardinality": self.cardinality,
            "rules": self.rules,
            "event_rules": self.event_rules,
            "suffix_rules": self.suffix_rules,
            "query_position": self.query_position,
        }


def make_world(
    *,
    family: Family,
    cardinality: int,
    rule_count: int,
    event_count: int,
    seed: int,
) -> UROMWorld:
    if (
        not MIN_OBJECTS <= cardinality <= MAX_OBJECTS
        or not 2 <= rule_count <= MAX_RULES
        or not 1 <= event_count < MAX_EVENTS
    ):
        raise UROMBoardError("world geometry differs")
    rng = random.Random(seed)
    used: set[str] = set()
    objects = tuple(opaque_name(rng, used) for _ in range(cardinality))
    rule_names = tuple(opaque_name(rng, used) for _ in range(rule_count))
    rules = tuple(
        generate_relation(family, cardinality, rng, index)
        for index in range(rule_count)
    )
    event_rules = tuple(rng.randrange(rule_count) for _ in range(event_count))
    suffix_count = MAX_EVENTS - event_count - 1
    suffix_rules = tuple(rng.randrange(rule_count) for _ in range(suffix_count))
    query_position = rng.randrange(cardinality)
    return UROMWorld(
        family,
        cardinality,
        objects,
        rule_names,
        rules,
        event_rules,
        suffix_rules,
        query_position,
    )


def _render_header(
    builder: TrackedText,
    world: UROMWorld,
    renderer: Renderer,
) -> None:
    if renderer == "ledger":
        builder.text("objects ")
        separator, trailer = " ", "\n"
    elif renderer == "prose":
        builder.text("The available markers are ")
        separator, trailer = ", ", ".\n"
    elif renderer == "symbolic":
        builder.text("O={")
        separator, trailer = "|", "}\n"
    elif renderer == "records":
        builder.text("domain:")
        separator, trailer = ";", "\n"
    else:
        raise UROMBoardError("renderer differs")
    for index, name in enumerate(world.objects):
        if index:
            builder.text(separator)
        builder.symbol(f"declaration.{index}", name)
    builder.text(trailer)


def _render_rule(
    builder: TrackedText,
    world: UROMWorld,
    renderer: Renderer,
    rule_index: int,
) -> None:
    name = world.rule_names[rule_index]
    edges = relation_edges(world.rules[rule_index])
    if renderer == "ledger":
        builder.text("rule ")
        builder.symbol(f"rule.{rule_index}.opcode", name)
        builder.text(" ")
        connector, separator, trailer = "<-", " ", "\n"
    elif renderer == "prose":
        builder.text("Transformation ")
        builder.symbol(f"rule.{rule_index}.opcode", name)
        builder.text(" contains ")
        connector, separator, trailer = " from ", ", ", ".\n"
    elif renderer == "symbolic":
        builder.text("R[")
        builder.symbol(f"rule.{rule_index}.opcode", name)
        builder.text("]={")
        connector, separator, trailer = "<", "|", "}\n"
    elif renderer == "records":
        builder.text("map:")
        builder.symbol(f"rule.{rule_index}.opcode", name)
        builder.text(":")
        connector, separator, trailer = "=", ";", "\n"
    else:
        raise UROMBoardError("renderer differs")
    for edge_index, (destination, source) in enumerate(edges):
        if edge_index:
            builder.text(separator)
        builder.symbol(
            f"rule.{rule_index}.edge.{edge_index}.destination",
            world.objects[destination],
        )
        builder.text(connector)
        builder.symbol(
            f"rule.{rule_index}.edge.{edge_index}.source",
            world.objects[source],
        )
    builder.text(trailer)


def render_program(
    world: UROMWorld,
    renderer: Renderer,
) -> tuple[str, dict[str, tuple[tuple[int, int], ...]]]:
    builder = TrackedText()
    _render_header(builder, world, renderer)
    for rule in range(len(world.rules)):
        _render_rule(builder, world, renderer, rule)

    if renderer == "ledger":
        builder.text("initial ")
        initial_connector, initial_separator = "<-", " "
    elif renderer == "prose":
        builder.text("Initially pair ")
        initial_connector, initial_separator = " with ", ", "
    elif renderer == "symbolic":
        builder.text("I={")
        initial_connector, initial_separator = "<", "|"
    else:
        builder.text("seed:")
        initial_connector, initial_separator = "=", ";"
    for index, name in enumerate(world.objects):
        if index:
            builder.text(initial_separator)
        builder.symbol(f"initial.{index}.destination", name)
        builder.text(initial_connector)
        builder.symbol(f"initial.{index}.source", name)
    builder.text("}\n" if renderer == "symbolic" else "\n")

    builder.text("program " if renderer in {"ledger", "records"} else "Run ")
    sequence = (*world.event_rules, None, *world.suffix_rules)
    for event, rule in enumerate(sequence):
        if event:
            builder.text(" ")
        if rule is None:
            builder.text("STOP")
        else:
            builder.symbol(
                f"event.{event}.opcode",
                world.rule_names[rule],
            )
    builder.text("\n")
    return builder.finish()


def render_query(
    world: UROMWorld,
    renderer: Renderer,
) -> tuple[str, dict[str, tuple[tuple[int, int], ...]]]:
    builder = TrackedText()
    if renderer == "ledger":
        builder.text("read ")
    elif renderer == "prose":
        builder.text("Which markers occupy ")
    elif renderer == "symbolic":
        builder.text("Q[")
    else:
        builder.text("inspect:")
    builder.symbol("query.position", world.objects[world.query_position])
    builder.text("]?\n" if renderer == "symbolic" else "?\n")
    return builder.finish()


def _pad_relation(relation: Relation) -> list[list[int]]:
    output = [[0] * MAX_OBJECTS for _ in range(MAX_OBJECTS)]
    for row, values in enumerate(relation):
        output[row][: len(values)] = values
    return output


def row_from_world(
    world: UROMWorld,
    *,
    split: str,
    axis_cell: AxisCell,
    renderer: Renderer,
    row_seed: int,
) -> dict[str, object]:
    program, program_spans = render_program(world, renderer)
    query, query_spans = render_query(world, renderer)
    initial = identity_relation(world.cardinality)
    terminal, trajectory = execute_relations(
        initial,
        world.rules,
        world.event_rules,
    )
    full_trajectory = (
        list(trajectory[1:])
        + [terminal] * (MAX_EVENTS - len(world.event_rules))
    )
    answer = terminal[world.query_position]
    event_kind = (
        [APPLY_KIND] * len(world.event_rules)
        + [STOP_KIND]
        + [APPLY_KIND] * len(world.suffix_rules)
    )
    event_rule = [
        *world.event_rules,
        0,
        *world.suffix_rules,
    ]
    if len(event_kind) != MAX_EVENTS or len(event_rule) != MAX_EVENTS:
        raise UROMBoardError("rendered event tape differs")
    rule_active = [index < len(world.rules) for index in range(MAX_RULES)]
    rules = [_pad_relation(rule) for rule in world.rules]
    rules.extend(
        [_pad_relation(identity_relation(world.cardinality))]
        * (MAX_RULES - len(rules))
    )
    payload = {
        "schema": "urom3_row_v2",
        "split": split,
        "axis_cell": axis_cell,
        "family": world.family,
        "renderer": renderer,
        "row_seed": int(row_seed),
        "semantic_sha256": sha256_json(world.semantic_payload()),
        "program_text": program,
        "query_text": query,
        "compiler_targets": {
            "cardinality": world.cardinality,
            "initial_edges": _pad_relation(initial),
            "rule_edges": rules,
            "rule_active": rule_active,
            "event_rule": event_rule,
            "event_kind": event_kind,
        },
        "late_query_target": {
            "position": world.query_position,
        },
        "oracle": {
            "terminal_state": _pad_relation(terminal),
            "answer_bits": list(answer) + [0] * (MAX_OBJECTS - len(answer)),
            "state_trajectory": [
                _pad_relation(value) for value in full_trajectory
            ],
        },
        "evidence_spans": {
            "program": program_spans,
            "query": query_spans,
        },
    }
    payload["row_sha256"] = sha256_json(payload)
    return payload


def axis_contract(split: str, axis_cell: AxisCell) -> dict[str, object]:
    if split == "train" and axis_cell == "in_domain":
        return {
            "cardinalities": (4, 5, 6),
            "rule_counts": (3, 4),
            "event_counts": (2, 8),
            "families": ("transport", "graph", "constraint"),
            "renderers": ("ledger", "prose"),
        }
    if split == "development" and axis_cell == "renderer":
        return {
            "cardinalities": (4, 5, 6),
            "rule_counts": (3, 4),
            "event_counts": (2, 8),
            "families": ("transport", "graph", "constraint"),
            "renderers": ("symbolic", "records"),
        }
    if split == "development" and axis_cell == "scale":
        return {
            "cardinalities": (7,),
            "rule_counts": (5, 6),
            "event_counts": (9, 16),
            "families": ("transport", "graph", "constraint"),
            "renderers": ("ledger", "prose"),
        }
    if split == "development" and axis_cell == "joint":
        return {
            "cardinalities": (7,),
            "rule_counts": (5, 6),
            "event_counts": (9, 16),
            "families": ("transport", "graph", "constraint"),
            "renderers": ("symbolic", "records"),
        }
    if split == "confirmation" and axis_cell == "joint":
        return {
            "cardinalities": (8,),
            "rule_counts": (7, 8),
            "event_counts": (17, 31),
            "families": ("transport", "graph", "constraint"),
            "renderers": ("symbolic", "records"),
        }
    if split == "confirmation" and axis_cell == "hybrid":
        return {
            "cardinalities": (8,),
            "rule_counts": (7, 8),
            "event_counts": (17, 31),
            "families": ("hybrid",),
            "renderers": ("symbolic", "records"),
        }
    raise UROMBoardError("unknown split/axis cell")


def split_contract(split: str) -> dict[str, object]:
    cells: tuple[AxisCell, ...]
    if split == "train":
        cells = ("in_domain",)
    elif split == "development":
        cells = ("renderer", "scale", "joint")
    elif split == "confirmation":
        cells = ("joint", "hybrid")
    else:
        raise UROMBoardError("unknown split")
    contracts = [axis_contract(split, cell) for cell in cells]
    return {
        "axis_cells": cells,
        "cardinalities": tuple(
            sorted(
                {
                    int(value)
                    for contract in contracts
                    for value in contract["cardinalities"]
                }
            )
        ),
        "rule_counts": tuple(
            sorted(
                {
                    int(value)
                    for contract in contracts
                    for value in contract["rule_counts"]
                }
            )
        ),
        "event_counts": (
            min(int(contract["event_counts"][0]) for contract in contracts),
            max(int(contract["event_counts"][1]) for contract in contracts),
        ),
        "families": tuple(
            sorted(
                {
                    str(value)
                    for contract in contracts
                    for value in contract["families"]
                }
            )
        ),
        "renderers": tuple(
            sorted(
                {
                    str(value)
                    for contract in contracts
                    for value in contract["renderers"]
                }
            )
        ),
    }


def generate_rows(
    *,
    split: str,
    count: int,
    seed: int,
) -> list[dict[str, object]]:
    if count < 1:
        raise UROMBoardError("row count must be positive")
    split_specification = split_contract(split)
    axis_cells = tuple(
        str(value) for value in split_specification["axis_cells"]
    )
    rows = []
    seen_semantics: set[str] = set()
    for index in range(count):
        row_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{split}:{index}".encode("ascii")).digest()[:8],
            "big",
        )
        rng = random.Random(row_seed)
        axis_cell = axis_cells[index % len(axis_cells)]
        contract = axis_contract(split, axis_cell)  # type: ignore[arg-type]
        cardinalities = tuple(
            int(value) for value in contract["cardinalities"]
        )
        rule_counts = tuple(int(value) for value in contract["rule_counts"])
        minimum_events, maximum_events = (
            int(value) for value in contract["event_counts"]
        )
        families = tuple(str(value) for value in contract["families"])
        renderers = tuple(str(value) for value in contract["renderers"])
        cell_index = index // len(axis_cells)
        family = families[cell_index % len(families)]
        renderer = renderers[
            (
                cell_index
                // (len(families) * len(cardinalities))
            )
            % len(renderers)
        ]
        cardinality = cardinalities[
            (cell_index // len(families)) % len(cardinalities)
        ]
        rule_count = rule_counts[rng.randrange(len(rule_counts))]
        event_count = rng.randint(minimum_events, maximum_events)
        world = make_world(
            family=family,  # type: ignore[arg-type]
            cardinality=cardinality,
            rule_count=rule_count,
            event_count=event_count,
            seed=row_seed,
        )
        row = row_from_world(
            world,
            split=split,
            axis_cell=axis_cell,  # type: ignore[arg-type]
            renderer=renderer,  # type: ignore[arg-type]
            row_seed=row_seed,
        )
        semantic = str(row["semantic_sha256"])
        if semantic in seen_semantics:
            raise UROMBoardError("semantic world collision")
        seen_semantics.add(semantic)
        rows.append(row)
    return rows


def validate_row(row: dict[str, object]) -> None:
    required = {
        "schema",
        "split",
        "axis_cell",
        "family",
        "renderer",
        "row_seed",
        "semantic_sha256",
        "program_text",
        "query_text",
        "compiler_targets",
        "late_query_target",
        "oracle",
        "evidence_spans",
        "row_sha256",
    }
    if set(row) != required or row.get("schema") != "urom3_row_v2":
        raise UROMBoardError("row schema differs")
    contract = axis_contract(
        str(row["split"]),
        str(row["axis_cell"]),  # type: ignore[arg-type]
    )
    if (
        row["family"] not in contract["families"]
        or row["renderer"] not in contract["renderers"]
    ):
        raise UROMBoardError("row leaves its axis-cell contract")
    expected_hash = row["row_sha256"]
    unhashed = dict(row)
    unhashed.pop("row_sha256")
    if expected_hash != sha256_json(unhashed):
        raise UROMBoardError("row hash differs")
    targets = row["compiler_targets"]
    query = row["late_query_target"]
    oracle = row["oracle"]
    if not isinstance(targets, dict) or not isinstance(query, dict) or not isinstance(
        oracle,
        dict,
    ):
        raise UROMBoardError("row target payload differs")
    cardinality = int(targets["cardinality"])
    rule_count = sum(bool(value) for value in targets["rule_active"])
    stop = list(targets["event_kind"]).index(STOP_KIND)
    if (
        cardinality not in contract["cardinalities"]
        or rule_count not in contract["rule_counts"]
        or not int(contract["event_counts"][0])
        <= stop
        <= int(contract["event_counts"][1])
    ):
        raise UROMBoardError("row scale leaves its axis-cell contract")
    initial = tuple(
        tuple(int(value) for value in values[:cardinality])
        for values in targets["initial_edges"][:cardinality]
    )
    rules = tuple(
        tuple(
            tuple(int(value) for value in values[:cardinality])
            for values in relation[:cardinality]
        )
        for relation, active in zip(
            targets["rule_edges"],
            targets["rule_active"],
            strict=True,
        )
        if active
    )
    event_rules = tuple(
        int(value) for value in targets["event_rule"][:stop]
    )
    terminal, trajectory = execute_relations(initial, rules, event_rules)
    full_trajectory = (
        list(trajectory[1:])
        + [terminal] * (MAX_EVENTS - len(event_rules))
    )
    expected_terminal = _pad_relation(terminal)
    expected_answer = (
        list(terminal[int(query["position"])])
        + [0] * (MAX_OBJECTS - cardinality)
    )
    if (
        oracle.get("terminal_state") != expected_terminal
        or oracle.get("answer_bits") != expected_answer
        or oracle.get("state_trajectory")
        != [_pad_relation(value) for value in full_trajectory]
    ):
        raise UROMBoardError("independent oracle differs")
    if "answer_bits" in str(row["program_text"]) or str(
        expected_answer
    ) in str(row["program_text"]):
        raise UROMBoardError("program source leaks answer representation")
