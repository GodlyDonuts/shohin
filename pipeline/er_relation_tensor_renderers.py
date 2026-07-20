"""Renderer cosets and independent grammar parser for the ER-TT fresh board."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Mapping, Sequence


MAX_CARDINALITY = 6
MAX_RULES = 4
EVENT_SLOTS = 13
TT_RECORDS = 18


@dataclass(frozen=True, slots=True)
class RelationTensorRenderer:
    declaration: int
    witness: int
    event: int
    query: int

    def __post_init__(self) -> None:
        if any(value not in (0, 1) for value in self.as_tuple()):
            raise ValueError("ER-TT renderer factors must be binary")

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.declaration, self.witness, self.event, self.query

    @property
    def parity(self) -> int:
        return sum(self.as_tuple()) % 2

    @property
    def name(self) -> str:
        d, w, e, q = self.as_tuple()
        return f"er-tt-d{d}w{w}e{e}q{q}-v1"


TRAIN_RENDERERS = tuple(
    RelationTensorRenderer(*values)
    for values in ((0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 0, 0), (1, 1, 1, 1))
)
SCORED_RENDERERS = tuple(
    RelationTensorRenderer(*values)
    for values in ((1, 0, 0, 0), (1, 0, 1, 1), (0, 1, 0, 0), (0, 1, 1, 1))
)
ALL_RENDERERS = TRAIN_RENDERERS + SCORED_RENDERERS

DECLARATION_PATTERNS = (
    re.compile(r"^D([3-6]) (\S+(?: \S+){2,5}) ; I (\S+(?: \S+){2,5})$"),
    re.compile(r"^R([3-6]) (\S+(?: \S+){2,5}) ; S (\S+(?: \S+){2,5})$"),
)
WITNESS_PATTERNS = (
    re.compile(r"^W([1-4]) (OFF|(\S+) (\S+(?: \S+){2,5}) > (\S+(?: \S+){2,5}))$"),
    re.compile(r"^L([1-4]) (VOID|(\S+) (\S+(?: \S+){2,5}) => (\S+(?: \S+){2,5}))$"),
)
EVENT_PATTERNS = (
    re.compile(r"^E(1[0-3]|[1-9]) (\S+)$"),
    re.compile(r"^T(1[0-3]|[1-9]) (\S+)$"),
)
QUERY_PATTERNS = (
    re.compile(r"^Q([1-6])$"),
    re.compile(r"^ASK ([1-6])$"),
)


def renderer_from_row(row: Mapping[str, object]) -> RelationTensorRenderer:
    value = row.get("renderer")
    if not isinstance(value, Mapping):
        raise ValueError("ER-TT row lacks renderer metadata")
    renderer = RelationTensorRenderer(
        *(int(value[key]) for key in ("declaration", "witness", "event", "query"))
    )
    if row.get("template_id") != renderer.name:
        raise ValueError("ER-TT renderer name differs")
    return renderer


def _target(row: Mapping[str, object]) -> Mapping[str, object]:
    value = row.get("compiler_targets")
    if not isinstance(value, Mapping):
        raise ValueError("ER-TT row lacks compiler targets")
    return value


def _rules(row: Mapping[str, object]) -> list[Mapping[str, object]]:
    values = _target(row).get("rule_cards")
    if not isinstance(values, list) or len(values) != MAX_RULES:
        raise ValueError("ER-TT row requires four physical rule slots")
    ordered = sorted(values, key=lambda item: int(item["slot"]))
    if [int(item["slot"]) for item in ordered] != list(range(MAX_RULES)):
        raise ValueError("ER-TT rule slots differ")
    return ordered


def _events(row: Mapping[str, object]) -> list[Mapping[str, object]]:
    values = _target(row).get("events")
    if not isinstance(values, list) or len(values) != EVENT_SLOTS:
        raise ValueError("ER-TT row requires thirteen event slots")
    ordered = sorted(values, key=lambda item: int(item["slot"]))
    if [int(item["slot"]) for item in ordered] != list(range(EVENT_SLOTS)):
        raise ValueError("ER-TT event slots differ")
    return ordered


def semantic_lines(
    row: Mapping[str, object],
    renderer: RelationTensorRenderer,
) -> tuple[str, ...]:
    target = _target(row)
    cardinality = int(target["cardinality"])
    bindings = tuple(map(str, target["entity_bindings"]))
    initial = tuple(bindings[int(index)] for index in target["initial_order"])
    if len(bindings) != cardinality or len(initial) != cardinality:
        raise ValueError("ER-TT declaration cardinality differs")
    if renderer.declaration == 0:
        declaration = f"D{cardinality} {' '.join(bindings)} ; I {' '.join(initial)}"
    else:
        declaration = f"R{cardinality} {' '.join(bindings)} ; S {' '.join(initial)}"

    rules = []
    for item in _rules(row):
        slot = int(item["slot"]) + 1
        if not bool(item["active"]):
            rules.append(f"W{slot} OFF" if renderer.witness == 0 else f"L{slot} VOID")
            continue
        opcode = str(item["opcode"])
        before = tuple(map(str, item["before"]))
        after = tuple(map(str, item["after"]))
        if len(before) != cardinality or len(after) != cardinality:
            raise ValueError("ER-TT witness cardinality differs")
        if renderer.witness == 0:
            rules.append(f"W{slot} {opcode} {' '.join(before)} > {' '.join(after)}")
        else:
            rules.append(f"L{slot} {opcode} {' '.join(before)} => {' '.join(after)}")

    events = []
    halt_word = "HALT" if renderer.event == 0 else "STOP"
    prefix = "E" if renderer.event == 0 else "T"
    for item in _events(row):
        value = halt_word if bool(item["halt"]) else str(item["opcode"])
        events.append(f"{prefix}{int(item['slot']) + 1} {value}")
    return (declaration, *rules, *events)


def render_query(
    position: int,
    renderer: RelationTensorRenderer,
) -> tuple[str, tuple[int, int]]:
    if not 0 <= position < MAX_CARDINALITY:
        raise ValueError("ER-TT query position differs")
    numeral = str(position + 1)
    text = f"Q{numeral}" if renderer.query == 0 else f"ASK {numeral}"
    start = text.index(numeral)
    return text, (start, start + 1)


def _spans(line: str, values: Sequence[str], start: int = 0) -> list[list[int]]:
    output = []
    cursor = start
    for value in values:
        found = line.index(value, cursor)
        output.append([found, found + len(value)])
        cursor = found + len(value)
    return output


def render_row(
    row: Mapping[str, object],
    renderer: RelationTensorRenderer,
    *,
    storage_order: Sequence[int],
    row_id: str,
    family_id: str,
) -> dict[str, object]:
    if sorted(map(int, storage_order)) != list(range(TT_RECORDS)):
        raise ValueError("ER-TT storage order differs")
    result = json.loads(json.dumps(dict(row)))
    lines = semantic_lines(result, renderer)
    physical = [lines[int(index)] for index in storage_order]
    target = dict(result["compiler_targets"])
    query, query_span = render_query(int(target["query_position"]), renderer)
    result.update(
        {
            "id": row_id,
            "family_id": family_id,
            "template_id": renderer.name,
            "variant": renderer.name,
            "program_text": "\n".join(physical),
            "late_query_text": query,
            "renderer": {
                "declaration": renderer.declaration,
                "witness": renderer.witness,
                "event": renderer.event,
                "query": renderer.query,
                "parity": renderer.parity,
            },
        }
    )
    target["storage_order"] = list(map(int, storage_order))
    target["physical_roles"] = list(map(int, storage_order))

    starts = []
    cursor = 0
    for line in physical:
        encoded = line.encode()
        starts.append((cursor, cursor + len(encoded)))
        cursor += len(encoded) + 1
    semantic_ranges: list[list[int]] = [[0, 0] for _ in range(TT_RECORDS)]
    for physical_index, semantic_role in enumerate(storage_order):
        semantic_ranges[int(semantic_role)] = list(starts[physical_index])
    target["line_ranges"] = semantic_ranges

    cardinality = int(target["cardinality"])
    declaration = lines[0]
    declaration_offset = semantic_ranges[0][0]
    bindings = tuple(map(str, target["entity_bindings"]))
    initial = tuple(bindings[int(index)] for index in target["initial_order"])
    binding_local = _spans(declaration, bindings)
    separator = " ; I " if renderer.declaration == 0 else " ; S "
    initial_local = _spans(declaration, initial, declaration.index(separator) + len(separator))
    target["binding_ranges"] = [
        [declaration_offset + start, declaration_offset + end]
        for start, end in binding_local
    ]
    target["initial_ranges"] = [
        [declaration_offset + start, declaration_offset + end]
        for start, end in initial_local
    ]

    before_ranges: list[list[list[int]]] = []
    after_ranges: list[list[list[int]]] = []
    for semantic_slot, item in enumerate(_rules(result), start=1):
        if not bool(item["active"]):
            before_ranges.append([])
            after_ranges.append([])
            continue
        line = lines[semantic_slot]
        offset = semantic_ranges[semantic_slot][0]
        opcode = str(item["opcode"])
        cursor = line.index(opcode) + len(opcode)
        before_local = _spans(line, tuple(map(str, item["before"])), cursor)
        cursor = before_local[-1][1]
        after_local = _spans(line, tuple(map(str, item["after"])), cursor)
        before_ranges.append([[offset + start, offset + end] for start, end in before_local])
        after_ranges.append([[offset + start, offset + end] for start, end in after_local])
    target["witness_before_ranges"] = before_ranges
    target["witness_after_ranges"] = after_ranges
    target["query_range"] = list(query_span)
    result["compiler_targets"] = target
    result["source_shape"] = {
        "program_bytes": len(result["program_text"].encode()),
        "program_lines": TT_RECORDS,
        "query_bytes": len(query.encode()),
        "max_line_bytes": max(len(line.encode()) for line in physical),
    }
    if cardinality != len(target["binding_ranges"]):
        raise ValueError("ER-TT rendered declaration spans differ")
    return result


def parse_rendered_row(row: Mapping[str, object]) -> dict[str, object]:
    """Parse only public bytes; do not consult compiler targets or oracle."""
    renderer = renderer_from_row(row)
    lines = str(row["program_text"]).splitlines()
    if len(lines) != TT_RECORDS:
        raise ValueError("ER-TT rendered program requires eighteen lines")
    declaration: tuple[str, ...] | None = None
    initial: tuple[str, ...] | None = None
    cardinality: int | None = None
    rules: dict[str, tuple[int, tuple[str, ...], tuple[str, ...]]] = {}
    inactive: set[int] = set()
    events: dict[int, str | None] = {}
    for line in lines:
        declaration_match = DECLARATION_PATTERNS[renderer.declaration].fullmatch(line)
        if declaration_match:
            if declaration is not None:
                raise ValueError("ER-TT rendered program repeats declaration")
            cardinality = int(declaration_match.group(1))
            declaration = tuple(declaration_match.group(2).split())
            initial = tuple(declaration_match.group(3).split())
            continue
        witness_match = WITNESS_PATTERNS[renderer.witness].fullmatch(line)
        if witness_match:
            slot = int(witness_match.group(1)) - 1
            if slot in inactive or any(value[0] == slot for value in rules.values()):
                raise ValueError("ER-TT rendered program repeats rule slot")
            if witness_match.group(2) in {"OFF", "VOID"}:
                inactive.add(slot)
            else:
                opcode = str(witness_match.group(3))
                if opcode in rules:
                    raise ValueError("ER-TT rendered program repeats opcode")
                rules[opcode] = (
                    slot,
                    tuple(str(witness_match.group(4)).split()),
                    tuple(str(witness_match.group(5)).split()),
                )
            continue
        event_match = EVENT_PATTERNS[renderer.event].fullmatch(line)
        if event_match:
            slot = int(event_match.group(1)) - 1
            if slot in events:
                raise ValueError("ER-TT rendered program repeats event")
            value = event_match.group(2)
            halt_word = "HALT" if renderer.event == 0 else "STOP"
            events[slot] = None if value == halt_word else value
            continue
        raise ValueError("ER-TT rendered line is outside its grammar")
    query_match = QUERY_PATTERNS[renderer.query].fullmatch(str(row["late_query_text"]))
    if query_match is None:
        raise ValueError("ER-TT rendered query differs")
    if cardinality is None or declaration is None or initial is None:
        raise ValueError("ER-TT rendered declaration is absent")
    if len(declaration) != cardinality or len(initial) != cardinality:
        raise ValueError("ER-TT rendered declaration cardinality differs")
    if len(rules) + len(inactive) != MAX_RULES or set(events) != set(range(EVENT_SLOTS)):
        raise ValueError("ER-TT rendered record cardinality differs")
    if {value[0] for value in rules.values()} | inactive != set(range(MAX_RULES)):
        raise ValueError("ER-TT rendered rule slots differ")
    if sum(value is None for value in events.values()) != 1:
        raise ValueError("ER-TT rendered program requires exactly one HALT")
    if any(value is not None and value not in rules for value in events.values()):
        raise ValueError("ER-TT event references an inactive or absent opcode")
    return {
        "cardinality": cardinality,
        "bindings": declaration,
        "initial": initial,
        "rules": rules,
        "inactive_rule_slots": tuple(sorted(inactive)),
        "events": tuple(events[index] for index in range(EVENT_SLOTS)),
        "query_position": int(query_match.group(1)) - 1,
    }


def independently_execute(row: Mapping[str, object]) -> dict[str, object]:
    """Infer relations by exact equality and execute without generator metadata."""
    parsed = parse_rendered_row(row)
    bindings = tuple(parsed["bindings"])
    state = tuple(parsed["initial"])
    inferred: dict[str, tuple[int, ...]] = {}
    slots: dict[int, tuple[int, ...]] = {}
    for opcode, (slot, before, after) in parsed["rules"].items():
        if len(before) != len(bindings) or len(after) != len(bindings):
            raise ValueError("ER-TT independent witness cardinality differs")
        if len(set(before)) != len(before) or any(value not in before for value in after):
            raise ValueError("ER-TT independent witness is not determining")
        relation = tuple(before.index(value) for value in after)
        inferred[str(opcode)] = relation
        slots[int(slot)] = relation
    trajectory = [state]
    halt_after = None
    for index, opcode in enumerate(parsed["events"]):
        if opcode is None:
            halt_after = index
            break
        relation = inferred[str(opcode)]
        state = tuple(state[position] for position in relation)
        trajectory.append(state)
    if halt_after is None:
        raise ValueError("ER-TT independent execution lacks HALT")
    query_position = int(parsed["query_position"])
    if not 0 <= query_position < len(bindings):
        raise ValueError("ER-TT independent query exceeds cardinality")
    return {
        **parsed,
        "rule_relations": slots,
        "final_state": state,
        "trajectory": tuple(trajectory),
        "halt_after": halt_after,
        "answer_role": bindings.index(state[query_position]),
    }
