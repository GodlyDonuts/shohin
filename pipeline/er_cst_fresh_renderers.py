"""Split-disjoint renderer compositions for the ER-CST fresh board."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RuleCardRenderer:
    declaration: int
    witness: int
    event: int
    query: int

    def __post_init__(self) -> None:
        if any(value not in (0, 1) for value in self.as_tuple()):
            raise ValueError("ER-CST renderer factors must be binary")

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.declaration, self.witness, self.event, self.query

    @property
    def parity(self) -> int:
        return sum(self.as_tuple()) % 2

    @property
    def name(self) -> str:
        d, w, e, q = self.as_tuple()
        return f"er-fresh-d{d}w{w}e{e}q{q}-v1"


TRAIN_RENDERERS = tuple(
    RuleCardRenderer(*values)
    for values in ((0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 0, 0), (1, 1, 1, 1))
)
SCORED_RENDERERS = tuple(
    RuleCardRenderer(*values)
    for values in ((1, 0, 0, 0), (1, 0, 1, 1), (0, 1, 0, 0), (0, 1, 1, 1))
)
ALL_RENDERERS = TRAIN_RENDERERS + SCORED_RENDERERS

DECLARATION_PATTERNS = (
    re.compile(r"^D (\S+) (\S+) (\S+) ; I (\S+) (\S+) (\S+)$"),
    re.compile(r"^R (\S+) (\S+) (\S+) ; S (\S+) (\S+) (\S+)$"),
)
WITNESS_PATTERNS = (
    re.compile(
        r"^W([1-3]) (\S+) (\S+) (\S+) (\S+) > (\S+) (\S+) (\S+)$"
    ),
    re.compile(
        r"^L([1-3]) (\S+) (\S+) (\S+) (\S+) => (\S+) (\S+) (\S+)$"
    ),
)
EVENT_PATTERNS = (
    re.compile(r"^E([1-9]) (\S+)$"),
    re.compile(r"^T([1-9]) (\S+)$"),
)
QUERY_PATTERNS = (
    re.compile(r"^Q([1-3])$"),
    re.compile(r"^ASK ([1-3])$"),
)


def renderer_from_row(row: Mapping[str, object]) -> RuleCardRenderer:
    value = row.get("renderer")
    if not isinstance(value, Mapping):
        raise ValueError("ER-CST row lacks renderer metadata")
    renderer = RuleCardRenderer(
        *(int(value[key]) for key in ("declaration", "witness", "event", "query"))
    )
    if row.get("template_id") != renderer.name:
        raise ValueError("ER-CST renderer name differs")
    return renderer


def _target(row: Mapping[str, object]) -> Mapping[str, object]:
    value = row.get("compiler_targets")
    if not isinstance(value, Mapping):
        raise ValueError("ER-CST row lacks compiler targets")
    return value


def _bindings(row: Mapping[str, object]) -> tuple[str, str, str]:
    values = _target(row).get("entity_bindings")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("ER-CST row requires three entity bindings")
    ordered = sorted(values, key=lambda item: int(item["role"]))
    names = tuple(str(item["name"]) for item in ordered)
    if len(set(names)) != 3:
        raise ValueError("ER-CST entity bindings are not distinct")
    return names  # type: ignore[return-value]


def _rule_targets(row: Mapping[str, object]) -> list[Mapping[str, object]]:
    values = _target(row).get("rule_cards")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("ER-CST row requires three rule cards")
    ordered = sorted(values, key=lambda item: int(item["slot"]))
    if [int(item["slot"]) for item in ordered] != [0, 1, 2]:
        raise ValueError("ER-CST rule-card slots differ")
    return ordered


def _event_targets(row: Mapping[str, object]) -> list[Mapping[str, object]]:
    values = _target(row).get("events")
    if not isinstance(values, list) or len(values) != 9:
        raise ValueError("ER-CST row requires nine events")
    ordered = sorted(values, key=lambda item: int(item["slot"]))
    if [int(item["slot"]) for item in ordered] != list(range(9)):
        raise ValueError("ER-CST event slots differ")
    return ordered


def semantic_lines(
    row: Mapping[str, object], renderer: RuleCardRenderer
) -> tuple[str, ...]:
    target = _target(row)
    names = _bindings(row)
    initial = tuple(names[int(role)] for role in target["initial_order"])
    if renderer.declaration == 0:
        declaration = f"D {' '.join(names)} ; I {' '.join(initial)}"
    else:
        declaration = f"R {' '.join(names)} ; S {' '.join(initial)}"

    rules = []
    for item in _rule_targets(row):
        opcode = str(item["opcode"])
        before = tuple(map(str, item["before"]))
        after = tuple(map(str, item["after"]))
        slot = int(item["slot"]) + 1
        if renderer.witness == 0:
            rules.append(
                f"W{slot} {opcode} {' '.join(before)} > {' '.join(after)}"
            )
        else:
            rules.append(
                f"L{slot} {opcode} {' '.join(before)} => {' '.join(after)}"
            )

    events = []
    halt_word = "HALT" if renderer.event == 0 else "STOP"
    prefix = "E" if renderer.event == 0 else "T"
    for item in _event_targets(row):
        value = halt_word if bool(item["halt"]) else str(item["opcode"])
        events.append(f"{prefix}{int(item['slot']) + 1} {value}")
    return (declaration, *rules, *events)


def render_query(position: int, renderer: RuleCardRenderer) -> tuple[str, tuple[int, int]]:
    if position not in range(3):
        raise ValueError("ER-CST query position differs")
    numeral = str(position + 1)
    text = f"Q{numeral}" if renderer.query == 0 else f"ASK {numeral}"
    start = text.index(numeral)
    return text, (start, start + 1)


def render_row(
    row: Mapping[str, object],
    renderer: RuleCardRenderer,
    *,
    storage_order: Sequence[int],
    row_id: str,
    family_id: str,
) -> dict[str, object]:
    if sorted(map(int, storage_order)) != list(range(13)):
        raise ValueError("ER-CST storage order differs")
    result = json.loads(json.dumps(dict(row)))
    lines = semantic_lines(result, renderer)
    physical = [lines[int(index)] for index in storage_order]
    query_position = int(_target(result)["query_position"])
    query, query_span = render_query(query_position, renderer)
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
    targets = dict(result["compiler_targets"])
    targets["storage_order"] = list(map(int, storage_order))
    targets["physical_roles"] = list(map(int, storage_order))

    starts = []
    cursor = 0
    for line in physical:
        encoded = line.encode("utf-8")
        starts.append((cursor, cursor + len(encoded)))
        cursor += len(encoded) + 1
    semantic_ranges: list[list[int]] = [[0, 0] for _ in range(13)]
    for physical_index, semantic_role in enumerate(storage_order):
        semantic_ranges[int(semantic_role)] = list(starts[physical_index])
    targets["line_ranges"] = semantic_ranges

    declaration_line = lines[0]
    declaration_offset = semantic_ranges[0][0]
    binding_ranges = []
    initial_ranges = []
    search_start = 0
    names = _bindings(result)
    for name in names:
        start = declaration_line.index(name, search_start)
        binding_ranges.append(
            [declaration_offset + start, declaration_offset + start + len(name)]
        )
        search_start = start + len(name)
    initial_start = declaration_line.index(" ; ") + 5
    for name in tuple(names[int(role)] for role in targets["initial_order"]):
        start = declaration_line.index(name, initial_start)
        initial_ranges.append(
            [declaration_offset + start, declaration_offset + start + len(name)]
        )
        initial_start = start + len(name)
    targets["binding_ranges"] = binding_ranges
    targets["initial_ranges"] = initial_ranges

    witness_before_ranges: list[list[list[int]]] = []
    witness_after_ranges: list[list[list[int]]] = []
    for semantic_slot, item in enumerate(_rule_targets(result), start=1):
        line = lines[semantic_slot]
        line_offset = semantic_ranges[semantic_slot][0]
        cursor = line.index(str(item["opcode"])) + len(str(item["opcode"]))
        before_ranges = []
        after_ranges = []
        for name in map(str, item["before"]):
            start = line.index(name, cursor)
            before_ranges.append(
                [line_offset + start, line_offset + start + len(name)]
            )
            cursor = start + len(name)
        for name in map(str, item["after"]):
            start = line.index(name, cursor)
            after_ranges.append(
                [line_offset + start, line_offset + start + len(name)]
            )
            cursor = start + len(name)
        witness_before_ranges.append(before_ranges)
        witness_after_ranges.append(after_ranges)
    targets["witness_before_ranges"] = witness_before_ranges
    targets["witness_after_ranges"] = witness_after_ranges
    targets["query_range"] = list(query_span)
    result["compiler_targets"] = targets
    result["source_shape"] = {
        "program_bytes": len(result["program_text"].encode("utf-8")),
        "program_lines": 13,
        "query_bytes": len(query.encode("utf-8")),
        "max_line_bytes": max(len(line.encode("utf-8")) for line in physical),
    }
    return result


def parse_rendered_row(row: Mapping[str, object]) -> dict[str, object]:
    renderer = renderer_from_row(row)
    lines = str(row["program_text"]).splitlines()
    if len(lines) != 13:
        raise ValueError("ER-CST rendered program requires thirteen lines")

    declaration: tuple[str, ...] | None = None
    initial: tuple[str, ...] | None = None
    rules: dict[str, tuple[int, tuple[str, ...], tuple[str, ...]]] = {}
    events: dict[int, str | None] = {}
    for line in lines:
        declaration_match = DECLARATION_PATTERNS[renderer.declaration].fullmatch(line)
        if declaration_match:
            if declaration is not None:
                raise ValueError("ER-CST rendered program repeats declaration")
            declaration = declaration_match.groups()[:3]
            initial = declaration_match.groups()[3:]
            continue
        witness_match = WITNESS_PATTERNS[renderer.witness].fullmatch(line)
        if witness_match:
            slot_text, opcode, *symbols = witness_match.groups()
            if opcode in rules:
                raise ValueError("ER-CST rendered program repeats rule")
            rules[opcode] = (
                int(slot_text) - 1,
                tuple(symbols[:3]),
                tuple(symbols[3:]),
            )
            continue
        event_match = EVENT_PATTERNS[renderer.event].fullmatch(line)
        if event_match:
            slot = int(event_match.group(1)) - 1
            value = event_match.group(2)
            halt_word = "HALT" if renderer.event == 0 else "STOP"
            if slot in events:
                raise ValueError("ER-CST rendered program repeats event")
            events[slot] = None if value == halt_word else value
            continue
        raise ValueError("ER-CST rendered line is outside its grammar")
    query_match = QUERY_PATTERNS[renderer.query].fullmatch(str(row["late_query_text"]))
    if query_match is None:
        raise ValueError("ER-CST rendered query differs")
    if declaration is None or initial is None or len(rules) != 3 or set(events) != set(range(9)):
        raise ValueError("ER-CST rendered record cardinality differs")
    if {value[0] for value in rules.values()} != set(range(3)):
        raise ValueError("ER-CST rendered rule slots differ")
    if sum(value is None for value in events.values()) != 1:
        raise ValueError("ER-CST rendered program requires exactly one HALT")
    return {
        "bindings": declaration,
        "initial": initial,
        "rules": rules,
        "events": tuple(events[index] for index in range(9)),
        "query_position": int(query_match.group(1)) - 1,
    }
