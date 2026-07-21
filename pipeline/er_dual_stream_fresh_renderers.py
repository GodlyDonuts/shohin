"""Neutral-namespace renderers for the ordinal-route fresh board."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Mapping, Sequence


MAX_CARDINALITY = 6
MAX_RULES = 4
EVENT_SLOTS = 13
TT_RECORDS = 18
NEUTRAL_NAME = re.compile(r"^z[0-9a-z]{5}$")


@dataclass(frozen=True, slots=True)
class DualStreamFreshRenderer:
    declaration: int
    witness: int
    event: int
    query: int

    def __post_init__(self) -> None:
        if any(value not in (0, 1) for value in self.as_tuple()):
            raise ValueError("dual-stream renderer factors must be binary")

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.declaration, self.witness, self.event, self.query

    @property
    def name(self) -> str:
        d, w, e, q = self.as_tuple()
        return f"er-ds-d{d}w{w}e{e}q{q}-v2"


# Each side spans every declaration/witness pair exactly once. The scored coset
# flips both the event and query correlations seen in training.
TRAIN_RENDERERS = tuple(
    DualStreamFreshRenderer(*values)
    for values in ((0, 0, 0, 0), (0, 1, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1))
)
SCORED_RENDERERS = tuple(
    DualStreamFreshRenderer(*values)
    for values in ((0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 0, 0), (1, 1, 1, 0))
)


def _target(row: Mapping[str, object]) -> Mapping[str, object]:
    value = row.get("compiler_targets")
    if not isinstance(value, Mapping):
        raise ValueError("dual-stream row lacks compiler targets")
    return value


def _rules(row: Mapping[str, object]) -> list[Mapping[str, object]]:
    values = _target(row).get("rule_cards")
    if not isinstance(values, list) or len(values) != MAX_RULES:
        raise ValueError("dual-stream row requires four rule slots")
    ordered = sorted(values, key=lambda item: int(item["slot"]))
    if [int(item["slot"]) for item in ordered] != list(range(MAX_RULES)):
        raise ValueError("dual-stream rule slots differ")
    return ordered


def _events(row: Mapping[str, object]) -> list[Mapping[str, object]]:
    values = _target(row).get("events")
    if not isinstance(values, list) or len(values) != EVENT_SLOTS:
        raise ValueError("dual-stream row requires thirteen event slots")
    ordered = sorted(values, key=lambda item: int(item["slot"]))
    if [int(item["slot"]) for item in ordered] != list(range(EVENT_SLOTS)):
        raise ValueError("dual-stream event slots differ")
    return ordered


def renderer_from_row(row: Mapping[str, object]) -> DualStreamFreshRenderer:
    value = row.get("renderer")
    if not isinstance(value, Mapping) or int(value.get("version", -1)) != 2:
        raise ValueError("dual-stream row lacks v2 renderer metadata")
    renderer = DualStreamFreshRenderer(
        *(int(value[key]) for key in ("declaration", "witness", "event", "query"))
    )
    if row.get("template_id") != renderer.name:
        raise ValueError("dual-stream renderer name differs")
    return renderer


def _validate_neutral(values: Sequence[str], name: str) -> tuple[str, ...]:
    output = tuple(map(str, values))
    if not output or any(NEUTRAL_NAME.fullmatch(value) is None for value in output):
        raise ValueError(f"dual-stream {name} leaves the neutral namespace")
    return output


def semantic_lines(
    row: Mapping[str, object],
    renderer: DualStreamFreshRenderer,
    *,
    rule_distractors: Sequence[str],
    event_distractor: str,
    event_distractor_slot: int,
) -> tuple[str, ...]:
    target = _target(row)
    cardinality = int(target["cardinality"])
    bindings = _validate_neutral(target["entity_bindings"], "bindings")
    initial = tuple(bindings[int(index)] for index in target["initial_order"])
    if len(bindings) != cardinality or len(initial) != cardinality:
        raise ValueError("dual-stream declaration cardinality differs")
    if renderer.declaration == 0:
        declaration = f"BOX{cardinality} {' '.join(bindings)} ; SEED {' '.join(initial)}"
    else:
        declaration = f"POOL{cardinality} {' '.join(bindings)} ; START {' '.join(initial)}"

    rule_noise = _validate_neutral(rule_distractors, "rule distractors")
    if len(rule_noise) != MAX_RULES:
        raise ValueError("dual-stream rule distractor count differs")
    rules = []
    for item in _rules(row):
        slot = int(item["slot"]) + 1
        if not bool(item["active"]):
            prefix = "R" if renderer.witness == 0 else "M"
            rules.append(f"{prefix}{slot} NONE {rule_noise[slot - 1]}")
            continue
        opcode = _validate_neutral([str(item["opcode"])], "opcode")[0]
        before = _validate_neutral(item["before"], "before witnesses")
        after = _validate_neutral(item["after"], "after witnesses")
        if len(before) != cardinality or len(after) != cardinality:
            raise ValueError("dual-stream witness cardinality differs")
        if renderer.witness == 0:
            rules.append(f"R{slot} {opcode} {' '.join(before)} > {' '.join(after)}")
        else:
            rules.append(f"M{slot} {' '.join(before)} : {opcode} : {' '.join(after)}")

    noise = _validate_neutral([event_distractor], "event distractor")[0]
    if not 0 <= event_distractor_slot < EVENT_SLOTS:
        raise ValueError("dual-stream event distractor slot differs")
    events = []
    for item in _events(row):
        slot = int(item["slot"])
        value = "END" if bool(item["halt"]) else str(item["opcode"])
        if value != "END":
            _validate_neutral([value], "event opcode")
        include_noise = slot == event_distractor_slot
        if renderer.event == 0:
            suffix = f" {noise}" if include_noise else ""
            events.append(f"S{slot + 1} {value}{suffix}")
        else:
            prefix = f"{noise} " if include_noise else ""
            events.append(f"A{slot + 1} {prefix}{value}")
    return (declaration, *rules, *events)


def render_query(
    position: int,
    renderer: DualStreamFreshRenderer,
    distractor: str,
) -> tuple[str, tuple[int, int]]:
    if not 0 <= position < MAX_CARDINALITY:
        raise ValueError("dual-stream query position differs")
    noise = _validate_neutral([distractor], "query distractor")[0]
    numeral = str(position + 1)
    text = (
        f"Q{numeral} {noise}"
        if renderer.query == 0
        else f"{noise} SELECT {numeral}"
    )
    start = text.rindex(numeral)
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
    renderer: DualStreamFreshRenderer,
    *,
    storage_order: Sequence[int],
    row_id: str,
    family_id: str,
    rule_distractors: Sequence[str],
    event_distractor: str,
    event_distractor_slot: int,
    query_distractor: str,
) -> dict[str, object]:
    if sorted(map(int, storage_order)) != list(range(TT_RECORDS)):
        raise ValueError("dual-stream storage order differs")
    result = json.loads(json.dumps(dict(row)))
    lines = semantic_lines(
        result,
        renderer,
        rule_distractors=rule_distractors,
        event_distractor=event_distractor,
        event_distractor_slot=event_distractor_slot,
    )
    physical = [lines[int(index)] for index in storage_order]
    target = dict(result["compiler_targets"])
    query, query_span = render_query(
        int(target["query_position"]), renderer, query_distractor
    )
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
                "version": 2,
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
    separator = " ; SEED " if renderer.declaration == 0 else " ; START "
    initial_local = _spans(
        declaration, initial, declaration.index(separator) + len(separator)
    )
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
        before = tuple(map(str, item["before"]))
        after = tuple(map(str, item["after"]))
        if renderer.witness == 0:
            cursor = line.index(opcode) + len(opcode)
            before_local = _spans(line, before, cursor)
            after_local = _spans(line, after, before_local[-1][1])
        else:
            before_local = _spans(line, before)
            after_local = _spans(line, after, line.index(opcode) + len(opcode))
        before_ranges.append(
            [[offset + start, offset + end] for start, end in before_local]
        )
        after_ranges.append(
            [[offset + start, offset + end] for start, end in after_local]
        )
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
        raise ValueError("dual-stream rendered declaration spans differ")
    return result


def parse_rendered_row(row: Mapping[str, object]) -> dict[str, object]:
    """Parse public bytes without consulting compiler targets or oracles."""
    renderer = renderer_from_row(row)
    lines = str(row["program_text"]).splitlines()
    if len(lines) != TT_RECORDS:
        raise ValueError("dual-stream program requires eighteen lines")
    declaration: tuple[str, ...] | None = None
    initial: tuple[str, ...] | None = None
    cardinality: int | None = None
    rules: dict[str, tuple[int, tuple[str, ...], tuple[str, ...]]] = {}
    inactive: set[int] = set()
    events: dict[int, str | None] = {}
    distractors: list[str] = []

    declaration_pattern = (
        re.compile(r"^BOX([3-6]) (.+) ; SEED (.+)$")
        if renderer.declaration == 0
        else re.compile(r"^POOL([3-6]) (.+) ; START (.+)$")
    )
    inactive_pattern = (
        re.compile(r"^R([1-4]) NONE (\S+)$")
        if renderer.witness == 0
        else re.compile(r"^M([1-4]) NONE (\S+)$")
    )
    active_pattern = (
        re.compile(r"^R([1-4]) (\S+) (.+) > (.+)$")
        if renderer.witness == 0
        else re.compile(r"^M([1-4]) (.+) : (\S+) : (.+)$")
    )
    event_pattern = (
        re.compile(r"^S(1[0-3]|[1-9]) (\S+)(?: (\S+))?$")
        if renderer.event == 0
        else re.compile(r"^A(1[0-3]|[1-9]) (?:(\S+) )?(\S+)$")
    )

    for line in lines:
        match = declaration_pattern.fullmatch(line)
        if match:
            if declaration is not None:
                raise ValueError("dual-stream declaration repeats")
            cardinality = int(match.group(1))
            declaration = _validate_neutral(match.group(2).split(), "parsed bindings")
            initial = _validate_neutral(match.group(3).split(), "parsed initial")
            continue
        match = inactive_pattern.fullmatch(line)
        if match:
            slot = int(match.group(1)) - 1
            noise = _validate_neutral([match.group(2)], "inactive distractor")[0]
            if slot in inactive or any(value[0] == slot for value in rules.values()):
                raise ValueError("dual-stream rule slot repeats")
            inactive.add(slot)
            distractors.append(noise)
            continue
        match = active_pattern.fullmatch(line)
        if match:
            slot = int(match.group(1)) - 1
            if renderer.witness == 0:
                opcode, before_raw, after_raw = match.group(2, 3, 4)
            else:
                before_raw, opcode, after_raw = match.group(2, 3, 4)
            opcode = _validate_neutral([opcode], "parsed opcode")[0]
            before = _validate_neutral(before_raw.split(), "parsed before")
            after = _validate_neutral(after_raw.split(), "parsed after")
            if slot in inactive or any(value[0] == slot for value in rules.values()):
                raise ValueError("dual-stream rule slot repeats")
            if opcode in rules:
                raise ValueError("dual-stream opcode repeats")
            rules[opcode] = (slot, before, after)
            continue
        match = event_pattern.fullmatch(line)
        if match:
            slot = int(match.group(1)) - 1
            if slot in events:
                raise ValueError("dual-stream event repeats")
            if renderer.event == 0:
                value, noise = match.group(2), match.group(3)
            else:
                noise, value = match.group(2), match.group(3)
            if noise is not None:
                distractors.append(
                    _validate_neutral([noise], "event distractor")[0]
                )
            if value != "END":
                value = _validate_neutral([value], "parsed event opcode")[0]
            events[slot] = None if value == "END" else value
            continue
        raise ValueError("dual-stream line is outside its grammar")

    query = str(row["late_query_text"])
    query_match = (
        re.fullmatch(r"Q([1-6]) (\S+)", query)
        if renderer.query == 0
        else re.fullmatch(r"(\S+) SELECT ([1-6])", query)
    )
    if query_match is None:
        raise ValueError("dual-stream query differs")
    if renderer.query == 0:
        query_position = int(query_match.group(1)) - 1
        query_noise = query_match.group(2)
    else:
        query_noise = query_match.group(1)
        query_position = int(query_match.group(2)) - 1
    distractors.append(
        _validate_neutral([query_noise], "query distractor")[0]
    )
    if cardinality is None or declaration is None or initial is None:
        raise ValueError("dual-stream declaration is absent")
    if len(declaration) != cardinality or len(initial) != cardinality:
        raise ValueError("dual-stream declaration cardinality differs")
    if len(rules) + len(inactive) != MAX_RULES or set(events) != set(range(EVENT_SLOTS)):
        raise ValueError("dual-stream record cardinality differs")
    if {value[0] for value in rules.values()} | inactive != set(range(MAX_RULES)):
        raise ValueError("dual-stream rule slots differ")
    if sum(value is None for value in events.values()) != 1:
        raise ValueError("dual-stream program requires exactly one END")
    if any(value is not None and value not in rules for value in events.values()):
        raise ValueError("dual-stream event references an absent opcode")
    semantic = set(declaration) | set(initial) | set(rules)
    for _, before, after in rules.values():
        semantic.update(before)
        semantic.update(after)
    if semantic & set(distractors):
        raise ValueError("dual-stream distractor aliases a semantic name")
    return {
        "cardinality": cardinality,
        "bindings": declaration,
        "initial": initial,
        "rules": rules,
        "inactive_rule_slots": tuple(sorted(inactive)),
        "events": tuple(events[index] for index in range(EVENT_SLOTS)),
        "query_position": query_position,
        "distractors": tuple(distractors),
    }


def independently_execute(row: Mapping[str, object]) -> dict[str, object]:
    parsed = parse_rendered_row(row)
    bindings = tuple(parsed["bindings"])
    state = tuple(parsed["initial"])
    inferred: dict[str, tuple[int, ...]] = {}
    slots: dict[int, tuple[int, ...]] = {}
    for opcode, (slot, before, after) in parsed["rules"].items():
        if len(before) != len(bindings) or len(after) != len(bindings):
            raise ValueError("dual-stream witness cardinality differs")
        if len(set(before)) != len(before) or any(value not in before for value in after):
            raise ValueError("dual-stream witness is not determining")
        relation = tuple(before.index(value) for value in after)
        inferred[str(opcode)] = relation
        slots[int(slot)] = relation
    trajectory = [state]
    halt_after = None
    for index, opcode in enumerate(parsed["events"]):
        if opcode is None:
            halt_after = index
            break
        state = tuple(state[position] for position in inferred[str(opcode)])
        trajectory.append(state)
    if halt_after is None:
        raise ValueError("dual-stream execution lacks END")
    query_position = int(parsed["query_position"])
    if not 0 <= query_position < len(bindings):
        raise ValueError("dual-stream query exceeds cardinality")
    return {
        **parsed,
        "rule_relations": slots,
        "final_state": state,
        "trajectory": tuple(trajectory),
        "halt_after": halt_after,
        "answer_role": bindings.index(state[query_position]),
    }
