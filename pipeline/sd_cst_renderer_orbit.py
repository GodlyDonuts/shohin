"""Factorized renderer orbit for the next SD-CST grounding experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RendererOrbitElement:
    declaration: int
    event: int
    query: int

    def __post_init__(self) -> None:
        if any(value not in (0, 1) for value in self.as_tuple()):
            raise ValueError("renderer factors must be binary")

    def as_tuple(self) -> tuple[int, int, int]:
        return self.declaration, self.event, self.query

    @property
    def parity(self) -> int:
        return sum(self.as_tuple()) % 2

    @property
    def name(self) -> str:
        return f"orbit-d{self.declaration}e{self.event}q{self.query}-v1"


RENDERER_ORBIT = tuple(
    RendererOrbitElement(declaration, event, query)
    for declaration in range(2)
    for event in range(2)
    for query in range(2)
)
TRAIN_RENDERERS = tuple(item for item in RENDERER_ORBIT if item.parity == 0)
HELD_OUT_RENDERERS = tuple(item for item in RENDERER_ORBIT if item.parity == 1)


def _targets(row: Mapping[str, object]) -> Mapping[str, object]:
    targets = row.get("compiler_targets")
    if not isinstance(targets, Mapping):
        raise ValueError("row lacks compiler targets")
    return targets


def _names_by_role(row: Mapping[str, object]) -> tuple[str, str, str]:
    values = _targets(row).get("entity_bindings")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("renderer orbit requires three entity bindings")
    ordered = sorted(values, key=lambda item: int(item["entity_role"]))
    names = tuple(str(item["entity"]) for item in ordered)
    if len(set(names)) != 3:
        raise ValueError("renderer orbit requires distinct entity names")
    return names  # type: ignore[return-value]


def _initial_names(row: Mapping[str, object]) -> tuple[str, str, str]:
    names = _names_by_role(row)
    order = _targets(row).get("initial_order_roles")
    if not isinstance(order, list) or sorted(map(int, order)) != [0, 1, 2]:
        raise ValueError("renderer orbit requires an initial role permutation")
    return tuple(names[int(role)] for role in order)  # type: ignore[return-value]


def _event_slots(row: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    slots = _targets(row).get("event_slots")
    if not isinstance(slots, list) or len(slots) != 8:
        raise ValueError("renderer orbit requires eight event slots")
    result = {int(item["semantic_ordinal"]): item for item in slots}
    if set(result) != set(range(1, 9)):
        raise ValueError("renderer orbit event ordinals differ")
    return result


def _storage_order(row: Mapping[str, object]) -> tuple[int, ...]:
    order = _targets(row).get("storage_order")
    if not isinstance(order, list) or sorted(map(int, order)) != list(range(1, 9)):
        raise ValueError("renderer orbit storage order differs")
    return tuple(map(int, order))


def render_program(
    row: Mapping[str, object],
    renderer: RendererOrbitElement,
) -> str:
    names = _names_by_role(row)
    initial = _initial_names(row)
    if renderer.declaration == 0:
        first = (
            f"Bindings: alpha {names[0]}; beta {names[1]}; gamma {names[2]}; "
            f"initial {', '.join(initial)}."
        )
    else:
        first = (
            f"Registry: {names[0]} is alpha; {names[1]} is beta; "
            f"{names[2]} is gamma; lineup {', '.join(initial)}."
        )

    lines = [first]
    slots = _event_slots(row)
    for ordinal in _storage_order(row):
        slot = slots[ordinal]
        if str(slot["kind"]) == "stop":
            noun = "event" if renderer.event == 0 else "action"
            lines.append(f"{noun.title()} {ordinal}: STOP.")
            continue
        entity = str(slot["entity"])
        direction = {"left": "west", "right": "east"}[str(slot["direction"])]
        amount = {1: "unit", 2: "pair"}[int(slot["amount"])]
        if renderer.event == 0:
            lines.append(f"Event {ordinal}: move {entity} {direction} by {amount}.")
        else:
            lines.append(f"Action {ordinal}: send {entity} {direction} for {amount}.")
    return "\n".join(lines)


def render_query(
    row: Mapping[str, object],
    renderer: RendererOrbitElement,
) -> tuple[str, tuple[int, int]]:
    target = row.get("late_query_target")
    if not isinstance(target, Mapping):
        raise ValueError("renderer orbit row lacks late-query target")
    numeral = str(int(target["position"]) + 1)
    if renderer.query == 0:
        text = f"Which entity now occupies position {numeral}?"
    else:
        text = f"Report the entity currently in slot {numeral}."
    encoded = text.encode("utf-8")
    start = encoded.rfind(numeral.encode("ascii"))
    if start < 0:
        raise RuntimeError("query numeral absent after rendering")
    return text, (start, start + len(numeral))


def render_row(
    row: Mapping[str, object],
    renderer: RendererOrbitElement,
    *,
    row_id: str | None = None,
    family_id: str | None = None,
) -> dict[str, object]:
    result = json.loads(json.dumps(dict(row)))
    query_text, query_span = render_query(row, renderer)
    result["id"] = row_id or f"{row['id']}::{renderer.name}"
    result["template_id"] = renderer.name
    result["variant"] = renderer.name
    result["program_text"] = render_program(row, renderer)
    result["late_query_text"] = query_text
    result["renderer_orbit"] = {
        "declaration": renderer.declaration,
        "event": renderer.event,
        "query": renderer.query,
        "parity": renderer.parity,
    }
    if family_id is not None:
        result["renderer_family_id"] = family_id
    target = dict(result["late_query_target"])
    target["byte_span"] = list(query_span)
    result["late_query_target"] = target
    result.pop("audit_only_combined_normalized_prompt", None)
    result["source_shape"] = {
        "program_line_count": len(str(result["program_text"]).splitlines()),
        "late_query_line_count": 1,
        "event_clause_count": 8,
        "program_character_count": len(str(result["program_text"])),
        "program_word_count": len(
            str(result["program_text"]).replace("\n", " ").split()
        ),
        "query_character_count": len(query_text),
        "query_word_count": len(query_text.split()),
    }
    return result


def orbit_atom_coverage(
    renderers: tuple[RendererOrbitElement, ...],
) -> dict[str, set[int]]:
    return {
        "declaration": {item.declaration for item in renderers},
        "event": {item.event for item in renderers},
        "query": {item.query for item in renderers},
    }
