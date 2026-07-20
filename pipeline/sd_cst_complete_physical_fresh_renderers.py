"""Fresh renderer orbit for the complete physical-record SD-CST compiler."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class FreshRenderer:
    declaration: int
    event: int
    query: int

    def __post_init__(self) -> None:
        if any(value not in (0, 1) for value in self.as_tuple()):
            raise ValueError("fresh renderer factors must be binary")

    def as_tuple(self) -> tuple[int, int, int]:
        return self.declaration, self.event, self.query

    @property
    def parity(self) -> int:
        return sum(self.as_tuple()) % 2

    @property
    def name(self) -> str:
        return f"physical-fresh-d{self.declaration}e{self.event}q{self.query}-v1"


RENDERERS = tuple(
    FreshRenderer(declaration, event, query)
    for declaration in range(2)
    for event in range(2)
    for query in range(2)
)
TRAIN_RENDERERS = tuple(item for item in RENDERERS if item.parity == 0)
SCORED_RENDERERS = tuple(item for item in RENDERERS if item.parity == 1)


def _targets(row: Mapping[str, object]) -> Mapping[str, object]:
    value = row.get("compiler_targets")
    if not isinstance(value, Mapping):
        raise ValueError("fresh renderer row lacks compiler targets")
    return value


def _names(row: Mapping[str, object]) -> tuple[str, str, str]:
    values = _targets(row).get("entity_bindings")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("fresh renderer requires three bindings")
    ordered = sorted(values, key=lambda item: int(item["entity_role"]))
    names = tuple(str(item["entity"]) for item in ordered)
    if len(set(names)) != 3:
        raise ValueError("fresh renderer names are not distinct")
    return names  # type: ignore[return-value]


def _initial_names(row: Mapping[str, object]) -> tuple[str, str, str]:
    names = _names(row)
    order = _targets(row).get("initial_order_roles")
    if not isinstance(order, list) or sorted(map(int, order)) != [0, 1, 2]:
        raise ValueError("fresh renderer initial order differs")
    return tuple(names[int(role)] for role in order)  # type: ignore[return-value]


def _event_slots(row: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    values = _targets(row).get("event_slots")
    if not isinstance(values, list) or len(values) != 8:
        raise ValueError("fresh renderer requires eight events")
    slots = {int(item["semantic_ordinal"]): item for item in values}
    if set(slots) != set(range(1, 9)):
        raise ValueError("fresh renderer ordinals differ")
    return slots


def _storage_order(row: Mapping[str, object]) -> tuple[int, ...]:
    value = _targets(row).get("storage_order")
    if not isinstance(value, list) or sorted(map(int, value)) != list(range(1, 9)):
        raise ValueError("fresh renderer storage order differs")
    return tuple(map(int, value))


def render_program(row: Mapping[str, object], renderer: FreshRenderer) -> str:
    names = _names(row)
    initial = _initial_names(row)
    if renderer.declaration == 0:
        declaration = (
            f"Manifest: first {names[0]}; second {names[1]}; third {names[2]}; "
            f"initial {' > '.join(initial)}."
        )
    else:
        declaration = (
            f"Directory: first {names[0]}; second {names[1]}; third {names[2]}; "
            f"lineup {' > '.join(initial)}."
        )

    lines = [declaration]
    slots = _event_slots(row)
    direction_words = (
        {"left": "port", "right": "starboard"}
        if renderer.event == 0
        else {"left": "inward", "right": "outward"}
    )
    amount_words = (
        {1: "single", 2: "double"}
        if renderer.event == 0
        else {1: "one-step", 2: "two-step"}
    )
    for ordinal in _storage_order(row):
        slot = slots[ordinal]
        if str(slot["kind"]) == "stop":
            noun = "Instruction" if renderer.event == 0 else "Command"
            lines.append(f"{noun} {ordinal}: HALT.")
            continue
        entity = str(slot["entity"])
        direction = direction_words[str(slot["direction"])]
        amount = amount_words[int(slot["amount"])]
        if renderer.event == 0:
            lines.append(
                f"Instruction {ordinal}: slide {entity} {direction} by {amount}."
            )
        else:
            lines.append(f"Command {ordinal}: carry {entity} {direction} for {amount}.")
    return "\n".join(lines)


def render_query(
    row: Mapping[str, object], renderer: FreshRenderer
) -> tuple[str, tuple[int, int]]:
    target = row.get("late_query_target")
    if not isinstance(target, Mapping):
        raise ValueError("fresh renderer row lacks query target")
    numeral = str(int(target["position"]) + 1)
    text = (
        f"Identify the member at rank {numeral}."
        if renderer.query == 0
        else f"Return whoever stands in place {numeral}."
    )
    encoded = text.encode("utf-8")
    start = encoded.rfind(numeral.encode("ascii"))
    if start < 0:
        raise RuntimeError("fresh query numeral is absent")
    return text, (start, start + len(numeral))


def render_row(
    row: Mapping[str, object],
    renderer: FreshRenderer,
    *,
    row_id: str,
    family_id: str,
) -> dict[str, object]:
    result = json.loads(json.dumps(dict(row)))
    query, query_span = render_query(row, renderer)
    result["id"] = row_id
    result["family_id"] = family_id
    result["template_id"] = renderer.name
    result["variant"] = renderer.name
    result["program_text"] = render_program(row, renderer)
    result["late_query_text"] = query
    result["fresh_renderer"] = {
        "declaration": renderer.declaration,
        "event": renderer.event,
        "query": renderer.query,
        "parity": renderer.parity,
    }
    query_target = dict(result["late_query_target"])
    query_target["byte_span"] = list(query_span)
    result["late_query_target"] = query_target
    result.pop("audit_only_combined_normalized_prompt", None)
    result["audit_only_combined_normalized_prompt"] = " ".join(
        (str(result["program_text"]) + "\n" + query).lower().split()
    )
    result["source_shape"] = {
        "program_line_count": 9,
        "late_query_line_count": 1,
        "event_clause_count": 8,
        "program_character_count": len(str(result["program_text"])),
        "program_word_count": len(
            str(result["program_text"]).replace("\n", " ").split()
        ),
        "query_character_count": len(query),
        "query_word_count": len(query.split()),
    }
    return result


def expand_rows(
    rows: Sequence[Mapping[str, object]],
    renderers: Sequence[FreshRenderer],
) -> list[dict[str, object]]:
    output = []
    for row in rows:
        family = str(row["id"])
        for renderer in renderers:
            output.append(
                render_row(
                    row,
                    renderer,
                    row_id=f"{family}::{renderer.name}",
                    family_id=family,
                )
            )
    return output
