"""Occurrence-quotient relation mechanics for the S9 grounding compiler.

The model-facing object is a set of source spans plus relation/slot labels. Exact
surface equality is the only identity primitive: it quotients repeated emitted
spans into classes. This module never infers a gold name, role, order, or link.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from s8_nil_linked_law_graph import (
    EventNode,
    LawCardNode,
    NilLinkedLawGraph,
    linked_path,
)


@dataclass(frozen=True)
class RelationRecord:
    kind: str
    ordinal: int
    arguments: tuple[int | None, ...]


@dataclass(frozen=True)
class OccurrenceQuotient:
    classes: tuple[str, ...]
    relations: tuple[RelationRecord, ...]


RELATION_ARITY = {
    "entity_roster": 1,
    "position_roster": 1,
    "state": 1,
    "card": 3,
    "event": 4,
    "entry": 1,
    "query": 1,
}


def _span_text(
    question: str, spans: Mapping[str, Mapping[str, object]], label: str
) -> str:
    if label not in spans:
        raise ValueError(f"S9 missing emitted span {label}")
    span = spans[label]
    start, end = int(span["start"]), int(span["end"])
    text = str(span["text"])
    if not text or question[start:end] != text:
        raise ValueError(f"S9 emitted span mismatch for {label}")
    return text


def _indices(spans: Mapping[str, object], prefix: str, suffix: str) -> tuple[int, ...]:
    result = set()
    for label in spans:
        if not label.startswith(prefix) or not label.endswith(suffix):
            continue
        middle = label[len(prefix):len(label) - len(suffix) if suffix else None]
        if not middle.isdigit():
            raise ValueError(f"S9 malformed indexed label {label}")
        result.add(int(middle))
    if result and result != set(range(max(result) + 1)):
        raise ValueError(f"S9 non-contiguous labels for {prefix}*{suffix}")
    return tuple(sorted(result))


def quotient_from_emitted_spans(
    question: str,
    spans: Mapping[str, Mapping[str, object]],
) -> OccurrenceQuotient:
    """Build the quotient from model-emitted spans and semantic relation labels.

    In the CPU theorem the frozen board's labeled spans stand in for model
    emissions. A neural S9 implementation must predict them from source tokens.
    """

    surface_records: list[tuple[str, int, tuple[str | None, ...]]] = []
    for prefix, kind in (
        ("entity.roster.", "entity_roster"),
        ("position.roster.", "position_roster"),
        ("state.entity.", "state"),
    ):
        for index in _indices(spans, prefix, ""):
            surface_records.append(
                (kind, index, (_span_text(question, spans, f"{prefix}{index}"),))
            )

    for index in _indices(spans, "card.", ".operation"):
        surface_records.append((
            "card",
            index,
            tuple(
                _span_text(question, spans, f"card.{index}.{slot}")
                for slot in ("operation", "y0", "y1")
            ),
        ))

    for index in _indices(spans, "event.", ".tag"):
        next_label = f"event.{index}.next"
        nil_label = f"event.{index}.nil"
        if (next_label in spans) == (nil_label in spans):
            raise ValueError("S9 event must emit exactly one next or nil slot")
        surface_records.append((
            "event",
            index,
            (
                _span_text(question, spans, f"event.{index}.tag"),
                _span_text(question, spans, f"event.{index}.operation"),
                _span_text(question, spans, f"event.{index}.entity"),
                _span_text(question, spans, next_label) if next_label in spans else None,
            ),
        ))

    surface_records.extend((
        ("entry", 0, (_span_text(question, spans, "entry.tag"),)),
        ("query", 0, (_span_text(question, spans, "query.position"),)),
    ))
    surfaces = sorted({value for _, _, args in surface_records for value in args if value})
    if not surfaces:
        raise ValueError("S9 quotient has no emitted surfaces")
    class_index = {value: index for index, value in enumerate(surfaces)}
    relations = tuple(
        RelationRecord(
            kind=kind,
            ordinal=ordinal,
            arguments=tuple(class_index[value] if value is not None else None for value in args),
        )
        for kind, ordinal, args in surface_records
    )
    return OccurrenceQuotient(classes=tuple(surfaces), relations=relations)


def _ordered(
    quotient: OccurrenceQuotient, kind: str
) -> tuple[RelationRecord, ...]:
    values = tuple(sorted(
        (record for record in quotient.relations if record.kind == kind),
        key=lambda record: record.ordinal,
    ))
    if not values or tuple(record.ordinal for record in values) != tuple(range(len(values))):
        raise ValueError(f"S9 {kind} relation ordinals are incomplete")
    if any(len(record.arguments) != RELATION_ARITY[kind] for record in values):
        raise ValueError(f"S9 {kind} relation arity mismatch")
    return values


def _one(quotient: OccurrenceQuotient, kind: str) -> RelationRecord:
    values = _ordered(quotient, kind)
    if len(values) != 1:
        raise ValueError(f"S9 requires exactly one {kind} relation")
    return values[0]


def compile_quotient(quotient: OccurrenceQuotient) -> NilLinkedLawGraph:
    """Compile class-level relation tuples into the unchanged S8 graph."""

    if len(set(quotient.classes)) != len(quotient.classes):
        raise ValueError("S9 quotient classes are not unique")
    if any(not value for value in quotient.classes):
        raise ValueError("S9 quotient has an empty class")
    for record in quotient.relations:
        if record.kind not in RELATION_ARITY:
            raise ValueError(f"S9 unknown relation kind {record.kind}")
        for argument in record.arguments:
            if argument is not None and not 0 <= argument < len(quotient.classes):
                raise ValueError("S9 relation argument outside quotient")

    entity_records = _ordered(quotient, "entity_roster")
    position_records = _ordered(quotient, "position_roster")
    state_records = _ordered(quotient, "state")
    entity_classes = tuple(int(record.arguments[0]) for record in entity_records)
    position_classes = tuple(int(record.arguments[0]) for record in position_records)
    state_classes = tuple(int(record.arguments[0]) for record in state_records)
    modulus = len(entity_classes)
    if modulus not in {5, 7, 11} or len(position_classes) != modulus:
        raise ValueError("S9 roster cardinality mismatch")
    if len(set(entity_classes)) != modulus or len(set(position_classes)) != modulus:
        raise ValueError("S9 roster identity collision")
    if len(state_classes) != modulus or set(state_classes) != set(entity_classes):
        raise ValueError("S9 state is not a permutation of entity classes")
    entity_index = {value: index for index, value in enumerate(entity_classes)}
    position_index = {value: index for index, value in enumerate(position_classes)}

    card_records = _ordered(quotient, "card")
    cards = []
    operation_classes = set()
    for record in card_records:
        operation, y0, y1 = record.arguments
        if None in record.arguments or operation in operation_classes:
            raise ValueError("S9 card operation is missing or duplicated")
        if y0 not in position_index or y1 not in position_index or y0 == y1:
            raise ValueError("S9 card witnesses are not distinct position classes")
        operation_classes.add(operation)
        cards.append(LawCardNode(
            operation=quotient.classes[int(operation)],
            y0=position_index[int(y0)],
            y1=position_index[int(y1)],
        ))

    event_records = _ordered(quotient, "event")
    tag_classes = tuple(int(record.arguments[0]) for record in event_records)
    if len(set(tag_classes)) != len(tag_classes):
        raise ValueError("S9 event tags collide")
    tag_index = {value: index for index, value in enumerate(tag_classes)}
    nodes = []
    for record in event_records:
        tag, operation, entity, next_tag = record.arguments
        if operation not in operation_classes or entity not in entity_index:
            raise ValueError("S9 event has an unbound operation or entity")
        if next_tag is not None and next_tag not in tag_index:
            raise ValueError("S9 event has an unbound next tag")
        nodes.append(EventNode(
            identity=entity_index[int(entity)],
            operation=quotient.classes[int(operation)],
            next_node=-1 if next_tag is None else tag_index[int(next_tag)],
        ))

    entry_class = _one(quotient, "entry").arguments[0]
    query_class = _one(quotient, "query").arguments[0]
    if entry_class not in tag_index or query_class not in position_index:
        raise ValueError("S9 entry or query class is unbound")
    graph = NilLinkedLawGraph(
        modulus=modulus,
        initial_state=tuple(entity_index[value] for value in state_classes),
        cards=tuple(cards),
        nodes=tuple(nodes),
        entry_node=tag_index[int(entry_class)],
        query_position=position_index[int(query_class)],
    )
    linked_path(graph)
    return graph


def reindex_classes(
    quotient: OccurrenceQuotient, permutation: Sequence[int]
) -> OccurrenceQuotient:
    if sorted(permutation) != list(range(len(quotient.classes))):
        raise ValueError("S9 class permutation is invalid")
    old_to_new = {old: new for new, old in enumerate(permutation)}
    return OccurrenceQuotient(
        classes=tuple(quotient.classes[old] for old in permutation),
        relations=tuple(replace(
            record,
            arguments=tuple(
                old_to_new[value] if value is not None else None
                for value in record.arguments
            ),
        ) for record in quotient.relations),
    )


def permute_relation_storage(
    quotient: OccurrenceQuotient, permutation: Sequence[int]
) -> OccurrenceQuotient:
    if sorted(permutation) != list(range(len(quotient.relations))):
        raise ValueError("S9 relation permutation is invalid")
    return replace(
        quotient,
        relations=tuple(quotient.relations[index] for index in permutation),
    )


def split_first_event_operation(quotient: OccurrenceQuotient) -> OccurrenceQuotient:
    records = list(quotient.relations)
    index = next(i for i, record in enumerate(records) if record.kind == "event")
    operation = records[index].arguments[1]
    classes = quotient.classes + (quotient.classes[int(operation)] + "_alias",)
    arguments = list(records[index].arguments)
    arguments[1] = len(classes) - 1
    records[index] = replace(records[index], arguments=tuple(arguments))
    return OccurrenceQuotient(classes, tuple(records))


def merge_first_two_entities(quotient: OccurrenceQuotient) -> OccurrenceQuotient:
    entities = _ordered(quotient, "entity_roster")
    first, second = (int(entities[index].arguments[0]) for index in range(2))
    return replace(quotient, relations=tuple(replace(
        record,
        arguments=tuple(first if value == second else value for value in record.arguments),
    ) for record in quotient.relations))


def unique_every_occurrence(quotient: OccurrenceQuotient) -> OccurrenceQuotient:
    """Free-word boundary: remove all exact-equality evidence without merging."""

    classes: list[str] = []
    records = []
    for record_index, record in enumerate(quotient.relations):
        arguments = []
        for slot, value in enumerate(record.arguments):
            if value is None:
                arguments.append(None)
                continue
            classes.append(f"{quotient.classes[value]}__r{record_index}s{slot}")
            arguments.append(len(classes) - 1)
        records.append(replace(record, arguments=tuple(arguments)))
    return OccurrenceQuotient(tuple(classes), tuple(records))


def swap_card_witnesses(quotient: OccurrenceQuotient) -> OccurrenceQuotient:
    records = []
    for record in quotient.relations:
        if record.kind != "card":
            records.append(record)
            continue
        operation, y0, y1 = record.arguments
        records.append(replace(record, arguments=(operation, y1, y0)))
    return replace(quotient, relations=tuple(records))


def corrupt_first_relation_kind(quotient: OccurrenceQuotient) -> OccurrenceQuotient:
    records = list(quotient.relations)
    index = next(i for i, record in enumerate(records) if record.kind == "card")
    records[index] = replace(records[index], kind="event")
    return replace(quotient, relations=tuple(records))
