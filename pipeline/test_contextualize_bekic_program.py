from __future__ import annotations

import copy

import pytest

from contextualize_bekic_program import (
    CARD_WITNESSES,
    PRIMITIVE_ORDER,
    ContextualizationError,
    contextualize_simultaneous_packet,
    identify_contextual_slots,
    reindex_contextual_objects,
    validate_contextual_packet,
    validate_contextual_packet_structure,
)
from contrastive_bekic_program_orbits import (
    generate_orbit,
    select_machine_input,
)


def _source(seed: int = 1301) -> dict[str, object]:
    row = generate_orbit(split="train", seed=seed)
    return select_machine_input(row, arm="p", form="simultaneous")


def _recursive_values(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(item) for item in value.values() if isinstance(item, str)} | set().union(
            *(_recursive_values(item) for item in value.values()),
            set(),
        )
    if isinstance(value, list):
        return set().union(*(_recursive_values(item) for item in value), set())
    return set()


def test_operation_names_are_deleted_and_cards_are_identifiable() -> None:
    packet = contextualize_simultaneous_packet(_source(), seed=812)
    validate_contextual_packet(packet)
    values = _recursive_values(packet)
    assert set(PRIMITIVE_ORDER).isdisjoint(values)
    bindings = identify_contextual_slots(packet)
    assert len(bindings) == len(packet["operation_cards"])
    assert all(
        len(card["witnesses"]) == CARD_WITNESSES
        for card in packet["operation_cards"]
    )
    assert set(bindings.values()) <= set(range(len(PRIMITIVE_ORDER)))


def test_contextualization_is_deterministic_but_seeded() -> None:
    source = _source(1302)
    first = contextualize_simultaneous_packet(source, seed=92)
    assert first == contextualize_simultaneous_packet(source, seed=92)
    second = contextualize_simultaneous_packet(source, seed=93)
    assert first != second
    assert identify_contextual_slots(first) == {
        card["slot"]: identify_contextual_slots(first)[card["slot"]]
        for card in first["operation_cards"]
    }


def test_node_and_card_order_do_not_define_semantics() -> None:
    packet = contextualize_simultaneous_packet(_source(1303), seed=94)
    changed = copy.deepcopy(packet)
    changed["nodes"].reverse()
    changed["operation_cards"].reverse()
    for card in changed["operation_cards"]:
        card["witnesses"].reverse()
    validate_contextual_packet(changed)
    assert identify_contextual_slots(changed) == identify_contextual_slots(packet)


def test_object_reindexing_preserves_card_binding() -> None:
    packet = contextualize_simultaneous_packet(_source(1304), seed=95)
    cardinality = packet["cardinality"]
    permutation = tuple(reversed(range(cardinality)))
    changed = reindex_contextual_objects(packet, permutation)
    assert identify_contextual_slots(changed) == identify_contextual_slots(packet)


def test_ambiguous_and_deranged_cards_fail_closed() -> None:
    packet = contextualize_simultaneous_packet(_source(1305), seed=96)
    binary = next(card for card in packet["operation_cards"] if card["arity"] == 2)
    ambiguous = copy.deepcopy(packet)
    target = next(
        card
        for card in ambiguous["operation_cards"]
        if card["slot"] == binary["slot"]
    )
    zero = [[0] * packet["cardinality"] for _ in range(packet["cardinality"])]
    target["witnesses"] = [
        {"left": zero, "right": zero, "output": zero}
        for _ in range(CARD_WITNESSES)
    ]
    validate_contextual_packet_structure(ambiguous)
    with pytest.raises(ContextualizationError, match="uniquely"):
        validate_contextual_packet(ambiguous)

    deranged = copy.deepcopy(packet)
    binary_cards = [
        card for card in deranged["operation_cards"] if card["arity"] == 2
    ]
    if len(binary_cards) >= 2:
        binary_cards[0]["witnesses"], binary_cards[1]["witnesses"] = (
            binary_cards[1]["witnesses"],
            binary_cards[0]["witnesses"],
        )
        # A whole-card swap changes the local laws but remains identifiable;
        # swapping only outputs makes the card contradictory.
        binary_cards[0]["witnesses"][0]["output"] = binary_cards[1][
            "witnesses"
        ][1]["output"]
        with pytest.raises(ContextualizationError, match="uniquely"):
            validate_contextual_packet(deranged)


def test_disconnected_nodes_fail_structural_validation() -> None:
    packet = contextualize_simultaneous_packet(_source(1307), seed=98)
    disconnected = copy.deepcopy(packet)
    leaf = next(
        node
        for node in disconnected["nodes"]
        if node["kind"] == "CONSTANT"
    )
    disconnected["nodes"].append(
        {
            "id": f"{leaf['id']}_disconnected",
            "kind": "CONSTANT",
            "constant": leaf["constant"],
        }
    )
    with pytest.raises(ContextualizationError, match="disconnected"):
        validate_contextual_packet_structure(disconnected)


def test_contextual_machine_packet_has_no_oracle_mapping_or_targets() -> None:
    packet = contextualize_simultaneous_packet(_source(1306), seed=97)
    keys = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            keys.update(str(key).lower() for key in value)
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(packet)
    assert not {
        "answer",
        "binding",
        "fixed_point",
        "oracle",
        "schedule",
        "target",
        "trajectory",
    } & keys
