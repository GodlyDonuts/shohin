"""Erase global operation names from a Bekić program using witness cards.

The returned machine packet contains fresh opaque operation slots and
relation-valued cards. Primitive names and the generation-time slot mapping
remain outside the packet. Validation independently identifies every card by
applying the complete compatible primitive bank to its witnesses.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Mapping, Sequence

from bekic_relational_fixed_point_board import (
    Relation,
    identity_relation,
    reindex_relation,
    relation_compose,
    relation_converse,
    relation_intersection,
    relation_union,
)
from contrastive_bekic_program_orbits import (
    BINARY_KINDS,
    ContrastiveOrbitError,
    evaluate_simultaneous,
)


CARD_WITNESSES = 8
PRIMITIVE_ORDER = (
    "UNION",
    "INTERSECTION",
    "COMPOSE",
    "CONVERSE",
    "IDENTITY",
)
PRIMITIVE_ARITY = {
    "UNION": 2,
    "INTERSECTION": 2,
    "COMPOSE": 2,
    "CONVERSE": 1,
    "IDENTITY": 0,
}


class ContextualizationError(ValueError):
    """Raised when operation-name deletion or card identification fails."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _opaque(prefix: str, rng: random.Random) -> str:
    return f"{prefix}_{rng.getrandbits(96):024x}"


def _serialize_relation(relation: Relation) -> list[list[int]]:
    return [list(row) for row in relation]


def _deserialize_relation(value: object, cardinality: int) -> Relation:
    if (
        not isinstance(value, list)
        or len(value) != cardinality
        or any(
            not isinstance(row, list)
            or len(row) != cardinality
            or any(bit not in (0, 1) for bit in row)
            for row in value
        )
    ):
        raise ContextualizationError("contextual relation geometry differs")
    return tuple(tuple(int(bit) for bit in row) for row in value)


def _apply_primitive(
    primitive: str,
    left: Relation,
    right: Relation,
    cardinality: int,
) -> Relation:
    if primitive == "UNION":
        return relation_union(left, right)
    if primitive == "INTERSECTION":
        return relation_intersection(left, right)
    if primitive == "COMPOSE":
        return relation_compose(left, right)
    if primitive == "CONVERSE":
        return relation_converse(left)
    if primitive == "IDENTITY":
        return identity_relation(cardinality)
    raise ContextualizationError("unknown contextual primitive")


def _random_relation(
    cardinality: int,
    rng: random.Random,
    density: float,
) -> Relation:
    if not 0.0 < density < 1.0:
        raise ContextualizationError("card witness density differs")
    return tuple(
        tuple(int(rng.random() < density) for _ in range(cardinality))
        for _ in range(cardinality)
    )


def _card_signature(
    primitive: str,
    witnesses: Sequence[Mapping[str, object]],
    cardinality: int,
) -> tuple[Relation, ...]:
    output: list[Relation] = []
    for witness in witnesses:
        left = _deserialize_relation(witness["left"], cardinality)
        right = _deserialize_relation(witness["right"], cardinality)
        output.append(_apply_primitive(primitive, left, right, cardinality))
    return tuple(output)


def _generate_cards(
    used_primitives: Sequence[str],
    *,
    cardinality: int,
    slot_ids: Mapping[str, str],
    rng: random.Random,
) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    for primitive in used_primitives:
        arity = PRIMITIVE_ARITY[primitive]
        empty = tuple(
            tuple(0 for _ in range(cardinality))
            for _ in range(cardinality)
        )
        for _ in range(128):
            witnesses = []
            densities = [0.08, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.92]
            left_densities = densities[:]
            right_densities = densities[:]
            rng.shuffle(left_densities)
            rng.shuffle(right_densities)
            for witness_index in range(CARD_WITNESSES):
                left = _random_relation(
                    cardinality,
                    rng,
                    left_densities[witness_index],
                )
                right = _random_relation(
                    cardinality,
                    rng,
                    right_densities[witness_index],
                )
                if arity < 1:
                    left = empty
                if arity < 2:
                    right = empty
                output = _apply_primitive(
                    primitive,
                    left,
                    right,
                    cardinality,
                )
                witnesses.append(
                    {
                        "left": _serialize_relation(left),
                        "right": _serialize_relation(right),
                        "output": _serialize_relation(output),
                    }
                )
            compatible = [
                candidate
                for candidate in PRIMITIVE_ORDER
                if PRIMITIVE_ARITY[candidate] == arity
                and all(
                    observed == expected
                    for observed, expected in zip(
                        _card_signature(
                            candidate,
                            witnesses,
                            cardinality,
                        ),
                        (
                            _deserialize_relation(
                                witness["output"],
                                cardinality,
                            )
                            for witness in witnesses
                        ),
                        strict=True,
                    )
                )
            ]
            if compatible == [primitive]:
                break
        else:
            raise ContextualizationError(
                "could not generate an identifiable operation card"
            )
        rng.shuffle(witnesses)
        cards.append(
            {
                "slot": slot_ids[primitive],
                "arity": arity,
                "witnesses": witnesses,
            }
        )
    rng.shuffle(cards)
    return cards


def _flatten_expression(
    expression: Mapping[str, object],
    *,
    slot_ids: Mapping[str, str],
    nodes: list[dict[str, object]],
) -> str:
    kind = expression.get("kind")
    node_id = expression.get("id")
    if not isinstance(node_id, str):
        raise ContextualizationError("contextual node id differs")
    if kind == "VARIABLE":
        nodes.append(
            {
                "id": node_id,
                "kind": "VARIABLE",
                "variable": expression["variable"],
            }
        )
        return node_id
    if kind == "CONSTANT":
        nodes.append(
            {
                "id": node_id,
                "kind": "CONSTANT",
                "constant": expression["constant"],
            }
        )
        return node_id
    if kind == "IDENTITY":
        nodes.append(
            {
                "id": node_id,
                "kind": "OPERATION",
                "slot": slot_ids["IDENTITY"],
                "inputs": [],
            }
        )
        return node_id
    if kind == "CONVERSE":
        child = _flatten_expression(
            expression["child"],
            slot_ids=slot_ids,
            nodes=nodes,
        )
        nodes.append(
            {
                "id": node_id,
                "kind": "OPERATION",
                "slot": slot_ids["CONVERSE"],
                "inputs": [child],
            }
        )
        return node_id
    if kind in BINARY_KINDS:
        inputs = [
            _flatten_expression(
                child,
                slot_ids=slot_ids,
                nodes=nodes,
            )
            for child in expression["children"]
        ]
        nodes.append(
            {
                "id": node_id,
                "kind": "OPERATION",
                "slot": slot_ids[str(kind)],
                "inputs": inputs,
            }
        )
        return node_id
    raise ContextualizationError("contextual expression kind differs")


def contextualize_simultaneous_packet(
    packet: Mapping[str, object],
    *,
    seed: int,
) -> dict[str, object]:
    """Return one deterministic operation-name-deleted machine packet."""

    try:
        evaluate_simultaneous(packet)
    except ContrastiveOrbitError as error:
        raise ContextualizationError(
            "source simultaneous packet is invalid"
        ) from error
    cardinality = packet.get("cardinality")
    constants = packet.get("constants")
    program = packet.get("program")
    if (
        not isinstance(cardinality, int)
        or not isinstance(constants, list)
        or not isinstance(program, dict)
        or program.get("form") != "simultaneous_lfp"
        or not isinstance(program.get("variables"), list)
        or not isinstance(program.get("equations"), list)
    ):
        raise ContextualizationError("source simultaneous packet differs")

    material = hashlib.sha256(
        f"{seed}:{sha256_json(packet)}:contextual-v1".encode()
    ).digest()
    rng = random.Random(int.from_bytes(material[:8], "big"))
    used_primitives = sorted(
        {
            str(node.get("kind"))
            for equation in program["equations"]
            for node in _walk_expression(equation["expression"])
            if node.get("kind") in PRIMITIVE_ARITY
        },
        key=PRIMITIVE_ORDER.index,
    )
    slot_ids = {
        primitive: _opaque("operation", rng)
        for primitive in used_primitives
    }
    nodes: list[dict[str, object]] = []
    equations = []
    for equation in program["equations"]:
        root = _flatten_expression(
            equation["expression"],
            slot_ids=slot_ids,
            nodes=nodes,
        )
        equations.append(
            {
                "variable": equation["variable"],
                "root": root,
            }
        )
    rng.shuffle(nodes)
    output = {
        "schema": "source_deleted_contextual_bekic_packet_v1",
        "cardinality": cardinality,
        "constants": copy.deepcopy(constants),
        "variables": copy.deepcopy(program["variables"]),
        "nodes": nodes,
        "equations": equations,
        "operation_cards": _generate_cards(
            used_primitives,
            cardinality=cardinality,
            slot_ids=slot_ids,
            rng=rng,
        ),
    }
    validate_contextual_packet(output)
    return output


def _walk_expression(
    expression: Mapping[str, object],
) -> Sequence[Mapping[str, object]]:
    output = [expression]
    kind = expression.get("kind")
    if kind == "CONVERSE":
        output.extend(_walk_expression(expression["child"]))
    elif kind in BINARY_KINDS:
        for child in expression["children"]:
            output.extend(_walk_expression(child))
    return output


def identify_contextual_slots(
    packet: Mapping[str, object],
) -> dict[str, int]:
    """Independently identify every opaque card from relation witnesses."""

    cardinality = packet.get("cardinality")
    cards = packet.get("operation_cards")
    if not isinstance(cardinality, int) or not isinstance(cards, list):
        raise ContextualizationError("contextual card packet differs")
    output: dict[str, int] = {}
    for card in cards:
        if (
            not isinstance(card, dict)
            or set(card) != {"slot", "arity", "witnesses"}
            or not isinstance(card["slot"], str)
            or card["slot"] in output
            or card["arity"] not in {0, 1, 2}
            or not isinstance(card["witnesses"], list)
            or len(card["witnesses"]) != CARD_WITNESSES
        ):
            raise ContextualizationError("opaque operation card differs")
        compatible = []
        for index, primitive in enumerate(PRIMITIVE_ORDER):
            if PRIMITIVE_ARITY[primitive] != card["arity"]:
                continue
            if all(
                set(witness) == {"left", "right", "output"}
                and _apply_primitive(
                    primitive,
                    _deserialize_relation(witness["left"], cardinality),
                    _deserialize_relation(witness["right"], cardinality),
                    cardinality,
                )
                == _deserialize_relation(witness["output"], cardinality)
                for witness in card["witnesses"]
            ):
                compatible.append(index)
        if len(compatible) != 1:
            raise ContextualizationError(
                "opaque operation card is not uniquely identifiable"
            )
        output[card["slot"]] = compatible[0]
    return output


def validate_contextual_packet_structure(
    packet: Mapping[str, object],
) -> None:
    """Validate only private packet structure, never card semantics."""

    if (
        packet.get("schema")
        != "source_deleted_contextual_bekic_packet_v1"
        or not isinstance(packet.get("cardinality"), int)
        or not 2 <= int(packet["cardinality"]) <= 8
        or not isinstance(packet.get("constants"), list)
        or not isinstance(packet.get("variables"), list)
        or len(packet["variables"]) != 2
        or len(set(packet["variables"])) != 2
        or not isinstance(packet.get("nodes"), list)
        or not isinstance(packet.get("equations"), list)
        or len(packet["equations"]) != 2
    ):
        raise ContextualizationError("contextual machine packet differs")
    cardinality = int(packet["cardinality"])
    constant_ids: set[str] = set()
    for item in packet["constants"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "relation"}
            or not isinstance(item["id"], str)
            or item["id"] in constant_ids
        ):
            raise ContextualizationError("contextual constant differs")
        constant_ids.add(item["id"])
        _deserialize_relation(item["relation"], cardinality)

    cards = packet.get("operation_cards")
    if not isinstance(cards, list):
        raise ContextualizationError("contextual operation cards differ")
    slot_arities: dict[str, int] = {}
    for card in cards:
        if (
            not isinstance(card, dict)
            or set(card) != {"slot", "arity", "witnesses"}
            or not isinstance(card["slot"], str)
            or card["slot"] in slot_arities
            or card["arity"] not in {0, 1, 2}
            or not isinstance(card["witnesses"], list)
            or len(card["witnesses"]) != CARD_WITNESSES
        ):
            raise ContextualizationError("opaque operation card differs")
        slot_arities[card["slot"]] = int(card["arity"])
        for witness in card["witnesses"]:
            if (
                not isinstance(witness, dict)
                or set(witness) != {"left", "right", "output"}
            ):
                raise ContextualizationError("operation-card witness differs")
            decoded = {
                field: _deserialize_relation(
                    witness[field],
                    cardinality,
                )
                for field in ("left", "right", "output")
            }
            if (
                card["arity"] < 1
                and any(any(row) for row in decoded["left"])
            ) or (
                card["arity"] < 2
                and any(any(row) for row in decoded["right"])
            ):
                raise ContextualizationError(
                    "unused card argument contains covert state"
                )

    node_ids: set[str] = set()
    nodes: dict[str, Mapping[str, object]] = {}
    variables = set(packet["variables"])
    for node in packet["nodes"]:
        if (
            not isinstance(node, dict)
            or not isinstance(node.get("id"), str)
            or node["id"] in node_ids
        ):
            raise ContextualizationError("contextual physical node differs")
        node_ids.add(node["id"])
        nodes[node["id"]] = node
    for node in nodes.values():
        kind = node.get("kind")
        if kind == "VARIABLE":
            if (
                set(node) != {"id", "kind", "variable"}
                or node.get("variable") not in variables
            ):
                raise ContextualizationError("contextual variable node differs")
        elif kind == "CONSTANT":
            if (
                set(node) != {"id", "kind", "constant"}
                or node.get("constant") not in constant_ids
            ):
                raise ContextualizationError("contextual constant node differs")
        elif kind == "OPERATION":
            if (
                set(node) != {"id", "kind", "slot", "inputs"}
                or node.get("slot") not in slot_arities
                or not isinstance(node.get("inputs"), list)
                or len(node["inputs"])
                != slot_arities[str(node["slot"])]
                or any(reference not in nodes for reference in node["inputs"])
            ):
                raise ContextualizationError("contextual operation node differs")
        else:
            raise ContextualizationError("contextual node kind differs")

    roots: dict[str, str] = {}
    for equation in packet["equations"]:
        if (
            not isinstance(equation, dict)
            or set(equation) != {"variable", "root"}
            or equation.get("variable") not in variables
            or equation.get("root") not in nodes
            or equation["variable"] in roots
        ):
            raise ContextualizationError("contextual equation differs")
        roots[equation["variable"]] = equation["root"]
    if set(roots) != variables:
        raise ContextualizationError("contextual equation roots differ")

    active: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in active:
            raise ContextualizationError("contextual graph is cyclic")
        active.add(node_id)
        node = nodes[node_id]
        for reference in node.get("inputs", []):
            visit(str(reference))
        active.remove(node_id)
        visited.add(node_id)

    for root in roots.values():
        visit(root)
    if visited != node_ids:
        raise ContextualizationError(
            "contextual packet contains disconnected nodes"
        )
    used_constants = {
        str(node["constant"])
        for node in nodes.values()
        if node.get("kind") == "CONSTANT"
    }
    used_slots = {
        str(node["slot"])
        for node in nodes.values()
        if node.get("kind") == "OPERATION"
    }
    if used_constants != constant_ids or used_slots != set(slot_arities):
        raise ContextualizationError(
            "contextual packet contains unused constants or slots"
        )


def validate_contextual_packet(packet: Mapping[str, object]) -> None:
    """Offline board admission including analytic card identification."""

    validate_contextual_packet_structure(packet)
    identify_contextual_slots(packet)


def reindex_contextual_objects(
    packet: Mapping[str, object],
    permutation: Sequence[int],
) -> dict[str, object]:
    output = copy.deepcopy(dict(packet))
    cardinality = int(output["cardinality"])
    if len(permutation) != cardinality or set(permutation) != set(
        range(cardinality)
    ):
        raise ContextualizationError("contextual object permutation differs")
    for constant in output["constants"]:
        constant["relation"] = _serialize_relation(
            reindex_relation(
                _deserialize_relation(constant["relation"], cardinality),
                permutation,
            )
        )
    for card in output["operation_cards"]:
        for witness in card["witnesses"]:
            for field in ("left", "right", "output"):
                witness[field] = _serialize_relation(
                    reindex_relation(
                        _deserialize_relation(witness[field], cardinality),
                        permutation,
                    )
                )
    validate_contextual_packet(output)
    return output
