"""Independent board for learned nonmonotone fixed-point execution.

Rows contain only raw source-deleted relations. They contain no operation
schedule, closure, halt time, trajectory, late query during execution, or
answer-equivalent compiler field. Targets are used by the trainer/evaluator,
never by the register machine.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Sequence


MAX_OBJECTS = 8
REGISTER_COUNT = 6


class FixedPointBoardError(ValueError):
    """Raised when a fixed-point episode violates its contract."""


Relation = tuple[tuple[int, ...], ...]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def identity_relation(cardinality: int) -> Relation:
    return tuple(
        tuple(int(row == column) for column in range(cardinality))
        for row in range(cardinality)
    )


def compose(left: Relation, right: Relation) -> Relation:
    cardinality = len(left)
    if (
        cardinality < 2
        or len(right) != cardinality
        or any(len(row) != cardinality for row in (*left, *right))
    ):
        raise FixedPointBoardError("fixed-point relation geometry differs")
    return tuple(
        tuple(
            int(
                any(
                    left[output][middle] and right[middle][source]
                    for middle in range(cardinality)
                )
            )
            for source in range(cardinality)
        )
        for output in range(cardinality)
    )


def relation_union(left: Relation, right: Relation) -> Relation:
    return tuple(
        tuple(int(first or second) for first, second in zip(a, b, strict=True))
        for a, b in zip(left, right, strict=True)
    )


def relation_difference(left: Relation, right: Relation) -> Relation:
    return tuple(
        tuple(
            int(first and not second)
            for first, second in zip(a, b, strict=True)
        )
        for a, b in zip(left, right, strict=True)
    )


def closure(relation: Relation) -> tuple[Relation, int]:
    current = identity_relation(len(relation))
    depth = 0
    while True:
        proposal = relation_union(current, compose(relation, current))
        if proposal == current:
            return current, depth
        current = proposal
        depth += 1
        if depth > len(relation):
            raise FixedPointBoardError("fixed-point closure did not converge")


def _empty(cardinality: int) -> list[list[int]]:
    return [[0] * cardinality for _ in range(cardinality)]


def _relation(value: Sequence[Sequence[int]]) -> Relation:
    return tuple(tuple(int(item) for item in row) for row in value)


def _pad(relation: Relation) -> list[list[int]]:
    output = [[0] * MAX_OBJECTS for _ in range(MAX_OBJECTS)]
    for row, values in enumerate(relation):
        output[row][: len(values)] = values
    return output


def _permute(relation: Relation, permutation: Sequence[int]) -> Relation:
    return tuple(
        tuple(relation[permutation[row]][permutation[column]] for column in range(len(relation)))
        for row in range(len(relation))
    )


def _chain_relation(
    cardinality: int,
    depth: int,
    rng: random.Random,
) -> tuple[Relation, tuple[tuple[int, int], ...]]:
    if not 1 <= depth < cardinality:
        raise FixedPointBoardError("requested closure depth differs")
    nodes = list(range(cardinality))
    rng.shuffle(nodes)
    backbone = tuple(
        (nodes[index], nodes[index + 1])
        for index in range(depth)
    )
    relation = _empty(cardinality)
    for source, destination in backbone:
        relation[destination][source] = 1

    order = {node: index for index, node in enumerate(nodes)}
    candidates = [
        (source, destination)
        for source in range(cardinality)
        for destination in range(cardinality)
        if order[source] < order[destination]
        and (source, destination) not in backbone
    ]
    rng.shuffle(candidates)
    for source, destination in candidates:
        if rng.random() >= 0.25:
            continue
        relation[destination][source] = 1
        if closure(_relation(relation))[1] != depth:
            relation[destination][source] = 0
    output = _relation(relation)
    if closure(output)[1] != depth:
        raise FixedPointBoardError("constructed graph depth differs")
    return output, backbone


def _subgraph(
    parent: Relation,
    backbone: Sequence[tuple[int, int]],
    depth: int,
    rng: random.Random,
) -> Relation:
    cardinality = len(parent)
    if not 0 <= depth <= len(backbone):
        raise FixedPointBoardError("subgraph depth differs")
    relation = _empty(cardinality)
    for source, destination in backbone[:depth]:
        relation[destination][source] = 1
    candidates = [
        (source, destination)
        for destination in range(cardinality)
        for source in range(cardinality)
        if parent[destination][source]
        and not relation[destination][source]
    ]
    rng.shuffle(candidates)
    for source, destination in candidates:
        if rng.random() >= 0.2:
            continue
        relation[destination][source] = 1
        if closure(_relation(relation))[1] != depth:
            relation[destination][source] = 0
    output = _relation(relation)
    if closure(output)[1] != depth:
        raise FixedPointBoardError("constructed subgraph depth differs")
    return output


def split_contract(split: str) -> dict[str, tuple[int, ...]]:
    if split == "train":
        return {
            "cardinalities": (3, 4, 5),
            "a_depths": (2, 3, 4),
        }
    if split == "development":
        return {
            "cardinalities": (6, 7),
            "a_depths": (5, 6),
        }
    if split == "confirmation":
        return {
            "cardinalities": (8,),
            "a_depths": (7,),
        }
    raise FixedPointBoardError("unknown fixed-point split")


def generate_row(*, split: str, seed: int) -> dict[str, object]:
    contract = split_contract(split)
    rng = random.Random(seed)
    eligible = [
        (cardinality, depth)
        for cardinality in contract["cardinalities"]
        for depth in contract["a_depths"]
        if depth < cardinality
    ]
    cardinality, a_depth = eligible[seed % len(eligible)]
    a_relation, backbone = _chain_relation(
        cardinality,
        a_depth,
        rng,
    )
    b_depth = rng.randint(0, a_depth - 1)
    b_relation = _subgraph(a_relation, backbone, b_depth, rng)
    a_closure, observed_a_depth = closure(a_relation)
    b_closure, observed_b_depth = closure(b_relation)
    target = relation_difference(a_closure, b_closure)
    if target in (
        a_relation,
        b_relation,
        identity_relation(cardinality),
        _relation(_empty(cardinality)),
    ):
        raise FixedPointBoardError("fixed-point target has a direct packet shortcut")
    candidates = [
        row
        for row, values in enumerate(target)
        if any(values) and not all(values)
    ]
    if not candidates:
        candidates = [row for row, values in enumerate(target) if any(values)]
    if not candidates:
        raise FixedPointBoardError("fixed-point target is trivial")
    query_position = rng.choice(candidates)

    permutation = list(range(cardinality))
    rng.shuffle(permutation)
    a_relation = _permute(a_relation, permutation)
    b_relation = _permute(b_relation, permutation)
    a_closure = _permute(a_closure, permutation)
    b_closure = _permute(b_closure, permutation)
    target = _permute(target, permutation)
    query_position = permutation.index(query_position)
    identity = identity_relation(cardinality)
    empty = _relation(_empty(cardinality))
    input_registers = (
        a_relation,
        b_relation,
        identity,
        empty,
        empty,
        empty,
    )
    target_registers = (
        a_relation,
        b_relation,
        identity,
        a_closure,
        b_closure,
        target,
    )
    semantic = {
        "cardinality": cardinality,
        "a": _pad(a_relation),
        "b": _pad(b_relation),
        "query_position": query_position,
    }
    payload = {
        "schema": "contrastive_fixed_point_v1",
        "split": split,
        "seed": seed,
        "cardinality": cardinality,
        "a_depth": observed_a_depth,
        "b_depth": observed_b_depth,
        "input_registers": [_pad(value) for value in input_registers],
        "target_registers": [_pad(value) for value in target_registers],
        "query": {
            "register": 5,
            "position": query_position,
        },
        "answer_bits": list(target[query_position])
        + [0] * (MAX_OBJECTS - cardinality),
        "semantic_sha256": sha256_json(semantic),
    }
    payload["row_sha256"] = sha256_json(payload)
    validate_row(payload)
    return payload


def generate_rows(*, split: str, count: int, seed: int) -> list[dict[str, object]]:
    if count < 1:
        raise FixedPointBoardError("fixed-point row count differs")
    rows = []
    seen: set[str] = set()
    cursor = 0
    while len(rows) < count:
        row_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{split}:{cursor}".encode()).digest()[:8],
            "big",
        )
        cursor += 1
        try:
            row = generate_row(split=split, seed=row_seed)
        except FixedPointBoardError:
            continue
        semantic = str(row["semantic_sha256"])
        if semantic in seen:
            continue
        seen.add(semantic)
        rows.append(row)
    return rows


def validate_row(row: dict[str, object]) -> None:
    required = {
        "schema",
        "split",
        "seed",
        "cardinality",
        "a_depth",
        "b_depth",
        "input_registers",
        "target_registers",
        "query",
        "answer_bits",
        "semantic_sha256",
        "row_sha256",
    }
    if set(row) != required or row["schema"] != "contrastive_fixed_point_v1":
        raise FixedPointBoardError("fixed-point row schema differs")
    unhashed = dict(row)
    expected_hash = unhashed.pop("row_sha256")
    if expected_hash != sha256_json(unhashed):
        raise FixedPointBoardError("fixed-point row hash differs")
    cardinality = int(row["cardinality"])
    contract = split_contract(str(row["split"]))
    if cardinality not in contract["cardinalities"]:
        raise FixedPointBoardError("fixed-point cardinality differs")
    inputs = row["input_registers"]
    targets = row["target_registers"]
    if (
        not isinstance(inputs, list)
        or not isinstance(targets, list)
        or len(inputs) != REGISTER_COUNT
        or len(targets) != REGISTER_COUNT
    ):
        raise FixedPointBoardError("fixed-point register count differs")
    cropped_inputs = tuple(
        tuple(
            tuple(int(value) for value in values[:cardinality])
            for values in relation[:cardinality]
        )
        for relation in inputs
    )
    a_relation, b_relation = cropped_inputs[:2]
    a_closure, a_depth = closure(a_relation)
    b_closure, b_depth = closure(b_relation)
    target = relation_difference(a_closure, b_closure)
    expected_targets = (
        a_relation,
        b_relation,
        identity_relation(cardinality),
        a_closure,
        b_closure,
        target,
    )
    if (
        int(row["a_depth"]) != a_depth
        or int(row["b_depth"]) != b_depth
        or a_depth not in contract["a_depths"]
        or targets != [_pad(value) for value in expected_targets]
    ):
        raise FixedPointBoardError("fixed-point independent oracle differs")
    query = row["query"]
    if (
        not isinstance(query, dict)
        or query != {
            "register": 5,
            "position": int(query["position"]),
        }
        or not 0 <= int(query["position"]) < cardinality
    ):
        raise FixedPointBoardError("fixed-point late query differs")
    expected_answer = list(target[int(query["position"])]) + [0] * (
        MAX_OBJECTS - cardinality
    )
    if row["answer_bits"] != expected_answer:
        raise FixedPointBoardError("fixed-point answer differs")
