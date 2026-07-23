"""Contrastive program-orbit board for monotone relational fixed points.

The board makes an episode-local program causally necessary.  Every orbit
contains:

* ``P``: a grammar-sampled pair of mutually recursive relation equations.
* ``P'``: a matched program with the same finite statistics but rewired
  operands and constant occurrences, and a materially different fixed point.
* ``Peq``: a semantics-preserving alpha/node/object/constant-list rewrite.
* an explicit nested ``LFP``/``LET`` Bekić term evaluated by a second oracle.

Only training and development splits exist.  Confirmation generation is
intentionally unavailable in this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import TypeAlias

from bekic_relational_fixed_point_board import (
    Relation,
    empty_relation,
    identity_relation,
    reindex_relation,
    relation_compose,
    relation_converse,
    relation_intersection,
    relation_subset,
    relation_union,
)


CONSTANT_COUNT = 8
MAX_OBJECTS = 8
COUNTERFACTUAL_MIN_HAMMING = 0.10
MIN_VARIABLE_CHANGE_STEPS = 1
MIN_CONVERGENCE_UPDATES = 2
MIN_TOTAL_VARIABLE_CHANGE_STEPS = 3
ISOLATED_COUNTERFACTUAL_ARMS = (
    "constant_rewire",
    "compose_reverse",
)
GRAMMAR_KINDS = frozenset(
    {
        "VARIABLE",
        "CONSTANT",
        "IDENTITY",
        "UNION",
        "INTERSECTION",
        "COMPOSE",
        "CONVERSE",
    }
)
BINARY_KINDS = frozenset({"UNION", "INTERSECTION", "COMPOSE"})
COMMUTATIVE_KINDS = frozenset({"UNION", "INTERSECTION"})
DEPENDENCY_TOPOLOGIES = (
    "independent_dag",
    "dag_x_to_y",
    "dag_y_to_x",
    "mutual_cycle",
)
DEVELOPMENT_CELLS = (
    "in_range",
    "motif",
    "scale",
    "depth",
    "joint",
)
HELD_OUT_MOTIF = ("COMPOSE", "CONVERSE", "INTERSECTION")

Json: TypeAlias = dict[str, object]
Environment: TypeAlias = dict[str, Relation]


class ContrastiveOrbitError(ValueError):
    """Raised when an orbit violates its source-deleted board contract."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def split_contract(split: str) -> Json:
    """Return the public train/development contract.

    There is deliberately no flag or capability that opens confirmation.
    """

    if split == "train":
        return {
            "split": "train",
            "cardinalities": [3, 4, 5, 6],
            "ast_depths": [4, 5, 6, 7],
            "dependency_topologies": list(DEPENDENCY_TOPOLOGIES),
        }
    if split == "development":
        return {
            "split": "development",
            "cells": {
                "in_range": {
                    "cardinalities": [3, 4, 5, 6],
                    "ast_depths": [4, 5, 6, 7],
                },
                "motif": {
                    "cardinalities": [3, 4, 5, 6],
                    "ast_depths": [4, 5, 6, 7],
                },
                "scale": {
                    "cardinalities": [7],
                    "ast_depths": [4, 5, 6, 7],
                },
                "depth": {
                    "cardinalities": [3, 4, 5, 6],
                    "ast_depths": [8, 9],
                },
                "joint": {
                    "cardinalities": [7],
                    "ast_depths": [8, 9],
                },
            },
            "dependency_topologies": list(DEPENDENCY_TOPOLOGIES),
        }
    raise ContrastiveOrbitError(
        "confirmation is fail-closed; only train and development exist"
    )


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
        raise ContrastiveOrbitError("serialized relation geometry differs")
    return tuple(tuple(int(bit) for bit in row) for row in value)


def _constants(packet: Mapping[str, object]) -> Environment:
    cardinality = packet.get("cardinality")
    items = packet.get("constants")
    if (
        not isinstance(cardinality, int)
        or not 2 <= cardinality <= MAX_OBJECTS
        or not isinstance(items, list)
        or len(items) != CONSTANT_COUNT
    ):
        raise ContrastiveOrbitError("constant packet differs")
    output: Environment = {}
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "relation"}
            or not isinstance(item["id"], str)
            or item["id"] in output
        ):
            raise ContrastiveOrbitError("constant entry differs")
        output[item["id"]] = _deserialize_relation(
            item["relation"],
            cardinality,
        )
    return output


def _sample_constant_world(
    cardinality: int,
    rng: random.Random,
) -> tuple[list[Json], float]:
    """Sample all constants from the same role-independent Bernoulli law."""

    density = rng.uniform(0.08, 0.30)
    constants: list[Json] = []
    for _ in range(CONSTANT_COUNT):
        relation = tuple(
            tuple(int(rng.random() < density) for _column in range(cardinality))
            for _row in range(cardinality)
        )
        constants.append(
            {
                "id": _opaque("rel", rng),
                "relation": _serialize_relation(relation),
            }
        )
    rng.shuffle(constants)
    return constants, density


def _leaf(kind: str, value: str | None = None) -> Json:
    node: Json = {"kind": kind}
    if kind == "VARIABLE":
        node["variable"] = value
    elif kind == "CONSTANT":
        node["constant"] = value
    elif kind != "IDENTITY":
        raise ContrastiveOrbitError("leaf kind differs")
    return node


def _minimum_depth(leaf_count: int) -> int:
    return math.ceil(math.log2(leaf_count)) + 1


def _build_exact_depth_tree(
    leaves: Sequence[Json],
    depth: int,
    rng: random.Random,
) -> Json:
    """Build a random typed AST with an exact maximum root-to-leaf depth."""

    if not leaves or depth < _minimum_depth(len(leaves)):
        raise ContrastiveOrbitError("requested AST depth cannot contain leaves")
    if len(leaves) == 1:
        expression = copy.deepcopy(leaves[0])
        for _ in range(depth - 1):
            expression = {"kind": "CONVERSE", "child": expression}
        return expression

    if depth > _minimum_depth(len(leaves)) and rng.random() < 0.16:
        return {
            "kind": "CONVERSE",
            "child": _build_exact_depth_tree(leaves, depth - 1, rng),
        }

    shuffled = list(leaves)
    rng.shuffle(shuffled)
    feasible_splits = [
        split
        for split in range(1, len(shuffled))
        if _minimum_depth(split) <= depth - 1
        and _minimum_depth(len(shuffled) - split) <= depth - 1
    ]
    if not feasible_splits:
        raise ContrastiveOrbitError(
            "requested AST depth has no feasible binary partition"
        )
    split = rng.choice(feasible_splits)
    left_leaves = shuffled[:split]
    right_leaves = shuffled[split:]

    left_min = _minimum_depth(len(left_leaves))
    right_min = _minimum_depth(len(right_leaves))
    deep_side = rng.randrange(2)
    if deep_side == 0:
        left_depth = depth - 1
        right_depth = rng.randint(right_min, depth - 1)
    else:
        right_depth = depth - 1
        left_depth = rng.randint(left_min, depth - 1)
    return {
        "kind": rng.choice(tuple(sorted(BINARY_KINDS))),
        "children": [
            _build_exact_depth_tree(left_leaves, left_depth, rng),
            _build_exact_depth_tree(right_leaves, right_depth, rng),
        ],
    }


def _walk(expression: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    yield expression
    kind = expression.get("kind")
    if kind == "CONVERSE":
        child = expression.get("child")
        if isinstance(child, dict):
            yield from _walk(child)
    elif kind in BINARY_KINDS:
        children = expression.get("children")
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    yield from _walk(child)


def contains_held_out_motif(program: Mapping[str, object]) -> bool:
    """Detect the noncommutative higher-order motif reserved for one dev cell."""

    equations = _equations(program)
    return any(
        node.get("kind") == HELD_OUT_MOTIF[0]
        and isinstance(node.get("children"), list)
        and len(node["children"]) == 2
        and node["children"][0].get("kind") == HELD_OUT_MOTIF[1]
        and node["children"][1].get("kind") == HELD_OUT_MOTIF[2]
        for expression in equations.values()
        for node in _walk(expression)
    )


def _expression_depth(expression: Mapping[str, object]) -> int:
    kind = expression.get("kind")
    if kind in {"VARIABLE", "CONSTANT", "IDENTITY"}:
        return 1
    if kind == "CONVERSE":
        child = expression.get("child")
        if not isinstance(child, dict):
            raise ContrastiveOrbitError("converse child differs")
        return 1 + _expression_depth(child)
    if kind in BINARY_KINDS:
        children = expression.get("children")
        if not isinstance(children, list) or len(children) != 2:
            raise ContrastiveOrbitError("binary children differ")
        return 1 + max(_expression_depth(child) for child in children)
    raise ContrastiveOrbitError("expression kind differs")


def _assign_node_ids(expression: Mapping[str, object], rng: random.Random) -> Json:
    output = copy.deepcopy(dict(expression))

    def assign(node: Json) -> None:
        node["id"] = _opaque("node", rng)
        kind = node.get("kind")
        if kind == "CONVERSE":
            assign(node["child"])
        elif kind in BINARY_KINDS:
            for child in node["children"]:
                assign(child)
        elif kind == "LFP":
            assign(node["body"])
        elif kind == "LET":
            assign(node["value"])
            assign(node["body"])
        elif kind == "PAIR":
            for entry in node["entries"]:
                assign(entry["expression"])

    assign(output)
    return output


def _strip_node_ids(expression: Mapping[str, object]) -> Json:
    output = copy.deepcopy(dict(expression))

    def strip(node: Json) -> None:
        node.pop("id", None)
        kind = node.get("kind")
        if kind == "CONVERSE":
            strip(node["child"])
        elif kind in BINARY_KINDS:
            for child in node["children"]:
                strip(child)
        elif kind == "LFP":
            strip(node["body"])
        elif kind == "LET":
            strip(node["value"])
            strip(node["body"])
        elif kind == "PAIR":
            for entry in node["entries"]:
                strip(entry["expression"])

    strip(output)
    return output


def _variable_leaves(
    topology: str,
    variables: Sequence[str],
    equation_index: int,
) -> list[Json]:
    own = variables[equation_index]
    other = variables[1 - equation_index]
    leaves = [_leaf("VARIABLE", own)]
    has_cross = (
        topology == "mutual_cycle"
        or topology == "dag_y_to_x"
        and equation_index == 0
        or topology == "dag_x_to_y"
        and equation_index == 1
    )
    if has_cross:
        leaves.append(_leaf("VARIABLE", other))
    return leaves


def _ensure_compose(program: Json, rng: random.Random) -> None:
    operations = [
        node
        for equation in program["equations"]
        for node in _walk(equation["expression"])
        if node.get("kind") in BINARY_KINDS
    ]
    if not operations:
        raise ContrastiveOrbitError("sampled program has no binary operation")
    if not any(node.get("kind") == "COMPOSE" for node in operations):
        mutable = rng.choice(operations)
        mutable["kind"] = "COMPOSE"


def _ensure_held_out_motif(program: Mapping[str, object], rng: random.Random) -> Json:
    """Construct the held-out motif without increasing declared AST depth."""

    output = copy.deepcopy(dict(program))
    candidates: list[tuple[Json, Json, Json]] = []
    for equation in output["equations"]:
        for node in _walk(equation["expression"]):
            if node.get("kind") not in BINARY_KINDS:
                continue
            children = node.get("children")
            if not isinstance(children, list) or len(children) != 2:
                continue
            for shallow_slot, binary_slot in ((0, 1), (1, 0)):
                shallow = children[shallow_slot]
                binary = children[binary_slot]
                if (
                    isinstance(shallow, dict)
                    and isinstance(binary, dict)
                    and binary.get("kind") in BINARY_KINDS
                    and _expression_depth(shallow) + 1
                    <= _expression_depth(binary)
                ):
                    candidates.append((node, shallow, binary))
    if not candidates:
        raise ContrastiveOrbitError(
            "sampled motif program has no depth-preserving insertion site"
        )

    motif_root, shallow, binary = rng.choice(candidates)
    motif_root["kind"] = "COMPOSE"
    binary["kind"] = "INTERSECTION"
    motif_root["children"] = [
        {"kind": "CONVERSE", "child": shallow},
        binary,
    ]

    # P' and the isolated compose control must be able to reverse a different
    # compose while retaining this score-bearing held-out motif.
    other_binary = [
        node
        for equation in output["equations"]
        for node in _walk(equation["expression"])
        if node is not motif_root
        and node is not binary
        and node.get("kind") in BINARY_KINDS
    ]
    if not other_binary:
        raise ContrastiveOrbitError(
            "sampled motif program has no isolated compose control site"
        )
    non_motif_composes = [
        node
        for node in other_binary
        if node.get("kind") == "COMPOSE"
        and not (
            node["children"][0].get("kind") == HELD_OUT_MOTIF[1]
            and node["children"][1].get("kind") == HELD_OUT_MOTIF[2]
        )
    ]
    if not non_motif_composes:
        control_compose = rng.choice(other_binary)
        control_compose["kind"] = "COMPOSE"
        if (
            control_compose["children"][0].get("kind")
            == HELD_OUT_MOTIF[1]
            and control_compose["children"][1].get("kind")
            == HELD_OUT_MOTIF[2]
        ):
            control_compose["children"].reverse()
    return _reidentify_program(output, rng)


def _sample_program(
    *,
    constant_ids: Sequence[str],
    variables: Sequence[str],
    depth: int,
    topology: str,
    rng: random.Random,
) -> Json:
    constant_assignment = list(constant_ids)
    rng.shuffle(constant_assignment)
    equations: list[Json] = []
    for equation_index in range(2):
        leaves = [
            *(
                _leaf("CONSTANT", constant)
                for constant in constant_assignment[
                    equation_index * 4 : (equation_index + 1) * 4
                ]
            ),
            *_variable_leaves(topology, variables, equation_index),
            _leaf("IDENTITY"),
        ]
        rng.shuffle(leaves)
        expression = _build_exact_depth_tree(leaves, depth, rng)
        equations.append(
            {
                "variable": variables[equation_index],
                "expression": expression,
            }
        )
    program: Json = {
        "schema": "contrastive_monotone_program_v1",
        "form": "simultaneous_lfp",
        "variables": list(variables),
        "equations": equations,
    }
    _ensure_compose(program, rng)
    return _reidentify_program(program, rng)


def _reidentify_program(program: Mapping[str, object], rng: random.Random) -> Json:
    output = copy.deepcopy(dict(program))
    if output.get("form") == "simultaneous_lfp":
        for equation in output["equations"]:
            equation["expression"] = _assign_node_ids(
                _strip_node_ids(equation["expression"]),
                rng,
            )
    elif output.get("form") == "nested_mu":
        output["expression"] = _assign_node_ids(
            _strip_node_ids(output["expression"]),
            rng,
        )
    else:
        raise ContrastiveOrbitError("program form differs")
    return output


def _equations(program: Mapping[str, object]) -> dict[str, Json]:
    if (
        program.get("schema") != "contrastive_monotone_program_v1"
        or program.get("form") != "simultaneous_lfp"
        or not isinstance(program.get("variables"), list)
        or len(program["variables"]) != 2
        or not isinstance(program.get("equations"), list)
        or len(program["equations"]) != 2
    ):
        raise ContrastiveOrbitError("simultaneous program schema differs")
    equations: dict[str, Json] = {}
    for equation in program["equations"]:
        if (
            not isinstance(equation, dict)
            or set(equation) != {"variable", "expression"}
            or not isinstance(equation["variable"], str)
            or not isinstance(equation["expression"], dict)
            or equation["variable"] in equations
        ):
            raise ContrastiveOrbitError("program equation differs")
        equations[equation["variable"]] = equation["expression"]
    if set(equations) != set(program["variables"]):
        raise ContrastiveOrbitError("program variables and equations differ")
    return equations


def _validate_expression(
    expression: Mapping[str, object],
    *,
    variables: set[str],
    constants: set[str],
    seen_ids: set[str],
) -> None:
    node_id = expression.get("id")
    kind = expression.get("kind")
    if not isinstance(node_id, str) or node_id in seen_ids:
        raise ContrastiveOrbitError("program node ids differ")
    seen_ids.add(node_id)
    if kind == "VARIABLE":
        if (
            set(expression) != {"id", "kind", "variable"}
            or expression.get("variable") not in variables
        ):
            raise ContrastiveOrbitError("variable expression differs")
        return
    if kind == "CONSTANT":
        if (
            set(expression) != {"id", "kind", "constant"}
            or expression.get("constant") not in constants
        ):
            raise ContrastiveOrbitError("constant expression differs")
        return
    if kind == "IDENTITY":
        if set(expression) != {"id", "kind"}:
            raise ContrastiveOrbitError("identity expression differs")
        return
    if kind == "CONVERSE":
        if set(expression) != {"id", "kind", "child"} or not isinstance(
            expression.get("child"), dict
        ):
            raise ContrastiveOrbitError("converse expression differs")
        _validate_expression(
            expression["child"],
            variables=variables,
            constants=constants,
            seen_ids=seen_ids,
        )
        return
    if kind in BINARY_KINDS:
        children = expression.get("children")
        if (
            set(expression) != {"id", "kind", "children"}
            or not isinstance(children, list)
            or len(children) != 2
            or any(not isinstance(child, dict) for child in children)
        ):
            raise ContrastiveOrbitError("binary expression differs")
        for child in children:
            _validate_expression(
                child,
                variables=variables,
                constants=constants,
                seen_ids=seen_ids,
            )
        return
    raise ContrastiveOrbitError("expression grammar differs")


def validate_simultaneous_program(
    program: Mapping[str, object],
    constant_ids: set[str],
) -> None:
    equations = _equations(program)
    variables = set(program["variables"])
    seen_ids: set[str] = set()
    for expression in equations.values():
        _validate_expression(
            expression,
            variables=variables,
            constants=constant_ids,
            seen_ids=seen_ids,
        )


def _apply_relation(
    kind: str,
    arguments: Sequence[Relation],
    cardinality: int,
) -> Relation:
    if kind == "IDENTITY":
        return identity_relation(cardinality)
    if kind == "UNION":
        return relation_union(arguments[0], arguments[1])
    if kind == "INTERSECTION":
        return relation_intersection(arguments[0], arguments[1])
    if kind == "COMPOSE":
        return relation_compose(arguments[0], arguments[1])
    if kind == "CONVERSE":
        return relation_converse(arguments[0])
    raise ContrastiveOrbitError("relation operation differs")


def _evaluate_sim_expression(
    expression: Mapping[str, object],
    constants: Mapping[str, Relation],
    state: Mapping[str, Relation],
    cardinality: int,
) -> Relation:
    kind = expression.get("kind")
    if kind == "VARIABLE":
        variable = expression.get("variable")
        if variable not in state:
            raise ContrastiveOrbitError("simultaneous variable is unbound")
        return state[str(variable)]
    if kind == "CONSTANT":
        constant = expression.get("constant")
        if constant not in constants:
            raise ContrastiveOrbitError("simultaneous constant is unbound")
        return constants[str(constant)]
    if kind == "IDENTITY":
        return identity_relation(cardinality)
    if kind == "CONVERSE":
        return relation_converse(
            _evaluate_sim_expression(
                expression["child"],
                constants,
                state,
                cardinality,
            )
        )
    if kind in BINARY_KINDS:
        arguments = [
            _evaluate_sim_expression(child, constants, state, cardinality)
            for child in expression["children"]
        ]
        return _apply_relation(str(kind), arguments, cardinality)
    raise ContrastiveOrbitError("simultaneous expression kind differs")


def evaluate_simultaneous(packet: Mapping[str, object]) -> Environment:
    """Independent simultaneous Kleene fixed-point oracle."""

    constants = _constants(packet)
    cardinality = int(packet["cardinality"])
    program = packet.get("program")
    if not isinstance(program, dict):
        raise ContrastiveOrbitError("simultaneous packet program differs")
    validate_simultaneous_program(program, set(constants))
    equations = _equations(program)
    variables = tuple(str(variable) for variable in program["variables"])
    current = {variable: empty_relation(cardinality) for variable in variables}
    for _ in range(2 * cardinality * cardinality + 2):
        proposal = {
            variable: _evaluate_sim_expression(
                equations[variable],
                constants,
                current,
                cardinality,
            )
            for variable in variables
        }
        if proposal == current:
            return proposal
        if any(
            not relation_subset(current[variable], proposal[variable])
            for variable in variables
        ):
            raise ContrastiveOrbitError("simultaneous iteration violated monotonicity")
        current = proposal
    raise ContrastiveOrbitError("simultaneous fixed point did not converge")


def _substitute_variable(
    expression: Mapping[str, object],
    variable: str,
    replacement: Mapping[str, object],
) -> Json:
    kind = expression.get("kind")
    if kind == "VARIABLE" and expression.get("variable") == variable:
        return copy.deepcopy(dict(replacement))
    output = _strip_node_ids(expression)
    if kind == "CONVERSE":
        output["child"] = _substitute_variable(
            expression["child"],
            variable,
            replacement,
        )
    elif kind in BINARY_KINDS:
        output["children"] = [
            _substitute_variable(child, variable, replacement)
            for child in expression["children"]
        ]
    return output


def build_nested_mu_program(
    simultaneous: Mapping[str, object],
    *,
    rng: random.Random,
) -> Json:
    """Compile equations to a genuine explicit nested Bekić ``LFP`` term."""

    equations = _equations(simultaneous)
    outer, inner = (str(item) for item in simultaneous["variables"])
    inner_lfp = {
        "kind": "LFP",
        "variable": inner,
        "body": _strip_node_ids(equations[inner]),
    }
    outer_body = _substitute_variable(
        equations[outer],
        inner,
        inner_lfp,
    )
    expression: Json = {
        "kind": "LET",
        "variable": outer,
        "value": {
            "kind": "LFP",
            "variable": outer,
            "body": outer_body,
        },
        "body": {
            "kind": "PAIR",
            "entries": [
                {
                    "variable": outer,
                    "expression": {
                        "kind": "VARIABLE",
                        "variable": outer,
                    },
                },
                {
                    "variable": inner,
                    "expression": {
                        "kind": "LFP",
                        "variable": inner,
                        "body": _strip_node_ids(equations[inner]),
                    },
                },
            ],
        },
    }
    program: Json = {
        "schema": "contrastive_nested_mu_program_v1",
        "form": "nested_mu",
        "variables": [outer, inner],
        "expression": expression,
    }
    return _reidentify_program(program, rng)


def _evaluate_mu_relation(
    expression: Mapping[str, object],
    *,
    constants: Mapping[str, Relation],
    environment: Mapping[str, Relation],
    cardinality: int,
) -> Relation:
    """Nested-term interpreter, separate from the simultaneous AST evaluator."""

    kind = expression.get("kind")
    if kind == "VARIABLE":
        name = expression.get("variable")
        if name not in environment:
            raise ContrastiveOrbitError("nested variable is unbound")
        return environment[str(name)]
    if kind == "CONSTANT":
        name = expression.get("constant")
        if name not in constants:
            raise ContrastiveOrbitError("nested constant is unbound")
        return constants[str(name)]
    if kind == "IDENTITY":
        return identity_relation(cardinality)
    if kind == "CONVERSE":
        return relation_converse(
            _evaluate_mu_relation(
                expression["child"],
                constants=constants,
                environment=environment,
                cardinality=cardinality,
            )
        )
    if kind in BINARY_KINDS:
        arguments = [
            _evaluate_mu_relation(
                child,
                constants=constants,
                environment=environment,
                cardinality=cardinality,
            )
            for child in expression["children"]
        ]
        return _apply_relation(str(kind), arguments, cardinality)
    if kind == "LFP":
        variable = expression.get("variable")
        body = expression.get("body")
        if not isinstance(variable, str) or not isinstance(body, dict):
            raise ContrastiveOrbitError("nested LFP syntax differs")
        current = empty_relation(cardinality)
        for _ in range(cardinality * cardinality + 2):
            scoped = dict(environment)
            scoped[variable] = current
            proposal = _evaluate_mu_relation(
                body,
                constants=constants,
                environment=scoped,
                cardinality=cardinality,
            )
            if proposal == current:
                return proposal
            if not relation_subset(current, proposal):
                raise ContrastiveOrbitError("nested LFP violated monotonicity")
            current = proposal
        raise ContrastiveOrbitError("nested LFP did not converge")
    raise ContrastiveOrbitError("nested relation expression differs")


def evaluate_nested_mu(packet: Mapping[str, object]) -> Environment:
    """Evaluate the explicit nested Bekić term without the simultaneous oracle."""

    constants = _constants(packet)
    cardinality = int(packet["cardinality"])
    program = packet.get("program")
    if (
        not isinstance(program, dict)
        or program.get("schema") != "contrastive_nested_mu_program_v1"
        or program.get("form") != "nested_mu"
        or not isinstance(program.get("variables"), list)
        or len(program["variables"]) != 2
    ):
        raise ContrastiveOrbitError("nested program schema differs")
    expression = program.get("expression")
    if (
        not isinstance(expression, dict)
        or expression.get("kind") != "LET"
        or not isinstance(expression.get("variable"), str)
        or not isinstance(expression.get("value"), dict)
        or expression["value"].get("kind") != "LFP"
        or not isinstance(expression.get("body"), dict)
        or expression["body"].get("kind") != "PAIR"
    ):
        raise ContrastiveOrbitError("nested Bekić syntax is not explicit")

    outer = str(expression["variable"])
    outer_value = _evaluate_mu_relation(
        expression["value"],
        constants=constants,
        environment={},
        cardinality=cardinality,
    )
    environment = {outer: outer_value}
    entries = expression["body"].get("entries")
    if not isinstance(entries, list) or len(entries) != 2:
        raise ContrastiveOrbitError("nested Bekić result pair differs")
    output: Environment = {}
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"variable", "expression"}
            or not isinstance(entry["variable"], str)
            or not isinstance(entry["expression"], dict)
        ):
            raise ContrastiveOrbitError("nested Bekić result entry differs")
        output[entry["variable"]] = _evaluate_mu_relation(
            entry["expression"],
            constants=constants,
            environment=environment,
            cardinality=cardinality,
        )
    if set(output) != set(program["variables"]):
        raise ContrastiveOrbitError("nested Bekić result variables differ")
    return output


def _expression_counts(
    expression: Mapping[str, object],
    counts: dict[str, Counter[str]],
) -> None:
    kind = str(expression.get("kind"))
    if kind == "VARIABLE":
        counts["variables"][str(expression["variable"])] += 1
    elif kind == "CONSTANT":
        counts["constants"][str(expression["constant"])] += 1
    elif kind in GRAMMAR_KINDS:
        if kind not in {"VARIABLE", "CONSTANT"}:
            counts["operations"][kind] += 1
    else:
        raise ContrastiveOrbitError("program statistics saw invalid kind")
    counts["nodes"]["total"] += 1
    if kind == "CONVERSE":
        _expression_counts(expression["child"], counts)
    elif kind in BINARY_KINDS:
        for child in expression["children"]:
            _expression_counts(child, counts)


def dependency_topology(program: Mapping[str, object]) -> str:
    equations = _equations(program)
    x_name, y_name = (str(item) for item in program["variables"])

    def variables(expression: Mapping[str, object]) -> set[str]:
        return {
            str(node["variable"])
            for node in _walk(expression)
            if node.get("kind") == "VARIABLE"
        }

    x_cross = y_name in variables(equations[x_name])
    y_cross = x_name in variables(equations[y_name])
    if x_cross and y_cross:
        return "mutual_cycle"
    if x_cross:
        return "dag_y_to_x"
    if y_cross:
        return "dag_x_to_y"
    return "independent_dag"


def program_statistics(program: Mapping[str, object]) -> Json:
    equations = _equations(program)
    counts = {
        "variables": Counter(),
        "constants": Counter(),
        "operations": Counter(),
        "nodes": Counter(),
    }
    for expression in equations.values():
        _expression_counts(expression, counts)
    return {
        "ast_depth": max(
            _expression_depth(expression) for expression in equations.values()
        ),
        "node_count": counts["nodes"]["total"],
        "operator_multiset": dict(sorted(counts["operations"].items())),
        "variable_use_counts": dict(sorted(counts["variables"].items())),
        "constant_use_counts": dict(sorted(counts["constants"].items())),
        "dependency_topology": dependency_topology(program),
    }


def _canonical_expression(
    expression: Mapping[str, object],
    variable_slots: Mapping[str, int],
    *,
    motif_depth: int | None = None,
) -> object:
    kind = str(expression.get("kind"))
    if motif_depth is not None and motif_depth <= 0:
        return ("CUT", kind)
    next_depth = None if motif_depth is None else motif_depth - 1
    if kind == "VARIABLE":
        return ("VARIABLE", variable_slots[str(expression["variable"])])
    if kind == "CONSTANT":
        return ("CONSTANT",)
    if kind == "IDENTITY":
        return ("IDENTITY",)
    if kind == "CONVERSE":
        return (
            "CONVERSE",
            _canonical_expression(
                expression["child"],
                variable_slots,
                motif_depth=next_depth,
            ),
        )
    if kind in BINARY_KINDS:
        children = [
            _canonical_expression(
                child,
                variable_slots,
                motif_depth=next_depth,
            )
            for child in expression["children"]
        ]
        if kind in COMMUTATIVE_KINDS:
            children.sort(key=canonical_json)
        return (kind, *children)
    raise ContrastiveOrbitError("canonical expression kind differs")


def canonical_skeleton(program: Mapping[str, object]) -> object:
    equations = _equations(program)
    variables = tuple(str(item) for item in program["variables"])
    slots = {variable: index for index, variable in enumerate(variables)}
    return tuple(
        _canonical_expression(equations[variable], slots) for variable in variables
    )


def _motif_receipt(program: Mapping[str, object], depth: int) -> str:
    equations = _equations(program)
    variables = tuple(str(item) for item in program["variables"])
    slots = {variable: index for index, variable in enumerate(variables)}
    motifs = [
        _canonical_expression(node, slots, motif_depth=depth)
        for expression in equations.values()
        for node in _walk(expression)
    ]
    motifs.sort(key=canonical_json)
    return sha256_json(motifs)


def _has_operation_prefix(
    expression: Mapping[str, object],
    depth: int,
) -> bool:
    """Return whether every node above the requested frontier is an operation."""

    kind = expression.get("kind")
    if depth <= 1:
        return kind == "CONVERSE" or kind in BINARY_KINDS
    if kind == "CONVERSE":
        child = expression.get("child")
        return isinstance(child, dict) and _has_operation_prefix(child, depth - 1)
    if kind in BINARY_KINDS:
        children = expression.get("children")
        return (
            isinstance(children, list)
            and len(children) == 2
            and all(
                isinstance(child, dict)
                and _has_operation_prefix(child, depth - 1)
                for child in children
            )
        )
    return False


def individual_motif_receipts(
    program: Mapping[str, object],
    depth: int,
) -> tuple[str, ...]:
    """Hash maximal, non-nested depth-N operation motifs independently.

    Nested windows from the same operation tree are not independent examples:
    forbidding every overlapping subwindow rapidly exhausts the small typed
    grammar.  The maximal antichain preserves individual motif disjointness
    while preventing one tree from contributing a combinatorial receipt bag.
    """

    if depth not in {2, 3}:
        raise ContrastiveOrbitError("individual motif depth differs")
    equations = _equations(program)
    variables = tuple(str(item) for item in program["variables"])
    slots = {variable: index for index, variable in enumerate(variables)}
    motifs: set[str] = set()

    def collect(
        node: Mapping[str, object],
        *,
        below_qualifying_motif: bool,
    ) -> None:
        qualifies = _has_operation_prefix(node, depth)
        if qualifies and not below_qualifying_motif:
            motifs.add(
                sha256_json(
                    _canonical_expression(
                        node,
                        slots,
                        motif_depth=depth,
                    )
                )
            )
        suppress_descendants = below_qualifying_motif or qualifies
        kind = node.get("kind")
        if kind == "CONVERSE":
            child = node.get("child")
            if isinstance(child, dict):
                collect(
                    child,
                    below_qualifying_motif=suppress_descendants,
                )
        elif kind in BINARY_KINDS:
            children = node.get("children")
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        collect(
                            child,
                            below_qualifying_motif=suppress_descendants,
                        )

    for expression in equations.values():
        collect(expression, below_qualifying_motif=False)
    return tuple(sorted(motifs))


def program_receipts(program: Mapping[str, object]) -> Json:
    return {
        "skeleton_sha256": sha256_json(canonical_skeleton(program)),
        "depth_2_motif_sha256": _motif_receipt(program, 2),
        "depth_3_motif_sha256": _motif_receipt(program, 3),
        "depth_2_individual_motif_sha256s": list(
            individual_motif_receipts(program, 2)
        ),
        "depth_3_individual_motif_sha256s": list(
            individual_motif_receipts(program, 3)
        ),
    }


def _constant_occurrence_order(program: Mapping[str, object]) -> list[str]:
    equations = _equations(program)
    order: list[str] = []
    seen: set[str] = set()
    for variable in program["variables"]:
        for node in _walk(equations[str(variable)]):
            if node.get("kind") == "CONSTANT":
                name = str(node["constant"])
                if name not in seen:
                    seen.add(name)
                    order.append(name)
    return order


def _rewrite_symbols(
    expression: Mapping[str, object],
    *,
    variables: Mapping[str, str],
    constants: Mapping[str, str],
    rng: random.Random | None = None,
    equivalent: bool = False,
) -> Json:
    output = _strip_node_ids(expression)
    kind = output.get("kind")
    if kind == "VARIABLE":
        output["variable"] = variables.get(
            str(output["variable"]),
            str(output["variable"]),
        )
    elif kind == "CONSTANT":
        output["constant"] = constants.get(
            str(output["constant"]),
            str(output["constant"]),
        )
    elif kind == "CONVERSE":
        output["child"] = _rewrite_symbols(
            output["child"],
            variables=variables,
            constants=constants,
            rng=rng,
            equivalent=equivalent,
        )
    elif kind in BINARY_KINDS:
        output["children"] = [
            _rewrite_symbols(
                child,
                variables=variables,
                constants=constants,
                rng=rng,
                equivalent=equivalent,
            )
            for child in output["children"]
        ]
        if (
            equivalent
            and kind in COMMUTATIVE_KINDS
            and rng is not None
            and rng.random() < 0.5
        ):
            output["children"].reverse()
    elif kind == "LFP":
        output["variable"] = variables.get(
            str(output["variable"]),
            str(output["variable"]),
        )
        output["body"] = _rewrite_symbols(
            output["body"],
            variables=variables,
            constants=constants,
            rng=rng,
            equivalent=equivalent,
        )
    elif kind == "LET":
        output["variable"] = variables.get(
            str(output["variable"]),
            str(output["variable"]),
        )
        output["value"] = _rewrite_symbols(
            output["value"],
            variables=variables,
            constants=constants,
            rng=rng,
            equivalent=equivalent,
        )
        output["body"] = _rewrite_symbols(
            output["body"],
            variables=variables,
            constants=constants,
            rng=rng,
            equivalent=equivalent,
        )
    elif kind == "PAIR":
        for entry in output["entries"]:
            entry["variable"] = variables.get(
                str(entry["variable"]),
                str(entry["variable"]),
            )
            entry["expression"] = _rewrite_symbols(
                entry["expression"],
                variables=variables,
                constants=constants,
                rng=rng,
                equivalent=equivalent,
            )
    return output


def _rewrite_program_symbols(
    program: Mapping[str, object],
    *,
    variables: Mapping[str, str],
    constants: Mapping[str, str],
    rng: random.Random,
    equivalent: bool = False,
) -> Json:
    output = copy.deepcopy(dict(program))
    output["variables"] = [
        variables.get(str(item), str(item)) for item in output["variables"]
    ]
    if output.get("form") == "simultaneous_lfp":
        for equation in output["equations"]:
            equation["variable"] = variables.get(
                str(equation["variable"]),
                str(equation["variable"]),
            )
            equation["expression"] = _rewrite_symbols(
                equation["expression"],
                variables=variables,
                constants=constants,
                rng=rng,
                equivalent=equivalent,
            )
    elif output.get("form") == "nested_mu":
        output["expression"] = _rewrite_symbols(
            output["expression"],
            variables=variables,
            constants=constants,
            rng=rng,
            equivalent=equivalent,
        )
    else:
        raise ContrastiveOrbitError("rewritten program form differs")
    return _reidentify_program(output, rng)


def _rotate_constant_occurrences(
    program: Mapping[str, object],
    rng: random.Random,
) -> Json:
    output = copy.deepcopy(dict(program))
    constants = _constant_occurrence_order(output)
    if len(constants) != CONSTANT_COUNT:
        raise ContrastiveOrbitError("counterfactual needs every constant once")
    shifted = constants[1:] + constants[:1]
    constant_mapping = dict(zip(constants, shifted, strict=True))
    return _rewrite_program_symbols(
        output,
        variables={},
        constants=constant_mapping,
        rng=rng,
    )


def _reverse_one_compose(
    program: Mapping[str, object],
    rng: random.Random,
    *,
    preserve_held_out_motif: bool = False,
) -> Json:
    output = copy.deepcopy(dict(program))
    compose_nodes = [
        node
        for equation in output["equations"]
        for node in _walk(equation["expression"])
        if node.get("kind") == "COMPOSE"
        and not (preserve_held_out_motif and (
            isinstance(node.get("children"), list)
            and len(node["children"]) == 2
            and node["children"][0].get("kind") == HELD_OUT_MOTIF[1]
            and node["children"][1].get("kind") == HELD_OUT_MOTIF[2]
        ))
    ]
    if not compose_nodes:
        raise ContrastiveOrbitError("counterfactual needs a compose node")
    chosen = rng.choice(compose_nodes)
    chosen["children"].reverse()
    return _reidentify_program(output, rng)


def _rewire_counterfactual(
    program: Mapping[str, object],
    rng: random.Random,
    *,
    preserve_held_out_motif: bool = False,
) -> Json:
    return _reverse_one_compose(
        _rotate_constant_occurrences(program, rng),
        rng,
        preserve_held_out_motif=preserve_held_out_motif,
    )


def _packet(
    cardinality: int,
    constants: Sequence[Mapping[str, object]],
    program: Mapping[str, object],
) -> Json:
    return {
        "schema": "contrastive_bekic_machine_input_v1",
        "cardinality": cardinality,
        "constants": copy.deepcopy(list(constants)),
        "program": copy.deepcopy(dict(program)),
    }


def fixed_point_pressure(packet: Mapping[str, object]) -> Json:
    """Return an auditable per-variable causal-change/convergence receipt."""

    constants = _constants(packet)
    cardinality = int(packet["cardinality"])
    program = packet.get("program")
    if not isinstance(program, dict):
        raise ContrastiveOrbitError("pressure packet program differs")
    validate_simultaneous_program(program, set(constants))
    equations = _equations(program)
    variables = tuple(str(variable) for variable in program["variables"])
    current = {
        variable: empty_relation(cardinality)
        for variable in variables
    }
    change_steps = [0 for _ in variables]
    changed_bits = [0 for _ in variables]
    last_change_update = [0 for _ in variables]
    for update in range(1, 2 * cardinality * cardinality + 3):
        proposal = {
            variable: _evaluate_sim_expression(
                equations[variable],
                constants,
                current,
                cardinality,
            )
            for variable in variables
        }
        for slot, variable in enumerate(variables):
            changed = sum(
                left_bit != right_bit
                for left_row, right_row in zip(
                    current[variable],
                    proposal[variable],
                    strict=True,
                )
                for left_bit, right_bit in zip(
                    left_row,
                    right_row,
                    strict=True,
                )
            )
            if changed:
                change_steps[slot] += 1
                changed_bits[slot] += changed
                last_change_update[slot] = update
        if proposal == current:
            return {
                "convergence_updates": update - 1,
                "minimum_variable_change_steps": min(change_steps),
                "total_variable_change_steps": sum(change_steps),
                "variables": [
                    {
                        "slot": slot,
                        "change_steps": change_steps[slot],
                        "changed_bits_total": changed_bits[slot],
                        "last_change_update": last_change_update[slot],
                    }
                    for slot in range(len(variables))
                ],
            }
        if any(
            not relation_subset(current[variable], proposal[variable])
            for variable in variables
        ):
            raise ContrastiveOrbitError("pressure iteration violated monotonicity")
        current = proposal
    raise ContrastiveOrbitError("pressure fixed point did not converge")


def _pressure_is_admissible(receipt: Mapping[str, object]) -> bool:
    return (
        receipt.get("minimum_variable_change_steps")
        is not None
        and int(receipt["minimum_variable_change_steps"])
        >= MIN_VARIABLE_CHANGE_STEPS
        and int(receipt.get("convergence_updates", -1))
        >= MIN_CONVERGENCE_UPDATES
        and int(receipt.get("total_variable_change_steps", -1))
        >= MIN_TOTAL_VARIABLE_CHANGE_STEPS
    )


def joint_target_hamming(
    first: Mapping[str, Relation],
    second: Mapping[str, Relation],
) -> float:
    if set(first) != set(second) or not first:
        raise ContrastiveOrbitError("target environments differ")
    changed = 0
    total = 0
    for variable in first:
        left = first[variable]
        right = second[variable]
        if len(left) != len(right):
            raise ContrastiveOrbitError("target cardinalities differ")
        for left_row, right_row in zip(left, right, strict=True):
            changed += sum(
                first_bit != second_bit
                for first_bit, second_bit in zip(
                    left_row,
                    right_row,
                    strict=True,
                )
            )
            total += len(left_row)
    return changed / total


def _serialized_targets(environment: Mapping[str, Relation]) -> list[Json]:
    return [
        {"variable": variable, "relation": _serialize_relation(relation)}
        for variable, relation in environment.items()
    ]


def deserialize_targets(value: object) -> Environment:
    if not isinstance(value, list) or len(value) != 2:
        raise ContrastiveOrbitError("target payload differs")
    output: Environment = {}
    cardinality: int | None = None
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"variable", "relation"}
            or not isinstance(item["variable"], str)
            or item["variable"] in output
            or not isinstance(item["relation"], list)
        ):
            raise ContrastiveOrbitError("target entry differs")
        observed = len(item["relation"])
        cardinality = observed if cardinality is None else cardinality
        output[item["variable"]] = _deserialize_relation(
            item["relation"],
            cardinality,
        )
    return output


def _equivalent_arm(
    *,
    cardinality: int,
    constants: Sequence[Mapping[str, object]],
    simultaneous: Mapping[str, object],
    nested: Mapping[str, object],
    target: Mapping[str, Relation],
    rng: random.Random,
) -> tuple[Json, Environment, Json]:
    variables = [str(item) for item in simultaneous["variables"]]
    variable_mapping = {variable: _opaque("var", rng) for variable in variables}
    object_permutation = list(range(cardinality))
    rng.shuffle(object_permutation)
    if object_permutation == list(range(cardinality)):
        object_permutation = object_permutation[1:] + object_permutation[:1]
    rewritten_constants = [
        {
            "id": item["id"],
            "relation": _serialize_relation(
                reindex_relation(
                    _deserialize_relation(item["relation"], cardinality),
                    object_permutation,
                )
            ),
        }
        for item in constants
    ]
    rng.shuffle(rewritten_constants)
    original_order = [str(item["id"]) for item in constants]
    if [str(item["id"]) for item in rewritten_constants] == original_order:
        rewritten_constants = rewritten_constants[1:] + rewritten_constants[:1]
    rewritten_simultaneous = _rewrite_program_symbols(
        simultaneous,
        variables=variable_mapping,
        constants={},
        rng=rng,
        equivalent=True,
    )
    rewritten_nested = _rewrite_program_symbols(
        nested,
        variables=variable_mapping,
        constants={},
        rng=rng,
        equivalent=True,
    )
    rewritten_target = {
        variable_mapping[variable]: reindex_relation(
            relation,
            object_permutation,
        )
        for variable, relation in target.items()
    }
    return (
        {
            "simultaneous": _packet(
                cardinality,
                rewritten_constants,
                rewritten_simultaneous,
            ),
            "nested": _packet(
                cardinality,
                rewritten_constants,
                rewritten_nested,
            ),
        },
        rewritten_target,
        {
            "variable_mapping": variable_mapping,
            "object_permutation": object_permutation,
        },
    )


def _axes(split: str, index: int, rng: random.Random) -> Json:
    contract = split_contract(split)
    if split == "train":
        cell = "factorial"
        cardinalities = contract["cardinalities"]
        depths = contract["ast_depths"]
    else:
        cell = DEVELOPMENT_CELLS[index % len(DEVELOPMENT_CELLS)]
        cell_contract = contract["cells"][cell]
        cardinalities = cell_contract["cardinalities"]
        depths = cell_contract["ast_depths"]
    return {
        "cell": cell,
        "cardinality": rng.choice(cardinalities),
        "ast_depth": rng.choice(depths),
        "dependency_topology": DEPENDENCY_TOPOLOGIES[
            index % len(DEPENDENCY_TOPOLOGIES)
        ],
    }


def _program_receipt_values(
    receipts: Mapping[str, object],
    *,
    include_individual_motifs: bool,
) -> set[str]:
    required = {
        "skeleton_sha256",
        "depth_2_motif_sha256",
        "depth_3_motif_sha256",
        "depth_2_individual_motif_sha256s",
        "depth_3_individual_motif_sha256s",
    }
    if set(receipts) != required:
        raise ContrastiveOrbitError("program receipt schema differs")
    output = {
        str(receipts["skeleton_sha256"]),
        str(receipts["depth_2_motif_sha256"]),
        str(receipts["depth_3_motif_sha256"]),
    }
    if include_individual_motifs:
        for key in (
            "depth_2_individual_motif_sha256s",
            "depth_3_individual_motif_sha256s",
        ):
            values = receipts[key]
            if not isinstance(values, list) or any(
                not isinstance(value, str) for value in values
            ):
                raise ContrastiveOrbitError("individual motif receipts differ")
            output.update(values)
    return output


def _receipt_values(
    row: Mapping[str, object],
    *,
    include_individual_motifs: bool = True,
) -> set[str]:
    receipts = row.get("receipts")
    if not isinstance(receipts, dict):
        raise ContrastiveOrbitError("orbit receipts differ")
    output: set[str] = set()
    for arm in ("p", "p_prime"):
        arm_receipts = receipts.get(arm)
        if not isinstance(arm_receipts, dict):
            raise ContrastiveOrbitError("arm receipts differ")
        output.update(
            _program_receipt_values(
                arm_receipts,
                include_individual_motifs=include_individual_motifs,
            )
        )
    isolated = receipts.get("isolated_counterfactuals")
    if not isinstance(isolated, dict) or set(isolated) != set(
        ISOLATED_COUNTERFACTUAL_ARMS
    ):
        raise ContrastiveOrbitError("isolated counterfactual receipts differ")
    for arm_receipts in isolated.values():
        if not isinstance(arm_receipts, dict):
            raise ContrastiveOrbitError("isolated arm receipts differ")
        output.update(
            _program_receipt_values(
                arm_receipts,
                include_individual_motifs=include_individual_motifs,
            )
        )
    return output


def generate_orbit(
    *,
    split: str,
    seed: int,
    index: int = 0,
    forbidden_receipts: set[str] | None = None,
    max_attempts: int = 1200,
) -> Json:
    """Generate one deterministic matched counterfactual orbit."""

    split_contract(split)
    seed_material = f"{seed}:{split}:{index}:contrastive-bekic-v1"
    base_seed = int.from_bytes(
        hashlib.sha256(seed_material.encode()).digest()[:8],
        "big",
    )
    axes_rng = random.Random(base_seed)
    axes = _axes(split, index, axes_rng)
    forbidden = forbidden_receipts or set()
    motif_required = split == "development" and axes["cell"] == "motif"

    for attempt in range(max_attempts):
        rng = random.Random(base_seed ^ (attempt * 0x9E3779B97F4A7C15))
        cardinality = int(axes["cardinality"])
        constants, density = _sample_constant_world(cardinality, rng)
        constant_ids = [str(item["id"]) for item in constants]
        variables = [_opaque("var", rng), _opaque("var", rng)]
        p_program = _sample_program(
            constant_ids=constant_ids,
            variables=variables,
            depth=int(axes["ast_depth"]),
            topology=str(axes["dependency_topology"]),
            rng=rng,
        )
        if motif_required:
            try:
                p_program = _ensure_held_out_motif(p_program, rng)
            except ContrastiveOrbitError:
                continue
        p_prime_program = _rewire_counterfactual(
            p_program,
            rng,
            preserve_held_out_motif=motif_required,
        )
        isolated_programs = {
            "constant_rewire": _rotate_constant_occurrences(p_program, rng),
            "compose_reverse": _reverse_one_compose(
                p_program,
                rng,
                preserve_held_out_motif=motif_required,
            ),
        }
        pre_equivalence_programs = {
            "p": p_program,
            "p_prime": p_prime_program,
            **isolated_programs,
        }
        motif_presence = {
            arm: contains_held_out_motif(program)
            for arm, program in pre_equivalence_programs.items()
        }
        if split == "train" and any(motif_presence.values()):
            continue
        if motif_required and not (
            motif_presence["p"] and motif_presence["p_prime"]
        ):
            continue
        if not motif_required and any(motif_presence.values()):
            continue

        p_nested = build_nested_mu_program(p_program, rng=rng)
        p_prime_nested = build_nested_mu_program(p_prime_program, rng=rng)
        isolated_nested = {
            arm: build_nested_mu_program(program, rng=rng)
            for arm, program in isolated_programs.items()
        }

        p_inputs = {
            "simultaneous": _packet(
                cardinality,
                constants,
                p_program,
            ),
            "nested": _packet(cardinality, constants, p_nested),
        }
        p_prime_inputs = {
            "simultaneous": _packet(
                cardinality,
                constants,
                p_prime_program,
            ),
            "nested": _packet(
                cardinality,
                constants,
                p_prime_nested,
            ),
        }
        isolated_inputs = {
            arm: {
                "simultaneous": _packet(
                    cardinality,
                    constants,
                    program,
                ),
                "nested": _packet(
                    cardinality,
                    constants,
                    isolated_nested[arm],
                ),
            }
            for arm, program in isolated_programs.items()
        }
        p_target = evaluate_simultaneous(p_inputs["simultaneous"])
        p_nested_target = evaluate_nested_mu(p_inputs["nested"])
        p_prime_target = evaluate_simultaneous(p_prime_inputs["simultaneous"])
        p_prime_nested_target = evaluate_nested_mu(p_prime_inputs["nested"])
        if p_target != p_nested_target or p_prime_target != p_prime_nested_target:
            raise ContrastiveOrbitError(
                "independent simultaneous and nested oracles disagree"
            )
        isolated_targets = {
            arm: evaluate_simultaneous(forms["simultaneous"])
            for arm, forms in isolated_inputs.items()
        }
        if any(
            isolated_targets[arm] != evaluate_nested_mu(forms["nested"])
            for arm, forms in isolated_inputs.items()
        ):
            raise ContrastiveOrbitError(
                "isolated simultaneous and nested oracles disagree"
            )
        hamming = joint_target_hamming(p_target, p_prime_target)
        if hamming < COUNTERFACTUAL_MIN_HAMMING:
            continue
        isolated_hamming = {
            arm: joint_target_hamming(p_target, target)
            for arm, target in isolated_targets.items()
        }
        if any(value <= 0.0 for value in isolated_hamming.values()):
            continue
        if set(p_target.values()) & set(_constants(p_inputs["simultaneous"]).values()):
            continue
        if set(p_prime_target.values()) & set(
            _constants(p_prime_inputs["simultaneous"]).values()
        ):
            continue

        p_stats = program_statistics(p_program)
        p_prime_stats = program_statistics(p_prime_program)
        if p_stats != p_prime_stats:
            raise ContrastiveOrbitError(
                "counterfactual programs are not finite-statistic matched"
            )
        isolated_stats = {
            arm: program_statistics(program)
            for arm, program in isolated_programs.items()
        }
        if any(stats != p_stats for stats in isolated_stats.values()):
            raise ContrastiveOrbitError(
                "isolated counterfactual statistics differ"
            )

        pressure_inputs = {
            "p": p_inputs["simultaneous"],
            "p_prime": p_prime_inputs["simultaneous"],
            **{
                arm: forms["simultaneous"]
                for arm, forms in isolated_inputs.items()
            },
        }
        pressure_receipts = {
            arm: fixed_point_pressure(packet)
            for arm, packet in pressure_inputs.items()
        }
        # Admit on the canonical P dynamics. P' is deliberately allowed to
        # simplify or disrupt recursion; forcing both counterfactual outcomes
        # through the same dynamical profile overselects a narrow orbit class.
        # Every arm still carries a recomputable per-variable receipt.
        if not _pressure_is_admissible(pressure_receipts["p"]):
            continue

        p_receipts = program_receipts(p_program)
        p_prime_receipts = program_receipts(p_prime_program)
        isolated_receipts = {
            arm: program_receipts(program)
            for arm, program in isolated_programs.items()
        }
        candidate_receipts = set()
        for receipts in (
            p_receipts,
            p_prime_receipts,
            *isolated_receipts.values(),
        ):
            candidate_receipts.update(
                _program_receipt_values(
                    receipts,
                    include_individual_motifs=True,
                )
            )
        if candidate_receipts & forbidden:
            continue

        peq_inputs, peq_target, peq_transform = _equivalent_arm(
            cardinality=cardinality,
            constants=constants,
            simultaneous=p_program,
            nested=p_nested,
            target=p_target,
            rng=rng,
        )
        if (
            evaluate_simultaneous(peq_inputs["simultaneous"]) != peq_target
            or evaluate_nested_mu(peq_inputs["nested"]) != peq_target
        ):
            raise ContrastiveOrbitError("equivalent rewrite changed program semantics")
        peq_receipts = program_receipts(peq_inputs["simultaneous"]["program"])
        if peq_receipts != p_receipts:
            raise ContrastiveOrbitError("equivalent rewrite changed canonical receipts")
        motif_presence["p_eq"] = contains_held_out_motif(
            peq_inputs["simultaneous"]["program"]
        )
        if motif_presence["p_eq"] != motif_required:
            raise ContrastiveOrbitError(
                "equivalent arm changed held-out motif contract"
            )
        pressure_receipts["p_eq"] = fixed_point_pressure(
            peq_inputs["simultaneous"]
        )
        if pressure_receipts["p_eq"] != pressure_receipts["p"]:
            raise ContrastiveOrbitError(
                "equivalent arm changed recursive-pressure receipt"
            )

        row: Json = {
            "schema": "contrastive_bekic_program_orbit_v1",
            "split": split,
            "orbit_id": hashlib.sha256(f"{base_seed}:{attempt}".encode()).hexdigest(),
            "axes": axes,
            "inputs": {
                "p": p_inputs,
                "p_prime": p_prime_inputs,
                "p_eq": peq_inputs,
            },
            "targets": {
                "p": _serialized_targets(p_target),
                "p_prime": _serialized_targets(p_prime_target),
                "p_eq": _serialized_targets(peq_target),
            },
            "counterfactual": {
                "joint_target_hamming": hamming,
                "isolated_joint_target_hamming": isolated_hamming,
                "matched_statistics_sha256": sha256_json(p_stats),
            },
            "isolated_counterfactuals": {
                arm: {
                    "inputs": isolated_inputs[arm],
                    "targets": _serialized_targets(isolated_targets[arm]),
                }
                for arm in ISOLATED_COUNTERFACTUAL_ARMS
            },
            "equivalence": peq_transform,
            "receipts": {
                "p": p_receipts,
                "p_prime": p_prime_receipts,
                "p_eq": peq_receipts,
                "isolated_counterfactuals": isolated_receipts,
                "p_statistics": p_stats,
                "p_prime_statistics": p_prime_stats,
                "isolated_statistics": isolated_stats,
                "recursive_pressure": pressure_receipts,
                "constant_sampling": {
                    "shared_bernoulli_probability": density,
                    "role_specific_probabilities": False,
                },
                "held_out_motif": {
                    "definition": list(HELD_OUT_MOTIF),
                    "required_in_score_bearing_arms": motif_required,
                    "present_by_arm": motif_presence,
                },
                "input_sha256": {
                    arm: {form: sha256_json(packet) for form, packet in forms.items()}
                    for arm, forms in {
                        "p": p_inputs,
                        "p_prime": p_prime_inputs,
                        "p_eq": peq_inputs,
                    }.items()
                },
                "isolated_input_sha256": {
                    arm: {
                        form: sha256_json(packet)
                        for form, packet in forms.items()
                    }
                    for arm, forms in isolated_inputs.items()
                },
            },
        }
        validate_orbit(row)
        return row
    raise ContrastiveOrbitError(
        "failed to sample a separated, leakage-free counterfactual orbit"
    )


def generate_orbits(
    *,
    split: str,
    count: int,
    seed: int,
    forbidden_receipts: set[str] | None = None,
) -> list[Json]:
    if count <= 0:
        raise ContrastiveOrbitError("orbit count must be positive")
    split_contract(split)
    split_forbidden = set(forbidden_receipts or ())
    within_split_forbidden: set[str] = set()
    rows: list[Json] = []
    for index in range(count):
        row = generate_orbit(
            split=split,
            seed=seed,
            index=index,
            forbidden_receipts=(
                split_forbidden | within_split_forbidden
            ),
        )
        rows.append(row)
        # Local motifs may repeat inside one split. Only cross-split local
        # reuse is prohibited; complete skeleton/bag receipts remain unique
        # inside each partition.
        within_split_forbidden.update(
            _receipt_values(
                row,
                include_individual_motifs=False,
            )
        )
    return rows


def generate_train_development(
    *,
    train_count: int,
    development_count: int,
    seed: int,
) -> tuple[list[Json], list[Json]]:
    """Generate deterministic partitions with disjoint program receipts."""

    train = generate_orbits(
        split="train",
        count=train_count,
        seed=seed,
    )
    train_receipts = set().union(*(_receipt_values(row) for row in train))
    development = generate_orbits(
        split="development",
        count=development_count,
        seed=seed,
        forbidden_receipts=train_receipts,
    )
    assert_split_disjoint(train, development)
    return train, development


def select_machine_input(
    row: Mapping[str, object],
    *,
    arm: str,
    form: str,
) -> Json:
    if arm not in {"p", "p_prime", "p_eq"}:
        raise ContrastiveOrbitError("machine arm differs")
    if form not in {"simultaneous", "nested"}:
        raise ContrastiveOrbitError("machine form differs")
    inputs = row.get("inputs")
    if not isinstance(inputs, dict):
        raise ContrastiveOrbitError("orbit inputs differ")
    forms = inputs.get(arm)
    if not isinstance(forms, dict) or not isinstance(forms.get(form), dict):
        raise ContrastiveOrbitError("orbit arm input differs")
    return copy.deepcopy(forms[form])


def select_isolated_counterfactual_input(
    row: Mapping[str, object],
    *,
    arm: str,
    form: str,
) -> Json:
    if arm not in ISOLATED_COUNTERFACTUAL_ARMS:
        raise ContrastiveOrbitError("isolated machine arm differs")
    if form not in {"simultaneous", "nested"}:
        raise ContrastiveOrbitError("isolated machine form differs")
    isolated = row.get("isolated_counterfactuals")
    if not isinstance(isolated, dict):
        raise ContrastiveOrbitError("isolated counterfactual packet differs")
    payload = isolated.get(arm)
    if not isinstance(payload, dict) or set(payload) != {"inputs", "targets"}:
        raise ContrastiveOrbitError("isolated arm payload differs")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get(form), dict):
        raise ContrastiveOrbitError("isolated arm input differs")
    return copy.deepcopy(inputs[form])


def _constant_slot_mapping(
    donor_program: Mapping[str, object],
    recipient_program: Mapping[str, object],
) -> dict[str, str]:
    donor = _constant_occurrence_order(donor_program)
    recipient = _constant_occurrence_order(recipient_program)
    if len(donor) != CONSTANT_COUNT or len(recipient) != CONSTANT_COUNT:
        raise ContrastiveOrbitError("transplant constant slots differ")
    return dict(zip(donor, recipient, strict=True))


def transplant_program(
    donor: Mapping[str, object],
    recipient: Mapping[str, object],
    *,
    donor_arm: str = "p",
    recipient_arm: str = "p",
) -> dict[str, Json]:
    """Put a donor program into a recipient constant world."""

    donor_sim = select_machine_input(
        donor,
        arm=donor_arm,
        form="simultaneous",
    )
    donor_nested = select_machine_input(
        donor,
        arm=donor_arm,
        form="nested",
    )
    recipient_sim = select_machine_input(
        recipient,
        arm=recipient_arm,
        form="simultaneous",
    )
    if donor_sim["cardinality"] != recipient_sim["cardinality"]:
        raise ContrastiveOrbitError("program transplant requires matching cardinality")
    mapping = _constant_slot_mapping(
        donor_sim["program"],
        recipient_sim["program"],
    )
    donor_variables = [str(variable) for variable in donor_sim["program"]["variables"]]
    recipient_variables = [
        str(variable) for variable in recipient_sim["program"]["variables"]
    ]
    variable_mapping = dict(zip(donor_variables, recipient_variables, strict=True))
    rng = random.Random(
        int.from_bytes(
            hashlib.sha256(
                canonical_json(
                    [donor_sim["program"], recipient_sim["program"]]
                ).encode()
            ).digest()[:8],
            "big",
        )
    )
    transplanted_sim = _rewrite_program_symbols(
        donor_sim["program"],
        variables=variable_mapping,
        constants=mapping,
        rng=rng,
    )
    transplanted_nested = _rewrite_program_symbols(
        donor_nested["program"],
        variables=variable_mapping,
        constants=mapping,
        rng=rng,
    )
    constants = recipient_sim["constants"]
    cardinality = int(recipient_sim["cardinality"])
    return {
        "simultaneous": _packet(
            cardinality,
            constants,
            transplanted_sim,
        ),
        "nested": _packet(
            cardinality,
            constants,
            transplanted_nested,
        ),
    }


def transplant_constants(
    donor: Mapping[str, object],
    recipient: Mapping[str, object],
    *,
    donor_arm: str = "p",
    recipient_arm: str = "p",
) -> dict[str, Json]:
    """Put donor constant values under a fixed recipient program."""

    donor_sim = select_machine_input(
        donor,
        arm=donor_arm,
        form="simultaneous",
    )
    recipient_sim = select_machine_input(
        recipient,
        arm=recipient_arm,
        form="simultaneous",
    )
    recipient_nested = select_machine_input(
        recipient,
        arm=recipient_arm,
        form="nested",
    )
    if donor_sim["cardinality"] != recipient_sim["cardinality"]:
        raise ContrastiveOrbitError("constant transplant requires matching cardinality")
    mapping = _constant_slot_mapping(
        donor_sim["program"],
        recipient_sim["program"],
    )
    donor_values = _constants(donor_sim)
    recipient_values = {
        recipient_name: donor_values[donor_name]
        for donor_name, recipient_name in mapping.items()
    }
    constants = [
        {
            "id": item["id"],
            "relation": _serialize_relation(recipient_values[str(item["id"])]),
        }
        for item in recipient_sim["constants"]
    ]
    cardinality = int(recipient_sim["cardinality"])
    return {
        "simultaneous": _packet(
            cardinality,
            constants,
            recipient_sim["program"],
        ),
        "nested": _packet(
            cardinality,
            constants,
            recipient_nested["program"],
        ),
    }


def assert_split_disjoint(
    train: Sequence[Mapping[str, object]],
    development: Sequence[Mapping[str, object]],
) -> None:
    train_receipts = set().union(*(_receipt_values(row) for row in train))
    development_receipts = set().union(*(_receipt_values(row) for row in development))
    if not train_receipts.isdisjoint(development_receipts):
        raise ContrastiveOrbitError(
            "train/development skeleton or motif receipts overlap"
        )
    train_inputs = {
        str(value)
        for row in train
        for arm in row["receipts"]["input_sha256"].values()
        for value in arm.values()
    }
    development_inputs = {
        str(value)
        for row in development
        for arm in row["receipts"]["input_sha256"].values()
        for value in arm.values()
    }
    train_inputs.update(
        str(value)
        for row in train
        for arm in row["receipts"]["isolated_input_sha256"].values()
        for value in arm.values()
    )
    development_inputs.update(
        str(value)
        for row in development
        for arm in row["receipts"]["isolated_input_sha256"].values()
        for value in arm.values()
    )
    if not train_inputs.isdisjoint(development_inputs):
        raise ContrastiveOrbitError("train/development machine input hashes overlap")


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key).lower() for key in value} | set().union(
            *(_recursive_keys(item) for item in value.values()),
            set(),
        )
    if isinstance(value, list):
        return set().union(
            *(_recursive_keys(item) for item in value),
            set(),
        )
    return set()


def validate_orbit(row: Mapping[str, object]) -> None:
    if row.get("schema") != "contrastive_bekic_program_orbit_v1" or row.get(
        "split"
    ) not in {"train", "development"}:
        raise ContrastiveOrbitError("orbit schema or split differs")
    split_contract(str(row["split"]))
    inputs = row.get("inputs")
    targets = row.get("targets")
    isolated = row.get("isolated_counterfactuals")
    if (
        not isinstance(inputs, dict)
        or set(inputs) != {"p", "p_prime", "p_eq"}
        or not isinstance(targets, dict)
        or set(targets) != {"p", "p_prime", "p_eq"}
        or not isinstance(isolated, dict)
        or set(isolated) != set(ISOLATED_COUNTERFACTUAL_ARMS)
    ):
        raise ContrastiveOrbitError("orbit arms differ")
    forbidden = {
        "answer",
        "fixed_point",
        "halt",
        "iteration",
        "query",
        "schedule",
        "target",
        "trajectory",
    }
    score_inputs = list(inputs.values())
    for arm in ISOLATED_COUNTERFACTUAL_ARMS:
        payload = isolated[arm]
        if (
            not isinstance(payload, dict)
            or set(payload) != {"inputs", "targets"}
            or not isinstance(payload["inputs"], dict)
            or set(payload["inputs"]) != {"simultaneous", "nested"}
        ):
            raise ContrastiveOrbitError("isolated counterfactual arm differs")
        score_inputs.append(payload["inputs"])
    for arm in score_inputs:
        if not isinstance(arm, dict) or set(arm) != {
            "simultaneous",
            "nested",
        }:
            raise ContrastiveOrbitError("paired oracle inputs differ")
        for packet in arm.values():
            keys = _recursive_keys(packet)
            if any(fragment in key for key in keys for fragment in forbidden):
                raise ContrastiveOrbitError("machine input leaks supervision")

    computed: dict[str, Environment] = {}
    for arm in ("p", "p_prime", "p_eq"):
        simultaneous = evaluate_simultaneous(inputs[arm]["simultaneous"])
        nested = evaluate_nested_mu(inputs[arm]["nested"])
        stored = deserialize_targets(targets[arm])
        if simultaneous != nested or simultaneous != stored:
            raise ContrastiveOrbitError("stored target or independent oracle differs")
        computed[arm] = simultaneous
    hamming = joint_target_hamming(computed["p"], computed["p_prime"])
    counterfactual = row.get("counterfactual")
    if (
        hamming < COUNTERFACTUAL_MIN_HAMMING
        or not isinstance(counterfactual, dict)
        or counterfactual.get("joint_target_hamming") != hamming
        ):
            raise ContrastiveOrbitError("counterfactual target separation differs")
    isolated_computed: dict[str, Environment] = {}
    isolated_hamming: dict[str, float] = {}
    for arm in ISOLATED_COUNTERFACTUAL_ARMS:
        payload = isolated[arm]
        simultaneous = evaluate_simultaneous(
            payload["inputs"]["simultaneous"]
        )
        nested = evaluate_nested_mu(payload["inputs"]["nested"])
        stored = deserialize_targets(payload["targets"])
        if simultaneous != nested or simultaneous != stored:
            raise ContrastiveOrbitError(
                "isolated stored target or independent oracle differs"
            )
        isolated_computed[arm] = simultaneous
        isolated_hamming[arm] = joint_target_hamming(
            computed["p"],
            simultaneous,
        )
        if isolated_hamming[arm] <= 0.0:
            raise ContrastiveOrbitError(
                "isolated counterfactual did not change target"
            )
    if counterfactual.get("isolated_joint_target_hamming") != isolated_hamming:
        raise ContrastiveOrbitError(
            "isolated counterfactual target separation differs"
        )
    receipts = row.get("receipts")
    if not isinstance(receipts, dict):
        raise ContrastiveOrbitError("orbit receipts differ")
    p_program = inputs["p"]["simultaneous"]["program"]
    p_prime_program = inputs["p_prime"]["simultaneous"]["program"]
    if program_statistics(p_program) != program_statistics(p_prime_program):
        raise ContrastiveOrbitError("counterfactual finite statistics differ")
    if program_receipts(p_program) != receipts.get("p"):
        raise ContrastiveOrbitError("P receipts differ")
    if program_receipts(p_prime_program) != receipts.get("p_prime"):
        raise ContrastiveOrbitError("P-prime receipts differ")
    peq_program = inputs["p_eq"]["simultaneous"]["program"]
    if program_receipts(peq_program) != receipts.get("p_eq"):
        raise ContrastiveOrbitError("Peq receipts differ")
    if receipts["p"] != receipts["p_eq"]:
        raise ContrastiveOrbitError("Peq canonical receipts are not invariant")
    isolated_receipts = receipts.get("isolated_counterfactuals")
    isolated_statistics = receipts.get("isolated_statistics")
    if (
        not isinstance(isolated_receipts, dict)
        or set(isolated_receipts) != set(ISOLATED_COUNTERFACTUAL_ARMS)
        or not isinstance(isolated_statistics, dict)
        or set(isolated_statistics) != set(ISOLATED_COUNTERFACTUAL_ARMS)
    ):
        raise ContrastiveOrbitError("isolated receipt schema differs")
    p_statistics = program_statistics(p_program)
    for arm in ISOLATED_COUNTERFACTUAL_ARMS:
        program = isolated[arm]["inputs"]["simultaneous"]["program"]
        if program_receipts(program) != isolated_receipts[arm]:
            raise ContrastiveOrbitError("isolated program receipts differ")
        if (
            program_statistics(program) != p_statistics
            or isolated_statistics[arm] != p_statistics
        ):
            raise ContrastiveOrbitError(
                "isolated finite statistics differ"
            )

    pressure = receipts.get("recursive_pressure")
    pressure_inputs = {
        "p": inputs["p"]["simultaneous"],
        "p_prime": inputs["p_prime"]["simultaneous"],
        "p_eq": inputs["p_eq"]["simultaneous"],
        **{
            arm: isolated[arm]["inputs"]["simultaneous"]
            for arm in ISOLATED_COUNTERFACTUAL_ARMS
        },
    }
    if (
        not isinstance(pressure, dict)
        or set(pressure) != set(pressure_inputs)
    ):
        raise ContrastiveOrbitError("recursive-pressure receipts differ")
    for arm, packet in pressure_inputs.items():
        observed = fixed_point_pressure(packet)
        if pressure[arm] != observed:
            raise ContrastiveOrbitError(
                "recursive-pressure admission differs"
            )
        if arm in {"p", "p_eq"} and not _pressure_is_admissible(
            observed
        ):
            raise ContrastiveOrbitError(
                "score-bearing recursive pressure is insufficient"
            )

    motif_receipt = receipts.get("held_out_motif")
    motif_required = (
        row["split"] == "development"
        and row.get("axes", {}).get("cell") == "motif"
    )
    motif_programs = {
        "p": p_program,
        "p_prime": p_prime_program,
        "p_eq": peq_program,
        **{
            arm: isolated[arm]["inputs"]["simultaneous"]["program"]
            for arm in ISOLATED_COUNTERFACTUAL_ARMS
        },
    }
    motif_presence = {
        arm: contains_held_out_motif(program)
        for arm, program in motif_programs.items()
    }
    expected_motif_receipt = {
        "definition": list(HELD_OUT_MOTIF),
        "required_in_score_bearing_arms": motif_required,
        "present_by_arm": motif_presence,
    }
    if motif_receipt != expected_motif_receipt:
        raise ContrastiveOrbitError("held-out motif receipt differs")
    if row["split"] == "train" and any(motif_presence.values()):
        raise ContrastiveOrbitError("training arm contains held-out motif")
    if motif_required and not all(
        motif_presence[arm] for arm in ("p", "p_prime", "p_eq")
    ):
        raise ContrastiveOrbitError(
            "motif development score-bearing arm lacks motif"
        )
    if not motif_required and any(motif_presence.values()):
        raise ContrastiveOrbitError(
            "non-motif arm contains held-out motif"
        )
