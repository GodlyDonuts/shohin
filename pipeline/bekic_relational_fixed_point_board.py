"""Sealed CPU board for Bekić-equivalent relational fixed points.

Each episode supplies raw relation constants and two source-deleted typed
program graphs. One graph denotes a simultaneous least fixed point; the other
denotes the mathematically equivalent nested Bekić construction. The graphs
contain no execution schedule, iteration count, trajectory, fixed point,
query answer, or target-equivalent relation.

The two oracle implementations below intentionally use different iteration
structures. They share only the primitive relation algebra.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Mapping, Sequence


MAX_OBJECTS = 8
VARIABLE_COUNT = 2
CONSTANT_COUNT = 8
ALLOWED_OPERATIONS = frozenset(
    {"UNION", "INTERSECTION", "COMPOSE", "CONVERSE", "IDENTITY"}
)
OPERATION_ARITY = {
    "UNION": 2,
    "INTERSECTION": 2,
    "COMPOSE": 2,
    "CONVERSE": 1,
    "IDENTITY": 0,
}
DEPENDENCY_TOPOLOGIES = (
    "independent_dag",
    "dag_x_to_y",
    "dag_y_to_x",
    "mutual_cycle",
)

Relation = tuple[tuple[int, ...], ...]
Environment = dict[str, Relation]


class BekicBoardError(ValueError):
    """Raised when a board episode violates its sealed contract."""


def canonical_json(value: object) -> str:
    """Return the canonical JSON encoding used by every board receipt."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    """Hash a JSON-compatible value using the board canonicalization."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def empty_relation(cardinality: int) -> Relation:
    _require_cardinality(cardinality)
    return tuple(tuple(0 for _ in range(cardinality)) for _ in range(cardinality))


def identity_relation(cardinality: int) -> Relation:
    _require_cardinality(cardinality)
    return tuple(
        tuple(int(row == column) for column in range(cardinality))
        for row in range(cardinality)
    )


def universal_relation(cardinality: int) -> Relation:
    _require_cardinality(cardinality)
    return tuple(tuple(1 for _ in range(cardinality)) for _ in range(cardinality))


def relation_union(left: Relation, right: Relation) -> Relation:
    _require_same_geometry(left, right)
    return tuple(
        tuple(int(first or second) for first, second in zip(a, b, strict=True))
        for a, b in zip(left, right, strict=True)
    )


def relation_intersection(left: Relation, right: Relation) -> Relation:
    _require_same_geometry(left, right)
    return tuple(
        tuple(int(first and second) for first, second in zip(a, b, strict=True))
        for a, b in zip(left, right, strict=True)
    )


def relation_compose(left: Relation, right: Relation) -> Relation:
    """Compose relations in output-by-source matrix convention."""

    cardinality = _require_same_geometry(left, right)
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


def relation_converse(relation: Relation) -> Relation:
    cardinality = _require_relation(relation)
    return tuple(
        tuple(relation[column][row] for column in range(cardinality))
        for row in range(cardinality)
    )


def relation_subset(left: Relation, right: Relation) -> bool:
    cardinality = _require_same_geometry(left, right)
    return all(
        left[row][column] <= right[row][column]
        for row in range(cardinality)
        for column in range(cardinality)
    )


def reindex_relation(
    relation: Relation,
    permutation: Sequence[int],
) -> Relation:
    """Relabel objects, with ``permutation[new_index] == old_index``."""

    cardinality = _require_relation(relation)
    _require_permutation(permutation, cardinality, "object")
    return tuple(
        tuple(
            relation[permutation[row]][permutation[column]]
            for column in range(cardinality)
        )
        for row in range(cardinality)
    )


def _require_cardinality(cardinality: int) -> None:
    if not isinstance(cardinality, int) or not 2 <= cardinality <= MAX_OBJECTS:
        raise BekicBoardError("relation cardinality is outside the board bound")


def _require_relation(value: object, cardinality: int | None = None) -> int:
    if not isinstance(value, tuple) or not value:
        raise BekicBoardError("relation must be a nonempty tuple matrix")
    observed = len(value)
    if cardinality is not None and observed != cardinality:
        raise BekicBoardError("relation cardinality differs")
    if any(
        not isinstance(row, tuple)
        or len(row) != observed
        or any(bit not in (0, 1) for bit in row)
        for row in value
    ):
        raise BekicBoardError("relation is not a square binary matrix")
    _require_cardinality(observed)
    return observed


def _require_same_geometry(left: Relation, right: Relation) -> int:
    cardinality = _require_relation(left)
    _require_relation(right, cardinality)
    return cardinality


def _as_relation(value: object, cardinality: int) -> Relation:
    if (
        not isinstance(value, list)
        or len(value) != cardinality
        or any(not isinstance(row, list) or len(row) != cardinality for row in value)
    ):
        raise BekicBoardError("serialized relation geometry differs")
    relation = tuple(tuple(bit for bit in row) for row in value)
    _require_relation(relation, cardinality)
    return relation


def _serialize_relation(relation: Relation) -> list[list[int]]:
    _require_relation(relation)
    return [list(row) for row in relation]


def _require_permutation(
    permutation: Sequence[int],
    length: int,
    label: str,
) -> None:
    if (
        len(permutation) != length
        or set(permutation) != set(range(length))
        or any(not isinstance(index, int) for index in permutation)
    ):
        raise BekicBoardError(f"{label} reindexing is not a permutation")


def _node_table(program: Mapping[str, object]) -> dict[str, dict[str, object]]:
    nodes = program.get("nodes")
    if not isinstance(nodes, list):
        raise BekicBoardError("program nodes differ")
    table: dict[str, dict[str, object]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise BekicBoardError("program node differs")
        node_id = node["id"]
        if node_id in table:
            raise BekicBoardError("program node ids are not unique")
        table[node_id] = node
    return table


def _equation_roots(program: Mapping[str, object]) -> dict[str, str]:
    equations = program.get("equations")
    if not isinstance(equations, list):
        raise BekicBoardError("program equations differ")
    roots: dict[str, str] = {}
    for equation in equations:
        if (
            not isinstance(equation, dict)
            or set(equation) != {"variable", "root"}
            or not isinstance(equation["variable"], str)
            or not isinstance(equation["root"], str)
        ):
            raise BekicBoardError("program equation differs")
        if equation["variable"] in roots:
            raise BekicBoardError("program equation variables are not unique")
        roots[equation["variable"]] = equation["root"]
    return roots


def _constant_environment(input_packet: Mapping[str, object]) -> Environment:
    cardinality = input_packet.get("cardinality")
    if not isinstance(cardinality, int):
        raise BekicBoardError("input cardinality differs")
    _require_cardinality(cardinality)
    constants = input_packet.get("constants")
    if not isinstance(constants, list) or len(constants) != CONSTANT_COUNT:
        raise BekicBoardError("raw relation constants differ")
    output: Environment = {}
    for item in constants:
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "relation"}
            or not isinstance(item["id"], str)
            or item["id"] in output
        ):
            raise BekicBoardError("raw relation constant differs")
        output[item["id"]] = _as_relation(item["relation"], cardinality)
    return output


def _apply_operation(operation: str, arguments: Sequence[Relation], cardinality: int) -> Relation:
    if operation == "UNION":
        return relation_union(arguments[0], arguments[1])
    if operation == "INTERSECTION":
        return relation_intersection(arguments[0], arguments[1])
    if operation == "COMPOSE":
        return relation_compose(arguments[0], arguments[1])
    if operation == "CONVERSE":
        return relation_converse(arguments[0])
    if operation == "IDENTITY":
        return identity_relation(cardinality)
    raise BekicBoardError("unknown relation operation")


def evaluate_program_equations(
    program: Mapping[str, object],
    constants: Mapping[str, Relation],
    variables: Mapping[str, Relation],
) -> Environment:
    """Evaluate one typed equation graph under a supplied variable state."""

    variable_ids = program.get("variables")
    if not isinstance(variable_ids, list) or len(variable_ids) != VARIABLE_COUNT:
        raise BekicBoardError("program variables differ")
    if set(variables) != set(variable_ids):
        raise BekicBoardError("variable environment differs")
    cardinalities = {_require_relation(value) for value in (*constants.values(), *variables.values())}
    if len(cardinalities) != 1:
        raise BekicBoardError("equation environment geometry differs")
    cardinality = cardinalities.pop()
    table = _node_table(program)
    roots = _equation_roots(program)
    memo: Environment = {}
    active: set[str] = set()

    def visit(node_id: str) -> Relation:
        if node_id in memo:
            return memo[node_id]
        if node_id in active or node_id not in table:
            raise BekicBoardError("program operation graph is cyclic or dangling")
        active.add(node_id)
        node = table[node_id]
        kind = node.get("kind")
        if kind == "variable":
            if set(node) != {"id", "kind", "variable"}:
                raise BekicBoardError("variable node schema differs")
            variable = node.get("variable")
            if variable not in variables:
                raise BekicBoardError("variable node is dangling")
            result = variables[str(variable)]
        elif kind == "constant":
            if set(node) != {"id", "kind", "constant"}:
                raise BekicBoardError("constant node schema differs")
            constant = node.get("constant")
            if constant not in constants:
                raise BekicBoardError("constant node is dangling")
            result = constants[str(constant)]
        elif kind == "operation":
            if set(node) != {"id", "kind", "operation", "inputs"}:
                raise BekicBoardError("operation node schema differs")
            operation = node.get("operation")
            inputs = node.get("inputs")
            if (
                operation not in ALLOWED_OPERATIONS
                or not isinstance(inputs, list)
                or len(inputs) != OPERATION_ARITY[str(operation)]
                or any(not isinstance(item, str) for item in inputs)
            ):
                raise BekicBoardError("typed relation operation differs")
            arguments = [visit(item) for item in inputs]
            result = _apply_operation(str(operation), arguments, cardinality)
        else:
            raise BekicBoardError("program node kind differs")
        active.remove(node_id)
        memo[node_id] = result
        return result

    if set(roots) != set(variable_ids):
        raise BekicBoardError("program does not define both variables")
    return {variable: visit(roots[variable]) for variable in variable_ids}


def evaluate_simultaneous(input_packet: Mapping[str, object]) -> Environment:
    """Compute the least simultaneous fixed point by Kleene iteration."""

    constants = _constant_environment(input_packet)
    program = _representation(input_packet, "simultaneous")
    variables = _program_variables(program)
    cardinality = int(input_packet["cardinality"])
    current = {variable: empty_relation(cardinality) for variable in variables}
    for _ in range(2 * cardinality * cardinality + 2):
        proposal = evaluate_program_equations(program, constants, current)
        if proposal == current:
            return proposal
        if any(
            not relation_subset(current[variable], proposal[variable])
            for variable in variables
        ):
            raise BekicBoardError("simultaneous iteration is not monotone")
        current = proposal
    raise BekicBoardError("simultaneous least fixed point did not converge")


def _evaluate_nested_expression(
    node_id: str,
    *,
    table: Mapping[str, dict[str, object]],
    constants: Mapping[str, Relation],
    state: Mapping[str, Relation],
    cardinality: int,
    memo: Environment,
    active: set[str],
) -> Relation:
    """Independent expression interpreter used only by the Bekić oracle."""

    cached = memo.get(node_id)
    if cached is not None:
        return cached
    if node_id in active:
        raise BekicBoardError("nested program operation graph is cyclic")
    node = table.get(node_id)
    if node is None:
        raise BekicBoardError("nested program node is dangling")
    active.add(node_id)
    kind = node.get("kind")
    if kind == "variable":
        variable = node.get("variable")
        if variable not in state:
            raise BekicBoardError("nested variable node is dangling")
        result = state[str(variable)]
    elif kind == "constant":
        constant = node.get("constant")
        if constant not in constants:
            raise BekicBoardError("nested constant node is dangling")
        result = constants[str(constant)]
    elif kind == "operation":
        operation = node.get("operation")
        references = node.get("inputs")
        if (
            operation not in ALLOWED_OPERATIONS
            or not isinstance(references, list)
            or len(references) != OPERATION_ARITY[str(operation)]
        ):
            raise BekicBoardError("nested typed operation differs")
        arguments = [
            _evaluate_nested_expression(
                str(reference),
                table=table,
                constants=constants,
                state=state,
                cardinality=cardinality,
                memo=memo,
                active=active,
            )
            for reference in references
        ]
        result = _apply_operation(str(operation), arguments, cardinality)
    else:
        raise BekicBoardError("nested program node kind differs")
    active.remove(node_id)
    memo[node_id] = result
    return result


def evaluate_bekic(input_packet: Mapping[str, object]) -> Environment:
    """Compute the same least fixed point using the nested Bekić identity."""

    constants = _constant_environment(input_packet)
    program = _representation(input_packet, "nested")
    variables = _program_variables(program)
    binding = program.get("binding")
    if (
        not isinstance(binding, dict)
        or set(binding) != {"outer", "inner"}
        or {binding["outer"], binding["inner"]} != set(variables)
    ):
        raise BekicBoardError("nested Bekić binding differs")
    outer = str(binding["outer"])
    inner = str(binding["inner"])
    roots = _equation_roots(program)
    table = _node_table(program)
    cardinality = int(input_packet["cardinality"])
    bottom = empty_relation(cardinality)

    def evaluate_root(root: str, state: Mapping[str, Relation]) -> Relation:
        return _evaluate_nested_expression(
            root,
            table=table,
            constants=constants,
            state=state,
            cardinality=cardinality,
            memo={},
            active=set(),
        )

    def inner_fixed_point(outer_value: Relation) -> Relation:
        current_inner = bottom
        for _ in range(cardinality * cardinality + 2):
            state = {outer: outer_value, inner: current_inner}
            proposal = evaluate_root(roots[inner], state)
            if proposal == current_inner:
                return proposal
            if not relation_subset(current_inner, proposal):
                raise BekicBoardError("inner Bekić iteration is not monotone")
            current_inner = proposal
        raise BekicBoardError("inner Bekić least fixed point did not converge")

    current_outer = bottom
    for _ in range(cardinality * cardinality + 2):
        current_inner = inner_fixed_point(current_outer)
        state = {outer: current_outer, inner: current_inner}
        proposal_outer = evaluate_root(roots[outer], state)
        if proposal_outer == current_outer:
            final_inner = inner_fixed_point(proposal_outer)
            return {outer: proposal_outer, inner: final_inner}
        if not relation_subset(current_outer, proposal_outer):
            raise BekicBoardError("outer Bekić iteration is not monotone")
        current_outer = proposal_outer
    raise BekicBoardError("outer Bekić least fixed point did not converge")


def _representation(
    input_packet: Mapping[str, object],
    form: str,
) -> dict[str, object]:
    representations = input_packet.get("representations")
    if not isinstance(representations, dict) or set(representations) != {
        "simultaneous",
        "nested",
    }:
        raise BekicBoardError("paired program representations differ")
    program = representations.get(form)
    if not isinstance(program, dict):
        raise BekicBoardError("program representation differs")
    return program


def select_machine_input(
    input_packet: Mapping[str, object],
    form: str,
) -> dict[str, object]:
    """Expose exactly one paired representation to a future neural arm."""

    if form not in {"simultaneous", "nested"}:
        raise BekicBoardError("machine representation differs")
    cardinality = input_packet.get("cardinality")
    constants = input_packet.get("constants")
    if not isinstance(cardinality, int) or not isinstance(constants, list):
        raise BekicBoardError("machine input packet differs")
    program = _representation(input_packet, form)
    return {
        "schema": "bekic_relational_machine_input_v1",
        "cardinality": cardinality,
        "constants": copy.deepcopy(constants),
        "program": copy.deepcopy(program),
    }


def _program_variables(program: Mapping[str, object]) -> tuple[str, str]:
    variables = program.get("variables")
    if (
        not isinstance(variables, list)
        or len(variables) != VARIABLE_COUNT
        or len(set(variables)) != VARIABLE_COUNT
        or any(not isinstance(variable, str) for variable in variables)
    ):
        raise BekicBoardError("program variables differ")
    return str(variables[0]), str(variables[1])


def canonical_program_semantics(program: Mapping[str, object]) -> object:
    """Normalize a program while erasing node ids and node list order."""

    variables = _program_variables(program)
    variable_slots = {variable: index for index, variable in enumerate(variables)}
    table = _node_table(program)
    roots = _equation_roots(program)
    active: set[str] = set()
    memo: dict[str, object] = {}

    def normalize(node_id: str) -> object:
        if node_id in memo:
            return memo[node_id]
        if node_id in active or node_id not in table:
            raise BekicBoardError("program graph cannot be normalized")
        active.add(node_id)
        node = table[node_id]
        kind = node.get("kind")
        if kind == "variable":
            variable = node.get("variable")
            if variable not in variable_slots:
                raise BekicBoardError("normalized variable differs")
            result: object = ("VARIABLE", variable_slots[str(variable)])
        elif kind == "constant":
            constant = node.get("constant")
            if not isinstance(constant, str):
                raise BekicBoardError("normalized constant differs")
            result = ("CONSTANT", constant)
        elif kind == "operation":
            operation = node.get("operation")
            references = node.get("inputs")
            if (
                operation not in ALLOWED_OPERATIONS
                or not isinstance(references, list)
                or len(references) != OPERATION_ARITY[str(operation)]
            ):
                raise BekicBoardError("normalized operation differs")
            result = (
                "OPERATION",
                operation,
                tuple(normalize(str(reference)) for reference in references),
            )
        else:
            raise BekicBoardError("normalized node differs")
        active.remove(node_id)
        memo[node_id] = result
        return result

    if set(roots) != set(variables):
        raise BekicBoardError("normalized equations differ")
    return tuple(normalize(roots[variable]) for variable in variables)


def reindex_program_nodes(
    program: Mapping[str, object],
    permutation: Sequence[int],
    *,
    namespace: str = "",
) -> dict[str, object]:
    """Rename and reorder operation-graph nodes without changing semantics."""

    output = copy.deepcopy(dict(program))
    nodes = output.get("nodes")
    if not isinstance(nodes, list):
        raise BekicBoardError("program nodes differ before reindexing")
    _require_permutation(permutation, len(nodes), "program-node")
    old_ids = [str(node["id"]) for node in nodes]
    salt = sha256_json([old_ids, namespace])[:10]
    fresh_ids = [f"node_{salt}_{index}" for index in range(len(nodes))]
    mapping = {
        old_ids[old_index]: fresh_ids[new_index]
        for new_index, old_index in enumerate(permutation)
    }

    def rewrite_node(node: dict[str, object]) -> dict[str, object]:
        rewritten = copy.deepcopy(node)
        rewritten["id"] = mapping[str(node["id"])]
        if rewritten.get("kind") == "operation":
            rewritten["inputs"] = [
                mapping[str(reference)] for reference in rewritten["inputs"]
            ]
        return rewritten

    output["nodes"] = [rewrite_node(nodes[index]) for index in permutation]
    output["equations"] = [
        {
            "variable": equation["variable"],
            "root": mapping[str(equation["root"])],
        }
        for equation in output["equations"]
    ]
    return output


def reindex_program_variables(
    program: Mapping[str, object],
    permutation: Sequence[int],
    *,
    namespace: str = "",
) -> tuple[dict[str, object], dict[str, str]]:
    """Alpha-rename and reorder variables while preserving their equations."""

    output = copy.deepcopy(dict(program))
    old_variables = list(_program_variables(output))
    _require_permutation(permutation, VARIABLE_COUNT, "variable")
    salt = sha256_json([old_variables, namespace])[:10]
    fresh_variables = [f"variable_{salt}_{index}" for index in range(VARIABLE_COUNT)]
    mapping = {
        old_variables[old_index]: fresh_variables[new_index]
        for new_index, old_index in enumerate(permutation)
    }
    output["variables"] = fresh_variables
    for node in output["nodes"]:
        if node.get("kind") == "variable":
            node["variable"] = mapping[str(node["variable"])]
    old_equations = {
        str(equation["variable"]): str(equation["root"])
        for equation in output["equations"]
    }
    output["equations"] = [
        {
            "variable": fresh_variables[new_index],
            "root": old_equations[old_variables[old_index]],
        }
        for new_index, old_index in enumerate(permutation)
    ]
    if output.get("form") == "bekic_nested":
        binding = output["binding"]
        output["binding"] = {
            "outer": mapping[str(binding["outer"])],
            "inner": mapping[str(binding["inner"])],
        }
    return output, mapping


def reindex_input_objects(
    input_packet: Mapping[str, object],
    permutation: Sequence[int],
) -> dict[str, object]:
    """Relabel every raw object relation in an input packet."""

    output = copy.deepcopy(dict(input_packet))
    cardinality = output.get("cardinality")
    if not isinstance(cardinality, int):
        raise BekicBoardError("input cardinality differs before reindexing")
    _require_permutation(permutation, cardinality, "object")
    for constant in output["constants"]:
        relation = _as_relation(constant["relation"], cardinality)
        constant["relation"] = _serialize_relation(
            reindex_relation(relation, permutation)
        )
    return output


def split_contract(split: str) -> dict[str, object]:
    """Return independent scale, graph-depth, and dependency-topology cells."""

    if split == "train":
        return {
            "axis_cells": {
                "factorial": {
                    "cardinalities": (3, 4, 5),
                    "expression_depths": (5, 6),
                    "dependency_topologies": DEPENDENCY_TOPOLOGIES,
                }
            }
        }
    if split == "development":
        return {
            "axis_cells": {
                "scale_only": {
                    "cardinalities": (6, 7),
                    "expression_depths": (5, 6),
                    "dependency_topologies": DEPENDENCY_TOPOLOGIES,
                },
                "depth_only": {
                    "cardinalities": (3, 4, 5),
                    "expression_depths": (7,),
                    "dependency_topologies": DEPENDENCY_TOPOLOGIES,
                },
                "joint": {
                    "cardinalities": (6, 7),
                    "expression_depths": (7,),
                    "dependency_topologies": DEPENDENCY_TOPOLOGIES,
                },
            }
        }
    if split == "confirmation":
        return {
            "axis_cells": {
                "sealed_joint": {
                    "cardinalities": (8,),
                    "expression_depths": (8,),
                    "dependency_topologies": DEPENDENCY_TOPOLOGIES,
                }
            }
        }
    raise BekicBoardError("unknown Bekić board split")


def _eligible_axes(split: str) -> list[dict[str, object]]:
    contract = split_contract(split)
    cells = contract["axis_cells"]
    output = []
    for cell_name, cell in cells.items():
        for cardinality in cell["cardinalities"]:
            for expression_depth in cell["expression_depths"]:
                for topology in cell["dependency_topologies"]:
                    output.append(
                        {
                            "axis_cell": cell_name,
                            "cardinality": cardinality,
                            "expression_depth": expression_depth,
                            "dependency_topology": topology,
                        }
                    )
    return output


class _ProgramBuilder:
    def __init__(self) -> None:
        self.nodes: list[dict[str, object]] = []
        self._leaves: dict[tuple[str, str], str] = {}

    def _add(self, payload: dict[str, object]) -> str:
        node_id = f"canonical_node_{len(self.nodes)}"
        self.nodes.append({"id": node_id, **payload})
        return node_id

    def variable(self, variable: str) -> str:
        key = ("variable", variable)
        if key not in self._leaves:
            self._leaves[key] = self._add(
                {"kind": "variable", "variable": variable}
            )
        return self._leaves[key]

    def constant(self, constant: str) -> str:
        key = ("constant", constant)
        if key not in self._leaves:
            self._leaves[key] = self._add(
                {"kind": "constant", "constant": constant}
            )
        return self._leaves[key]

    def operation(self, operation: str, *inputs: str) -> str:
        if operation not in ALLOWED_OPERATIONS or len(inputs) != OPERATION_ARITY[operation]:
            raise BekicBoardError("builder operation differs")
        return self._add(
            {
                "kind": "operation",
                "operation": operation,
                "inputs": list(inputs),
            }
        )


def _build_equation(
    builder: _ProgramBuilder,
    *,
    own_variable: str,
    cross_variable: str | None,
    constants: Sequence[str],
    expression_depth: int,
) -> str:
    self_node = builder.variable(own_variable)
    driver = builder.operation(
        "UNION",
        builder.constant(constants[0]),
        builder.operation("CONVERSE", builder.constant(constants[1])),
    )
    seed = builder.operation(
        "UNION",
        builder.operation("IDENTITY"),
        builder.constant(constants[2]),
    )
    power = self_node
    for _ in range(expression_depth - 4):
        power = builder.operation("COMPOSE", driver, power)
    base = builder.operation("UNION", self_node, seed)
    with_power = builder.operation("UNION", base, power)
    if cross_variable is None:
        garnish_left = builder.constant(constants[3])
    else:
        garnish_left = builder.variable(cross_variable)
    garnish = builder.operation(
        "INTERSECTION",
        garnish_left,
        builder.constant(constants[3]),
    )
    return builder.operation("UNION", with_power, garnish)


def _canonical_program(
    *,
    topology: str,
    expression_depth: int,
    outer_index: int,
) -> dict[str, object]:
    if topology not in DEPENDENCY_TOPOLOGIES or expression_depth < 5:
        raise BekicBoardError("program axes differ")
    variables = ["x", "y"]
    cross = {
        "independent_dag": (None, None),
        "dag_x_to_y": (None, "x"),
        "dag_y_to_x": ("y", None),
        "mutual_cycle": ("y", "x"),
    }[topology]
    builder = _ProgramBuilder()
    x_root = _build_equation(
        builder,
        own_variable="x",
        cross_variable=cross[0],
        constants=("c0", "c1", "c2", "c3"),
        expression_depth=expression_depth,
    )
    y_root = _build_equation(
        builder,
        own_variable="y",
        cross_variable=cross[1],
        constants=("c4", "c5", "c6", "c7"),
        expression_depth=expression_depth,
    )
    outer = variables[outer_index]
    inner = variables[1 - outer_index]
    return {
        "schema": "bekic_typed_program_v1",
        "form": "bekic_nested",
        "variables": variables,
        "nodes": builder.nodes,
        "equations": [
            {"variable": "x", "root": x_root},
            {"variable": "y", "root": y_root},
        ],
        "binding": {"outer": outer, "inner": inner},
    }


def _program_depth(program: Mapping[str, object]) -> int:
    table = _node_table(program)
    roots = _equation_roots(program)
    memo: dict[str, int] = {}

    def depth(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        node = table[node_id]
        if node["kind"] != "operation" or not node.get("inputs"):
            result = 0
        else:
            result = 1 + max(depth(str(item)) for item in node["inputs"])
        memo[node_id] = result
        return result

    return max(depth(root) for root in roots.values())


def _random_relation(
    cardinality: int,
    rng: random.Random,
    density: float,
) -> Relation:
    matrix = [[0] * cardinality for _ in range(cardinality)]
    candidates = [
        (row, column)
        for row in range(cardinality)
        for column in range(cardinality)
        if row != column
    ]
    rng.shuffle(candidates)
    for row, column in candidates:
        if rng.random() < density:
            matrix[row][column] = 1
    if not any(any(row) for row in matrix):
        row, column = candidates[0]
        matrix[row][column] = 1
    return tuple(tuple(row) for row in matrix)


def _opaque_identifier(prefix: str, seed: int, index: int) -> str:
    digest = hashlib.sha256(f"{prefix}:{seed}:{index}".encode()).hexdigest()[:14]
    return f"{prefix}_{digest}"


def _build_input(
    *,
    axes: Mapping[str, object],
    seed: int,
    attempt: int,
) -> dict[str, object]:
    cardinality = int(axes["cardinality"])
    rng_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{attempt}:bekic".encode()).digest()[:8],
        "big",
    )
    rng = random.Random(rng_seed)
    constants = [
        _random_relation(cardinality, rng, 0.10 + 0.035 * (index % 4))
        for index in range(CONSTANT_COUNT)
    ]
    object_permutation = list(range(cardinality))
    rng.shuffle(object_permutation)
    constants = [
        reindex_relation(relation, object_permutation) for relation in constants
    ]
    constant_ids = [
        _opaque_identifier("constant", rng_seed, index)
        for index in range(CONSTANT_COUNT)
    ]

    canonical = _canonical_program(
        topology=str(axes["dependency_topology"]),
        expression_depth=int(axes["expression_depth"]),
        outer_index=rng.randrange(VARIABLE_COUNT),
    )
    constant_mapping = {
        f"c{index}": constant_ids[index] for index in range(CONSTANT_COUNT)
    }
    for node in canonical["nodes"]:
        if node["kind"] == "constant":
            node["constant"] = constant_mapping[str(node["constant"])]

    variable_permutation = [0, 1]
    nested, _ = reindex_program_variables(
        canonical,
        variable_permutation,
        namespace=f"{rng_seed}:variables",
    )
    simultaneous = copy.deepcopy(nested)
    simultaneous["form"] = "simultaneous"
    simultaneous.pop("binding")
    simultaneous_permutation = list(range(len(simultaneous["nodes"])))
    nested_permutation = list(range(len(nested["nodes"])))
    rng.shuffle(simultaneous_permutation)
    rng.shuffle(nested_permutation)
    simultaneous = reindex_program_nodes(
        simultaneous,
        simultaneous_permutation,
        namespace=f"{rng_seed}:simultaneous",
    )
    nested = reindex_program_nodes(
        nested,
        nested_permutation,
        namespace=f"{rng_seed}:nested",
    )
    return {
        "schema": "bekic_relational_input_v1",
        "cardinality": cardinality,
        "constants": [
            {
                "id": constant_ids[index],
                "relation": _serialize_relation(constants[index]),
            }
            for index in range(CONSTANT_COUNT)
        ],
        "representations": {
            "simultaneous": simultaneous,
            "nested": nested,
        },
    }


def _target_payload(terminal: Mapping[str, Relation], variables: Sequence[str]) -> dict[str, object]:
    return {
        "variables": [
            {
                "id": variable,
                "relation": _serialize_relation(terminal[variable]),
            }
            for variable in variables
        ]
    }


def _target_environment(
    payload: Mapping[str, object],
    cardinality: int,
) -> Environment:
    variables = payload.get("variables")
    if not isinstance(variables, list) or len(variables) != VARIABLE_COUNT:
        raise BekicBoardError("target variables differ")
    output: Environment = {}
    for item in variables:
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "relation"}
            or not isinstance(item["id"], str)
            or item["id"] in output
        ):
            raise BekicBoardError("target variable differs")
        output[item["id"]] = _as_relation(item["relation"], cardinality)
    return output


def _is_nontrivial_terminal(
    input_packet: Mapping[str, object],
    terminal: Mapping[str, Relation],
) -> bool:
    cardinality = int(input_packet["cardinality"])
    forbidden = {
        empty_relation(cardinality),
        identity_relation(cardinality),
        universal_relation(cardinality),
        *_constant_environment(input_packet).values(),
    }
    values = list(terminal.values())
    return (
        len(values) == VARIABLE_COUNT
        and values[0] != values[1]
        and all(value not in forbidden for value in values)
    )


def _row_hashes(payload: Mapping[str, object]) -> dict[str, str]:
    input_packet = payload["input"]
    targets = payload["targets"]
    simultaneous = _representation(input_packet, "simultaneous")
    receipts = {
        "input_sha256": sha256_json(input_packet),
        "targets_sha256": sha256_json(targets),
        "program_semantics_sha256": sha256_json(
            canonical_program_semantics(simultaneous)
        ),
    }
    receipts["row_sha256"] = sha256_json(
        {"payload": payload, "receipts": receipts}
    )
    return receipts


def generate_row(
    *,
    split: str,
    seed: int,
    sealed_confirmation: bool = False,
) -> dict[str, object]:
    """Generate one deterministic row, failing closed on confirmation."""

    if split == "confirmation" and not sealed_confirmation:
        raise BekicBoardError(
            "confirmation generation requires sealed_confirmation=True"
        )
    axes_options = _eligible_axes(split)
    axes = axes_options[seed % len(axes_options)]
    for attempt in range(512):
        input_packet = _build_input(axes=axes, seed=seed, attempt=attempt)
        simultaneous = evaluate_simultaneous(input_packet)
        nested = evaluate_bekic(input_packet)
        if simultaneous != nested or not _is_nontrivial_terminal(
            input_packet, simultaneous
        ):
            continue
        variables = _program_variables(
            _representation(input_packet, "simultaneous")
        )
        payload = {
            "schema": "bekic_relational_fixed_point_row_v1",
            "split": split,
            "seed": seed,
            "axes": dict(axes),
            "input": input_packet,
            "targets": _target_payload(simultaneous, variables),
        }
        row = {**payload, "hashes": _row_hashes(payload)}
        validate_row(row, sealed_confirmation=sealed_confirmation)
        return row
    raise BekicBoardError("could not generate a nontrivial Bekić episode")


def generate_rows(
    *,
    split: str,
    count: int,
    seed: int,
    sealed_confirmation: bool = False,
) -> list[dict[str, object]]:
    if count < 1:
        raise BekicBoardError("row count differs")
    if split == "confirmation" and not sealed_confirmation:
        raise BekicBoardError(
            "confirmation generation requires sealed_confirmation=True"
        )
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    cursor = 0
    while len(rows) < count:
        row_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{split}:{cursor}".encode()).digest()[:8],
            "big",
        )
        cursor += 1
        row = generate_row(
            split=split,
            seed=row_seed,
            sealed_confirmation=sealed_confirmation,
        )
        input_hash = str(row["hashes"]["input_sha256"])
        if input_hash in seen:
            continue
        seen.add(input_hash)
        rows.append(row)
    return rows


def _validate_program(
    program: Mapping[str, object],
    *,
    form: str,
    constants: Mapping[str, Relation],
) -> None:
    expected = {"schema", "form", "variables", "nodes", "equations"}
    if form == "bekic_nested":
        expected.add("binding")
    if (
        set(program) != expected
        or program.get("schema") != "bekic_typed_program_v1"
        or program.get("form") != form
    ):
        raise BekicBoardError("typed program schema differs")
    variables = _program_variables(program)
    table = _node_table(program)
    roots = _equation_roots(program)
    if set(roots) != set(variables):
        raise BekicBoardError("typed program roots differ")
    for node in table.values():
        kind = node.get("kind")
        if kind == "variable":
            if (
                set(node) != {"id", "kind", "variable"}
                or node.get("variable") not in variables
            ):
                raise BekicBoardError("typed variable node differs")
        elif kind == "constant":
            if (
                set(node) != {"id", "kind", "constant"}
                or node.get("constant") not in constants
            ):
                raise BekicBoardError("typed constant node differs")
        elif kind == "operation":
            operation = node.get("operation")
            inputs = node.get("inputs")
            if (
                set(node) != {"id", "kind", "operation", "inputs"}
                or operation not in ALLOWED_OPERATIONS
                or not isinstance(inputs, list)
                or len(inputs) != OPERATION_ARITY[str(operation)]
                or any(reference not in table for reference in inputs)
            ):
                raise BekicBoardError("typed operation node differs")
        else:
            raise BekicBoardError("typed node kind differs")
    cardinality = _require_relation(next(iter(constants.values())))
    bottom = empty_relation(cardinality)
    evaluate_program_equations(
        program,
        constants,
        {variable: bottom for variable in variables},
    )
    if form == "bekic_nested":
        binding = program.get("binding")
        if (
            not isinstance(binding, dict)
            or set(binding) != {"outer", "inner"}
            or {binding["outer"], binding["inner"]} != set(variables)
        ):
            raise BekicBoardError("typed nested binding differs")


def _variable_dependencies(program: Mapping[str, object]) -> dict[str, set[str]]:
    variables = set(_program_variables(program))
    table = _node_table(program)
    roots = _equation_roots(program)
    memo: dict[str, set[str]] = {}

    def dependencies(node_id: str) -> set[str]:
        if node_id in memo:
            return memo[node_id]
        node = table[node_id]
        if node["kind"] == "variable":
            result = {str(node["variable"])}
        elif node["kind"] == "operation":
            result = set().union(
                *(dependencies(str(item)) for item in node["inputs"])
            )
        else:
            result = set()
        memo[node_id] = result
        return result

    output = {variable: dependencies(roots[variable]) for variable in variables}
    if any(variable not in output[variable] for variable in variables):
        raise BekicBoardError("equation lost its recursive state")
    return output


def _validate_topology(program: Mapping[str, object], topology: str) -> None:
    variables = _program_variables(program)
    dependencies = _variable_dependencies(program)
    cross = (
        variables[1] in dependencies[variables[0]],
        variables[0] in dependencies[variables[1]],
    )
    expected = {
        "independent_dag": (False, False),
        "dag_x_to_y": (False, True),
        "dag_y_to_x": (True, False),
        "mutual_cycle": (True, True),
    }.get(topology)
    if cross != expected:
        raise BekicBoardError("cross-variable dependency topology differs")


def _validate_axes(split: str, axes: Mapping[str, object]) -> None:
    if set(axes) != {
        "axis_cell",
        "cardinality",
        "expression_depth",
        "dependency_topology",
    }:
        raise BekicBoardError("board axes differ")
    cells = split_contract(split)["axis_cells"]
    cell = cells.get(axes["axis_cell"]) if isinstance(cells, dict) else None
    if (
        not isinstance(cell, dict)
        or axes["cardinality"] not in cell["cardinalities"]
        or axes["expression_depth"] not in cell["expression_depths"]
        or axes["dependency_topology"] not in cell["dependency_topologies"]
    ):
        raise BekicBoardError("board axes violate the split contract")


def _reject_input_leakage(input_packet: Mapping[str, object]) -> None:
    forbidden_key_fragments = {
        "answer",
        "fixed_point",
        "halt",
        "iteration_count",
        "query",
        "schedule",
        "source",
        "target",
        "trajectory",
    }

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = str(key).lower()
                if any(fragment in lowered for fragment in forbidden_key_fragments):
                    raise BekicBoardError("input contains a forbidden oracle field")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(input_packet)


def validate_row(
    row: Mapping[str, object],
    *,
    sealed_confirmation: bool = False,
) -> None:
    """Replay all contracts and both independent oracles for one row."""

    if set(row) != {
        "schema",
        "split",
        "seed",
        "axes",
        "input",
        "targets",
        "hashes",
    } or row.get("schema") != "bekic_relational_fixed_point_row_v1":
        raise BekicBoardError("Bekić row schema differs")
    split = row.get("split")
    if split == "confirmation" and not sealed_confirmation:
        raise BekicBoardError(
            "confirmation validation requires sealed_confirmation=True"
        )
    if split not in {"train", "development", "confirmation"}:
        raise BekicBoardError("Bekić row split differs")
    axes = row.get("axes")
    input_packet = row.get("input")
    targets = row.get("targets")
    hashes = row.get("hashes")
    if (
        not isinstance(axes, dict)
        or not isinstance(input_packet, dict)
        or not isinstance(targets, dict)
        or not isinstance(hashes, dict)
    ):
        raise BekicBoardError("Bekić row payload differs")
    _validate_axes(str(split), axes)
    if (
        set(input_packet) != {
            "schema",
            "cardinality",
            "constants",
            "representations",
        }
        or input_packet.get("schema") != "bekic_relational_input_v1"
        or input_packet.get("cardinality") != axes["cardinality"]
    ):
        raise BekicBoardError("Bekić input schema differs")
    _reject_input_leakage(input_packet)
    constants = _constant_environment(input_packet)
    simultaneous_program = _representation(input_packet, "simultaneous")
    nested_program = _representation(input_packet, "nested")
    _validate_program(
        simultaneous_program,
        form="simultaneous",
        constants=constants,
    )
    _validate_program(
        nested_program,
        form="bekic_nested",
        constants=constants,
    )
    if _program_variables(simultaneous_program) != _program_variables(nested_program):
        raise BekicBoardError("paired program variable order differs")
    if canonical_program_semantics(
        simultaneous_program
    ) != canonical_program_semantics(nested_program):
        raise BekicBoardError("paired program semantics differ")
    if _program_depth(simultaneous_program) != axes["expression_depth"]:
        raise BekicBoardError("program expression depth differs")
    _validate_topology(
        simultaneous_program,
        str(axes["dependency_topology"]),
    )

    cardinality = int(input_packet["cardinality"])
    target_environment = _target_environment(targets, cardinality)
    variables = _program_variables(simultaneous_program)
    if set(target_environment) != set(variables):
        raise BekicBoardError("target variable ids differ")
    simultaneous = evaluate_simultaneous(input_packet)
    nested = evaluate_bekic(input_packet)
    if simultaneous != nested:
        raise BekicBoardError("simultaneous and nested Bekić oracles disagree")
    if target_environment != simultaneous:
        raise BekicBoardError("stored targets differ from independent oracles")
    if not _is_nontrivial_terminal(input_packet, simultaneous):
        raise BekicBoardError("terminal relations admit a direct packet shortcut")

    payload = {key: copy.deepcopy(row[key]) for key in row if key != "hashes"}
    expected_hashes = _row_hashes(payload)
    if hashes != expected_hashes:
        raise BekicBoardError("Bekić row hashes differ")
