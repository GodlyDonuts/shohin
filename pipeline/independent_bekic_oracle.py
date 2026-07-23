"""Independent set-based oracle for contextual Bekić program boards.

This module deliberately does not import the board's matrix relation
implementation. Its purpose is to catch shared primitive and fixed-point bugs,
not to provide a score-bearing executor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


Pair = tuple[int, int]
SetRelation = frozenset[Pair]
MatrixRelation = tuple[tuple[int, ...], ...]
Environment = dict[str, MatrixRelation]


class IndependentBekicOracleError(ValueError):
    """Raised when an oracle packet or program violates its contract."""


def _matrix_to_set(value: object, cardinality: int) -> SetRelation:
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
        raise IndependentBekicOracleError("relation matrix differs")
    return frozenset(
        (row, column)
        for row, values in enumerate(value)
        for column, bit in enumerate(values)
        if bit
    )


def _set_to_matrix(value: SetRelation, cardinality: int) -> MatrixRelation:
    return tuple(
        tuple(int((row, column) in value) for column in range(cardinality))
        for row in range(cardinality)
    )


def _constants(
    packet: Mapping[str, object],
) -> tuple[int, dict[str, SetRelation]]:
    cardinality = packet.get("cardinality")
    items = packet.get("constants")
    if (
        not isinstance(cardinality, int)
        or cardinality < 1
        or not isinstance(items, list)
    ):
        raise IndependentBekicOracleError("packet constants differ")
    output: dict[str, SetRelation] = {}
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "relation"}
            or not isinstance(item["id"], str)
            or item["id"] in output
        ):
            raise IndependentBekicOracleError("constant entry differs")
        output[item["id"]] = _matrix_to_set(
            item["relation"],
            cardinality,
        )
    return cardinality, output


def _compose(left: SetRelation, right: SetRelation) -> SetRelation:
    right_by_middle: dict[int, set[int]] = {}
    for middle, destination in right:
        right_by_middle.setdefault(middle, set()).add(destination)
    return frozenset(
        (source, destination)
        for source, middle in left
        for destination in right_by_middle.get(middle, ())
    )


def _primitive(
    kind: str,
    arguments: Sequence[SetRelation],
    cardinality: int,
) -> SetRelation:
    if kind == "IDENTITY":
        return frozenset((index, index) for index in range(cardinality))
    if kind == "UNION" and len(arguments) == 2:
        return arguments[0] | arguments[1]
    if kind == "INTERSECTION" and len(arguments) == 2:
        return arguments[0] & arguments[1]
    if kind == "COMPOSE" and len(arguments) == 2:
        return _compose(arguments[0], arguments[1])
    if kind == "CONVERSE" and len(arguments) == 1:
        return frozenset((right, left) for left, right in arguments[0])
    raise IndependentBekicOracleError("primitive application differs")


def _expression(
    expression: Mapping[str, object],
    constants: Mapping[str, SetRelation],
    environment: Mapping[str, SetRelation],
    cardinality: int,
) -> SetRelation:
    kind = expression.get("kind")
    if kind == "VARIABLE":
        variable = expression.get("variable")
        if variable not in environment:
            raise IndependentBekicOracleError("variable is unbound")
        return environment[str(variable)]
    if kind == "CONSTANT":
        constant = expression.get("constant")
        if constant not in constants:
            raise IndependentBekicOracleError("constant is unbound")
        return constants[str(constant)]
    if kind == "IDENTITY":
        return _primitive("IDENTITY", (), cardinality)
    if kind == "CONVERSE":
        child = expression.get("child")
        if not isinstance(child, dict):
            raise IndependentBekicOracleError("converse child differs")
        return _primitive(
            "CONVERSE",
            (_expression(child, constants, environment, cardinality),),
            cardinality,
        )
    if kind in {"UNION", "INTERSECTION", "COMPOSE"}:
        children = expression.get("children")
        if (
            not isinstance(children, list)
            or len(children) != 2
            or any(not isinstance(child, dict) for child in children)
        ):
            raise IndependentBekicOracleError("binary children differ")
        return _primitive(
            str(kind),
            tuple(
                _expression(
                    child,
                    constants,
                    environment,
                    cardinality,
                )
                for child in children
            ),
            cardinality,
        )
    if kind == "LFP":
        variable = expression.get("variable")
        body = expression.get("body")
        if not isinstance(variable, str) or not isinstance(body, dict):
            raise IndependentBekicOracleError("LFP syntax differs")
        current: SetRelation = frozenset()
        for _ in range(cardinality * cardinality + 2):
            scoped = dict(environment)
            scoped[variable] = current
            proposal = _expression(
                body,
                constants,
                scoped,
                cardinality,
            )
            if proposal == current:
                return proposal
            if not current <= proposal:
                raise IndependentBekicOracleError("LFP is not monotone")
            current = proposal
        raise IndependentBekicOracleError("LFP did not converge")
    raise IndependentBekicOracleError("expression kind differs")


def evaluate_simultaneous_independently(
    packet: Mapping[str, object],
) -> Environment:
    cardinality, constants = _constants(packet)
    program = packet.get("program")
    if (
        not isinstance(program, dict)
        or program.get("form") != "simultaneous_lfp"
        or not isinstance(program.get("variables"), list)
        or len(program["variables"]) != 2
        or not isinstance(program.get("equations"), list)
        or len(program["equations"]) != 2
    ):
        raise IndependentBekicOracleError("simultaneous program differs")
    variables = tuple(str(item) for item in program["variables"])
    equations = {
        str(item["variable"]): item["expression"] for item in program["equations"]
    }
    if set(equations) != set(variables) or any(
        not isinstance(expression, dict) for expression in equations.values()
    ):
        raise IndependentBekicOracleError("simultaneous equations differ")
    current = {variable: frozenset() for variable in variables}
    for _ in range(2 * cardinality * cardinality + 2):
        proposal = {
            variable: _expression(
                equations[variable],
                constants,
                current,
                cardinality,
            )
            for variable in variables
        }
        if proposal == current:
            return {
                variable: _set_to_matrix(proposal[variable], cardinality)
                for variable in variables
            }
        if any(not current[variable] <= proposal[variable] for variable in variables):
            raise IndependentBekicOracleError("simultaneous program is not monotone")
        current = proposal
    raise IndependentBekicOracleError("simultaneous fixed point did not converge")


def evaluate_nested_independently(
    packet: Mapping[str, object],
) -> Environment:
    cardinality, constants = _constants(packet)
    program = packet.get("program")
    if (
        not isinstance(program, dict)
        or program.get("form") != "nested_mu"
        or not isinstance(program.get("variables"), list)
        or len(program["variables"]) != 2
        or not isinstance(program.get("expression"), dict)
    ):
        raise IndependentBekicOracleError("nested program differs")
    expression = program["expression"]
    if (
        expression.get("kind") != "LET"
        or not isinstance(expression.get("variable"), str)
        or not isinstance(expression.get("value"), dict)
        or not isinstance(expression.get("body"), dict)
        or expression["body"].get("kind") != "PAIR"
        or not isinstance(expression["body"].get("entries"), list)
    ):
        raise IndependentBekicOracleError("nested Bekić syntax differs")
    outer = str(expression["variable"])
    outer_value = _expression(
        expression["value"],
        constants,
        {},
        cardinality,
    )
    scope = {outer: outer_value}
    output: Environment = {}
    for entry in expression["body"]["entries"]:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("variable"), str)
            or not isinstance(entry.get("expression"), dict)
        ):
            raise IndependentBekicOracleError("nested result entry differs")
        output[str(entry["variable"])] = _set_to_matrix(
            _expression(
                entry["expression"],
                constants,
                scope,
                cardinality,
            ),
            cardinality,
        )
    if set(output) != {str(item) for item in program["variables"]}:
        raise IndependentBekicOracleError("nested result variables differ")
    return output
