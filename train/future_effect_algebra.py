"""Compact affine operators for future-effect reasoning experiments.

The two-register microcode domain is affine. Each language event denotes a
3x3 homogeneous operator over ``[register_0, register_1, 1]`` and a complete
event sequence is the chronological matrix product. This gives future work a
precise source-dropped target: the carried object is what the text *does* to
all possible states and queries, not a generated rationale or an unconstrained
latent vector.

This module is an exact research contract, not a language compiler and not a
reasoning result. Structured opcodes are used only to validate the algebra;
an admitted model must infer operators from text and pass counterfactual
normal/zero/shuffle controls.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from categorical_microcode import OPCODES, QUERIES


def _name(value, vocabulary):
    return vocabulary[int(value)] if isinstance(value, int) else str(value)


def operation_operator(opcode, value=0, *, dtype=torch.float64, device=None):
    """Return the homogeneous affine operator for one chronological event."""
    name = _name(opcode, OPCODES)
    value = int(value)
    operator = torch.eye(3, dtype=dtype, device=device)
    if name.startswith("add_"):
        operator[int(name[-1]), 2] = value
    elif name.startswith("sub_"):
        operator[int(name[-1]), 2] = -value
    elif name.startswith("move_"):
        source, target = map(int, name.split("_")[1:])
        operator[source, 2] = -value
        operator[target, 2] = value
    elif name.startswith("merge_"):
        source, target = map(int, name.split("_")[1:])
        operator[target, source] = 1
    elif name == "swap":
        operator = torch.tensor(
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=dtype, device=device,
        )
    else:
        raise ValueError("unknown opcode {}".format(name))
    return operator


def query_operator(query, *, dtype=torch.float64, device=None):
    """Return a row operator that reads the requested scalar from final state."""
    name = _name(query, QUERIES)
    rows = {
        "read_0": (1, 0, 0),
        "read_1": (0, 1, 0),
        "sum": (1, 1, 0),
        "difference_0_1": (1, -1, 0),
        "difference_1_0": (-1, 1, 0),
    }
    if name not in rows:
        raise ValueError("unknown query {}".format(name))
    return torch.tensor(rows[name], dtype=dtype, device=device)


def compose_operators(operators: Iterable[torch.Tensor], *, dtype=torch.float64, device=None):
    """Compose chronological operators into one source-droppable operator."""
    total = torch.eye(3, dtype=dtype, device=device)
    for operator in operators:
        if operator.shape != (3, 3):
            raise ValueError("operators must be 3x3")
        total = operator.to(dtype=dtype, device=device) @ total
    return total


def program_operator(opcodes, values, *, dtype=torch.float64, device=None):
    if len(opcodes) != len(values):
        raise ValueError("opcode/value lengths differ")
    return compose_operators(
        (
            operation_operator(opcode, value, dtype=dtype, device=device)
            for opcode, value in zip(opcodes, values)
        ),
        dtype=dtype,
        device=device,
    )


def execute_operator(initial_values, opcodes, values, query):
    """Execute through the compact operator; intended for exact CPU audits."""
    if len(initial_values) != 2:
        raise ValueError("exactly two initial values are required")
    state = torch.tensor([initial_values[0], initial_values[1], 1], dtype=torch.int64)
    operator = program_operator(opcodes, values, dtype=torch.int64)
    readout = query_operator(query, dtype=torch.int64)
    return int((readout @ operator @ state).item())


def effect_signature(operator, states=None, queries=None):
    """Identify an operator by its effects over state/query probe bases.

    With the default standard bases this signature is exactly the full linear
    operator. Larger or held-out probe banks can test whether a learned compact
    thought preserves future behavior without inspecting its coordinates.
    """
    if operator.shape != (3, 3):
        raise ValueError("operator must be 3x3")
    states = torch.eye(3, dtype=operator.dtype, device=operator.device) if states is None else states
    queries = torch.eye(3, dtype=operator.dtype, device=operator.device) if queries is None else queries
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("states must have shape [probes, 3]")
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("queries must have shape [probes, 3]")
    return queries @ operator @ states.transpose(0, 1)
