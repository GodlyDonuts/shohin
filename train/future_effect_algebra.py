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


def redundant_probe_bank(*, dtype=torch.float64, device=None):
    """Return fixed overcomplete state/query probes for effect coding.

    These probes are deliberately not learned. Their 64 scalar effects encode
    a nine-coordinate operator with enough redundancy to expose an error
    syndrome and correct one arbitrarily corrupted scalar in the exact CPU
    contract.
    """
    hadamard = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, -1, 1, -1, -1, 1, -1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1, -1, 1, -1, 1, 1, -1],
    ], dtype=dtype, device=device)
    states = hadamard[:, [1, 2, 0]]
    queries = hadamard[:, [3, 4, 0]]
    return states, queries


def effect_measurement_matrix(states, queries):
    """Linear map from row-major operator coordinates to flattened effects."""
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("states must have shape [probes, 3]")
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("queries must have shape [probes, 3]")
    if states.device != queries.device or states.dtype != queries.dtype:
        raise ValueError("state and query probes must share dtype and device")
    return torch.einsum("ia,jb->ijab", queries, states).reshape(-1, 9)


def random_orthogonal_measurement_matrix(
    channels=64, coordinates=9, *, seed=20260714, scale=1.0,
    dtype=torch.float64, device=None,
):
    """Return a deterministic random code with orthogonal columns.

    This is useful as an exact control and, more importantly, for proving a
    limitation of representation-only comparisons: any two full-rank linear
    operator codes are connected by a fixed linear transport. A random code is
    therefore not a distinct reasoning mechanism when both arms decode to the
    same operator and use the same composition law.
    """
    channels = int(channels)
    coordinates = int(coordinates)
    if channels < coordinates or coordinates <= 0:
        raise ValueError("orthogonal code requires channels >= coordinates > 0")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    raw = torch.randn((channels, coordinates), generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(raw, mode="reduced")
    return (float(scale) * basis).to(dtype=dtype, device=device)


def encode_operator_code(operator, measurements):
    """Encode a 3x3 operator with an arbitrary full-rank linear code."""
    if operator.shape != (3, 3):
        raise ValueError("operator must be 3x3")
    if measurements.ndim != 2 or measurements.shape[1] != 9:
        raise ValueError("measurements must have shape [channels, 9]")
    coordinates = operator.reshape(-1).to(
        dtype=measurements.dtype, device=measurements.device,
    )
    return measurements @ coordinates


def decode_operator_code(code, measurements):
    """Decode the least-squares operator represented by a linear code."""
    if measurements.ndim != 2 or measurements.shape[1] != 9:
        raise ValueError("measurements must have shape [channels, 9]")
    if code.shape != (measurements.shape[0],):
        raise ValueError("code must have one scalar per measurement channel")
    coordinates = torch.linalg.lstsq(
        measurements, code.to(dtype=measurements.dtype, device=measurements.device),
    ).solution
    return coordinates.reshape(3, 3)


def transport_operator_code(code, source_measurements, target_measurements):
    """Move a valid code between arbitrary full-rank linear codebooks."""
    operator = decode_operator_code(code, source_measurements)
    return encode_operator_code(operator, target_measurements)


def compose_operator_codes(left_code, right_code, measurements):
    """Compose chronological code chunks after exact decode and re-encode."""
    left = decode_operator_code(left_code, measurements)
    right = decode_operator_code(right_code, measurements)
    return encode_operator_code(right @ left, measurements)


def decode_effect_signature(signature, states, queries, max_outliers=0):
    """Project a redundant effect code onto its nearest valid operator.

    ``max_outliers=1`` enumerates one omitted scalar and minimizes the residual
    after trimming the largest error. It is intentionally small and exact for
    mechanism tests; a future neural implementation may use a learned or
    vectorized decoder but must match this reference on admitted examples.
    """
    expected = (queries.shape[0], states.shape[0])
    if signature.shape != expected:
        raise ValueError("signature must have shape {}".format(expected))
    max_outliers = int(max_outliers)
    measurements = effect_measurement_matrix(states, queries)
    values = signature.reshape(-1).to(dtype=measurements.dtype, device=measurements.device)
    if max_outliers not in (0, 1):
        raise ValueError("reference decoder supports zero or one outlier")
    candidates = [None] if max_outliers == 0 else list(range(values.numel()))
    best = None
    for omitted in candidates:
        mask = torch.ones(values.numel(), dtype=torch.bool, device=values.device)
        if omitted is not None:
            mask[omitted] = False
        solution = torch.linalg.lstsq(measurements[mask], values[mask]).solution
        projected = measurements @ solution
        residual = values - projected
        trimmed = residual.square().sort().values[:values.numel() - max_outliers]
        score = trimmed.mean()
        tie_break = residual.square().mean()
        key = (float(score.item()), float(tie_break.item()))
        if best is None or key < best[0]:
            best = (key, solution, projected, residual, omitted)
    _, solution, projected, residual, omitted = best
    return {
        "operator": solution.reshape(3, 3),
        "projected_signature": projected.reshape(expected),
        "syndrome": residual.reshape(expected),
        "omitted_index": omitted,
    }
