"""Sound affine ambiguity propagation for R10 ACAW mechanics.

The workspace carries an affine hull of every complete 3x3 transform still
consistent with local candidate sets. Chronological products are propagated by
an exact rational overapproximation. A query may be answered only when every
ambiguity direction is annihilated by that query and initial state.

This module deliberately retains every unresolved source in the hot frontier
and every source's immutable retrieval pointer. It does not claim that one
witness per affine basis direction is sufficient for replay. That stronger
context-scaling contract requires a separate provenance proof.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from version_space_product_tree import (
    ExactAffineTransform,
    as_exact_transform,
    operation_transform,
    query_row,
)


Scalar = Fraction
Vector = tuple[Scalar, ...]
RationalMatrix = tuple[
    tuple[Scalar, Scalar, Scalar],
    tuple[Scalar, Scalar, Scalar],
    tuple[Scalar, Scalar, Scalar],
]


def _exact_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError("{} must be an integer, not bool".format(name))
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError("{} must be an exact integer".format(name)) from error


def _matrix(value) -> RationalMatrix:
    if isinstance(value, ExactAffineTransform):
        value = value.rows
    if hasattr(value, "tolist"):
        value = value.tolist()
    rows = tuple(tuple(Fraction(entry) for entry in row) for row in value)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("ambiguity matrices must be 3x3")
    return rows  # type: ignore[return-value]


def _flatten(matrix: RationalMatrix) -> Vector:
    return tuple(entry for row in matrix for entry in row)


def _unflatten(vector: Sequence[Scalar]) -> RationalMatrix:
    if len(vector) != 9:
        raise ValueError("a flattened 3x3 matrix must have nine entries")
    values = tuple(Fraction(value) for value in vector)
    return (
        values[0:3],
        values[3:6],
        values[6:9],
    )  # type: ignore[return-value]


def _subtract(left: RationalMatrix, right: RationalMatrix) -> RationalMatrix:
    return tuple(
        tuple(left[row][column] - right[row][column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _matmul(left: RationalMatrix, right: RationalMatrix) -> RationalMatrix:
    return tuple(
        tuple(
            sum(
                (left[row][inner] * right[inner][column] for inner in range(3)),
                Fraction(0),
            )
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _rref_basis(vectors: Iterable[Sequence[Scalar]]) -> tuple[Vector, ...]:
    """Return a deterministic reduced-row-echelon basis for a rational span."""
    rows = [list(map(Fraction, vector)) for vector in vectors]
    if any(len(row) != 9 for row in rows):
        raise ValueError("ambiguity vectors must have nine entries")
    rows = [row for row in rows if any(row)]
    pivot_row = 0
    for column in range(9):
        pivot = next(
            (index for index in range(pivot_row, len(rows)) if rows[index][column]),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        divisor = rows[pivot_row][column]
        rows[pivot_row] = [value / divisor for value in rows[pivot_row]]
        for index in range(len(rows)):
            if index == pivot_row or not rows[index][column]:
                continue
            multiplier = rows[index][column]
            rows[index] = [
                value - multiplier * pivot_value
                for value, pivot_value in zip(rows[index], rows[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return tuple(tuple(row) for row in rows[:pivot_row])


def _canonical_matrices(matrices: Iterable[RationalMatrix]) -> tuple[RationalMatrix, ...]:
    return tuple(_unflatten(vector) for vector in _rref_basis(_flatten(item) for item in matrices))


def _in_span(vector: RationalMatrix, basis: Sequence[RationalMatrix]) -> bool:
    current = _rref_basis(_flatten(item) for item in basis)
    extended = _rref_basis([*current, _flatten(vector)])
    return len(current) == len(extended)


def _matrix_vector(matrix: RationalMatrix, vector: Sequence[Scalar]) -> tuple[Scalar, ...]:
    if len(vector) != 3:
        raise ValueError("homogeneous state must have three coordinates")
    values = tuple(Fraction(value) for value in vector)
    return tuple(
        sum((matrix[row][column] * values[column] for column in range(3)), Fraction(0))
        for row in range(3)
    )


def _query_effect(
    matrix: RationalMatrix,
    initial_values: Sequence[int],
    query,
) -> Scalar:
    if len(initial_values) != 2:
        raise ValueError("exactly two initial values are required")
    state = (
        Fraction(_exact_int(initial_values[0], "initial value")),
        Fraction(_exact_int(initial_values[1], "initial value")),
        Fraction(1),
    )
    output = _matrix_vector(matrix, state)
    row = query_row(query)
    return Fraction(row[0]) * output[0] + Fraction(row[1]) * output[1]


@dataclass(frozen=True)
class AffineQueryCertificate:
    certified: bool
    answer: Scalar | None
    ambiguity_rank: int
    hot_context_evictable: bool

    @property
    def integer_answer(self) -> int | None:
        if self.answer is None or self.answer.denominator != 1:
            return None
        return int(self.answer)


@dataclass(frozen=True)
class AffineAmbiguityWorkspace:
    """A sound affine overapproximation with conservative source provenance."""

    anchor: RationalMatrix
    basis: tuple[RationalMatrix, ...]
    retained_source_indices: tuple[int, ...]
    retrieval_source_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        anchor = _matrix(self.anchor)
        basis = _canonical_matrices(_matrix(item) for item in self.basis)
        retained = tuple(
            sorted({_exact_int(index, "source index") for index in self.retained_source_indices})
        )
        retrieval = tuple(
            sorted({_exact_int(index, "retrieval source index") for index in self.retrieval_source_indices})
        )
        if anchor[2] != (Fraction(0), Fraction(0), Fraction(1)):
            raise ValueError("affine anchors require homogeneous bottom row (0, 0, 1)")
        if any(direction[2] != (Fraction(0), Fraction(0), Fraction(0)) for direction in basis):
            raise ValueError("affine ambiguity directions require a zero bottom row")
        if len(basis) > 6:
            raise ValueError("a homogeneous 2D affine ambiguity rank cannot exceed six")
        if any(index not in retrieval for index in retained):
            raise ValueError("hot retained sources must have immutable retrieval pointers")
        if not basis and retained:
            raise ValueError("rank-zero workspaces must evict all sources from the hot frontier")
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "retained_source_indices", retained)
        object.__setattr__(self, "retrieval_source_indices", retrieval)

    @property
    def ambiguity_rank(self) -> int:
        return len(self.basis)

    @property
    def hot_context_evictable(self) -> bool:
        return self.ambiguity_rank == 0

    @property
    def source_droppable(self) -> bool:
        """Compatibility name for candidate-conditional hot eviction only."""
        return self.hot_context_evictable

    def contains(self, transform) -> bool:
        difference = _subtract(_matrix(as_exact_transform(transform)), self.anchor)
        return _in_span(difference, self.basis)

    def query_certificate(self, initial_values: Sequence[int], query) -> AffineQueryCertificate:
        certified = all(
            _query_effect(direction, initial_values, query) == 0
            for direction in self.basis
        )
        answer = _query_effect(self.anchor, initial_values, query) if certified else None
        return AffineQueryCertificate(
            certified=certified,
            answer=answer,
            ambiguity_rank=self.ambiguity_rank,
            hot_context_evictable=self.hot_context_evictable,
        )

    def query_set_certificate(
        self,
        initial_values: Sequence[int],
        queries: Iterable[object],
    ) -> AffineQueryCertificate:
        """Certify only when every hull transform and candidate query agree."""
        queries = tuple(queries)
        if not queries:
            raise ValueError("a query candidate set must be nonempty")
        direction_invariant = all(
            _query_effect(direction, initial_values, query) == 0
            for direction in self.basis
            for query in queries
        )
        anchor_answers = {
            _query_effect(self.anchor, initial_values, query) for query in queries
        }
        certified = direction_invariant and len(anchor_answers) == 1
        answer = next(iter(anchor_answers)) if certified else None
        return AffineQueryCertificate(
            certified=certified,
            answer=answer,
            ambiguity_rank=self.ambiguity_rank,
            hot_context_evictable=self.hot_context_evictable,
        )


def workspace_from_transforms(
    transforms: Iterable[ExactAffineTransform],
    *,
    source_index: int | None = None,
) -> AffineAmbiguityWorkspace:
    unique = tuple(sorted({as_exact_transform(transform) for transform in transforms}))
    if not unique:
        raise ValueError("an ambiguity workspace requires at least one transform")
    anchor = _matrix(unique[0])
    basis = _canonical_matrices(
        _subtract(_matrix(transform), anchor) for transform in unique[1:]
    )
    if source_index is None:
        raise ValueError("leaf workspaces require an immutable retrieval source index")
    source_index = _exact_int(source_index, "source index")
    retained = (source_index,) if basis else ()
    return AffineAmbiguityWorkspace(anchor, basis, retained, (source_index,))


def workspace_from_operations(
    candidates: Iterable[tuple[object, int]],
    *,
    source_index: int | None = None,
) -> AffineAmbiguityWorkspace:
    return workspace_from_transforms(
        (operation_transform(opcode, value) for opcode, value in candidates),
        source_index=source_index,
    )


def compose_workspaces(
    earlier: AffineAmbiguityWorkspace,
    later: AffineAmbiguityWorkspace,
) -> AffineAmbiguityWorkspace:
    """Return a sound hull for every chronological ``later @ earlier`` product."""
    if not isinstance(earlier, AffineAmbiguityWorkspace) or not isinstance(
        later, AffineAmbiguityWorkspace,
    ):
        raise TypeError("workspace composition requires two ambiguity workspaces")
    anchor = _matmul(later.anchor, earlier.anchor)
    generators = []
    generators.extend(_matmul(later.anchor, direction) for direction in earlier.basis)
    generators.extend(_matmul(direction, earlier.anchor) for direction in later.basis)
    generators.extend(
        _matmul(later_direction, earlier_direction)
        for later_direction in later.basis
        for earlier_direction in earlier.basis
    )
    basis = _canonical_matrices(generators)
    retained = () if not basis else tuple(sorted(set(
        earlier.retained_source_indices + later.retained_source_indices
    )))
    retrieval = tuple(sorted(set(
        earlier.retrieval_source_indices + later.retrieval_source_indices
    )))
    return AffineAmbiguityWorkspace(anchor, basis, retained, retrieval)


def build_workspace(
    candidate_sets: Iterable[Iterable[tuple[object, int]]],
    *,
    start: int = 0,
) -> AffineAmbiguityWorkspace:
    candidate_sets = tuple(tuple(candidates) for candidates in candidate_sets)
    if not candidate_sets:
        raise ValueError("at least one event candidate set is required")
    start = _exact_int(start, "source start")
    level = [
        workspace_from_operations(candidates, source_index=start + offset)
        for offset, candidates in enumerate(candidate_sets)
    ]
    while len(level) > 1:
        next_level = []
        for offset in range(0, len(level), 2):
            if offset + 1 == len(level):
                next_level.append(level[offset])
            else:
                next_level.append(compose_workspaces(level[offset], level[offset + 1]))
        level = next_level
    return level[0]
