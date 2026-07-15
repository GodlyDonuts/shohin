"""Exact version-space folding for chronological affine microcode.

Each source event owns a finite set of lawful affine hypotheses. Product-tree
parents form every chronological product (later @ earlier), deduplicate equal
complete transforms, and bind every supporting derivation into a fixed-size
commitment. Exact replay comes from the retained factorized children, never a
flattened opcode witness. A subtree may evict its hot source only when its
complete version space is known and contains exactly one transform.

The cap is deliberately fail closed. Once more than ``K`` distinct transforms
are observed, the node becomes overflowed and discards its incomplete candidate
sample. Its exact factorized children remain available, so already certified
singleton siblings do not need their hot source resurrected. Overflow is sticky
under ordinary composition; only a monotone leaf refinement followed by exact
path recomputation can clear it. This is a pure Python integer mechanics
contract, not a neural compiler or evidence that irreversible source deletion
works from text.
"""

from __future__ import annotations

import hashlib
import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from categorical_microcode import OPCODES, QUERIES


DEFAULT_CAP = 32
NUMERIC_OPCODES = frozenset((
    "add_0", "add_1", "sub_0", "sub_1", "move_0_1", "move_1_0",
))
STRUCTURAL_OPCODES = frozenset(("merge_0_1", "merge_1_0", "swap"))

MatrixRows = tuple[
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]
AtomLabels = tuple[str, ...]


def _exact_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError("{} must be an integer, not bool".format(name))
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError("{} must be an exact integer".format(name)) from error


def _matrix_rows(rows) -> MatrixRows:
    if hasattr(rows, "tolist"):
        rows = rows.tolist()
    try:
        normalized = tuple(
            tuple(_exact_int(value, "matrix entry") for value in row)
            for row in rows
        )
    except TypeError as error:
        raise TypeError("matrix must be an iterable of rows") from error
    if len(normalized) != 3 or any(len(row) != 3 for row in normalized):
        raise ValueError("affine transforms must be 3x3")
    return normalized  # type: ignore[return-value]


def _matmul(left: MatrixRows, right: MatrixRows) -> MatrixRows:
    return tuple(
        tuple(sum(left[row][inner] * right[inner][column] for inner in range(3))
              for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


@dataclass(frozen=True, order=True)
class ExactAffineTransform:
    """An exact, microcode-reachable integral affine state transform.

    The categorical opcode family generates integral affine maps whose linear
    2x2 block is unimodular.  Requiring determinant ``+/-1`` keeps every
    admitted transform invertible and preserves that lawful family under
    product-tree composition.
    """

    rows: MatrixRows

    def __post_init__(self) -> None:
        rows = _matrix_rows(self.rows)
        if rows[2] != (0, 0, 1):
            raise ValueError("homogeneous affine bottom row must be (0, 0, 1)")
        determinant = rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
        if determinant not in (-1, 1):
            raise ValueError("lawful microcode transforms must be unimodular")
        object.__setattr__(self, "rows", rows)

    @classmethod
    def identity(cls) -> "ExactAffineTransform":
        return cls(((1, 0, 0), (0, 1, 0), (0, 0, 1)))

    @property
    def flat(self) -> tuple[int, ...]:
        return tuple(value for row in self.rows for value in row)

    def followed_by(self, later: "ExactAffineTransform") -> "ExactAffineTransform":
        """Return the chronological product for ``self`` and then ``later``."""
        if not isinstance(later, ExactAffineTransform):
            raise TypeError("later transform must be ExactAffineTransform")
        return ExactAffineTransform(_matmul(later.rows, self.rows))

    def apply(self, initial_values: Sequence[int]) -> tuple[int, int]:
        if len(initial_values) != 2:
            raise ValueError("exactly two initial values are required")
        state = (
            _exact_int(initial_values[0], "initial value"),
            _exact_int(initial_values[1], "initial value"),
            1,
        )
        output = tuple(
            sum(self.rows[row][column] * state[column] for column in range(3))
            for row in range(3)
        )
        if output[2] != 1:
            raise AssertionError("lawful affine transform changed homogeneous coordinate")
        return output[0], output[1]

    def answer(self, initial_values: Sequence[int], query) -> int:
        state = self.apply(initial_values)
        row = query_row(query)
        return row[0] * state[0] + row[1] * state[1]


def as_exact_transform(transform) -> ExactAffineTransform:
    if isinstance(transform, ExactAffineTransform):
        return transform
    return ExactAffineTransform(transform)


def _vocabulary_name(value, vocabulary: tuple[str, ...], kind: str) -> str:
    if isinstance(value, str):
        name = value
    else:
        index = _exact_int(value, "{} id".format(kind))
        if not 0 <= index < len(vocabulary):
            raise ValueError("{} id is out of range".format(kind))
        name = vocabulary[index]
    if name not in vocabulary:
        raise ValueError("unknown {} {}".format(kind, name))
    return name


def operation_transform(opcode, value=0) -> ExactAffineTransform:
    """Return the exact integer transform for one categorical opcode."""
    name = _vocabulary_name(opcode, OPCODES, "opcode")
    value = _exact_int(value, "operation value")
    rows = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if name.startswith("add_"):
        rows[int(name[-1])][2] = value
    elif name.startswith("sub_"):
        rows[int(name[-1])][2] = -value
    elif name.startswith("move_"):
        source, target = map(int, name.split("_")[1:])
        rows[source][2] = -value
        rows[target][2] = value
    elif name.startswith("merge_"):
        if value != 0:
            raise ValueError("structural opcodes require value 0")
        source, target = map(int, name.split("_")[1:])
        rows[target][source] = 1
    elif name == "swap":
        if value != 0:
            raise ValueError("structural opcodes require value 0")
        rows = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    else:  # pragma: no cover - guarded by the shared vocabulary
        raise ValueError("unknown opcode {}".format(name))
    return ExactAffineTransform(rows)


def query_row(query) -> tuple[int, int]:
    name = _vocabulary_name(query, QUERIES, "query")
    rows = {
        "read_0": (1, 0),
        "read_1": (0, 1),
        "sum": (1, 1),
        "difference_0_1": (1, -1),
        "difference_1_0": (-1, 1),
    }
    return rows[name]


def chronological_compose(
    transforms: Iterable[ExactAffineTransform],
) -> ExactAffineTransform:
    total = ExactAffineTransform.identity()
    for transform in transforms:
        total = total.followed_by(as_exact_transform(transform))
    return total


def _normalize_atom_labels(labels: str | Iterable[str]) -> AtomLabels:
    labels = (labels,) if isinstance(labels, str) else tuple(labels)
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("candidate atoms require one or more nonempty labels")
    return labels


def _commit(domain: str, *parts: object) -> str:
    """Return a length-delimited SHA-256 commitment over canonical text parts."""
    digest = hashlib.sha256()
    for part in (domain, *parts):
        payload = str(part).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _validate_commitment(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("support commitments must be 64-character SHA-256 hex")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("support commitments must be SHA-256 hex") from error
    return value.lower()


def _transform_key(transform: ExactAffineTransform) -> str:
    return ",".join(map(str, transform.flat))


def _aggregate_supports(domain: str, commitments: Iterable[str]) -> str:
    commitments = tuple(sorted({_validate_commitment(item) for item in commitments}))
    if not commitments:
        raise ValueError("a transform must retain at least one support commitment")
    if len(commitments) == 1:
        return commitments[0]
    return _commit(domain, *commitments)


@dataclass(frozen=True)
class TransformCandidate:
    transform: ExactAffineTransform
    support_commitment: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform", as_exact_transform(self.transform))
        object.__setattr__(
            self,
            "support_commitment",
            _validate_commitment(self.support_commitment),
        )


def transform_candidate(transform, witness: str | Iterable[str]) -> TransformCandidate:
    transform = as_exact_transform(transform)
    labels = _normalize_atom_labels(witness)
    commitment = _commit("vspt-leaf-atom", _transform_key(transform), *labels)
    return TransformCandidate(transform, commitment)


def opcode_candidate(opcode, value=0, *, witness=None) -> TransformCandidate:
    name = _vocabulary_name(opcode, OPCODES, "opcode")
    value = _exact_int(value, "operation value")
    if name in STRUCTURAL_OPCODES and value != 0:
        raise ValueError("structural opcodes require value 0")
    if witness is None:
        witness = name if name in STRUCTURAL_OPCODES else "{}({})".format(name, value)
    return transform_candidate(operation_transform(name, value), witness)


def _coerce_candidate(candidate) -> TransformCandidate:
    if isinstance(candidate, TransformCandidate):
        return candidate
    if isinstance(candidate, tuple):
        if len(candidate) == 2 and isinstance(candidate[0], str):
            return opcode_candidate(candidate[0], candidate[1])
        if len(candidate) == 3 and isinstance(candidate[0], str):
            return opcode_candidate(candidate[0], candidate[1], witness=candidate[2])
        if len(candidate) == 2:
            return transform_candidate(candidate[0], candidate[1])
    raise TypeError(
        "candidates must be TransformCandidate or (opcode, value[, witness]) tuples"
    )


def _validate_cap(cap) -> int:
    cap = _exact_int(cap, "candidate cap")
    if cap <= 0:
        raise ValueError("candidate cap must be positive")
    return cap


def _canonical_candidates(candidates: Iterable[TransformCandidate], cap: int):
    by_transform: dict[ExactAffineTransform, set[str]] = {}
    overflow = False
    for raw_candidate in candidates:
        candidate = _coerce_candidate(raw_candidate)
        if overflow:
            continue
        by_transform.setdefault(candidate.transform, set()).add(
            candidate.support_commitment,
        )
        if len(by_transform) > cap:
            overflow = True
            by_transform.clear()
    if overflow:
        return (), True
    ordered = tuple(
        TransformCandidate(
            transform,
            _aggregate_supports("vspt-leaf-aliases", commitments),
        )
        for transform, commitments in sorted(by_transform.items())
    )
    return ordered, False


@dataclass(frozen=True, order=True)
class SourceLeaf:
    index: int
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _exact_int(self.index, "source index"))
        object.__setattr__(self, "source", str(self.source))

    @property
    def commitment(self) -> str:
        return _commit("vspt-source", self.index, self.source)


@dataclass(frozen=True)
class VersionSpaceProductTree:
    start: int
    end: int
    cap: int
    candidates: tuple[TransformCandidate, ...]
    overflow: bool
    version_space_lower_bound: int
    source_leaf: SourceLeaf | None = None
    left: "VersionSpaceProductTree | None" = None
    right: "VersionSpaceProductTree | None" = None

    def __post_init__(self) -> None:
        start = _exact_int(self.start, "tree start")
        end = _exact_int(self.end, "tree end")
        cap = _validate_cap(self.cap)
        if end <= start:
            raise ValueError("tree range must be nonempty")
        if self.overflow:
            if self.candidates:
                raise ValueError("overflowed nodes cannot expose an incomplete candidate sample")
            if self.version_space_lower_bound <= cap:
                raise ValueError("overflow lower bound must exceed the candidate cap")
        else:
            if not self.candidates:
                raise ValueError("exact version spaces must contain at least one transform")
            if len(self.candidates) > cap:
                raise ValueError("exact version space exceeds the candidate cap")
            if self.version_space_lower_bound != len(self.candidates):
                raise ValueError("exact node lower bound must equal its version-space size")
            if len({candidate.transform for candidate in self.candidates}) != len(self.candidates):
                raise ValueError("version-space transforms must be deduplicated")
        if (self.left is None) != (self.right is None):
            raise ValueError("internal nodes require both children")
        if self.left is None and end - start != 1:
            raise ValueError("only one-event ranges may be leaves")
        if self.left is None:
            if not isinstance(self.source_leaf, SourceLeaf):
                raise ValueError("leaves require exactly one external source reference")
            if self.source_leaf.index != start:
                raise ValueError("leaf source index must equal the tree start")
        elif self.source_leaf is not None:
            raise ValueError("internal nodes retain source references only in their leaves")
        if self.left is not None:
            if self.left.start != start or self.right.end != end:
                raise ValueError("children do not cover the parent range")
            if self.left.end != self.right.start:
                raise ValueError("children must be chronologically contiguous")
            if self.left.cap != cap or self.right.cap != cap:
                raise ValueError("children must use the parent's frozen candidate cap")
            if not self.overflow and (self.left.overflow or self.right.overflow):
                raise ValueError("overflow must propagate to every ancestor")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "cap", cap)

    @property
    def events(self) -> int:
        return self.end - self.start

    @property
    def leaf(self) -> bool:
        return self.events == 1

    @property
    def node_commitment(self) -> str:
        candidate_parts = tuple(
            "{}:{}".format(
                _transform_key(candidate.transform),
                candidate.support_commitment,
            )
            for candidate in self.candidates
        )
        if self.leaf:
            topology = ("leaf", self.source_leaf.commitment)
        else:
            topology = ("internal", self.left.node_commitment, self.right.node_commitment)
        return _commit(
            "vspt-node",
            self.start,
            self.end,
            self.cap,
            int(self.overflow),
            self.version_space_lower_bound,
            *topology,
            *candidate_parts,
        )

    @property
    def version_space_size(self) -> int | None:
        """Return the exact distinct-transform count, or ``None`` on overflow."""
        return None if self.overflow else len(self.candidates)

    @property
    def source_droppable(self) -> bool:
        """Candidate-conditional hot eviction; retrieval provenance remains."""
        return not self.overflow and len(self.candidates) == 1

    @property
    def hot_context_evictable(self) -> bool:
        return self.source_droppable

    @property
    def certified(self) -> bool:
        """Compatibility alias for the stronger full-transform certificate."""
        return self.source_droppable

    @property
    def unique_transform(self) -> ExactAffineTransform | None:
        return self.candidates[0].transform if self.source_droppable else None

    @property
    def support_commitment(self) -> str | None:
        return self.candidates[0].support_commitment if self.source_droppable else None

    @property
    def source_leaves(self) -> tuple[SourceLeaf, ...]:
        """Materialize external provenance only when an audit explicitly requests it."""
        if self.leaf:
            return (self.source_leaf,)
        return self.left.source_leaves + self.right.source_leaves

    @property
    def retained_source_indices(self) -> tuple[int, ...]:
        """Materialize the hot frontier; it is not duplicated in stored tree nodes."""
        if self.source_droppable:
            return ()
        if self.leaf:
            return (self.start,)
        return self.left.retained_source_indices + self.right.retained_source_indices

    @property
    def retained_source_count(self) -> int:
        return len(self.retained_source_indices)

    @property
    def retained_source_leaves(self) -> tuple[SourceLeaf, ...]:
        retained = set(self.retained_source_indices)
        return tuple(source for source in self.source_leaves if source.index in retained)


def leaf_node(index, candidates, source=None, *, cap=DEFAULT_CAP) -> VersionSpaceProductTree:
    index = _exact_int(index, "leaf index")
    cap = _validate_cap(cap)
    candidates, overflow = _canonical_candidates(candidates, cap)
    source_leaf = SourceLeaf(index, "event:{}".format(index) if source is None else source)
    return VersionSpaceProductTree(
        start=index,
        end=index + 1,
        cap=cap,
        candidates=candidates,
        overflow=overflow,
        version_space_lower_bound=cap + 1 if overflow else len(candidates),
        source_leaf=source_leaf,
    )


def _overflow_parent(
    left: VersionSpaceProductTree,
    right: VersionSpaceProductTree,
    lower_bound: int,
) -> VersionSpaceProductTree:
    return VersionSpaceProductTree(
        start=left.start,
        end=right.end,
        cap=left.cap,
        candidates=(),
        overflow=True,
        version_space_lower_bound=max(left.cap + 1, lower_bound),
        left=left,
        right=right,
    )


def merge_nodes(
    left: VersionSpaceProductTree,
    right: VersionSpaceProductTree,
    *,
    cap=None,
) -> VersionSpaceProductTree:
    """Compose contiguous chunks in chronological order and deduplicate effects."""
    if not isinstance(left, VersionSpaceProductTree) or not isinstance(
        right, VersionSpaceProductTree,
    ):
        raise TypeError("merge_nodes requires version-space tree nodes")
    if left.end != right.start:
        raise ValueError("version-space ranges must be chronologically contiguous")
    requested_cap = left.cap if cap is None else _validate_cap(cap)
    if left.cap != right.cap or requested_cap != left.cap:
        raise ValueError("composed chunks must use the same frozen candidate cap")

    if left.overflow or right.overflow:
        lower_bound = max(
            left.version_space_lower_bound,
            right.version_space_lower_bound,
            left.cap + 1,
        )
        return _overflow_parent(left, right, lower_bound)

    supports: dict[ExactAffineTransform, set[str]] = {}
    for earlier in left.candidates:
        for later in right.candidates:
            transform = earlier.transform.followed_by(later.transform)
            derivation = _commit(
                "vspt-derivation",
                earlier.support_commitment,
                later.support_commitment,
            )
            supports.setdefault(transform, set()).add(derivation)
            if len(supports) > left.cap:
                return _overflow_parent(left, right, left.cap + 1)

    candidates = tuple(
        TransformCandidate(
            transform,
            _aggregate_supports("vspt-alternate-derivations", commitments),
        )
        for transform, commitments in sorted(supports.items())
    )
    return VersionSpaceProductTree(
        start=left.start,
        end=right.end,
        cap=left.cap,
        candidates=candidates,
        overflow=False,
        version_space_lower_bound=len(candidates),
        left=left,
        right=right,
    )


def compose_chunks(chunks: Iterable[VersionSpaceProductTree]) -> VersionSpaceProductTree:
    """Build a balanced product tree from already-built contiguous chunks."""
    level = list(chunks)
    if not level:
        raise ValueError("at least one nonempty chunk is required")
    while len(level) > 1:
        next_level = []
        for offset in range(0, len(level), 2):
            if offset + 1 == len(level):
                next_level.append(level[offset])
            else:
                next_level.append(merge_nodes(level[offset], level[offset + 1]))
        level = next_level
    return level[0]


def build_tree(candidate_sets, sources=None, *, start=0, cap=DEFAULT_CAP):
    """Build a balanced tree from one candidate iterable per source event."""
    candidate_sets = tuple(candidate_sets)
    if not candidate_sets:
        raise ValueError("a version-space product tree requires at least one leaf")
    start = _exact_int(start, "tree start")
    if sources is None:
        sources = tuple("event:{}".format(start + offset) for offset in range(len(candidate_sets)))
    else:
        sources = tuple(sources)
    if len(sources) != len(candidate_sets):
        raise ValueError("source count differs from candidate-set count")
    leaves = (
        leaf_node(start + offset, candidates, sources[offset], cap=cap)
        for offset, candidates in enumerate(candidate_sets)
    )
    return compose_chunks(leaves)


def refine_leaf(
    node: VersionSpaceProductTree,
    index: int,
    candidates,
) -> VersionSpaceProductTree:
    """Apply a monotone leaf-local reduction and rebuild the exact root path."""
    if not isinstance(node, VersionSpaceProductTree):
        raise TypeError("refinement requires a version-space product tree")
    index = _exact_int(index, "refined source index")
    if not node.start <= index < node.end:
        raise ValueError("refined source index is outside the tree range")
    if node.leaf:
        if node.overflow:
            raise ValueError("an overflowed leaf lacks a complete candidate universe")
        normalized, overflow = _canonical_candidates(candidates, node.cap)
        if overflow:
            raise ValueError("monotone refinement cannot increase a leaf into overflow")
        if not normalized:
            raise ValueError("monotone refinement cannot remove every leaf candidate")
        old_transforms = {candidate.transform for candidate in node.candidates}
        new_transforms = {candidate.transform for candidate in normalized}
        if not new_transforms.issubset(old_transforms):
            raise ValueError("leaf refinement may remove candidates but never add them")
        return leaf_node(
            index,
            normalized,
            source=node.source_leaf.source,
            cap=node.cap,
        )
    if index < node.left.end:
        left = refine_leaf(node.left, index, candidates)
        right = node.right
    else:
        left = node.left
        right = refine_leaf(node.right, index, candidates)
    return merge_nodes(left, right)


@dataclass(frozen=True)
class QueryAgreement:
    """A query-local answer result, explicitly separate from source certification."""

    complete: bool
    query_agrees: bool
    answers: tuple[int, ...]
    answer: int | None
    source_droppable: bool

    @property
    def agrees(self) -> bool:
        return self.query_agrees

    @property
    def full_transform_certified(self) -> bool:
        return self.source_droppable


def query_agreement(
    node: VersionSpaceProductTree,
    initial_values: Sequence[int],
    query,
) -> QueryAgreement:
    if node.overflow:
        return QueryAgreement(False, False, (), None, False)
    answers = tuple(sorted({
        candidate.transform.answer(initial_values, query)
        for candidate in node.candidates
    }))
    agrees = len(node.candidates) > 0 and len(answers) == 1
    return QueryAgreement(
        complete=True,
        query_agrees=agrees,
        answers=answers,
        answer=answers[0] if agrees else None,
        source_droppable=node.source_droppable,
    )


def query_set_agreement(
    node: VersionSpaceProductTree,
    initial_values: Sequence[int],
    queries: Iterable[object],
) -> QueryAgreement:
    """Certify only when every transform and every candidate query agree."""
    queries = tuple(queries)
    if not queries:
        raise ValueError("a query candidate set must be nonempty")
    if node.overflow:
        return QueryAgreement(False, False, (), None, False)
    answers = tuple(sorted({
        candidate.transform.answer(initial_values, query)
        for candidate in node.candidates
        for query in queries
    }))
    agrees = bool(node.candidates) and len(answers) == 1
    return QueryAgreement(
        complete=True,
        query_agrees=agrees,
        answers=answers,
        answer=answers[0] if agrees else None,
        source_droppable=node.source_droppable,
    )


def read_tree(node: VersionSpaceProductTree, initial_values: Sequence[int], query) -> int:
    """Read an answer only when the complete version space agrees for this query."""
    agreement = query_agreement(node, initial_values, query)
    if not agreement.complete:
        raise ValueError("overflowed version space cannot certify a query answer")
    if not agreement.query_agrees:
        raise ValueError("version space does not agree on the requested query")
    return agreement.answer  # type: ignore[return-value]


def compact_frontier(node: VersionSpaceProductTree):
    """Return exact hot summaries plus committed external retrieval records."""
    if node.overflow:
        if not node.leaf:
            return compact_frontier(node.left) + compact_frontier(node.right)
        source = node.source_leaf
        return ({
            "kind": "source",
            "start": source.index,
            "end": source.index + 1,
            "source": source.source,
            "source_commitment": source.commitment,
            "overflow": True,
            "version_space_lower_bound": node.version_space_lower_bound,
        },)
    if node.source_droppable:
        candidate = node.candidates[0]
        return ({
            "kind": "transform",
            "start": node.start,
            "end": node.end,
            "transform": candidate.transform,
            "support_commitment": candidate.support_commitment,
            "node_commitment": node.node_commitment,
            "version_space_size": 1,
            "retrieval_reference": {
                "start": node.start,
                "end": node.end,
                "node_commitment": node.node_commitment,
            },
        },)
    if node.leaf:
        source = node.source_leaf
        return ({
            "kind": "source",
            "start": source.index,
            "end": source.index + 1,
            "source": source.source,
            "source_commitment": source.commitment,
            "overflow": False,
            "version_space_size": node.version_space_size,
        },)
    return compact_frontier(node.left) + compact_frontier(node.right)


def retained_sources(node: VersionSpaceProductTree) -> int:
    return node.retained_source_count


def version_space_size(node: VersionSpaceProductTree) -> int | None:
    return node.version_space_size
