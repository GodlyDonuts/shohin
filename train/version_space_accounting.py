"""Canonical byte accounting for the exact R10 version-space product tree.

The product tree has four logically distinct stores:

* the active hot frontier (raw unresolved sources or exact singleton summaries),
* the complete factorized tree used as replay provenance,
* the immutable external source payloads, and
* range retrieval references for hot-evicted singleton subtrees.

Each byte figure is the sum of compact, sorted-key canonical JSON payloads for
the records in that store. Transport framing is deliberately excluded. Exact
integers are serialized as unbounded decimal JSON integers, so coefficient
growth changes the byte count; magnitude-bit statistics make that growth
explicit as well. SHA-256 commitments stay in the 64-character hexadecimal
form exposed by :mod:`version_space_product_tree`.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from version_space_product_tree import (
    ExactAffineTransform,
    VersionSpaceProductTree,
    compact_frontier,
)


ACCOUNTING_SCHEMA = "r10-version-space-accounting-v1"


class AccountingContractError(ValueError):
    """Raised when exact accounting would otherwise be partial or ambiguous."""


def _exact_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise AccountingContractError("{} must be an integer, not bool".format(name))
    try:
        return operator.index(value)
    except TypeError as error:
        raise AccountingContractError("{} must be an exact integer".format(name)) from error


def _normalize_canonical(value, active: set[int] | None = None):
    """Return the strict JSON value used by every accounting category."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if active is None:
        active = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise AccountingContractError("canonical payloads cannot contain cycles")
        keys = tuple(value)
        if any(not isinstance(key, str) for key in keys):
            raise AccountingContractError("canonical mapping keys must be strings")
        active.add(identity)
        try:
            normalized = {}
            for key in sorted(keys):
                normalized[key] = _normalize_canonical(value[key], active)
            return normalized
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise AccountingContractError("canonical payloads cannot contain cycles")
        active.add(identity)
        try:
            return [_normalize_canonical(item, active) for item in value]
        finally:
            active.remove(identity)
    raise AccountingContractError(
        "canonical payloads support only null, bool, integer, string, list, and mapping"
    )


def canonical_json_bytes(payload) -> bytes:
    """Encode one strict canonical record as compact ASCII JSON."""
    normalized = _normalize_canonical(payload)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True)
class IntegerBitGrowth:
    """Magnitude-bit use for exact integers in canonical logical records."""

    values: int = 0
    magnitude_bits: int = 0
    max_magnitude_bits: int = 0

    def __add__(self, other: "IntegerBitGrowth") -> "IntegerBitGrowth":
        if not isinstance(other, IntegerBitGrowth):
            return NotImplemented
        return IntegerBitGrowth(
            values=self.values + other.values,
            magnitude_bits=self.magnitude_bits + other.magnitude_bits,
            max_magnitude_bits=max(
                self.max_magnitude_bits, other.max_magnitude_bits,
            ),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "integer_values": self.values,
            "integer_magnitude_bits": self.magnitude_bits,
            "max_integer_magnitude_bits": self.max_magnitude_bits,
        }


def integer_bit_growth(payload) -> IntegerBitGrowth:
    """Measure exact integer magnitudes in a canonical payload.

    Zero requires one magnitude bit. A negative sign is already represented in
    the canonical JSON byte count and is not an additional magnitude bit.
    """
    normalized = _normalize_canonical(payload)
    values = []

    def visit(item) -> None:
        if isinstance(item, bool) or item is None or isinstance(item, str):
            return
        if isinstance(item, int):
            values.append(max(1, abs(item).bit_length()))
            return
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if isinstance(item, dict):
            for key in sorted(item):
                visit(item[key])
            return
        raise AssertionError("canonical normalization produced an unknown value")

    visit(normalized)
    return IntegerBitGrowth(
        values=len(values),
        magnitude_bits=sum(values),
        max_magnitude_bits=max(values, default=0),
    )


@dataclass(frozen=True)
class CanonicalCategory:
    """Canonical byte and integer accounting for one disjoint store."""

    canonical_bytes: int
    records: int
    integer_growth: IntegerBitGrowth

    def as_dict(self) -> dict[str, int]:
        return {
            "bytes": self.canonical_bytes,
            "records": self.records,
            **self.integer_growth.as_dict(),
        }


@dataclass(frozen=True)
class VersionSpaceAccounting:
    """Complete canonical accounting for one exact factorized product tree."""

    active_hot_frontier: CanonicalCategory
    factorized_provenance: CanonicalCategory
    external_source: CanonicalCategory
    retrieval_reference: CanonicalCategory
    transform_integer_growth: IntegerBitGrowth
    events: int
    retained_source_events: int
    evicted_source_events: int

    @property
    def active_hot_frontier_bytes(self) -> int:
        return self.active_hot_frontier.canonical_bytes

    @property
    def canonical_hot_bytes(self) -> int:
        return self.active_hot_frontier_bytes

    @property
    def factorized_provenance_bytes(self) -> int:
        return self.factorized_provenance.canonical_bytes

    @property
    def external_source_bytes(self) -> int:
        return self.external_source.canonical_bytes

    @property
    def source_bytes(self) -> int:
        return self.external_source_bytes

    @property
    def retrieval_reference_bytes(self) -> int:
        return self.retrieval_reference.canonical_bytes

    @property
    def retrieval_provenance_bytes(self) -> int:
        return self.retrieval_reference_bytes

    @property
    def retrieval_reference_count(self) -> int:
        return self.retrieval_reference.records

    @property
    def factorized_node_count(self) -> int:
        return self.factorized_provenance.records

    @property
    def total_canonical_bytes(self) -> int:
        return (
            self.active_hot_frontier_bytes
            + self.factorized_provenance_bytes
            + self.external_source_bytes
            + self.retrieval_reference_bytes
        )

    @property
    def hot_plus_retrieval_reference_bytes(self) -> int:
        return self.active_hot_frontier_bytes + self.retrieval_reference_bytes

    @property
    def total_integer_growth(self) -> IntegerBitGrowth:
        return (
            self.active_hot_frontier.integer_growth
            + self.factorized_provenance.integer_growth
            + self.external_source.integer_growth
            + self.retrieval_reference.integer_growth
        )

    def as_dict(self) -> dict:
        """Return a report-ready object without merging the four byte stores."""
        return {
            "accounting_schema": ACCOUNTING_SCHEMA,
            "active_hot_frontier_bytes": self.active_hot_frontier_bytes,
            "evicted_source_events": self.evicted_source_events,
            "events": self.events,
            "external_source_bytes": self.external_source_bytes,
            "factorized_node_count": self.factorized_node_count,
            "factorized_provenance_bytes": self.factorized_provenance_bytes,
            "hot_plus_retrieval_reference_bytes": (
                self.hot_plus_retrieval_reference_bytes
            ),
            "irreversible_source_deletions": 0,
            "retained_source_events": self.retained_source_events,
            "retrieval_reference_bytes": self.retrieval_reference_bytes,
            "retrieval_reference_count": self.retrieval_reference_count,
            "stores": {
                "active_hot_frontier": self.active_hot_frontier.as_dict(),
                "external_source": self.external_source.as_dict(),
                "factorized_provenance": self.factorized_provenance.as_dict(),
                "retrieval_reference": self.retrieval_reference.as_dict(),
            },
            "total_canonical_bytes": self.total_canonical_bytes,
            "total_integer_growth": self.total_integer_growth.as_dict(),
            "transform_integer_growth": self.transform_integer_growth.as_dict(),
        }


def _validate_tree(node) -> VersionSpaceProductTree:
    if not isinstance(node, VersionSpaceProductTree):
        raise AccountingContractError(
            "accounting requires a VersionSpaceProductTree root"
        )
    return node


def _validate_commitment(value, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AccountingContractError("{} must be a SHA-256 commitment".format(name))
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise AccountingContractError(
            "{} must be hexadecimal SHA-256".format(name)
        ) from error
    if len(raw) != 32:
        raise AccountingContractError("{} must be a SHA-256 commitment".format(name))
    return value.lower()


def _transform_payload(transform) -> list[list[int]]:
    if not isinstance(transform, ExactAffineTransform):
        raise AccountingContractError(
            "frontier and provenance transforms must be ExactAffineTransform"
        )
    return [list(row) for row in transform.rows]


def _tree_nodes(node: VersionSpaceProductTree) -> tuple[VersionSpaceProductTree, ...]:
    nodes = []
    stack = [node]
    while stack:
        current = stack.pop()
        nodes.append(current)
        if not current.leaf:
            if current.left is None or current.right is None:
                raise AccountingContractError("internal tree node lacks a child")
            stack.append(current.right)
            stack.append(current.left)
    return tuple(nodes)


def _tree_leaves(node: VersionSpaceProductTree) -> tuple[VersionSpaceProductTree, ...]:
    return tuple(current for current in _tree_nodes(node) if current.leaf)


def _resolve_external_sources(
    node: VersionSpaceProductTree,
    external_sources,
) -> dict[int, object]:
    leaves = _tree_leaves(node)
    indices = tuple(leaf.start for leaf in leaves)
    if indices != tuple(range(node.start, node.end)):
        raise AccountingContractError("tree leaves do not cover a contiguous source range")

    if external_sources is None:
        raw = {
            leaf.start: leaf.source_leaf.source  # type: ignore[union-attr]
            for leaf in leaves
        }
    elif isinstance(external_sources, Mapping):
        raw = {}
        for key, payload in external_sources.items():
            index = _exact_int(key, "external source index")
            if index in raw:
                raise AccountingContractError("duplicate external source index")
            raw[index] = payload
        if set(raw) != set(indices):
            raise AccountingContractError(
                "external source mapping must cover the tree range exactly"
            )
    else:
        if isinstance(external_sources, (str, bytes, bytearray)):
            raise AccountingContractError(
                "external sources must be a sequence of payloads, not one scalar"
            )
        try:
            payloads = tuple(external_sources)
        except TypeError as error:
            raise AccountingContractError(
                "external sources must be a mapping or ordered iterable"
            ) from error
        if len(payloads) != len(indices):
            raise AccountingContractError(
                "external source count must equal the tree event count"
            )
        raw = dict(zip(indices, payloads))

    return {
        index: _normalize_canonical(raw[index])
        for index in indices
    }


def external_source_records(
    node: VersionSpaceProductTree,
    external_sources=None,
) -> tuple[object, ...]:
    """Return external payloads in chronological, range-addressable order."""
    node = _validate_tree(node)
    sources = _resolve_external_sources(node, external_sources)
    return tuple(sources[index] for index in range(node.start, node.end))


def factorized_provenance_records(
    node: VersionSpaceProductTree,
) -> tuple[dict, ...]:
    """Serialize each logical factorized node exactly once, without raw source."""
    node = _validate_tree(node)
    nodes = _tree_nodes(node)
    commitments = {
        id(current): _validate_commitment(
            current.node_commitment, "node commitment",
        )
        for current in nodes
    }
    records = []
    for current in nodes:
        candidates = [{
            "support_commitment": _validate_commitment(
                candidate.support_commitment, "support commitment",
            ),
            "transform": _transform_payload(candidate.transform),
        } for candidate in current.candidates]
        record = {
            "candidates": candidates,
            "cap": current.cap,
            "end": current.end,
            "kind": "leaf" if current.leaf else "internal",
            "node_commitment": commitments[id(current)],
            "overflow": current.overflow,
            "start": current.start,
            "version_space_lower_bound": current.version_space_lower_bound,
        }
        if current.leaf:
            if current.source_leaf is None:
                raise AccountingContractError("leaf lacks its source commitment")
            record["source_reference"] = {
                "source_commitment": _validate_commitment(
                    current.source_leaf.commitment, "source commitment",
                ),
                "source_index": current.source_leaf.index,
            }
        else:
            if current.left is None or current.right is None:
                raise AccountingContractError("internal tree node lacks a child")
            record["children"] = [{
                "end": child.end,
                "node_commitment": commitments[id(child)],
                "start": child.start,
            } for child in (current.left, current.right)]
        records.append(record)
    return tuple(records)


def _frontier_records(
    node: VersionSpaceProductTree,
    sources: Mapping[int, object],
) -> tuple[tuple[dict, ...], tuple[dict, ...], int, int]:
    hot_records = []
    retrieval_records = []
    retained_events = 0
    evicted_events = 0
    expected_start = node.start

    for item in compact_frontier(node):
        if not isinstance(item, dict):
            raise AccountingContractError("compact frontier records must be mappings")
        start = _exact_int(item.get("start"), "frontier range start")
        end = _exact_int(item.get("end"), "frontier range end")
        if start != expected_start or not start < end <= node.end:
            raise AccountingContractError(
                "compact frontier ranges must be ordered, contiguous, and nonempty"
            )
        expected_start = end
        kind = item.get("kind")
        if kind == "source":
            if end != start + 1 or start not in sources:
                raise AccountingContractError(
                    "source frontier records must address exactly one tree leaf"
                )
            record = {
                "end": end,
                "kind": "source",
                "overflow": bool(item.get("overflow")),
                "source": sources[start],
                "source_commitment": _validate_commitment(
                    item.get("source_commitment"), "source commitment",
                ),
                "start": start,
            }
            if record["overflow"]:
                record["version_space_lower_bound"] = _exact_int(
                    item.get("version_space_lower_bound"),
                    "overflow version-space lower bound",
                )
            else:
                record["version_space_size"] = _exact_int(
                    item.get("version_space_size"), "version-space size",
                )
            hot_records.append(record)
            retained_events += 1
            continue
        if kind != "transform":
            raise AccountingContractError("unknown compact frontier record kind")

        support_commitment = _validate_commitment(
            item.get("support_commitment"), "support commitment",
        )
        node_commitment = _validate_commitment(
            item.get("node_commitment"), "node commitment",
        )
        version_space_size = _exact_int(
            item.get("version_space_size"), "version-space size",
        )
        if version_space_size != 1:
            raise AccountingContractError(
                "only exact singleton transforms may enter the hot frontier"
            )
        hot_records.append({
            "end": end,
            "kind": "transform",
            "node_commitment": node_commitment,
            "start": start,
            "support_commitment": support_commitment,
            "transform": _transform_payload(item.get("transform")),
            "version_space_size": version_space_size,
        })

        reference = item.get("retrieval_reference")
        if not isinstance(reference, Mapping):
            raise AccountingContractError(
                "hot-evicted transforms require a range retrieval reference"
            )
        reference_start = _exact_int(
            reference.get("start"), "retrieval range start",
        )
        reference_end = _exact_int(reference.get("end"), "retrieval range end")
        reference_commitment = _validate_commitment(
            reference.get("node_commitment"), "retrieval node commitment",
        )
        if (
            reference_start != start
            or reference_end != end
            or reference_commitment != node_commitment
        ):
            raise AccountingContractError(
                "retrieval reference must bind the hot transform's exact range and node"
            )
        retrieval_records.append({
            "end": reference_end,
            "node_commitment": reference_commitment,
            "start": reference_start,
        })
        evicted_events += end - start

    if expected_start != node.end:
        raise AccountingContractError("compact frontier does not cover the root range")
    if retained_events + evicted_events != node.events:
        raise AccountingContractError("compact frontier event accounting is incomplete")
    return (
        tuple(hot_records),
        tuple(retrieval_records),
        retained_events,
        evicted_events,
    )


def active_hot_frontier_records(
    node: VersionSpaceProductTree,
    external_sources=None,
) -> tuple[dict, ...]:
    """Return canonical hot records with retrieval references removed."""
    node = _validate_tree(node)
    sources = _resolve_external_sources(node, external_sources)
    hot, _, _, _ = _frontier_records(node, sources)
    return hot


def retrieval_reference_records(
    node: VersionSpaceProductTree,
) -> tuple[dict, ...]:
    """Return exact contiguous range references for every hot summary."""
    node = _validate_tree(node)
    sources = _resolve_external_sources(node, None)
    _, retrieval, _, _ = _frontier_records(node, sources)
    return retrieval


def _measure(records: Sequence[object]) -> CanonicalCategory:
    growth = IntegerBitGrowth()
    byte_count = 0
    for record in records:
        byte_count += len(canonical_json_bytes(record))
        growth = growth + integer_bit_growth(record)
    return CanonicalCategory(byte_count, len(records), growth)


def _transform_growth(
    hot_records: Sequence[dict],
    provenance_records: Sequence[dict],
) -> IntegerBitGrowth:
    growth = IntegerBitGrowth()
    for record in hot_records:
        if record["kind"] == "transform":
            growth = growth + integer_bit_growth(record["transform"])
    for record in provenance_records:
        for candidate in record["candidates"]:
            growth = growth + integer_bit_growth(candidate["transform"])
    return growth


def account_version_space_tree(
    node: VersionSpaceProductTree,
    external_sources=None,
) -> VersionSpaceAccounting:
    """Measure all four R10 stores for one exact product tree.

    ``external_sources`` may be an ordered iterable aligned to ``node.start`` or
    a mapping keyed by absolute source index. When omitted, the tree's own leaf
    source strings are the external payloads. A supplied collection must cover
    the tree range exactly; partial accounting fails closed.
    """
    node = _validate_tree(node)
    sources = _resolve_external_sources(node, external_sources)
    external_records = tuple(sources[index] for index in range(node.start, node.end))
    provenance_records = factorized_provenance_records(node)
    hot_records, retrieval_records, retained, evicted = _frontier_records(
        node, sources,
    )
    return VersionSpaceAccounting(
        active_hot_frontier=_measure(hot_records),
        factorized_provenance=_measure(provenance_records),
        external_source=_measure(external_records),
        retrieval_reference=_measure(retrieval_records),
        transform_integer_growth=_transform_growth(hot_records, provenance_records),
        events=node.events,
        retained_source_events=retained,
        evicted_source_events=evicted,
    )


# Concise alias for evaluators that already name the subject in local context.
account_tree = account_version_space_tree
