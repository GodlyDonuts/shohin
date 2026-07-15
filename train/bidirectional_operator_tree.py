"""Bidirectional certified folding for noncommutative event microcode.

Forward and backward semantic channels independently predict one affine event
operator.  A leaf is certifiable only when those predictions agree.  Internal
nodes compose chronological products but inherit certification only when both
children were independently certified, preventing product-level cancellation
from hiding a bad leaf.  Certified subtrees may discard all source text and
retain one fixed 3x3 operator; uncertified paths retain only their suspect
leaves and certified siblings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from future_effect_algebra import effect_signature, query_operator


@dataclass(frozen=True)
class OperatorTree:
    start: int
    end: int
    forward: torch.Tensor
    backward: torch.Tensor
    certified: bool
    syndrome_max: float
    source: str | None = None
    left: "OperatorTree | None" = None
    right: "OperatorTree | None" = None

    @property
    def events(self) -> int:
        return self.end - self.start

    @property
    def leaf(self) -> bool:
        return self.events == 1


def _validate_operator(operator: torch.Tensor) -> None:
    if operator.shape != (3, 3) or not bool(torch.isfinite(operator).all()):
        raise ValueError("event operators must be finite 3x3 tensors")


def _agreement(forward: torch.Tensor, backward: torch.Tensor, tolerance: float):
    _validate_operator(forward)
    _validate_operator(backward)
    syndrome = effect_signature(forward) - effect_signature(backward)
    maximum = float(syndrome.abs().max().item())
    return maximum <= float(tolerance), maximum


def leaf_node(index, forward, backward, source, *, tolerance=1e-9):
    agreed, maximum = _agreement(forward, backward, tolerance)
    return OperatorTree(
        start=int(index), end=int(index) + 1,
        forward=forward.clone(), backward=backward.clone(),
        certified=agreed, syndrome_max=maximum, source=str(source),
    )


def merge_nodes(left: OperatorTree, right: OperatorTree, *, tolerance=1e-9):
    if left.end != right.start:
        raise ValueError("operator-tree ranges must be contiguous")
    forward = right.forward @ left.forward
    backward = right.backward @ left.backward
    agreed, maximum = _agreement(forward, backward, tolerance)
    # Product equality alone is insufficient: opposite leaf errors can cancel.
    certified = bool(left.certified and right.certified and agreed)
    return OperatorTree(
        start=left.start, end=right.end,
        forward=forward, backward=backward,
        certified=certified, syndrome_max=maximum,
        left=left, right=right,
    )


def build_tree(forward, backward, sources=None, *, start=0, tolerance=1e-9):
    if len(forward) != len(backward) or not forward:
        raise ValueError("forward/backward event sequences must be equally nonempty")
    if sources is None:
        sources = tuple("event:{}".format(start + index) for index in range(len(forward)))
    if len(sources) != len(forward):
        raise ValueError("source count differs from event count")
    level = [
        leaf_node(start + index, left, right, sources[index], tolerance=tolerance)
        for index, (left, right) in enumerate(zip(forward, backward))
    ]
    while len(level) > 1:
        next_level = []
        for offset in range(0, len(level), 2):
            if offset + 1 == len(level):
                next_level.append(level[offset])
            else:
                next_level.append(merge_nodes(level[offset], level[offset + 1], tolerance=tolerance))
        level = next_level
    return level[0]


def suspect_leaves(node: OperatorTree):
    if node.certified:
        return ()
    if node.leaf:
        return (node.start,)
    suspects = []
    for child in (node.left, node.right):
        if child is not None:
            suspects.extend(suspect_leaves(child))
    return tuple(suspects)


def compact_frontier(node: OperatorTree):
    """Return certified summaries plus source-retaining suspect leaves."""
    if node.certified:
        return ({
            "kind": "operator", "start": node.start, "end": node.end,
            "operator": node.forward.clone(),
        },)
    if node.leaf:
        return ({
            "kind": "source", "start": node.start, "end": node.end,
            "source": node.source, "syndrome_max": node.syndrome_max,
        },)
    frontier = []
    for child in (node.left, node.right):
        if child is not None:
            frontier.extend(compact_frontier(child))
    return tuple(frontier)


def scalar_payload(frontier) -> int:
    """Count numeric state scalars; source strings are reported separately."""
    return sum(9 if item["kind"] == "operator" else 1 for item in frontier)


def retained_sources(frontier) -> int:
    return sum(item["kind"] == "source" for item in frontier)


def read_tree(node: OperatorTree, initial_values, query, *, channel="forward") -> int:
    if len(initial_values) != 2 or channel not in {"forward", "backward"}:
        raise ValueError("invalid tree readout")
    operator = node.forward if channel == "forward" else node.backward
    state = torch.tensor([initial_values[0], initial_values[1], 1], dtype=operator.dtype)
    readout = query_operator(query, dtype=operator.dtype)
    return int((readout @ operator @ state).item())
