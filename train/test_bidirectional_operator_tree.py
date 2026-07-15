#!/usr/bin/env python3
"""Exact tests for bidirectional certified operator folding."""

import torch

from bidirectional_operator_tree import (
    build_tree,
    compact_frontier,
    read_tree,
    retained_sources,
    suspect_leaves,
)
from future_effect_algebra import operation_operator


def main():
    add = operation_operator("add_0", 4, dtype=torch.int64)
    swap = operation_operator("swap", dtype=torch.int64)
    clean = build_tree([add, swap], [add, swap], ["add", "swap"])
    assert clean.certified and suspect_leaves(clean) == ()
    assert len(compact_frontier(clean)) == 1 and retained_sources(compact_frontier(clean)) == 0
    assert read_tree(clean, [3, 8], "read_1") == 7

    corrupt = build_tree([add, swap], [add, add], ["add", "swap"])
    assert not corrupt.certified and suspect_leaves(corrupt) == (1,)
    frontier = compact_frontier(corrupt)
    assert len(frontier) == 2 and retained_sources(frontier) == 1

    # Product cancellation does not override failed leaf certification.
    inverse = operation_operator("sub_0", 4, dtype=torch.int64)
    canceled = build_tree([add, inverse], [torch.eye(3, dtype=torch.int64), torch.eye(3, dtype=torch.int64)])
    assert torch.equal(canceled.forward, canceled.backward)
    assert not canceled.certified and suspect_leaves(canceled) == (0, 1)
    print("bidirectional operator tree mechanics: passed")


if __name__ == "__main__":
    main()
