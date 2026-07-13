#!/usr/bin/env python3
"""Focused contracts for source-memory packet data generation."""

from generate_source_memory_packet_v1 import (
    HELDOUT_DOMAINS, HELDOUT_STYLES, TRAIN_DOMAINS, TRAIN_STYLES, build_rows, source_key,
)


def main():
    train = build_rows(96, (2, 3, 4), TRAIN_DOMAINS, TRAIN_STYLES, (3, 55), False, 4)
    heldout = build_rows(48, (5, 6, 8), HELDOUT_DOMAINS, HELDOUT_STYLES, (56, 99), True, 5,
                         {source_key(row) for row in train})
    assert len({source_key(row) for row in train}) == len(train)
    assert not ({source_key(row) for row in train} & {source_key(row) for row in heldout})
    assert {row["chunk_count"] for row in train} == {2, 3, 4}
    assert {row["chunk_count"] for row in heldout} == {5, 6, 8}
    assert all(len(row["chunks"]) == len(row["operations"]) == row["chunk_count"] for row in train + heldout)
    assert all(row["response"] == "The answer is {}.".format(row["answer"]) for row in train + heldout)
    assert all(not row["heldout"] for row in train)
    assert all(row["heldout"] for row in heldout)
    print("source memory packet generator tests passed")


if __name__ == "__main__":
    main()
