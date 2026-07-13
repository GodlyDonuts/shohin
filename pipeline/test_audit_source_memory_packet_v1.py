#!/usr/bin/env python3
"""Solver audit fixtures for source-memory packet rows."""

import random

from audit_source_memory_packet_v1 import valid
from generate_source_memory_packet_v1 import TRAIN_DOMAINS, make_row


def main():
    row = make_row(1, random.Random(3), TRAIN_DOMAINS[0], 3, 0, (3, 55), False)
    assert valid(row, False)
    altered = dict(row)
    altered["answer"] = str(int(row["answer"]) + 1)
    assert not valid(altered, False)
    heldout = dict(row)
    heldout["heldout"] = True
    assert valid(heldout, True)
    print("source memory packet audit tests passed")


if __name__ == "__main__":
    main()
