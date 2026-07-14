#!/usr/bin/env python3
"""Contract checks for microcode admission summaries."""

import torch

from audit_categorical_microcode_v1 import exact_table
from categorical_microcode import alu_basis_accuracy, execute_program, OPCODE_TO_ID, QUERY_TO_ID


def main():
    table = exact_table()
    assert alu_basis_accuracy(table) == (400, 400)
    assert execute_program(
        (12, 11),
        [OPCODE_TO_ID["add_0"], OPCODE_TO_ID["move_0_1"]],
        (3, 2), QUERY_TO_ID["difference_0_1"], table,
    ) == 0
    print("categorical microcode admission tests passed")


if __name__ == "__main__":
    main()
