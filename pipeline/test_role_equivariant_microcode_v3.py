#!/usr/bin/env python3
"""Generator contracts for role-equivariant microcode views."""

import random

from generate_latent_operator_v1 import TRAIN_DOMAINS, make_row
from generate_role_equivariant_microcode_v3 import make_rows, permute_source


def main():
    source = make_row(3, random.Random(17), TRAIN_DOMAINS[0], 4, False)
    transformed = permute_source(source)
    keys = source["keys"]
    assert transformed["initial"][keys[0]] == source["initial"][keys[1]]
    assert transformed["initial"][keys[1]] == source["initial"][keys[0]]
    assert str(transformed["answer"]) == str(source["answer"])
    rows = make_rows(source, 3)
    assert len(rows) == 6
    assert {(row["semantic_view"], row["register_permutation"]) for row in rows} == {
        (view, permutation)
        for view in ("anchor", "paraphrase_a", "paraphrase_b")
        for permutation in (0, 1)
    }
    assert len({row["question"] for row in rows}) == 6
    print("role-equivariant microcode generator tests passed")


if __name__ == "__main__":
    main()
