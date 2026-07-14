#!/usr/bin/env python3
"""Generator contracts for role-equivariant microcode views."""

import random

from generate_latent_operator_v1 import TRAIN_DOMAINS, make_row
from generate_role_equivariant_microcode_v3 import make_rows, normalized_question, permute_source, select_rows


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

    duplicate = make_row(3, random.Random(17), TRAIN_DOMAINS[0], 4, False)
    replacement = make_row(5, random.Random(29), TRAIN_DOMAINS[1], 4, False)
    selected, source_indices, skipped = select_rows([source, duplicate, replacement], 2, set())
    assert len(selected) == 12
    assert source_indices == [0, 2]
    assert skipped == {"duplicate_prior_group": 1}
    assert len({normalized_question(row["question"]) for row in selected}) == 12

    exact_eval = {normalized_question(rows[0]["question"])}
    selected, source_indices, skipped = select_rows([source, replacement], 1, exact_eval)
    assert len(selected) == 6
    assert source_indices == [1]
    assert skipped == {"exact_eval_prompt": 1}
    print("role-equivariant microcode generator tests passed")


if __name__ == "__main__":
    main()
