#!/usr/bin/env python3
"""Focused solver/split contracts for latent-operator generation."""

from generate_latent_operator_v1 import build_rows, normalized


def main():
    train = build_rows(36, (1, 2, 3, 4), 9, False)
    heldout = build_rows(27, (5, 6, 8), 10, True)
    assert {row["depth"] for row in train} == {1, 2, 3, 4}
    assert {row["depth"] for row in heldout} == {5, 6, 8}
    assert not ({normalized(row["question"]) for row in train} & {normalized(row["question"]) for row in heldout})
    larger_train = build_rows(2_000, (1, 2, 3, 4), 11, False)
    assert len({normalized(row["question"]) for row in larger_train}) == len(larger_train)
    assert all(row["response"] == "The answer is {}.".format(row["answer"]) for row in train + heldout)
    assert all(len(row["operations"]) == row["depth"] for row in train + heldout)
    print("latent operator generator tests passed")


if __name__ == "__main__":
    main()
