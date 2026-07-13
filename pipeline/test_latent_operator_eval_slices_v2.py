#!/usr/bin/env python3
"""Focused contracts for decomposed latent-operator evaluation slices."""

from generate_latent_operator_eval_slices_v2 import build_slices
from generate_latent_operator_v1 import build_rows, normalized


def main():
    train = build_rows(240, (1, 2, 3, 4), 11, False)
    rows = build_slices(train, per_depth=3, seed=17)
    by_regime = {}
    for row in rows:
        by_regime.setdefault(row["eval_regime"], []).append(row)
    assert set(by_regime) == {"fit_iid", "depth_ood", "language_ood", "full_ood"}
    assert {row["depth"] for row in by_regime["fit_iid"]} == {1, 2, 3, 4}
    assert {row["depth"] for row in by_regime["depth_ood"]} == {5, 6, 8}
    assert {row["depth"] for row in by_regime["language_ood"]} == {1, 2, 3, 4}
    assert {row["depth"] for row in by_regime["full_ood"]} == {5, 6, 8}
    assert all(row["heldout"] for row in rows)
    assert len({normalized(row["question"]) for row in rows}) == len(rows)
    assert not ({normalized(row["question"]) for row in train} & {normalized(row["question"]) for row in rows})
    assert all(3 <= value <= 29 for row in by_regime["fit_iid"] for value in row["initial"].values())
    assert all(3 <= value <= 29 for row in by_regime["language_ood"] for value in row["initial"].values())
    assert all(37 <= value <= 79 for row in by_regime["full_ood"] for value in row["initial"].values())
    print("latent operator diagnostic slice tests passed")


if __name__ == "__main__":
    main()
