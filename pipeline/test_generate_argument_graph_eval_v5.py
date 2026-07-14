#!/usr/bin/env python3
"""Contracts for the fresh R5 argument-graph board generator."""

from generate_argument_graph_eval_v5 import build_fresh
from generate_latent_operator_v1 import apply_operation


def main():
    rows = build_fresh(32, "language_ood", (1, 2, 3, 4), 17)
    assert len(rows) == 32
    assert len({row["question"] for row in rows}) == 32
    assert {row["eval_regime"] for row in rows} == {"language_ood"}
    assert {row["family"] for row in rows} == {
        "greenhouse", "depot", "laboratory", "library",
    }
    for row in rows:
        values = dict(row["initial"])
        for operation in row["operations"]:
            values = apply_operation(values, operation)
        query = row["query"]
        if query["kind"] == "read":
            answer = values[query["key"]]
        elif query["kind"] == "sum":
            answer = sum(values.values())
        else:
            answer = values[query["high"]] - values[query["low"]]
        assert answer == int(row["answer"])
        assert row["question"].count("\nResult:") == 1
    print("argument-graph fresh-board generator tests passed")


if __name__ == "__main__":
    main()
