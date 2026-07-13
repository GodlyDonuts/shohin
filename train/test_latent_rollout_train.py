#!/usr/bin/env python3
"""Unit tests for pairing and curriculum mechanics, without a GPU."""

from latent_rollout_train import bucketed_batches, progressive_latent_steps


def main():
    assert [progressive_latent_steps(step, 100, 4) for step in (0, 14, 15, 34, 35, 59, 60, 99)] == [
        0, 0, 1, 1, 2, 2, 4, 4
    ]
    assert progressive_latent_steps(90, 100, 0) == 0
    examples = [
        {"prompt": [1, 2], "answer": [3], "line": 1},
        {"prompt": [4, 5], "answer": [6], "line": 2},
        {"prompt": [7, 8], "answer": [9], "line": 3},
        {"prompt": [10, 11], "answer": [12], "line": 4},
        {"prompt": [13], "answer": [14, 15], "line": 5},
        {"prompt": [16], "answer": [17, 18], "line": 6},
        {"prompt": [19], "answer": [20, 21], "line": 7},
    ]
    first, first_report = bucketed_batches(examples, batch_size=2, seed=9)
    second, second_report = bucketed_batches(examples, batch_size=2, seed=9)
    assert first == second and first_report == second_report
    assert first_report == {"buckets": 2, "full_batches": 3, "dropped_examples": 1}
    for batch in first:
        shapes = {(len(examples[index]["prompt"]), len(examples[index]["answer"])) for index in batch}
        assert len(shapes) == 1
    print("latent rollout trainer tests passed")


if __name__ == "__main__":
    main()
