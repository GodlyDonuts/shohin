#!/usr/bin/env python3
"""CPU contracts for NRR trainer batching and exact shape preservation."""
from train_native_residual_relay import bucketed_batches, make_batch


def main():
    examples = [
        {"source": [1, 2], "suffix": [3], "answer": [4], "shape": (2, 1, 1)},
        {"source": [5, 6], "suffix": [7], "answer": [8], "shape": (2, 1, 1)},
        {"source": [9, 10, 11], "suffix": [12], "answer": [13], "shape": (3, 1, 1)},
    ]
    batches, report = bucketed_batches(examples, 2, 17)
    assert report["full_batches"] == 1 and report["dropped_examples"] == 1
    source, suffix, answer = make_batch(examples, batches[0], "cpu")
    assert tuple(source.shape) == (2, 2) and tuple(suffix.shape) == (2, 1) and tuple(answer.shape) == (2, 1)
    print("native residual relay trainer checks: passed")


if __name__ == "__main__":
    main()
