"""CPU contracts for equal-work causal prefix readback controls."""

import torch

from causal_prefix_readback_memory_train import readback_label_assignment


def main():
    labels = (
        {"query": [1], "answer": [7]},
        {"query": [2, 3], "answer": [8]},
        {"query": [4], "answer": [9, 0]},
    )
    verified = readback_label_assignment(labels, "verified")
    assert verified == labels
    torch.manual_seed(4)
    shuffled = readback_label_assignment(labels, "shuffled")
    assert len(shuffled) == len(labels) and shuffled != labels
    print("causal prefix readback trainer controls passed")


if __name__ == "__main__":
    main()
