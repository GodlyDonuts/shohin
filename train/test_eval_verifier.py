#!/usr/bin/env python3
"""Regression test for verifier inference against GPT's tuple-return contract."""
from types import SimpleNamespace

import torch

from eval_verifier import score_candidate


class FakeTokenizer:
    def encode(self, _prompt):
        return SimpleNamespace(ids=[1, 2, 3])


class FakeModel:
    cfg = SimpleNamespace(seq_len=16)

    def __call__(self, token_ids):
        logits = torch.zeros((1, token_ids.shape[1], 8))
        logits[0, -1, 4] = 2.0
        logits[0, -1, 5] = -1.0
        return logits, None


def main():
    score = score_candidate(FakeModel(), FakeTokenizer(), "q", "candidate", "cpu", 4, 5)
    assert score > 0
    print("verifier tuple-return contract: passed")


if __name__ == "__main__":
    main()
