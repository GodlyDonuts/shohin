#!/usr/bin/env python3
"""CPU contract test for same-prompt batched decoding."""
from types import SimpleNamespace

import torch

from eval_suite import generate, generate_batch


class FakeTokenizer:
    def encode(self, prompt):
        return SimpleNamespace(ids=[1, 2])

    def decode(self, ids):
        return " ".join(map(str, ids))

    def decode_batch(self, batch):
        return [self.decode(ids) for ids in batch]

    def token_to_id(self, token):
        assert token == "<|endoftext|>"
        return 4


class SpecialTokenizer(FakeTokenizer):
    def decode(self, ids, skip_special_tokens=True):
        mapping = {3: "<think>", 4: "<|endoftext|>"}
        return "".join(
            "" if skip_special_tokens and token in mapping else mapping.get(token, str(token))
            for token in ids
        )

    def decode_batch(self, batch, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch]


class FakeModel:
    cfg = SimpleNamespace(seq_len=16)

    def __init__(self):
        self.positions = []

    def __call__(self, ids, cache=None, pos=0, return_cache=False):
        self.positions.append(pos)
        batch, length = ids.shape
        logits = torch.full((batch, length, 8), -100.0)
        # Force one cached decode update before EOS. This validates the actual
        # batch-shaped KV-cache call rather than only the prefill/stop path.
        logits[:, -1, 3 if pos == 0 else 4] = 0.0
        return logits, []


def main():
    model, tok = FakeModel(), FakeTokenizer()
    many = generate_batch(model, tok, "Question: x\nAnswer:", "cpu", n=3, max_new=4)
    assert many == ["3", "3", "3"], many
    assert model.positions[:2] == [0, 2], model.positions
    one = generate(model, tok, "Question: x\nAnswer:", "cpu", max_new=4)
    assert one == many[0], (one, many)
    special = SpecialTokenizer()
    assert generate(model, special, "x", "cpu", max_new=4) == ""
    assert generate(model, special, "x", "cpu", max_new=4, skip_special_tokens=False) == "<think>"
    assert generate_batch(model, tok, "x", "cpu", n=0) == []
    print("batched generation contract: passed")


if __name__ == "__main__":
    main()
