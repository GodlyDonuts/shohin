#!/usr/bin/env python3
"""CPU-only shape and pairing invariants for the LSA trainer surface."""

import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from latent_state_algebra_train import (
    bucketed_batches,
    load_pairs,
    make_pair_batch,
    nonidentity_permutation,
)


def main():
    root = Path(__file__).resolve().parents[1]
    smoke = Path("/tmp/shohin-lsa-smoke/train.jsonl")
    if not smoke.exists():
        raise SystemExit("run the generator smoke before this test")
    tokenizer = Tokenizer.from_file(str(root / "artifacts" / "shohin-tok-32k.json"))
    pairs, skipped = load_pairs(str(smoke), tokenizer, slots=8, max_chunks=8, seq_len=2048)
    assert not skipped
    assert len(pairs) == 32
    assert {pair["equivalent"] for pair in pairs} == {False, True}
    batches, report = bucketed_batches(pairs, batch_size=2, seed=17)
    assert report["full_batches"] > 0
    batch = batches[0]
    values = make_pair_batch(pairs, batch, "cpu")
    state_a, state_b, equivalent = values[3], values[7], values[8]
    assert state_a.shape == state_b.shape == (2, 2)
    assert equivalent.dtype == torch.bool and equivalent.shape == (2,)
    for _ in range(20):
        order = nonidentity_permutation(4, "cpu")
        assert not torch.equal(order, torch.arange(4))
    print("latent state algebra trainer CPU invariants passed")


if __name__ == "__main__":
    main()
