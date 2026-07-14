#!/usr/bin/env python3
"""CPU-only data and exact-shape batching checks for CRA training."""
import json
import os
import tempfile

from pipeline.generate_counterfactual_residual_algebra_v1 import ANCHOR, build, factor_audit_sets
from train_counterfactual_residual_algebra import bucketed_batches, load_examples, make_batch


class _Encoded(object):
    def __init__(self, ids):
        self.ids = ids


class CharacterTokenizer(object):
    def encode(self, text):
        return _Encoded([ord(char) for char in text])


def main():
    tokenizer = CharacterTokenizer()
    rows = build("train", 24, 17)
    handle, path = tempfile.mkstemp(prefix="cra-data-", suffix=".jsonl")
    os.close(handle)
    try:
        with open(path, "w") as output:
            for row in rows:
                output.write(json.dumps(row) + "\n")
        anchor_ids = tokenizer.encode(ANCHOR).ids
        examples, skipped = load_examples(path, tokenizer, 512, anchor_ids)
        assert len(examples) == 48 and not skipped
        batches, report = bucketed_batches(examples, 2, 31)
        assert report["full_batches"] > 0 and report["dropped_examples"] < 2 * report["buckets"]
        base, edited, donor, suffix, answer = make_batch(examples, batches[0], "cpu")
        assert base.shape[0] == edited.shape[0] == donor.shape[0] == suffix.shape[0] == answer.shape[0] == 2
        assert all(row["base"][-len(anchor_ids):] == anchor_ids for row in examples)
        factor = build("factor_delta", 24, 47, delta_heldout=True)
        factor_audit = factor_audit_sets(rows, factor)
        assert not factor_audit["train_factor_exact_bundle_hits"]
        assert not factor_audit["train_factor_exact_state_hits"]
    finally:
        os.unlink(path)
    print("CRA trainer data checks: passed")


if __name__ == "__main__":
    main()
