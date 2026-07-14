#!/usr/bin/env python3
"""Focused unit contracts for paraphrase-state alignment mechanics."""
import json
import tempfile
from pathlib import Path

import torch

from model import GPT, GPTConfig
from sft_paraphrase_state_alignment import (
    BoundaryCapture, alignment_statistics, collect_state_pairs, contrastive_statistics,
    mismatch_order, pad_prompts, select_distinct_pair_indices,
)


def row(episode_id, phase, response):
    return {
        "schema": "semantic_basis_transport_v2", "episode_id": episode_id, "phase": phase,
        "response": response, "question": "{} prompt".format(phase),
    }


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pairs.jsonl"
        path.write_text("\n".join(json.dumps(item) for item in (
            row("a", "compile", "ledger:P=1;Q=2"), row("a", "reflect", "ledger:P=1;Q=2"),
            row("b", "compile", "ledger:P=3;Q=4"), row("b", "reflect", "ledger:P=3;Q=4"),
        )) + "\n")
        pairs = collect_state_pairs([path])
    assert len(pairs) == 2
    assert mismatch_order(pairs) == [1, 0]
    try:
        mismatch_order(pairs[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("single-pair mismatch control was accepted")
    duplicates = [pairs[0], pairs[0], pairs[1]]
    indices, cursor = select_distinct_pair_indices(duplicates, [0, 1, 2], 0, 2)
    assert indices == [0, 2] and cursor == 0

    cfg = GPTConfig(vocab_size=32, n_layer=2, n_head=2, n_kv_head=1, d_model=16, d_ff=32, seq_len=8, zloss=0.0)
    model = GPT(cfg)
    capture = BoundaryCapture(model, 1)
    try:
        batch = [{"ids": [1, 2, 3]}, {"ids": [4, 5]}]
        ids, lengths = pad_prompts(batch, "ids", 0, "cpu")
        hidden = capture.boundary(model, ids, lengths)
        assert hidden.shape == (2, cfg.d_model)
        loss, stats = alignment_statistics(hidden, hidden.clone())
        assert loss.item() < 1e-5 and stats["variance"].item() >= 0
        contrastive, contrastive_stats = contrastive_statistics(hidden, hidden.clone(), 0.1)
        assert contrastive_stats["positive_cosine"].item() >= contrastive_stats["negative_cosine"].item()
        (loss + contrastive).backward()
        assert model.tok.weight.grad is not None
    finally:
        capture.close()
    print("paraphrase state alignment checks: passed")


if __name__ == "__main__":
    main()
