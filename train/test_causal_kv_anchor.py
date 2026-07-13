#!/usr/bin/env python3
"""CPU-only contracts for exact cache-native anchor transport."""

import torch

from causal_kv_anchor import append_exact_tokens, assert_exact_replay, full_replay_last_logits, prefill_anchor
from model import GPT, GPTConfig


def tiny_model():
    torch.manual_seed(7)
    model = GPT(GPTConfig(
        vocab_size=64,
        seq_len=24,
        d_model=32,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        d_ff=64,
    ))
    return model.eval()


def main():
    model = tiny_model()
    anchor_ids = torch.tensor([[3, 5, 7], [11, 13, 17]], dtype=torch.long)
    updates = torch.tensor([[19, 23, 29], [31, 37, 41]], dtype=torch.long)

    state = assert_exact_replay(model, anchor_ids, updates)
    assert state.pos == 6
    assert torch.equal(state.tokens, torch.cat((anchor_ids, updates), dim=1))

    # The immutable root cache must remain reusable for a causal swap control.
    root, _ = prefill_anchor(model, anchor_ids)
    left, left_logits = append_exact_tokens(model, root, updates[:, :1])
    right, right_logits = append_exact_tokens(model, root, updates[:, 1:2])
    assert root.pos == anchor_ids.shape[1]
    assert left.pos == right.pos == root.pos + 1
    torch.testing.assert_close(left_logits, full_replay_last_logits(model, left.tokens))
    torch.testing.assert_close(right_logits, full_replay_last_logits(model, right.tokens))

    try:
        append_exact_tokens(model, root, torch.ones((1, 1), dtype=torch.long))
    except ValueError as exc:
        assert "batch size" in str(exc)
    else:
        raise AssertionError("mismatched batches must be rejected")
    print("causal KV-anchor contracts passed")


if __name__ == "__main__":
    main()
