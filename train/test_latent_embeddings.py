#!/usr/bin/env python3
"""Focused contracts for continuous-input transformer execution."""

import torch

from model import GPT, GPTConfig


def config():
    return GPTConfig(
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        d_model=32,
        d_ff=64,
        seq_len=32,
        zloss=0.0,
    )


def main():
    torch.manual_seed(17)
    model = GPT(config()).eval()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    targets = torch.tensor([[2, 3, 4, -1], [6, 7, 8, -1]])
    with torch.no_grad():
        token_logits, token_loss = model(ids, targets)
        embed_logits, embed_loss = model.forward_embeds(model.tok(ids), targets)
    assert torch.allclose(token_logits, embed_logits, rtol=0.0, atol=0.0)
    assert torch.allclose(token_loss, embed_loss, rtol=0.0, atol=0.0)

    model.train()
    embeds = model.tok(ids).detach().requires_grad_(True)
    _, loss, hidden = model.forward_embeds(embeds, targets, return_hidden=True)
    assert hidden.shape == embeds.shape
    loss.backward()
    assert embeds.grad is not None and torch.isfinite(embeds.grad).all()
    assert model.blocks[0].attn.q.weight.grad is not None
    print("latent embedding model contracts passed")


if __name__ == "__main__":
    main()
