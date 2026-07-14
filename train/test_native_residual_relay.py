#!/usr/bin/env python3
"""Mechanical source-cut contracts for the native residual relay."""
import torch

from model import GPT, GPTConfig
from native_residual_relay import encode_relay, generate_from_relay, relay_suffix_logits, supervised_relay_loss


def main():
    torch.manual_seed(23)
    cfg = GPTConfig(vocab_size=64, n_layer=4, n_head=4, n_kv_head=2, d_model=32, d_ff=64, seq_len=32, zloss=0.0)
    model = GPT(cfg)
    source = torch.tensor([[3, 5, 7]], dtype=torch.long)
    suffix_prompt = torch.tensor([[11, 13]], dtype=torch.long)
    answer = torch.tensor([[17, 19]], dtype=torch.long)

    relay = encode_relay(model, source, layer=1)
    assert relay.shape == (1, 1, cfg.d_model)
    logits, loss, captured, targets = supervised_relay_loss(model, source, suffix_prompt, answer, layer=1, eos_id=0)
    assert torch.isfinite(loss)
    assert torch.equal(relay, captured)
    assert logits.shape == (1, 1 + suffix_prompt.shape[1] + answer.shape[1], cfg.vocab_size)
    assert torch.equal(targets[:, :2], torch.full((1, 2), -1, dtype=torch.long))
    assert torch.equal(targets[:, 2:4], answer)
    assert int(targets[0, 4]) == 0

    # The suffix receives only the explicit relay.  Reusing it gives bitwise
    # identical logits regardless of what source happened to produce it.
    suffix_embeds = torch.cat((model.tok(suffix_prompt), model.tok(answer)), dim=1)
    one = relay_suffix_logits(model, relay, suffix_embeds, layer=1)
    two = relay_suffix_logits(model, relay.clone(), suffix_embeds, layer=1)
    assert torch.equal(one, two)
    assert generate_from_relay(model, relay, suffix_prompt, layer=1, eos_id=0, max_new=4).ndim == 1

    loss.backward()
    assert model.tok.weight.grad is not None
    assert torch.isfinite(model.tok.weight.grad).all()
    print("native residual relay checks: passed")


if __name__ == "__main__":
    main()
