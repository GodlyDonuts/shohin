#!/usr/bin/env python3
"""Mechanical checks for continuous latent rollout bookkeeping."""

import torch

from latent_rollout import (
    append_latents,
    build_answer_targets,
    generate_with_latents,
    supervised_latent_loss,
)
from model import GPT, GPTConfig


def tiny_model():
    torch.manual_seed(7)
    return GPT(GPTConfig(vocab_size=32, n_layer=2, n_head=3, n_kv_head=1, d_model=24, d_ff=48,
                         seq_len=32, zloss=0.0))


def main():
    model = tiny_model()
    prompt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    answer = torch.tensor([[9, 10, 11], [12, 13, 14]], dtype=torch.long)
    eos = 0

    targets = build_answer_targets(answer, prompt_len=4, latent_steps=2, eos_id=eos)
    assert targets.shape == (2, 9)
    assert torch.equal(targets[:, :5], torch.full((2, 5), -1, dtype=torch.long))
    assert torch.equal(targets[:, 5:8], answer)
    assert torch.equal(targets[:, 8], torch.zeros(2, dtype=torch.long))

    prompt_embeds = model.tok(prompt)
    context = append_latents(model, prompt_embeds, latent_steps=2)
    assert context.shape == (2, 6, model.cfg.d_model)
    assert torch.isfinite(context).all()

    full_ids = torch.cat((prompt, answer), dim=1)
    direct_targets = build_answer_targets(answer, prompt_len=4, latent_steps=0, eos_id=eos)
    direct_logits, direct_loss = model(full_ids, direct_targets)
    latent_logits, latent_loss, latent_targets = supervised_latent_loss(model, prompt, answer, 0, eos)
    assert torch.equal(direct_targets, latent_targets)
    assert torch.equal(direct_logits, latent_logits)
    assert torch.equal(direct_loss, latent_loss)

    model.zero_grad(set_to_none=True)
    _, loss, _ = supervised_latent_loss(model, prompt, answer, 2, eos)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.tok.weight.grad is not None and torch.isfinite(model.tok.weight.grad).all()
    assert model.blocks[0].attn.q.weight.grad is not None
    assert torch.isfinite(model.blocks[0].attn.q.weight.grad).all()

    model.eval()
    generated = generate_with_latents(model, prompt[:1], latent_steps=2, eos_id=eos, max_new=5)
    assert generated.ndim == 1 and len(generated) <= 5
    print("latent rollout tests passed")


if __name__ == "__main__":
    main()
