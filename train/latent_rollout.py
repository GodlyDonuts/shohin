"""Continuous latent-rollout helpers for isolated Shohin research experiments.

The rollout is deliberately not a serialized ``<think>`` channel.  It takes
the final normalized hidden state for a prompt, rescales it to the token-input
scale, appends that vector as a soft token, and repeats.  The answer loss is
applied only after the final soft token, so gradients train the intermediate
continuous states end-to-end.

This module does not alter the regular token-id/KV-cache ``GPT.forward`` path.
"""

from __future__ import annotations

import torch


def _check_ids(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2 or value.dtype != torch.long:
        raise ValueError("{} must be a rank-2 torch.long tensor".format(name))


def _token_rms(model, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Match the soft token's magnitude to the model's ordinary token inputs.

    ``model.norm`` produces approximately unit-RMS states while the input
    embedding table is initialized and trained at a much smaller magnitude.
    Feeding an unscaled normalized state would dominate every residual stream
    before the model can learn the latent protocol.  The detached scalar keeps
    the differentiable state direction while preserving the established input
    scale.
    """
    scale = model.tok.weight.detach().float().square().mean().sqrt()
    return scale.to(device=device, dtype=dtype)


def append_latents(model, prompt_embeds: torch.Tensor, latent_steps: int) -> torch.Tensor:
    """Append ``latent_steps`` differentiable model states to prompt embeddings."""
    if prompt_embeds.ndim != 3:
        raise ValueError("prompt_embeds must have shape [batch, tokens, d_model]")
    if latent_steps < 0:
        raise ValueError("latent_steps must be non-negative")
    context = prompt_embeds
    if context.shape[1] + latent_steps > model.cfg.seq_len:
        raise ValueError("latent rollout exceeds configured sequence length")
    for _ in range(latent_steps):
        _, _, hidden = model.forward_embeds(context, return_hidden=True)
        soft_token = hidden[:, -1:, :] * _token_rms(model, hidden.device, hidden.dtype)
        context = torch.cat((context, soft_token), dim=1)
    return context


def build_answer_targets(
    answer_ids: torch.Tensor,
    prompt_len: int,
    latent_steps: int,
    eos_id: int,
) -> torch.Tensor:
    """Build next-token labels for ``prompt + latent + answer`` inputs.

    The final prompt/latent position predicts the first answer token.  The
    final answer token predicts EOS.  Prompt and latent tokens themselves are
    ignored, leaving an answer-only objective exactly like the SFT path.
    """
    _check_ids("answer_ids", answer_ids)
    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive")
    if latent_steps < 0:
        raise ValueError("latent_steps must be non-negative")
    batch, answer_len = answer_ids.shape
    if answer_len <= 0:
        raise ValueError("answer_ids must include at least one token")
    total_len = prompt_len + latent_steps + answer_len
    targets = torch.full((batch, total_len), -1, dtype=torch.long, device=answer_ids.device)
    start = prompt_len + latent_steps - 1
    targets[:, start:start + answer_len] = answer_ids
    targets[:, start + answer_len] = int(eos_id)
    return targets


def supervised_latent_loss(
    model,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    latent_steps: int,
    eos_id: int,
):
    """Return logits, loss, and labels for a continuous latent rollout batch."""
    _check_ids("prompt_ids", prompt_ids)
    _check_ids("answer_ids", answer_ids)
    if prompt_ids.shape[0] != answer_ids.shape[0]:
        raise ValueError("prompt_ids and answer_ids batch sizes differ")
    if prompt_ids.shape[1] + latent_steps + answer_ids.shape[1] > model.cfg.seq_len:
        raise ValueError("latent supervised sequence exceeds configured sequence length")
    prompt_embeds = model.tok(prompt_ids)
    context = append_latents(model, prompt_embeds, latent_steps)
    full_embeds = torch.cat((context, model.tok(answer_ids)), dim=1)
    targets = build_answer_targets(answer_ids, prompt_ids.shape[1], latent_steps, eos_id)
    logits, loss = model.forward_embeds(full_embeds, targets=targets)
    return logits, loss, targets


@torch.no_grad()
def generate_with_latents(
    model,
    prompt_ids: torch.Tensor,
    latent_steps: int,
    eos_id: int,
    max_new: int,
) -> torch.Tensor:
    """Greedily decode after a continuous latent rollout.

    This intentionally re-forwards the compact context after every generated
    token instead of using the token-id KV cache.  It keeps the soft-token
    experiment isolated from the production inference path and makes each
    latent step explicit and auditable.
    """
    _check_ids("prompt_ids", prompt_ids)
    if prompt_ids.shape[0] != 1:
        raise ValueError("latent generation currently accepts one prompt at a time")
    if max_new <= 0:
        raise ValueError("max_new must be positive")
    context = append_latents(model, model.tok(prompt_ids), latent_steps)
    generated = []
    for _ in range(max_new):
        logits, _ = model.forward_embeds(context)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token = int(next_id.item())
        if token == int(eos_id):
            break
        generated.append(token)
        if context.shape[1] >= model.cfg.seq_len:
            break
        context = torch.cat((context, model.tok(next_id)), dim=1)
    return torch.tensor(generated, dtype=torch.long, device=prompt_ids.device)
