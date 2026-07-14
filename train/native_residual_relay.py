"""Native residual relay for isolated causal-bottleneck experiments.

The source is encoded only through a chosen transformer layer.  The final
source residual is then the *sole* continuous input to a fresh suffix pass
containing an event/query and answer tokens; no source token or source KV cache
is supplied to that suffix.  This differs from the rejected source-memory work:
it introduces neither slots nor trainable packet parameters.

The module leaves ``GPT.forward`` untouched.  It is intentionally a small,
auditable primitive for future isolated experiments, not a flagship path.
"""
from __future__ import annotations

import torch

from latent_rollout import build_answer_targets


def _check_ids(name: str, ids: torch.Tensor) -> None:
    if ids.ndim != 2 or ids.dtype != torch.long or not ids.shape[1]:
        raise ValueError("{} must be a nonempty rank-2 torch.long tensor".format(name))


def _check_layer(model, layer: int) -> None:
    if not 0 <= int(layer) < len(model.blocks) - 1:
        raise ValueError("relay layer must leave at least one suffix block")
    if model.cfg.n_loop != 1:
        raise ValueError("native residual relay requires n_loop=1")


def encode_relay(model, source_ids: torch.Tensor, layer: int) -> torch.Tensor:
    """Encode only source tokens through ``layer`` and return its final residual.

    The output has shape ``[batch, 1, d_model]`` and is deliberately not
    normalized or projected: it is the model's native intermediate state.
    """
    _check_ids("source_ids", source_ids)
    _check_layer(model, layer)
    if source_ids.shape[1] > model.cfg.seq_len:
        raise ValueError("source exceeds configured sequence length")
    x = model.tok(source_ids)
    cos = model.cos[:source_ids.shape[1]].to(x.device)
    sin = model.sin[:source_ids.shape[1]].to(x.device)
    for block in model.blocks[:layer + 1]:
        x, _ = block(x, cos, sin)
    return x[:, -1:, :]


def relay_suffix_logits(model, relay: torch.Tensor, suffix_embeds: torch.Tensor, layer: int) -> torch.Tensor:
    """Run the remaining blocks from ``[relay, suffix]`` with source absent.

    This function accepts embeddings rather than source ids by construction.
    Consequently no source K/V cache can enter the suffix computation.
    """
    _check_layer(model, layer)
    if relay.ndim != 3 or relay.shape[1] != 1 or relay.shape[-1] != model.cfg.d_model:
        raise ValueError("relay must have shape [batch, 1, d_model]")
    if suffix_embeds.ndim != 3 or suffix_embeds.shape[0] != relay.shape[0] or suffix_embeds.shape[-1] != model.cfg.d_model:
        raise ValueError("suffix embeddings must match relay batch and model width")
    x = torch.cat((relay, suffix_embeds), dim=1)
    if x.shape[1] > model.cfg.seq_len:
        raise ValueError("relay suffix exceeds configured sequence length")
    cos = model.cos[:x.shape[1]].to(x.device)
    sin = model.sin[:x.shape[1]].to(x.device)
    for block in model.blocks[layer + 1:]:
        x, _ = block(x, cos, sin)
    return model.head(model.norm(x))


def supervised_relay_loss(
    model,
    source_ids: torch.Tensor,
    suffix_prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    layer: int,
    eos_id: int,
):
    """Completion-only loss after a hard source-to-suffix residual cut."""
    _check_ids("source_ids", source_ids)
    _check_ids("suffix_prompt_ids", suffix_prompt_ids)
    _check_ids("answer_ids", answer_ids)
    if not (source_ids.shape[0] == suffix_prompt_ids.shape[0] == answer_ids.shape[0]):
        raise ValueError("source, suffix, and answer batches must match")
    relay = encode_relay(model, source_ids, layer)
    suffix = torch.cat((model.tok(suffix_prompt_ids), model.tok(answer_ids)), dim=1)
    logits = relay_suffix_logits(model, relay, suffix, layer)
    targets = build_answer_targets(answer_ids, 1 + suffix_prompt_ids.shape[1], 0, eos_id)
    loss = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1,
    )
    return logits, loss, relay, targets


@torch.no_grad()
def generate_from_relay(model, relay: torch.Tensor, suffix_prompt_ids: torch.Tensor, layer: int, eos_id: int, max_new: int):
    """Greedily decode a suffix from a supplied native relay with no source input."""
    _check_ids("suffix_prompt_ids", suffix_prompt_ids)
    if relay.shape[0] != 1 or suffix_prompt_ids.shape[0] != 1:
        raise ValueError("native relay generation currently accepts one example")
    if max_new <= 0:
        raise ValueError("max_new must be positive")
    suffix = model.tok(suffix_prompt_ids)
    generated = []
    for _ in range(max_new):
        logits = relay_suffix_logits(model, relay, suffix, layer)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if int(next_id.item()) == int(eos_id) or suffix.shape[1] + 2 > model.cfg.seq_len:
            break
        generated.append(int(next_id.item()))
        suffix = torch.cat((suffix, model.tok(next_id)), dim=1)
    return torch.tensor(generated, dtype=torch.long, device=suffix_prompt_ids.device)
