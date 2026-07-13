#!/usr/bin/env python3
"""Exact, model-authored KV-anchor transport for future context experiments.

This module deliberately does *not* compress, parse, rank, or repair a state.
It transports a token sequence already authored by the model as an immutable
attention prefix, then appends later controller/model tokens with the model's
own KV cache.  It is an inference substrate for a future causal experiment,
not a learned-memory mechanism and not a context-window extension.

The current GPT cache path uses the fast causal kernel when there is no past
and is exact for a single token appended to a past cache.  A multi-token append
against a past cache would let those new tokens attend to one another without
the required triangular mask.  Therefore every append here is deliberately
serial.  That choice is auditable and makes cached decoding equivalent to a
full replay of the same token history.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CausalKVAnchor:
    """An immutable token history and its exact per-layer KV cache."""

    tokens: torch.Tensor
    cache: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    pos: int


def _validate_ids(ids: torch.Tensor, name: str) -> None:
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.dtype != torch.long:
        raise ValueError("{} must have shape [batch, tokens] and dtype torch.long".format(name))
    if not ids.shape[1]:
        raise ValueError("{} must contain at least one token".format(name))


def _validate_anchor(model, anchor: CausalKVAnchor) -> None:
    _validate_ids(anchor.tokens, "anchor tokens")
    if anchor.pos != anchor.tokens.shape[1]:
        raise ValueError("anchor position must equal its token-history length")
    if anchor.pos > model.cfg.seq_len:
        raise ValueError("anchor exceeds model sequence length")
    expected_layers = model.cfg.n_loop * len(model.blocks)
    if len(anchor.cache) != expected_layers:
        raise ValueError("anchor cache has the wrong number of transformer layers")


@torch.inference_mode()
def prefill_anchor(model, anchor_ids: torch.Tensor) -> tuple[CausalKVAnchor, torch.Tensor]:
    """Create an immutable anchor from exact model-authored token IDs.

    The caller owns how the IDs were generated.  This function does not change
    them or derive a shorter representation.  The returned logits are those of
    the final anchor token, useful when the next token is to be sampled.
    """

    _validate_ids(anchor_ids, "anchor ids")
    if anchor_ids.shape[1] > model.cfg.seq_len:
        raise ValueError("anchor exceeds model sequence length")
    logits, cache = model(anchor_ids, pos=0, return_cache=True)
    anchor = CausalKVAnchor(
        tokens=anchor_ids.detach().clone(),
        cache=tuple(cache),
        pos=int(anchor_ids.shape[1]),
    )
    _validate_anchor(model, anchor)
    return anchor, logits[:, -1:, :]


@torch.inference_mode()
def append_exact_tokens(
    model,
    anchor: CausalKVAnchor,
    token_ids: torch.Tensor,
) -> tuple[CausalKVAnchor, torch.Tensor]:
    """Append controller/model tokens serially, preserving exact causality.

    A cache is not mutated in place: the original anchor remains usable for a
    matched cache-swap or no-cache control.  The final logits predict the token
    after the appended sequence.
    """

    _validate_anchor(model, anchor)
    _validate_ids(token_ids, "appended token ids")
    if token_ids.shape[0] != anchor.tokens.shape[0]:
        raise ValueError("appended tokens must have the anchor batch size")
    if anchor.pos + token_ids.shape[1] > model.cfg.seq_len:
        raise ValueError("anchor plus appended tokens exceeds model sequence length")
    if token_ids.device != anchor.tokens.device:
        raise ValueError("appended tokens and anchor must share a device")

    cache = anchor.cache
    pos = anchor.pos
    logits = None
    # GPT.forward's past-cache attention contract is exact for one token only.
    for token in token_ids.split(1, dim=1):
        logits, cache = model(token, cache=cache, pos=pos, return_cache=True)
        pos += 1
    assert logits is not None
    combined = torch.cat((anchor.tokens, token_ids.detach()), dim=1)
    updated = CausalKVAnchor(tokens=combined, cache=tuple(cache), pos=pos)
    _validate_anchor(model, updated)
    return updated, logits


@torch.inference_mode()
def full_replay_last_logits(model, ids: torch.Tensor) -> torch.Tensor:
    """Reference next-token logits from a normal causal full replay."""

    _validate_ids(ids, "replay ids")
    if ids.shape[1] > model.cfg.seq_len:
        raise ValueError("replay exceeds model sequence length")
    logits, _ = model(ids)
    return logits[:, -1:, :]


@torch.inference_mode()
def assert_exact_replay(
    model,
    anchor_ids: torch.Tensor,
    appended_ids: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> CausalKVAnchor:
    """Verify cached serial transport against every full-replay prefix.

    This is a mechanical prerequisite for a later semantic anchor experiment.
    It proves neither semantic retention nor context scaling.
    """

    _validate_ids(anchor_ids, "anchor ids")
    _validate_ids(appended_ids, "appended ids")
    if anchor_ids.shape[0] != appended_ids.shape[0]:
        raise ValueError("anchor and appended ids must have matching batch sizes")
    anchor, _ = prefill_anchor(model, anchor_ids)
    state = anchor
    for index in range(appended_ids.shape[1]):
        state, cached_logits = append_exact_tokens(model, state, appended_ids[:, index:index + 1])
        replay_logits = full_replay_last_logits(model, state.tokens)
        torch.testing.assert_close(cached_logits, replay_logits, atol=atol, rtol=rtol)
    return state
