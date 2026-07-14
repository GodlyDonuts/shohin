"""Native-tape residual algebra for isolated causal-composition experiments.

The source path exports the final ``tape_len`` residual positions from a fixed
ordinary-language anchor.  The suffix receives only the native composition
``donor + edited - base`` plus query/answer embeddings.  It receives no source
ids, source embeddings, source K/V cache, learned slot, parser, or controller.
"""
from __future__ import annotations

import torch

from latent_rollout import build_answer_targets


def _check_ids(name, ids):
    if ids.ndim != 2 or ids.dtype != torch.long or not ids.shape[1]:
        raise ValueError("{} must be a nonempty rank-2 torch.long tensor".format(name))


def _check_layer(model, layer):
    if not 0 <= int(layer) < len(model.blocks) - 1:
        raise ValueError("algebra layer must leave at least one suffix block")
    if model.cfg.n_loop != 1:
        raise ValueError("counterfactual residual algebra requires n_loop=1")


def _check_tape(model, tape, tape_len):
    if tape.ndim != 3 or tape.shape[1] != tape_len or tape.shape[-1] != model.cfg.d_model:
        raise ValueError("tape must have shape [batch, tape_len, d_model]")


def encode_residual_tape(model, source_ids, layer, tape_len):
    """Encode a source to a native tape at one fixed intermediate layer."""
    _check_ids("source_ids", source_ids)
    _check_layer(model, layer)
    if tape_len <= 0 or tape_len > source_ids.shape[1]:
        raise ValueError("tape_len must be positive and no longer than source")
    if source_ids.shape[1] > model.cfg.seq_len:
        raise ValueError("source exceeds configured sequence length")
    x = model.tok(source_ids)
    cos = model.cos[:source_ids.shape[1]].to(x.device)
    sin = model.sin[:source_ids.shape[1]].to(x.device)
    for block in model.blocks[:layer + 1]:
        x, _ = block(x, cos, sin)
    return x[:, -tape_len:, :]


def compose_counterfactual_tape(base, edited, donor):
    """Apply ``base -> edited`` as a residual intervention to ``donor``."""
    if base.shape != edited.shape or base.shape != donor.shape or base.ndim != 3:
        raise ValueError("base, edited, and donor tapes must have the same rank-3 shape")
    return donor + edited - base


def compose_two_edit_counterfactual_tape(base, primary_edited, secondary_edited, donor):
    """Apply two independently compiled edits to a donor tape in either order."""
    if any(tape.shape != base.shape or tape.ndim != 3 for tape in (primary_edited, secondary_edited, donor)):
        raise ValueError("all two-edit tapes must have the same rank-3 shape")
    return donor + primary_edited + secondary_edited - 2 * base


def algebra_suffix_logits(model, tape, suffix_embeds, layer, tape_len):
    """Run the tail from a composed tape and source-free suffix embeddings."""
    _check_layer(model, layer)
    _check_tape(model, tape, tape_len)
    if suffix_embeds.ndim != 3 or suffix_embeds.shape[0] != tape.shape[0] or suffix_embeds.shape[-1] != model.cfg.d_model:
        raise ValueError("suffix embeddings must match tape batch and width")
    x = torch.cat((tape, suffix_embeds), dim=1)
    if x.shape[1] > model.cfg.seq_len:
        raise ValueError("tape suffix exceeds configured sequence length")
    cos = model.cos[:x.shape[1]].to(x.device)
    sin = model.sin[:x.shape[1]].to(x.device)
    for block in model.blocks[layer + 1:]:
        x, _ = block(x, cos, sin)
    return model.head(model.norm(x))


def supervised_algebra_loss(model, base_ids, edited_ids, donor_ids, suffix_prompt_ids, answer_ids, layer, tape_len, eos_id):
    """Completion loss after a three-source residual-algebra hard cut."""
    for name, ids in (("base_ids", base_ids), ("edited_ids", edited_ids), ("donor_ids", donor_ids),
                      ("suffix_prompt_ids", suffix_prompt_ids), ("answer_ids", answer_ids)):
        _check_ids(name, ids)
    batch = base_ids.shape[0]
    if any(ids.shape[0] != batch for ids in (edited_ids, donor_ids, suffix_prompt_ids, answer_ids)):
        raise ValueError("all algebra inputs must share the same batch")
    base = encode_residual_tape(model, base_ids, layer, tape_len)
    edited = encode_residual_tape(model, edited_ids, layer, tape_len)
    donor = encode_residual_tape(model, donor_ids, layer, tape_len)
    tape = compose_counterfactual_tape(base, edited, donor)
    suffix = torch.cat((model.tok(suffix_prompt_ids), model.tok(answer_ids)), dim=1)
    logits = algebra_suffix_logits(model, tape, suffix, layer, tape_len)
    targets = build_answer_targets(answer_ids, tape_len + suffix_prompt_ids.shape[1], 0, eos_id)
    loss = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1,
    )
    return logits, loss, tape, targets


def answer_loss_for_algebra_tape(model, tape, suffix_prompt_ids, answer_ids, layer, tape_len, eos_id):
    """Return answer-only CE after a previously composed native tape."""
    _check_ids("suffix_prompt_ids", suffix_prompt_ids)
    _check_ids("answer_ids", answer_ids)
    if tape.shape[0] != suffix_prompt_ids.shape[0] or tape.shape[0] != answer_ids.shape[0]:
        raise ValueError("tape, suffix, and answer batch sizes must agree")
    suffix = torch.cat((model.tok(suffix_prompt_ids), model.tok(answer_ids)), dim=1)
    logits = algebra_suffix_logits(model, tape, suffix, layer, tape_len)
    targets = build_answer_targets(answer_ids, tape_len + suffix_prompt_ids.shape[1], 0, eos_id)
    loss = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1,
    )
    return logits, loss, targets


def answer_nll_per_example_for_algebra_tape(model, tape, suffix_prompt_ids, answer_ids, layer, tape_len, eos_id):
    """Return one answer-only mean NLL per batch item after a fixed tape."""
    _check_ids("suffix_prompt_ids", suffix_prompt_ids)
    _check_ids("answer_ids", answer_ids)
    if tape.shape[0] != suffix_prompt_ids.shape[0] or tape.shape[0] != answer_ids.shape[0]:
        raise ValueError("tape, suffix, and answer batch sizes must agree")
    suffix = torch.cat((model.tok(suffix_prompt_ids), model.tok(answer_ids)), dim=1)
    logits = algebra_suffix_logits(model, tape, suffix, layer, tape_len)
    targets = build_answer_targets(answer_ids, tape_len + suffix_prompt_ids.shape[1], 0, eos_id)
    token_nll = torch.nn.functional.cross_entropy(
        logits.float().transpose(1, 2), targets, ignore_index=-1, reduction="none",
    )
    mask = targets.ne(-1)
    return (token_nll * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def paired_counterfactual_algebra_loss(
    model, base_ids, edited_ids, counter_edited_ids, donor_ids, suffix_prompt_ids,
    answer_ids, counter_answer_ids, layer, tape_len, eos_id, margin,
):
    """Train both edit directions plus a functional paired-answer margin.

    This does not align hidden vectors.  Each source-free algebra tape must
    assign lower answer CE to its own solver target than to the paired
    counterfactual target, preventing a decoder from treating +d and -d as one
    template while preserving the original hard cut.
    """
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    for name, ids in (
        ("base_ids", base_ids), ("edited_ids", edited_ids), ("counter_edited_ids", counter_edited_ids),
        ("donor_ids", donor_ids), ("suffix_prompt_ids", suffix_prompt_ids), ("answer_ids", answer_ids),
        ("counter_answer_ids", counter_answer_ids),
    ):
        _check_ids(name, ids)
    batch = base_ids.shape[0]
    if any(ids.shape[0] != batch for ids in (
        edited_ids, counter_edited_ids, donor_ids, suffix_prompt_ids, answer_ids, counter_answer_ids,
    )):
        raise ValueError("all paired algebra inputs must share a batch")
    base = encode_residual_tape(model, base_ids, layer, tape_len)
    edited = encode_residual_tape(model, edited_ids, layer, tape_len)
    counter_edited = encode_residual_tape(model, counter_edited_ids, layer, tape_len)
    donor = encode_residual_tape(model, donor_ids, layer, tape_len)
    normal_tape = compose_counterfactual_tape(base, edited, donor)
    counter_tape = compose_counterfactual_tape(base, counter_edited, donor)
    normal_ce_each = answer_nll_per_example_for_algebra_tape(
        model, normal_tape, suffix_prompt_ids, answer_ids, layer, tape_len, eos_id,
    )
    normal_foil_each = answer_nll_per_example_for_algebra_tape(
        model, normal_tape, suffix_prompt_ids, counter_answer_ids, layer, tape_len, eos_id,
    )
    counter_ce_each = answer_nll_per_example_for_algebra_tape(
        model, counter_tape, suffix_prompt_ids, counter_answer_ids, layer, tape_len, eos_id,
    )
    counter_foil_each = answer_nll_per_example_for_algebra_tape(
        model, counter_tape, suffix_prompt_ids, answer_ids, layer, tape_len, eos_id,
    )
    normal_ce, normal_foil = normal_ce_each.mean(), normal_foil_each.mean()
    counter_ce, counter_foil = counter_ce_each.mean(), counter_foil_each.mean()
    rank = (torch.relu(margin + normal_ce_each - normal_foil_each) +
            torch.relu(margin + counter_ce_each - counter_foil_each)).mean()
    return {
        "loss": normal_ce + counter_ce + rank,
        "normal_ce": normal_ce,
        "counter_ce": counter_ce,
        "normal_foil": normal_foil,
        "counter_foil": counter_foil,
        "rank": rank,
        "normal_tape": normal_tape,
        "counter_tape": counter_tape,
    }


@torch.no_grad()
def generate_from_algebra_tape(model, tape, suffix_prompt_ids, layer, tape_len, eos_id, max_new):
    """Greedily decode from a supplied composed tape with sources absent."""
    _check_ids("suffix_prompt_ids", suffix_prompt_ids)
    _check_tape(model, tape, tape_len)
    if tape.shape[0] != 1 or suffix_prompt_ids.shape[0] != 1:
        raise ValueError("algebra generation currently accepts one example")
    if max_new <= 0:
        raise ValueError("max_new must be positive")
    suffix = model.tok(suffix_prompt_ids)
    generated = []
    for _ in range(max_new):
        logits = algebra_suffix_logits(model, tape, suffix, layer, tape_len)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if int(next_id.item()) == int(eos_id) or tape_len + suffix.shape[1] + 1 > model.cfg.seq_len:
            break
        generated.append(int(next_id.item()))
        suffix = torch.cat((suffix, model.tok(next_id)), dim=1)
    return torch.tensor(generated, dtype=torch.long, device=suffix_prompt_ids.device)
