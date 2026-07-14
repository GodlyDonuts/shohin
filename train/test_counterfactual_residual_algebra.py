#!/usr/bin/env python3
"""Mechanical source-cut contracts for Counterfactual Residual Algebra."""
import torch

from counterfactual_residual_algebra import (
    algebra_suffix_logits,
    compose_counterfactual_tape,
    compose_two_edit_counterfactual_tape,
    encode_residual_tape,
    generate_from_algebra_tape,
    paired_counterfactual_algebra_loss,
    supervised_algebra_loss,
)
from model import GPT, GPTConfig


def main():
    torch.manual_seed(29)
    cfg = GPTConfig(vocab_size=64, n_layer=4, n_head=4, n_kv_head=2, d_model=32, d_ff=64, seq_len=32, zloss=0.0)
    model = GPT(cfg)
    base = torch.tensor([[3, 5, 7, 9, 11]], dtype=torch.long)
    edited = torch.tensor([[3, 5, 13, 9, 11]], dtype=torch.long)
    donor = torch.tensor([[17, 19, 23, 29, 31]], dtype=torch.long)
    counter_edited = torch.tensor([[3, 5, 15, 9, 11]], dtype=torch.long)
    prompt = torch.tensor([[37, 41]], dtype=torch.long)
    answer = torch.tensor([[43, 47]], dtype=torch.long)
    tape_len = 2

    base_tape = encode_residual_tape(model, base, layer=1, tape_len=tape_len)
    edited_tape = encode_residual_tape(model, edited, layer=1, tape_len=tape_len)
    donor_tape = encode_residual_tape(model, donor, layer=1, tape_len=tape_len)
    tape = compose_counterfactual_tape(base_tape, edited_tape, donor_tape)
    assert tape.shape == (1, tape_len, cfg.d_model)
    assert torch.equal(tape, donor_tape + edited_tape - base_tape)
    # A full-width window has no prefix and therefore preserves the old path
    # exactly; a wider window aligns the terminal anchor at a fixed RoPE phase.
    assert torch.equal(
        base_tape, encode_residual_tape(model, base, layer=1, tape_len=tape_len, source_window=base.shape[1]),
    )
    phase_base = encode_residual_tape(model, base, layer=1, tape_len=tape_len, source_window=8)
    phase_edited = encode_residual_tape(model, edited, layer=1, tape_len=tape_len, source_window=8)
    phase_donor = encode_residual_tape(model, donor, layer=1, tape_len=tape_len, source_window=8)
    phase_tape = compose_counterfactual_tape(phase_base, phase_edited, phase_donor)
    assert phase_tape.shape == tape.shape
    two_edit = compose_two_edit_counterfactual_tape(base_tape, edited_tape, donor_tape, donor_tape)
    assert torch.equal(two_edit, donor_tape + edited_tape + donor_tape - 2 * base_tape)

    logits, loss, captured, targets = supervised_algebra_loss(
        model, base, edited, donor, prompt, answer, layer=1, tape_len=tape_len, eos_id=0,
    )
    assert torch.isfinite(loss)
    assert torch.equal(tape, captured)
    assert logits.shape == (1, tape_len + prompt.shape[1] + answer.shape[1], cfg.vocab_size)
    first_answer = tape_len + prompt.shape[1] - 1
    assert torch.equal(targets[:, :first_answer], torch.full((1, first_answer), -1, dtype=torch.long))
    assert torch.equal(targets[:, first_answer:first_answer + answer.shape[1]], answer)

    suffix = torch.cat((model.tok(prompt), model.tok(answer)), dim=1)
    one = algebra_suffix_logits(model, tape, suffix, layer=1, tape_len=tape_len)
    two = algebra_suffix_logits(model, tape.clone(), suffix, layer=1, tape_len=tape_len)
    assert torch.equal(one, two)
    phase_logits = algebra_suffix_logits(model, phase_tape, suffix, layer=1, tape_len=tape_len, tape_start_pos=6)
    assert phase_logits.shape == one.shape
    assert generate_from_algebra_tape(model, tape, prompt, layer=1, tape_len=tape_len, eos_id=0, max_new=4).ndim == 1
    assert generate_from_algebra_tape(
        model, phase_tape, prompt, layer=1, tape_len=tape_len, eos_id=0, max_new=4, tape_start_pos=6,
    ).ndim == 1

    paired = paired_counterfactual_algebra_loss(
        model, base, edited, counter_edited, donor, prompt, answer, torch.tensor([[53, 59]], dtype=torch.long),
        layer=1, tape_len=tape_len, eos_id=0, margin=0.2,
    )
    assert all(torch.isfinite(paired[field]) for field in ("loss", "normal_ce", "counter_ce", "normal_foil", "counter_foil", "rank"))
    assert paired["normal_tape"].shape == tape.shape and paired["counter_tape"].shape == tape.shape

    phase_logits, phase_loss, _, _ = supervised_algebra_loss(
        model, base, edited, donor, prompt, answer, layer=1, tape_len=tape_len, eos_id=0, source_window=8,
    )
    assert phase_logits.shape == logits.shape and torch.isfinite(phase_loss)

    loss.backward()
    paired["loss"].backward()
    assert model.tok.weight.grad is not None
    assert torch.isfinite(model.tok.weight.grad).all()
    print("counterfactual residual algebra checks: passed")


if __name__ == "__main__":
    main()
