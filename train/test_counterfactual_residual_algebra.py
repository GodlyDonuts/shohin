#!/usr/bin/env python3
"""Mechanical source-cut contracts for Counterfactual Residual Algebra."""
import torch

from counterfactual_residual_algebra import (
    algebra_suffix_logits,
    compose_counterfactual_tape,
    compose_two_edit_counterfactual_tape,
    encode_residual_tape,
    generate_from_algebra_tape,
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
    prompt = torch.tensor([[37, 41]], dtype=torch.long)
    answer = torch.tensor([[43, 47]], dtype=torch.long)
    tape_len = 2

    base_tape = encode_residual_tape(model, base, layer=1, tape_len=tape_len)
    edited_tape = encode_residual_tape(model, edited, layer=1, tape_len=tape_len)
    donor_tape = encode_residual_tape(model, donor, layer=1, tape_len=tape_len)
    tape = compose_counterfactual_tape(base_tape, edited_tape, donor_tape)
    assert tape.shape == (1, tape_len, cfg.d_model)
    assert torch.equal(tape, donor_tape + edited_tape - base_tape)
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
    assert generate_from_algebra_tape(model, tape, prompt, layer=1, tape_len=tape_len, eos_id=0, max_new=4).ndim == 1

    loss.backward()
    assert model.tok.weight.grad is not None
    assert torch.isfinite(model.tok.weight.grad).all()
    print("counterfactual residual algebra checks: passed")


if __name__ == "__main__":
    main()
