#!/usr/bin/env python3
"""Tensor and gradient contracts for the selected-probe effect head."""

import torch

from future_effect_compiler import ProbeConditionedEffectCompiler, probe_descriptor


def main():
    torch.manual_seed(17)
    hidden = 32
    effect_hidden = 16
    compiler = ProbeConditionedEffectCompiler.__new__(ProbeConditionedEffectCompiler)
    torch.nn.Module.__init__(compiler)
    compiler.effect_hidden = effect_hidden
    compiler.effect_text = torch.nn.Sequential(
        torch.nn.Linear(2 * hidden + 4, effect_hidden, bias=False),
        torch.nn.SiLU(),
        torch.nn.Linear(effect_hidden, effect_hidden, bias=False),
    )
    compiler.effect_probe = torch.nn.Sequential(
        torch.nn.Linear(15, effect_hidden, bias=False),
        torch.nn.SiLU(),
        torch.nn.Linear(effect_hidden, effect_hidden, bias=False),
    )
    compiler.effect_interaction = torch.nn.Sequential(
        torch.nn.Linear(4 * effect_hidden, effect_hidden, bias=False),
        torch.nn.SiLU(),
        torch.nn.Linear(effect_hidden, 1),
    )
    operation = {
        "kind_context": torch.randn(hidden, requires_grad=True),
        "target_context": torch.randn(hidden, requires_grad=True),
        "target_weights": torch.ones(3) / 3,
        "role_logits": torch.tensor([2.0, -1.0], requires_grad=True),
        "slot_presence_scores": torch.tensor([0.9, 0.1], requires_grad=True),
    }
    states = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    descriptor = probe_descriptor(states, queries)
    assert descriptor.shape == (2, 15)
    predictions = compiler.predict_effect(operation, states, queries)
    assert predictions.shape == (2,)
    assert torch.isfinite(predictions).all()
    assert predictions[0].item() != predictions[1].item()
    predictions.square().mean().backward()
    assert operation["kind_context"].grad is not None
    assert operation["role_logits"].grad is not None
    assert all(parameter.grad is not None for parameter in compiler.parameters())

    bank = compiler.predict_effect_bank(operation, states, queries)
    assert bank.shape == (2, 2)
    assert torch.isfinite(bank).all()
    print("probe-conditioned effect compiler tests passed")


if __name__ == "__main__":
    main()
