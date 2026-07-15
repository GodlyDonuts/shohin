#!/usr/bin/env python3
"""CPU tests for the text bridge's exact causal targets and losses."""

from types import SimpleNamespace

import torch

from bidirectional_syndrome_microcode import expected_operators
from categorical_microcode import OPCODES, QUERIES
from referential_syndrome_microcode import (
    assemble_event_feature,
    directional_effect_losses,
    oracle_operator_tensor,
    oracle_prefix_suffix,
    query_goal_from_logits,
)


def main():
    fake = {
        "kind_context": torch.zeros(8),
        "target_context": torch.ones(8),
        "kind_logits": torch.arange(5.),
        "role_logits": torch.arange(2.),
        "slot_presence_scores": torch.arange(2.),
    }
    assert assemble_event_feature(fake).shape == (25,)

    query_logits = torch.full((2, len(QUERIES)), -100., dtype=torch.float64)
    query_logits[0, QUERIES.index("read_0")] = 100.
    query_logits[1, QUERIES.index("difference_1_0")] = 100.
    goals = query_goal_from_logits(query_logits)
    assert torch.equal(goals, torch.tensor([[1., 0., 0.], [-1., 1., 0.]], dtype=torch.float64))

    values = torch.tensor([[2., 3.], [4., 1.]], dtype=torch.float64)
    targets = torch.tensor([
        [OPCODES.index("add_0"), OPCODES.index("swap")],
        [OPCODES.index("move_1_0"), OPCODES.index("sub_1")],
    ])
    oracle = oracle_operator_tensor(targets, values)
    initial = torch.tensor([[5., 7.], [3., 9.]], dtype=torch.float64)
    prefix, suffix = oracle_prefix_suffix(initial, goals, oracle)
    assert prefix.shape == suffix.shape == (2, 3, 3)

    logits = torch.full((2, 2, len(OPCODES)), -100., dtype=torch.float64)
    logits.scatter_(2, targets.unsqueeze(-1), 100.)
    predicted = expected_operators(logits, values)
    run = SimpleNamespace(
        forward_logits=logits,
        backward_logits=logits,
        forward_operators=predicted,
        backward_operators=predicted,
        syndrome=predicted - predicted,
    )
    answers = (goals * prefix[:, -1]).sum(dim=-1)
    losses = directional_effect_losses(run, values, initial, goals, targets, answers)
    assert float(losses["forward_effect"]) < 1e-12
    assert float(losses["backward_effect"]) < 1e-12
    assert float(losses["agreement"]) == 0.0
    assert float(losses["endpoint"]) < 1e-12
    assert float(losses["total"]) < 1e-10

    wrong_logits = logits.roll(1, dims=-1)
    wrong = expected_operators(wrong_logits, values)
    wrong_run = SimpleNamespace(
        forward_logits=wrong_logits,
        backward_logits=wrong_logits,
        forward_operators=wrong,
        backward_operators=wrong,
        syndrome=wrong - wrong,
    )
    wrong_losses = directional_effect_losses(wrong_run, values, initial, goals, targets, answers)
    assert float(wrong_losses["total"]) > 0.1
    print("referential syndrome microcode bridge: passed")


if __name__ == "__main__":
    main()
