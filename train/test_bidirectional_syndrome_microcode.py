#!/usr/bin/env python3
"""Exact and causal tests for R9c bidirectional syndrome microcode."""

import torch

from audit_bidirectional_syndrome_dynamics import audit
from audit_bidirectional_syndrome_identifiability import audit as audit_identifiability
from bidirectional_syndrome_microcode import (
    BidirectionalSyndromeMicrocode,
    apply_operator,
    expected_operators,
    opcode_operator_bank,
    pull_back_goal,
)
from categorical_microcode import OPCODES
from future_effect_algebra import operation_operator


def main():
    values = torch.tensor([[3., 5.]], dtype=torch.float64)
    bank = opcode_operator_bank(values)
    for event, value in enumerate((3, 5)):
        for opcode, name in enumerate(OPCODES):
            expected = operation_operator(name, value, dtype=torch.float64)
            assert torch.equal(bank[0, event, opcode], expected)

    logits = torch.full((1, 2, len(OPCODES)), -100., dtype=torch.float64)
    logits[0, 0, OPCODES.index("move_0_1")] = 100.
    logits[0, 1, OPCODES.index("swap")] = 100.
    operators = expected_operators(logits, values)
    state = torch.tensor([[7., 2., 1.]], dtype=torch.float64)
    state = apply_operator(operators[:, 0], state)
    state = apply_operator(operators[:, 1], state)
    assert torch.equal(state, torch.tensor([[5., 4., 1.]], dtype=torch.float64))
    goal = pull_back_goal(torch.tensor([[1., 0., 0.]], dtype=torch.float64), operators[:, 1])
    goal = pull_back_goal(goal, operators[:, 0])
    assert torch.equal(goal, torch.tensor([[0., 1., 3.]], dtype=torch.float64))

    torch.manual_seed(4)
    model = BidirectionalSyndromeMicrocode(event_dim=7, memory_dim=16).double()
    features = torch.randn(2, 4, 7, dtype=torch.float64)
    run = model(
        features, torch.ones(2, 4, dtype=torch.float64),
        torch.tensor([[2., 3.], [5., 7.]], dtype=torch.float64),
        torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float64),
        rounds=2, conditioning="directional", use_syndrome=True,
    )
    assert run.forward_logits.shape == (2, 4, len(OPCODES))
    assert run.backward_logits.shape == run.forward_logits.shape
    assert run.prefix_states.shape == (2, 5, 3)
    assert run.suffix_goals.shape == (2, 5, 3)
    assert run.syndrome.shape == (2, 4, 3, 3)
    assert not torch.equal(run.forward_logit_history[0], run.forward_logit_history[1])

    report = audit()
    assert report["mechanics_pass"] and not report["authorize_language_fit"]
    assert all(report["gates"].values())
    identifiability = audit_identifiability()
    assert not identifiability["fail_closed_argmax_certificate_valid"]
    assert not identifiability["authorize_r9c_reuse"]
    assert identifiability["collision"]["argmaxes_differ"]
    assert identifiability["collision"]["exact_collision_at_tolerance_1e_12"]
    assert identifiability["collision"]["runtime_collision_below_r9c_threshold_0_05"]
    assert identifiability["goal_roll_control"]["unchanged_semantic_goal_fraction"] == 5 / 6
    print("bidirectional syndrome microcode dynamics: passed")


if __name__ == "__main__":
    main()
