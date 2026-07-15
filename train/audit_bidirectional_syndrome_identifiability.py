#!/usr/bin/env python3
"""Prove the R9c expected-operator syndrome cannot certify argmax decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bidirectional_syndrome_microcode import expected_operators, opcode_operator_bank
from categorical_microcode import OPCODES


def audit():
    ranks = {}
    for value in (1.0, 3.0, 99.0):
        bank = opcode_operator_bank(torch.tensor([[value]], dtype=torch.float64))[0, 0]
        coordinates = bank.reshape(len(OPCODES), -1).T
        ranks[str(int(value))] = {
            "linear_rank": int(torch.linalg.matrix_rank(coordinates)),
            "affine_rank": int(torch.linalg.matrix_rank(coordinates[:, 1:] - coordinates[:, :1])),
        }

    # These categorical distributions have different executed argmax operations
    # but exactly the same expected affine operator. Mixing in uniform mass makes
    # every probability strictly positive, so both are realizable by finite logits.
    left = torch.tensor(
        [0.40, 0.15, 0.30, 0.15, 0.00, 0.00, 0.00, 0.00, 0.00],
        dtype=torch.float64,
    )
    right = torch.tensor(
        [0.15, 0.10, 0.15, 0.00, 0.25, 0.35, 0.00, 0.00, 0.00],
        dtype=torch.float64,
    )
    epsilon = 1e-3
    left = (1.0 - epsilon * len(OPCODES)) * left + epsilon
    right = (1.0 - epsilon * len(OPCODES)) * right + epsilon
    bank = opcode_operator_bank(torch.tensor([[3.0]], dtype=torch.float64))[0, 0]
    exact_left = torch.einsum("c,cij->ij", left, bank)
    exact_right = torch.einsum("c,cij->ij", right, bank)
    exact_collision_error = float((exact_left - exact_right).abs().max())
    logits = torch.stack((left.log(), right.log())).reshape(1, 2, len(OPCODES))
    operators = expected_operators(logits, torch.full((1, 2), 3.0, dtype=torch.float64))[0]
    runtime_collision_error = float((operators[0] - operators[1]).abs().max())
    runtime_syndrome_norm = float((operators[0] - operators[1]).square().mean().sqrt())

    # Training rolls goals over batches laid out as six adjacent equivalent
    # views. Five of six examples therefore receive an unchanged semantic goal.
    semantic_group = torch.arange(4).repeat_interleave(6)
    unchanged = int(semantic_group.roll(1).eq(semantic_group).sum())
    unchanged_goal_fraction = unchanged / semantic_group.numel()

    left_argmax = int(left.argmax())
    right_argmax = int(right.argmax())
    return {
        "audit": "bidirectional_syndrome_identifiability_r9c",
        "operator_probability_simplex_dimension": len(OPCODES) - 1,
        "operator_projection_ranks": ranks,
        "collision": {
            "value": 3,
            "left_argmax": OPCODES[left_argmax],
            "right_argmax": OPCODES[right_argmax],
            "argmaxes_differ": left_argmax != right_argmax,
            "all_probabilities_strictly_positive": bool((left > 0).all() and (right > 0).all()),
            "categorical_l1_distance": float((left - right).abs().sum()),
            "exact_expected_operator_max_abs_error": exact_collision_error,
            "exact_collision_at_tolerance_1e_12": exact_collision_error <= 1e-12,
            "runtime_expected_operator_max_abs_error": runtime_collision_error,
            "runtime_syndrome_norm": runtime_syndrome_norm,
            "runtime_collision_below_r9c_threshold_0_05": runtime_syndrome_norm <= 0.05,
        },
        "goal_roll_control": {
            "views_per_semantic_group": 6,
            "batch_groups": 4,
            "unchanged_semantic_goal_fraction": unchanged_goal_fraction,
            "destroys_semantic_goal_for_every_example": unchanged_goal_fraction == 0.0,
        },
        "fail_closed_argmax_certificate_valid": False,
        "authorize_r9c_reuse": False,
        "decision": "reject_noninjective_syndrome_and_weak_goal_shuffle",
        "claim_boundary": (
            "This exact counterexample invalidates R9c's matrix-syndrome certificate for categorical "
            "argmax execution. It does not assess version-space or exact-transform certificates."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    args = parser.parse_args()
    report = audit()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
