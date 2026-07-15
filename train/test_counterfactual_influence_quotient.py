#!/usr/bin/env python3
"""Pure mechanics tests for the interventional semantic quotient."""

import torch

from counterfactual_influence_quotient import (
    event_candidates,
    informative_channel_order,
    normalized_curvature,
    normalized_signature,
    operation_intervention_bundle,
    operation_intervention_text,
    query_candidates,
    rank_signature_candidates,
)


def main():
    row = {
        "question": (
            "Scenario: The greenhouse ledger begins with seedlings at 64 and planters at 76.\n"
            "Event 1: Remove 5 plants from seedlings.\n"
            "Request: Report the final tally for seedlings.\nResult:"
        ),
        "keys": ["seedlings", "planters"],
        "initial": {"seedlings": 64, "planters": 76},
    }
    value, candidates = event_candidates(row["question"].splitlines()[1], row["keys"])
    assert value == 5 and tuple(candidates) == (
        "add_0", "add_1", "sub_0", "sub_1", "move_0_1", "move_1_0",
    )
    _, structural = event_candidates("Event 1: Swap seedlings with planters.", row["keys"])
    assert tuple(structural) == ("merge_0_1", "merge_1_0", "swap")
    assert tuple(query_candidates(row["question"].splitlines()[2], row["keys"])) == (
        "read_0", "read_1", "sum", "difference_0_1", "difference_1_0",
    )

    original = operation_intervention_bundle(row, 0)
    candidate = operation_intervention_bundle(row, 0, "sub_0")
    assert original["value"] == candidate["value"] == 5
    assert set(original["interventions"]) == {"initial_0", "initial_1", "event_value"}
    assert "Subtract 5 from seedlings" in candidate["baseline"]
    assert "seedlings at 65" in original["interventions"]["initial_0"]
    assert "Remove 6 plants" in original["interventions"]["event_value"]
    joint = operation_intervention_text(row, 0, channels=("event_value", "initial_0"))
    assert "seedlings at 65" in joint and "Remove 6 plants" in joint
    canonical_joint = operation_intervention_text(
        row, 0, "sub_0", channels=("event_value", "initial_1"),
    )
    assert "planters at 77" in canonical_joint and "Subtract 6 from seedlings" in canonical_joint

    baseline = torch.zeros(2, 3)
    original_signature = normalized_signature(
        baseline, torch.tensor([[[1., 0., 0.], [1., 0., 0.]], [[0., 1., 0.], [0., 1., 0.]]]),
    )
    candidates = torch.stack((
        original_signature,
        normalized_signature(
            baseline, torch.tensor([[[-1., 0., 0.], [-1., 0., 0.]], [[0., 1., 0.], [0., 1., 0.]]]),
        ),
        normalized_signature(
            baseline, torch.tensor([[[0., 1., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 0., 0.]]]),
        ),
    ))
    order = informative_channel_order(candidates)
    assert order[0] == 0
    ranking, _ = rank_signature_candidates(original_signature, candidates, order[:2])
    assert int(ranking[0]) == 0
    curvature, norms = normalized_curvature(
        baseline,
        torch.ones(2, 2, 3),
        torch.full((2, 2, 3), 2.0),
        torch.full((2, 2, 3), 4.0),
    )
    assert curvature.shape == (2, 2, 3) and torch.all(norms > 0)
    print("counterfactual influence quotient mechanics: passed")


if __name__ == "__main__":
    main()
