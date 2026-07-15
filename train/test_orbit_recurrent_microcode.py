#!/usr/bin/env python3
"""Exact tests for the R9a static-orbit equivalence audit."""

import torch

from compare_orbit_recurrent_microcode_equivalence import audit
from orbit_recurrent_microcode import (
    closed_form_static_consensus,
    orbit_cross_entropy,
    pull_back_logits,
    recurrent_static_consensus,
    transformed_label_cross_entropy,
)


def main():
    logits = torch.tensor([[[3., 1., 0.], [0., 3., 1.]]], dtype=torch.float64)
    permutations = torch.tensor([[0, 1, 2], [1, 2, 0]])
    pulled = pull_back_logits(logits, permutations)
    assert torch.equal(pulled[0, 1], torch.tensor([3., 1., 0.], dtype=torch.float64))
    initial = torch.zeros(1, 3, dtype=torch.float64)
    assert torch.allclose(
        recurrent_static_consensus(initial, pulled, 7, 0.3),
        closed_form_static_consensus(initial, pulled, 7, 0.3),
        atol=1e-14,
        rtol=0,
    )
    targets = torch.tensor([0])
    assert torch.allclose(
        orbit_cross_entropy(logits, targets, permutations),
        transformed_label_cross_entropy(logits, targets, permutations),
        atol=1e-14,
        rtol=0,
    )
    report = audit(batch=8, views=3, hypotheses=11)
    assert report["decision"] == "reject_static_orbit_recurrence_as_reparameterized_classifier"
    assert not report["authorize_neural_fit"] and all(report["gates"].values())
    print("orbit recurrent microcode equivalence: passed")


if __name__ == "__main__":
    main()
