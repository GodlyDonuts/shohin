#!/usr/bin/env python3
"""Executable pre-neural equivalence audit for the initial R9 proposal."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from orbit_recurrent_microcode import (
    closed_form_static_consensus,
    orbit_cross_entropy,
    pull_back_logits,
    recurrent_static_consensus,
    static_orbit_syndrome,
    transformed_label_cross_entropy,
)


def audit(seed=20260714, batch=64, views=5, hypotheses=97):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    raw = torch.randn(batch, views, hypotheses, generator=generator, dtype=torch.float64)
    permutations = torch.stack([
        torch.randperm(hypotheses, generator=generator) for _ in range(views)
    ])
    pulled = pull_back_logits(raw, permutations)
    initial = torch.randn(batch, hypotheses, generator=generator, dtype=torch.float64)
    max_forward_error = 0.0
    max_gradient_error = 0.0
    for steps in (1, 2, 4, 8, 16):
        for rate in (0.2, 0.5, 1.0):
            recurrent = recurrent_static_consensus(initial, pulled, steps, rate)
            closed = closed_form_static_consensus(initial, pulled, steps, rate)
            max_forward_error = max(max_forward_error, float((recurrent - closed).abs().max()))

            rec_initial = initial.clone().requires_grad_(True)
            rec_views = pulled.clone().requires_grad_(True)
            rec_loss = recurrent_static_consensus(rec_initial, rec_views, steps, rate).square().mean()
            rec_grad = torch.autograd.grad(rec_loss, (rec_initial, rec_views))
            one_initial = initial.clone().requires_grad_(True)
            one_views = pulled.clone().requires_grad_(True)
            one_loss = closed_form_static_consensus(one_initial, one_views, steps, rate).square().mean()
            one_grad = torch.autograd.grad(one_loss, (one_initial, one_views))
            max_gradient_error = max(
                max_gradient_error,
                *(float((left - right).abs().max()) for left, right in zip(rec_grad, one_grad)),
            )

    targets = torch.randint(0, hypotheses, (batch,), generator=generator)
    orbit_ce = orbit_cross_entropy(raw, targets, permutations)
    augmented_ce = transformed_label_cross_entropy(raw, targets, permutations)
    ce_error = float((orbit_ce - augmented_ce).abs())
    syndrome = static_orbit_syndrome(pulled)
    gates = {
        "recurrence_has_exact_feedforward_closed_form": max_forward_error < 1e-12,
        "recurrent_and_closed_form_gradients_match": max_gradient_error < 1e-12,
        "orbit_output_ce_equals_transformed_label_augmentation": ce_error < 1e-12,
        "syndrome_uses_only_static_view_logits": syndrome.shape == (batch,) and bool(torch.isfinite(syndrome).all()),
    }
    return {
        "audit": "orbit_recurrent_microcode_pre_neural_equivalence_r9a",
        "seed": int(seed),
        "batch": int(batch),
        "views": int(views),
        "hypotheses": int(hypotheses),
        "max_forward_error": max_forward_error,
        "max_gradient_error": max_gradient_error,
        "orbit_vs_augmented_ce_error": ce_error,
        "syndrome_min": float(syndrome.min()),
        "syndrome_max": float(syndrome.max()),
        "gates": gates,
        "authorize_neural_fit": not all(gates.values()),
        "decision": (
            "reject_static_orbit_recurrence_as_reparameterized_classifier"
            if all(gates.values()) else "equivalence_not_proven"
        ),
        "required_revision": (
            "A future cell must change the evidence as a function of the evolving state or apply "
            "noncommuting transitions to a carried state. Replaying static orbit logits is not reasoning."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing output")
    result = audit(seed=args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
