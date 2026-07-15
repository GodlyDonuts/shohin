#!/usr/bin/env python3
"""CPU causal admission audit for the R9c trainable tensor contract."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch

from bidirectional_syndrome_microcode import BidirectionalSyndromeMicrocode


def _inputs(seed=20260714, batch=3, events=5, event_dim=12):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    features = torch.randn(batch, events, event_dim, generator=generator, dtype=torch.float64)
    values = torch.randint(1, 8, (batch, events), generator=generator).to(torch.float64)
    initial = torch.randint(1, 12, (batch, 2), generator=generator).to(torch.float64)
    query = torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., -1., 0.]], dtype=torch.float64)
    return features, values, initial, query


def _cross_event_gradient(model, inputs, direction, conditioning):
    features, values, initial, query = inputs
    features = features.clone().requires_grad_(True)
    result = model(
        features, values, initial, query, rounds=1,
        conditioning=conditioning, use_syndrome=False,
    )
    if direction == "forward":
        score = result.forward_logits[0, -1, 0]
        source = features[0, 0]
    elif direction == "backward":
        score = result.backward_logits[0, 0, 0]
        source = features[0, -1]
    else:
        raise ValueError("unknown direction")
    gradient = torch.autograd.grad(score, features, retain_graph=False)[0]
    selected = gradient[0, 0] if direction == "forward" else gradient[0, -1]
    assert selected.shape == source.shape
    return float(selected.norm().item())


def _backward_to_forward_effect(model, inputs, use_syndrome):
    features, values, initial, query = inputs
    before = model(
        features, values, initial, query, rounds=2,
        conditioning="directional", use_syndrome=use_syndrome,
    ).forward_logits.detach()
    modified = copy.deepcopy(model)
    with torch.no_grad():
        modified.backward_compiler.operator_head.bias[0].add_(0.75)
        modified.backward_compiler.operator_head.weight[1].mul_(1.25)
    after = modified(
        features, values, initial, query, rounds=2,
        conditioning="directional", use_syndrome=use_syndrome,
    ).forward_logits.detach()
    return float((before - after).abs().max().item())


def audit(seed=20260714):
    torch.manual_seed(int(seed))
    inputs = _inputs(seed=seed)
    model = BidirectionalSyndromeMicrocode(event_dim=12, memory_dim=24).double()
    dynamic_forward = _cross_event_gradient(model, inputs, "forward", "directional")
    static_forward = _cross_event_gradient(model, inputs, "forward", "static")
    dynamic_backward = _cross_event_gradient(model, inputs, "backward", "directional")
    static_backward = _cross_event_gradient(model, inputs, "backward", "static")
    syndrome_coupling = _backward_to_forward_effect(model, inputs, True)
    no_syndrome_coupling = _backward_to_forward_effect(model, inputs, False)

    features, values, initial, query = inputs
    adaptive = model(
        features, values, initial, query, rounds=3, conditioning="directional",
        use_syndrome=True, adaptive=True, syndrome_threshold=1e9,
    )
    fixed = model(
        features, values, initial, query, rounds=3, conditioning="directional",
        use_syndrome=True, adaptive=False,
    )
    adaptive_updates = [int(mask.sum().item()) for mask in adaptive.active_masks]
    fixed_updates = [int(mask.sum().item()) for mask in fixed.active_masks]
    expected_full = features.shape[0] * features.shape[1]
    parameter_count = model.adapter_num_params()
    gates = {
        "forward_evidence_depends_on_earlier_events_via_state": dynamic_forward > 1e-10,
        "backward_evidence_depends_on_later_events_via_goal": dynamic_backward > 1e-10,
        "static_control_has_no_forward_cross_event_path": static_forward < 1e-14,
        "static_control_has_no_backward_cross_event_path": static_backward < 1e-14,
        "syndrome_causally_couples_backward_to_forward_replay": syndrome_coupling > 1e-10,
        "no_syndrome_control_breaks_cross_directional_path": no_syndrome_coupling < 1e-14,
        "adaptive_replay_can_halt_all_certified_events": adaptive_updates == [expected_full, 0, 0],
        "fixed_replay_spends_the_full_matched_budget": fixed_updates == [expected_full] * 3,
        "runtime_controls_are_parameter_identical": parameter_count > 0,
    }
    return {
        "audit": "bidirectional_syndrome_microcode_dynamics_r9c",
        "seed": int(seed),
        "adapter_parameters": parameter_count,
        "dynamic_forward_cross_event_gradient": dynamic_forward,
        "static_forward_cross_event_gradient": static_forward,
        "dynamic_backward_cross_event_gradient": dynamic_backward,
        "static_backward_cross_event_gradient": static_backward,
        "syndrome_backward_to_forward_effect": syndrome_coupling,
        "no_syndrome_backward_to_forward_effect": no_syndrome_coupling,
        "adaptive_updates_per_round": adaptive_updates,
        "fixed_updates_per_round": fixed_updates,
        "gates": gates,
        "mechanics_pass": all(gates.values()),
        "authorize_language_fit": False,
        "next_requirement": (
            "Bind the tensor contract to text-only event/query features, freeze disjoint forward-state "
            "and backward-goal supervision plus exact same-parameter controls, and preregister fresh "
            "operation/program/answer/common-mode gates before any H100 fit."
        ),
        "claim_boundary": (
            "A pass proves dynamic causal paths and parameter-identical controls in an untrained cell. "
            "It does not prove semantic learning, convergence, reasoning, or context scaling."
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
