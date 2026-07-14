"""Active counterfactual discrimination for compact effect reasoning.

The cell treats latent deliberation as an experiment-selection problem. Given
several lawful event operators that remain compatible with current evidence,
it chooses the state/query probe whose possible effects best partition those
hypotheses. A future neural compiler will answer that selected counterfactual
from text, update the compatible set, and repeat only while ambiguity remains.

This CPU module defines exact mechanics and controls. It does not consume text
and is not evidence that a trained model can answer the selected probes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from future_effect_algebra import effect_signature, operation_operator, redundant_probe_bank


NUMERIC_OPCODES = (
    "add_0", "add_1", "sub_0", "sub_1", "move_0_1", "move_1_0",
)
STRUCTURAL_OPCODES = ("merge_0_1", "merge_1_0", "swap")


@dataclass(frozen=True)
class OperatorHypothesis:
    opcode: str
    value: int
    operator: torch.Tensor


def legal_operator_hypotheses(values=range(1, 100), *, dtype=torch.float64):
    """Enumerate distinct lawful event effects without using an example label."""
    values = tuple(int(value) for value in values)
    if not values or any(value <= 0 for value in values):
        raise ValueError("numeric hypothesis values must be positive")
    hypotheses = [
        OperatorHypothesis(opcode, value, operation_operator(opcode, value, dtype=dtype))
        for opcode in NUMERIC_OPCODES
        for value in values
    ]
    hypotheses.extend(
        OperatorHypothesis(opcode, 0, operation_operator(opcode, 0, dtype=dtype))
        for opcode in STRUCTURAL_OPCODES
    )
    flattened = torch.stack([hypothesis.operator.reshape(-1) for hypothesis in hypotheses])
    if torch.unique(flattened, dim=0).shape[0] != len(hypotheses):
        raise ValueError("lawful operator bank contains duplicate effects")
    return tuple(hypotheses)


def hypothesis_effect_codes(hypotheses, states=None, queries=None):
    """Return one flattened counterfactual effect code per hypothesis."""
    if states is None or queries is None:
        default_states, default_queries = redundant_probe_bank()
        states = default_states if states is None else states
        queries = default_queries if queries is None else queries
    return torch.stack([
        effect_signature(
            hypothesis.operator.to(dtype=states.dtype, device=states.device),
            states=states,
            queries=queries,
        ).reshape(-1)
        for hypothesis in hypotheses
    ])


def _partition_score(values):
    _, counts = torch.unique(values, sorted=True, return_counts=True)
    probabilities = counts.double() / counts.sum()
    entropy = float((-(probabilities * probabilities.log2())).sum().item())
    return entropy, int(counts.numel()), -int(counts.max().item())


def select_discriminating_probe(codes, candidates, observed=()):
    """Choose the probe with maximal hypothesis-partition information."""
    if codes.ndim != 2:
        raise ValueError("codes must have shape [hypotheses, probes]")
    candidates = tuple(int(index) for index in candidates)
    if not candidates:
        raise ValueError("candidate set is empty")
    observed = set(map(int, observed))
    best = None
    candidate_index = torch.tensor(candidates, dtype=torch.long, device=codes.device)
    for probe in range(codes.shape[1]):
        if probe in observed:
            continue
        score = _partition_score(codes.index_select(0, candidate_index)[:, probe])
        key = score + (-probe,)
        if best is None or key > best[0]:
            best = (key, probe)
    if best is None:
        raise ValueError("no unobserved probes remain")
    return best[1]


def compatible_hypotheses(codes, candidates, probe, effect, *, atol=1e-9):
    """Retain hypotheses compatible with one observed counterfactual effect."""
    candidates = tuple(int(index) for index in candidates)
    values = codes[list(candidates), int(probe)]
    keep = torch.isclose(
        values,
        torch.as_tensor(effect, dtype=values.dtype, device=values.device),
        atol=float(atol),
        rtol=0,
    )
    return tuple(index for index, valid in zip(candidates, keep.tolist()) if valid)


def identify_with_oracle(codes, target, *, max_probes=64, policy="active", seed=20260714):
    """Run an exact active or random probe trace against a hidden target code."""
    if policy not in {"active", "random"}:
        raise ValueError("policy must be active or random")
    target = int(target)
    if not 0 <= target < codes.shape[0]:
        raise ValueError("target hypothesis is out of range")
    candidates = tuple(range(codes.shape[0]))
    observed = []
    if policy == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        order = torch.randperm(codes.shape[1], generator=generator).tolist()
    else:
        order = None
    trace = []
    while len(candidates) > 1 and len(observed) < int(max_probes):
        probe = (
            select_discriminating_probe(codes, candidates, observed)
            if policy == "active"
            else order[len(observed)]
        )
        effect = codes[target, probe]
        before = len(candidates)
        candidates = compatible_hypotheses(codes, candidates, probe, effect)
        observed.append(probe)
        trace.append({
            "probe": int(probe),
            "effect": float(effect.item()),
            "candidates_before": before,
            "candidates_after": len(candidates),
            "information_bits": math.log2(before / len(candidates)),
        })
        if target not in candidates:
            raise AssertionError("oracle evidence removed its own target")
    return {
        "target": target,
        "resolved": candidates == (target,),
        "remaining": candidates,
        "observed": tuple(observed),
        "trace": tuple(trace),
    }

