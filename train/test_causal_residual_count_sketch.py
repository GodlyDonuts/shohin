#!/usr/bin/env python3
"""CPU-only CRCS invariants; no model or CUDA allocation is required."""
from __future__ import annotations

import torch

from causal_residual_count_sketch import assignment, collision_summary, flat_sum, sketch_events


events = torch.arange(2 * 5 * 3 * 4, dtype=torch.float32).reshape(2, 5, 3, 4)
locations, signs = assignment(5, rows=3, buckets=4, seed=11)
assert locations.shape == signs.shape == (3, 5)
assert set(int(value) for value in signs.flatten()) == {-1, 1}
first = sketch_events(events, rows=3, buckets=4, seed=11)
second = sketch_events(events, rows=3, buckets=4, seed=11)
assert first.shape == (2, 3 * 4 * 3, 4)
assert torch.equal(first, second)
assert torch.equal(sketch_events(torch.zeros_like(events), rows=3, buckets=4, seed=11), torch.zeros_like(first))
assert torch.equal(flat_sum(events), events.sum(dim=1))
swapped = events.clone()
swapped[:, [0, 1]] = swapped[:, [1, 0]]
assert not torch.equal(first, sketch_events(swapped, rows=3, buckets=4, seed=11))
gradient_events = events.clone().requires_grad_(True)
sketch_events(gradient_events, rows=3, buckets=4, seed=11).square().mean().backward()
assert gradient_events.grad is not None and torch.isfinite(gradient_events.grad).all()
summary = collision_summary(16, rows=4, buckets=4, seed=11)
assert summary["lanes"] == 16 and summary["max_bucket_load"] >= 1
assert len(summary["occupancies"]) == 4
print("causal residual count-sketch checks: passed")
