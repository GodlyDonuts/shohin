"""Training-only causal prefix targets for source-free memory research.

The failure mode of final-state-only memory training is delayed credit
assignment: every source write is judged only by a final answer.  This module
turns a solver-recomputed two-register trajectory into supervision after each
write, while retaining the same inference boundary: the answer decoder sees
only the final continuous packet and a fresh query.

This is deliberately a building block, not a promoted model recipe.  Any
future GPU run must compare final-only supervision against prefix targets and
shuffled prefix labels with the same data, initialization, and answer loss.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as functional


def apply_register_operation(values: Mapping[str, int], operation: Mapping[str, object]) -> dict[str, int]:
    """Apply the solver's bounded two-register operation without model input."""
    result = {str(key): int(value) for key, value in values.items()}
    kind = operation["kind"]
    if kind == "add":
        result[str(operation["target"])] += int(operation["value"])
    elif kind == "sub":
        result[str(operation["target"])] -= int(operation["value"])
    elif kind == "move":
        source, target = str(operation["source"]), str(operation["target"])
        value = int(operation["value"])
        result[source] -= value
        result[target] += value
    elif kind == "merge":
        source, target = str(operation["source"]), str(operation["target"])
        result[target] += result[source]
    elif kind == "swap":
        left, right = str(operation["left"]), str(operation["right"])
        result[left], result[right] = result[right], result[left]
    else:
        raise ValueError("unknown register operation: {}".format(kind))
    return result


def prefix_state_targets(
    initial: Mapping[str, int],
    operations: Iterable[Mapping[str, object]],
    keys: Sequence[str],
    state_scale: int,
) -> list[list[float]]:
    """Return the exact normalized state after every serialized source chunk."""
    if not keys or state_scale <= 0:
        raise ValueError("keys and state_scale must be positive")
    values = {str(key): int(initial[str(key)]) for key in keys}
    targets = []
    for operation in operations:
        values = apply_register_operation(values, operation)
        targets.append([float(values[str(key)]) / float(state_scale) for key in keys])
    if not targets:
        raise ValueError("prefix supervision requires at least one source operation")
    return targets


def prefix_trajectory_losses(
    packets: Sequence[torch.Tensor],
    targets: torch.Tensor,
    state_predictor: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Measure state and state-delta error at every source write.

    ``packets`` contains one `[batch, slots, hidden]` packet for each source
    prefix; ``targets`` is `[batch, prefixes, registers]`.  The final packet
    remains the only object available to answer decoding.
    """
    if not packets:
        raise ValueError("packets must include at least one source prefix")
    if targets.ndim != 3 or targets.shape[1] != len(packets):
        raise ValueError("targets must have shape [batch, prefixes, registers]")
    predicted = torch.stack([state_predictor(packet) for packet in packets], dim=1)
    if predicted.shape != targets.shape:
        raise ValueError("state predictor output does not match prefix targets")
    state = functional.mse_loss(predicted, targets)
    if predicted.shape[1] == 1:
        delta = state.new_zeros(())
    else:
        delta = functional.mse_loss(predicted[:, 1:] - predicted[:, :-1], targets[:, 1:] - targets[:, :-1])
    return {"state": state, "delta": delta, "predicted": predicted}
