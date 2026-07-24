"""Proof-carrying law projection for the primary EFC board.

Neural components may propose key-role and witness costs, but they do not emit
an unconstrained machine.  This zero-parameter layer projects action costs onto
the Birkhoff polytope and observer costs onto a balanced transport polytope.
Hard scoring uses deterministic maximum-weight assignments, so every emitted
action is a permutation and every observer uses each of four answers twice.

The layer knows only the public K=8/M=3/P=2/Y=4 laws.  It never parses source
grammar, inspects targets, executes a query, or retains a proof certificate
after the ordinary fixed-width machine is serialized.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn

from episode_functor_machine import (
    HardFunctorKeys,
    HardFunctorMachine,
    MAX_ACTIONS,
    MAX_ANSWERS,
    MAX_OBSERVERS,
    MAX_STATES,
    SoftFunctorMachine,
)
from episode_functor_pointer_compiler import MAX_UNIQUE_KEYS


PRIMARY_STATES = 8
PRIMARY_ACTIONS = 3
PRIMARY_OBSERVERS = 2
PRIMARY_ANSWERS = 4
PRIMARY_KEYS = PRIMARY_STATES + PRIMARY_ACTIONS + PRIMARY_OBSERVERS


class ConstrainedTransportError(ValueError):
    """A public-law projection or hard certificate failed."""


@dataclass(frozen=True, slots=True)
class LawfulProjection:
    machine: SoftFunctorMachine
    transition_transport: torch.Tensor
    observer_transport: torch.Tensor


@dataclass(frozen=True, slots=True)
class HardKeyProjection:
    keys: HardFunctorKeys
    active_unique_indices: torch.Tensor


def _check_logits(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ConstrainedTransportError(
            f"{name} must be finite floating point {shape}"
        )


def _balanced_sinkhorn(
    logits: torch.Tensor,
    *,
    column_mass: float,
    iterations: int,
) -> torch.Tensor:
    centered = (
        logits.float()
        - logits.float().amax((-2, -1), keepdim=True)
    ).clamp(min=-60.0, max=0.0)
    values = centered.exp()
    for _ in range(iterations):
        values = values / values.sum(-1, keepdim=True).clamp_min(
            torch.finfo(values.dtype).tiny
        )
        values = (
            values
            / values.sum(-2, keepdim=True).clamp_min(
                torch.finfo(values.dtype).tiny
            )
            * column_mass
        )
    values = values / values.sum(-1, keepdim=True).clamp_min(
        torch.finfo(values.dtype).tiny
    )
    return values


def _active_logits(
    *,
    batch: int,
    maximum: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    output = torch.full(
        (batch, maximum, 2),
        -20.0,
        dtype=torch.float32,
        device=device,
    )
    output[:, :, 0] = 20.0
    output[:, :count, 0] = -20.0
    output[:, :count, 1] = 20.0
    return output


def _maximum_assignment(weights: Sequence[Sequence[float]]) -> tuple[int, ...]:
    size = len(weights)
    if size == 0 or any(len(row) != size for row in weights):
        raise ConstrainedTransportError("assignment matrix must be nonempty square")
    if any(not math.isfinite(value) for row in weights for value in row):
        raise ConstrainedTransportError("assignment matrix is not finite")
    best: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for row in range(size):
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (score, assignment) in best.items():
            for column in range(size):
                bit = 1 << column
                if mask & bit:
                    continue
                candidate = (
                    score + float(weights[row][column]),
                    assignment + (column,),
                )
                prior = updated.get(mask | bit)
                if (
                    prior is None
                    or candidate[0] > prior[0]
                    or (
                        candidate[0] == prior[0]
                        and candidate[1] < prior[1]
                    )
                ):
                    updated[mask | bit] = candidate
        best = updated
    return best[(1 << size) - 1][1]


class LawfulMachineProjector(nn.Module):
    """Zero-parameter soft and hard projection onto the primary public laws."""

    def __init__(self, *, sinkhorn_iterations: int = 64) -> None:
        super().__init__()
        if sinkhorn_iterations < 8:
            raise ConstrainedTransportError("too few Sinkhorn iterations")
        self.sinkhorn_iterations = int(sinkhorn_iterations)

    def parameter_count(self) -> int:
        return 0

    def forward(
        self,
        transition_logits: torch.Tensor,
        observer_logits: torch.Tensor,
        *,
        straight_through: bool = False,
    ) -> LawfulProjection:
        if transition_logits.ndim != 4:
            raise ConstrainedTransportError(
                "transition logits must be rank four"
            )
        batch = int(transition_logits.shape[0])
        _check_logits(
            "transition logits",
            transition_logits,
            (
                batch,
                PRIMARY_ACTIONS,
                PRIMARY_STATES,
                PRIMARY_STATES,
            ),
        )
        _check_logits(
            "observer logits",
            observer_logits,
            (
                batch,
                PRIMARY_OBSERVERS,
                PRIMARY_STATES,
                PRIMARY_ANSWERS,
            ),
        )
        if transition_logits.device != observer_logits.device:
            raise ConstrainedTransportError(
                "law projection tensors must share one device"
            )
        transitions = _balanced_sinkhorn(
            transition_logits,
            column_mass=1.0,
            iterations=self.sinkhorn_iterations,
        )
        observers = _balanced_sinkhorn(
            observer_logits,
            column_mass=2.0,
            iterations=self.sinkhorn_iterations,
        )
        transition_values = transitions.clamp_min(
            torch.finfo(transitions.dtype).tiny
        ).log()
        observer_values = observers.clamp_min(
            torch.finfo(observers.dtype).tiny
        ).log()
        if straight_through:
            hard = self.hard_project(
                transition_logits,
                observer_logits,
            )
            hard_transition = torch.full_like(
                transition_values,
                -20.0,
            )
            hard_transition.scatter_(
                -1,
                hard.action_next[
                    :,
                    :PRIMARY_ACTIONS,
                    :PRIMARY_STATES,
                ].long().unsqueeze(-1),
                20.0,
            )
            hard_observer = torch.full_like(
                observer_values,
                -20.0,
            )
            hard_observer.scatter_(
                -1,
                hard.observer_answer[
                    :,
                    :PRIMARY_OBSERVERS,
                    :PRIMARY_STATES,
                ].long().unsqueeze(-1),
                20.0,
            )
            transition_values = (
                hard_transition
                + transition_values
                - transition_values.detach()
            )
            observer_values = (
                hard_observer
                + observer_values
                - observer_values.detach()
            )
        transition_table = torch.full(
            (
                batch,
                MAX_ACTIONS,
                MAX_STATES,
                MAX_STATES,
            ),
            -60.0,
            dtype=transitions.dtype,
            device=transitions.device,
        )
        transition_table[
            :,
            :PRIMARY_ACTIONS,
            :PRIMARY_STATES,
            :PRIMARY_STATES,
        ] = transition_values
        observer_table = torch.full(
            (
                batch,
                MAX_OBSERVERS,
                MAX_STATES,
                MAX_ANSWERS,
            ),
            -60.0,
            dtype=observers.dtype,
            device=observers.device,
        )
        observer_table[
            :,
            :PRIMARY_OBSERVERS,
            :PRIMARY_STATES,
            :PRIMARY_ANSWERS,
        ] = observer_values
        machine = SoftFunctorMachine(
            state_active=_active_logits(
                batch=batch,
                maximum=MAX_STATES,
                count=PRIMARY_STATES,
                device=transitions.device,
            ),
            action_active=_active_logits(
                batch=batch,
                maximum=MAX_ACTIONS,
                count=PRIMARY_ACTIONS,
                device=transitions.device,
            ),
            observer_active=_active_logits(
                batch=batch,
                maximum=MAX_OBSERVERS,
                count=PRIMARY_OBSERVERS,
                device=transitions.device,
            ),
            action_next=transition_table,
            observer_answer=observer_table,
        )
        return LawfulProjection(
            machine=machine,
            transition_transport=transitions,
            observer_transport=observers,
        )

    @torch.no_grad()
    def hard_project(
        self,
        transition_logits: torch.Tensor,
        observer_logits: torch.Tensor,
    ) -> HardFunctorMachine:
        if transition_logits.ndim != 4:
            raise ConstrainedTransportError(
                "transition logits must be rank four"
            )
        batch = int(transition_logits.shape[0])
        _check_logits(
            "transition logits",
            transition_logits,
            (
                batch,
                PRIMARY_ACTIONS,
                PRIMARY_STATES,
                PRIMARY_STATES,
            ),
        )
        _check_logits(
            "observer logits",
            observer_logits,
            (
                batch,
                PRIMARY_OBSERVERS,
                PRIMARY_STATES,
                PRIMARY_ANSWERS,
            ),
        )
        if transition_logits.device != observer_logits.device:
            raise ConstrainedTransportError(
                "law projection tensors must share one device"
            )
        transitions = torch.zeros(
            (batch, MAX_ACTIONS, MAX_STATES),
            dtype=torch.uint8,
            device=transition_logits.device,
        )
        observers = torch.zeros(
            (batch, MAX_OBSERVERS, MAX_STATES),
            dtype=torch.uint8,
            device=observer_logits.device,
        )
        transition_cpu = transition_logits.detach().float().cpu()
        observer_cpu = observer_logits.detach().float().cpu()
        for row in range(batch):
            for action in range(PRIMARY_ACTIONS):
                assignment = _maximum_assignment(
                    transition_cpu[row, action].tolist()
                )
                transitions[row, action, :PRIMARY_STATES] = torch.tensor(
                    assignment,
                    dtype=torch.uint8,
                    device=transitions.device,
                )
            expanded_answers = tuple(
                answer
                for answer in range(PRIMARY_ANSWERS)
                for _ in range(PRIMARY_STATES // PRIMARY_ANSWERS)
            )
            for observer in range(PRIMARY_OBSERVERS):
                expanded = [
                    [
                        float(observer_cpu[row, observer, state, answer])
                        for answer in expanded_answers
                    ]
                    for state in range(PRIMARY_STATES)
                ]
                assignment = _maximum_assignment(expanded)
                observers[
                    row,
                    observer,
                    :PRIMARY_STATES,
                ] = torch.tensor(
                    tuple(expanded_answers[column] for column in assignment),
                    dtype=torch.uint8,
                    device=observers.device,
                )
        state_active = torch.zeros(
            (batch, MAX_STATES),
            dtype=torch.uint8,
            device=transition_logits.device,
        )
        action_active = torch.zeros(
            (batch, MAX_ACTIONS),
            dtype=torch.uint8,
            device=transition_logits.device,
        )
        observer_active = torch.zeros(
            (batch, MAX_OBSERVERS),
            dtype=torch.uint8,
            device=transition_logits.device,
        )
        state_active[:, :PRIMARY_STATES] = 1
        action_active[:, :PRIMARY_ACTIONS] = 1
        observer_active[:, :PRIMARY_OBSERVERS] = 1
        return HardFunctorMachine(
            state_active=state_active,
            action_active=action_active,
            observer_active=observer_active,
            action_next=transitions,
            observer_answer=observers,
        )


def project_key_assignment_logits(
    *,
    slot_assignment_logits: torch.Tensor,
    source_unique_key_valid: torch.Tensor,
    sinkhorn_iterations: int = 64,
    straight_through: bool = False,
) -> torch.Tensor:
    """Project the thirteen active semantic slots onto thirteen unique keys."""

    if slot_assignment_logits.ndim != 3:
        raise ConstrainedTransportError(
            "slot assignment logits must be rank three"
        )
    batch = int(slot_assignment_logits.shape[0])
    _check_logits(
        "slot assignment logits",
        slot_assignment_logits,
        (
            batch,
            MAX_STATES + MAX_ACTIONS + MAX_OBSERVERS,
            MAX_UNIQUE_KEYS,
        ),
    )
    if (
        source_unique_key_valid.shape != (batch, MAX_UNIQUE_KEYS)
        or source_unique_key_valid.dtype != torch.bool
        or source_unique_key_valid.device != slot_assignment_logits.device
    ):
        raise ConstrainedTransportError(
            "source key validity geometry differs"
        )
    if sinkhorn_iterations < 8:
        raise ConstrainedTransportError(
            "too few key-transport Sinkhorn iterations"
        )
    active_slots = (
        tuple(range(PRIMARY_STATES))
        + tuple(
            MAX_STATES + index
            for index in range(PRIMARY_ACTIONS)
        )
        + tuple(
            MAX_STATES + MAX_ACTIONS + index
            for index in range(PRIMARY_OBSERVERS)
        )
    )
    projected = torch.full_like(slot_assignment_logits, -60.0)
    slot_index = torch.tensor(
        active_slots,
        dtype=torch.long,
        device=slot_assignment_logits.device,
    )
    for row in range(batch):
        unique_index = source_unique_key_valid[row].nonzero().flatten()
        if unique_index.numel() != PRIMARY_KEYS:
            raise ConstrainedTransportError(
                "primary source must expose exactly thirteen unique keys"
            )
        selected = slot_assignment_logits[row].index_select(
            0,
            slot_index,
        ).index_select(1, unique_index)
        transport = _balanced_sinkhorn(
            selected[None],
            column_mass=1.0,
            iterations=sinkhorn_iterations,
        )[0]
        selected_values = transport.clamp_min(
            torch.finfo(transport.dtype).tiny
        ).log()
        if straight_through:
            assignment = _maximum_assignment(
                selected.detach().float().cpu().tolist()
            )
            hard_values = torch.full_like(selected_values, -20.0)
            hard_values[
                torch.arange(
                    PRIMARY_KEYS,
                    device=hard_values.device,
                ),
                torch.tensor(
                    assignment,
                    dtype=torch.long,
                    device=hard_values.device,
                ),
            ] = 20.0
            selected_values = (
                hard_values
                + selected_values
                - selected_values.detach()
            )
        rows = slot_index[:, None].expand(-1, PRIMARY_KEYS)
        columns = unique_index[None].expand(PRIMARY_KEYS, -1)
        projected[row, rows, columns] = selected_values
    return projected


@torch.no_grad()
def hard_assign_keys(
    *,
    slot_assignment_logits: torch.Tensor,
    source_unique_key_bytes: torch.Tensor,
    source_unique_key_valid: torch.Tensor,
) -> HardKeyProjection:
    """Select one unique copied key for every active semantic machine slot."""

    if slot_assignment_logits.ndim != 3:
        raise ConstrainedTransportError(
            "slot assignment logits must be rank three"
        )
    batch = int(slot_assignment_logits.shape[0])
    _check_logits(
        "slot assignment logits",
        slot_assignment_logits,
        (
            batch,
            MAX_STATES + MAX_ACTIONS + MAX_OBSERVERS,
            MAX_UNIQUE_KEYS,
        ),
    )
    if (
        source_unique_key_bytes.shape != (batch, MAX_UNIQUE_KEYS, 8)
        or source_unique_key_bytes.dtype != torch.uint8
        or source_unique_key_valid.shape != (batch, MAX_UNIQUE_KEYS)
        or source_unique_key_valid.dtype != torch.bool
    ):
        raise ConstrainedTransportError("source key inventory geometry differs")
    if len(
        {
            slot_assignment_logits.device,
            source_unique_key_bytes.device,
            source_unique_key_valid.device,
        }
    ) != 1:
        raise ConstrainedTransportError(
            "hard key projection tensors must share one device"
        )
    active_slots = (
        tuple(range(PRIMARY_STATES))
        + tuple(
            MAX_STATES + index
            for index in range(PRIMARY_ACTIONS)
        )
        + tuple(
            MAX_STATES + MAX_ACTIONS + index
            for index in range(PRIMARY_OBSERVERS)
        )
    )
    selected = torch.zeros(
        (batch, PRIMARY_KEYS),
        dtype=torch.long,
        device=slot_assignment_logits.device,
    )
    state_keys = torch.zeros(
        (batch, MAX_STATES, 8),
        dtype=torch.uint8,
        device=slot_assignment_logits.device,
    )
    action_keys = torch.zeros(
        (batch, MAX_ACTIONS, 8),
        dtype=torch.uint8,
        device=slot_assignment_logits.device,
    )
    observer_keys = torch.zeros(
        (batch, MAX_OBSERVERS, 8),
        dtype=torch.uint8,
        device=slot_assignment_logits.device,
    )
    logits_cpu = slot_assignment_logits.detach().float().cpu()
    valid_cpu = source_unique_key_valid.detach().cpu()
    for row in range(batch):
        valid = tuple(
            index
            for index, present in enumerate(valid_cpu[row].tolist())
            if present
        )
        if len(valid) != PRIMARY_KEYS:
            raise ConstrainedTransportError(
                "primary source must expose exactly thirteen unique keys"
            )
        weights = [
            [float(logits_cpu[row, slot, unique]) for unique in valid]
            for slot in active_slots
        ]
        assignment = _maximum_assignment(weights)
        chosen = tuple(valid[column] for column in assignment)
        selected[row] = torch.tensor(
            chosen,
            dtype=torch.long,
            device=selected.device,
        )
        state_keys[row, :PRIMARY_STATES] = source_unique_key_bytes[
            row,
            torch.tensor(
                chosen[:PRIMARY_STATES],
                device=selected.device,
            ),
        ]
        action_keys[row, :PRIMARY_ACTIONS] = source_unique_key_bytes[
            row,
            torch.tensor(
                chosen[
                    PRIMARY_STATES : PRIMARY_STATES + PRIMARY_ACTIONS
                ],
                device=selected.device,
            ),
        ]
        observer_keys[row, :PRIMARY_OBSERVERS] = source_unique_key_bytes[
            row,
            torch.tensor(
                chosen[-PRIMARY_OBSERVERS:],
                device=selected.device,
            ),
        ]
    return HardKeyProjection(
        keys=HardFunctorKeys(
            state_keys=state_keys,
            action_keys=action_keys,
            observer_keys=observer_keys,
        ),
        active_unique_indices=selected,
    )


__all__ = [
    "ConstrainedTransportError",
    "HardKeyProjection",
    "LawfulMachineProjector",
    "LawfulProjection",
    "PRIMARY_ACTIONS",
    "PRIMARY_ANSWERS",
    "PRIMARY_KEYS",
    "PRIMARY_OBSERVERS",
    "PRIMARY_STATES",
    "hard_assign_keys",
    "project_key_assignment_logits",
]
