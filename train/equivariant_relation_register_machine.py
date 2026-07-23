"""Query-blind learned controller over an equivariant relation-algebra machine.

This module is an architectural executor, not a language compiler. It receives
only a source-deleted packet of raw relations and a cardinality. A learned
permutation-invariant controller selects algebra operations, operands,
destinations, and HALT. The late query is applied only after execution.

Every operation is evaluated tensorially. No Python branch is controlled by a
predicted semantic action, and a missing HALT is retained as invalid rather
than repaired.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_OBJECTS = 8
REGISTER_COUNT = 6
READ_ONLY_REGISTERS = 3
WRITABLE_REGISTER_COUNT = REGISTER_COUNT - READ_ONLY_REGISTERS
PHASE_COUNT = 3
MAX_STEPS = 24
CONTINUE = 0
HALT = 1


class RelationOperation(IntEnum):
    COMPOSE = 0
    UNION = 1
    INTERSECTION = 2
    DIFFERENCE = 3
    CONVERSE = 4
    COPY = 5
    CLEAR = 6
    IDENTITY = 7
    EXPAND = 8


OPERATION_COUNT = len(RelationOperation)


class RelationRegisterError(ValueError):
    """Raised when a source-deleted register packet leaves its contract."""


def _active_square(cardinality: torch.Tensor) -> torch.Tensor:
    if cardinality.ndim != 1 or cardinality.dtype not in {
        torch.uint8,
        torch.int32,
        torch.int64,
    }:
        raise RelationRegisterError("register cardinality differs")
    if (
        cardinality.numel() == 0
        or int(cardinality.min()) < 2
        or int(cardinality.max()) > MAX_OBJECTS
    ):
        raise RelationRegisterError("register cardinality leaves its domain")
    positions = torch.arange(MAX_OBJECTS, device=cardinality.device)
    active = positions[None] < cardinality.long()[:, None]
    return active[:, :, None] & active[:, None, :]


def boolean_relation_compose(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    if (
        left.ndim != 3
        or right.shape != left.shape
        or left.shape[-2:] != (MAX_OBJECTS, MAX_OBJECTS)
    ):
        raise RelationRegisterError("register relation geometry differs")
    paths = left[:, :, :, None] * right[:, None, :, :]
    return 1.0 - (1.0 - paths).prod(dim=2)


@dataclass(frozen=True, slots=True)
class DeletedRelationRegisterPacket:
    """Raw source-deleted relations available before late query disclosure."""

    cardinality: torch.Tensor
    registers: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.registers, torch.Tensor)
            or self.registers.ndim != 4
            or self.registers.shape[1:] != (
                REGISTER_COUNT,
                MAX_OBJECTS,
                MAX_OBJECTS,
            )
            or not self.registers.is_floating_point()
            or not bool(torch.isfinite(self.registers).all())
        ):
            raise RelationRegisterError("source-deleted registers differ")
        if self.cardinality.shape != (self.registers.shape[0],):
            raise RelationRegisterError("register packet batch differs")
        active = _active_square(self.cardinality)
        outside = ~active[:, None]
        if bool(self.registers.masked_select(outside).ne(0).any()):
            raise RelationRegisterError("register packet has covert outside state")
        if bool(
            (self.registers < 0).any() or (self.registers > 1).any()
        ):
            raise RelationRegisterError("register values leave probability domain")

    @property
    def batch_size(self) -> int:
        return int(self.registers.shape[0])


@dataclass(frozen=True, slots=True)
class LateRelationRegisterQuery:
    register: torch.Tensor
    position: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.register.ndim != 1
            or self.position.ndim != 1
            or self.register.shape != self.position.shape
            or self.register.dtype != torch.long
            or self.position.dtype != torch.long
            or (
                self.register.numel() > 0
                and (
                    int(self.register.min()) < 0
                    or int(self.register.max()) >= REGISTER_COUNT
                    or int(self.position.min()) < 0
                    or int(self.position.max()) >= MAX_OBJECTS
                )
            )
        ):
            raise RelationRegisterError("late register query differs")


@dataclass(frozen=True, slots=True)
class ControllerAction:
    operation: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    destination: torch.Tensor
    halt: torch.Tensor
    phase: torch.Tensor
    operation_logits: torch.Tensor
    left_logits: torch.Tensor
    right_logits: torch.Tensor
    destination_logits: torch.Tensor
    halt_logits: torch.Tensor
    phase_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class RelationRegisterRollout:
    final_registers: torch.Tensor
    answer: torch.Tensor
    actions: tuple[ControllerAction, ...]
    register_trajectory: tuple[torch.Tensor, ...]
    alive_trajectory: tuple[torch.Tensor, ...]
    halt_trajectory: tuple[torch.Tensor, ...]
    halted_by_deadline: torch.Tensor


def _straight_through_one_hot(
    logits: torch.Tensor,
    *,
    hard: bool,
) -> torch.Tensor:
    probabilities = logits.float().softmax(-1)
    if not hard:
        return probabilities
    selected = F.one_hot(
        probabilities.argmax(-1),
        probabilities.shape[-1],
    ).float()
    return selected + probabilities - probabilities.detach()


def relation_algebra_candidates(
    left: torch.Tensor,
    right: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Return every primitive operation without semantic host branching."""

    if left.shape != right.shape or left.shape != active.shape:
        raise RelationRegisterError("relation-algebra operands differ")
    compose = boolean_relation_compose(left, right)
    union = 1.0 - (1.0 - left) * (1.0 - right)
    intersection = left * right
    difference = left * (1.0 - right)
    converse = left.transpose(-1, -2)
    copy = left
    clear = torch.zeros_like(left)
    identity = torch.eye(
        MAX_OBJECTS,
        device=left.device,
        dtype=left.dtype,
    )[None].expand_as(left)
    expand = 1.0 - (1.0 - right) * (
        1.0 - boolean_relation_compose(left, right)
    )
    candidates = torch.stack(
        (
            compose,
            union,
            intersection,
            difference,
            converse,
            copy,
            clear,
            identity,
            expand,
        ),
        dim=1,
    )
    return candidates * active[:, None]


def _invariant_features(
    registers: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    active_float = active[:, None].float()
    count = active_float.sum(dim=(-1, -2)).clamp_min(1.0)
    density = (registers * active_float).sum(dim=(-1, -2)) / count
    diagonal = registers.diagonal(dim1=-2, dim2=-1)
    active_diagonal = active.diagonal(dim1=-2, dim2=-1)[:, None]
    diagonal_density = (
        diagonal * active_diagonal.float()
    ).sum(-1) / active_diagonal.float().sum(-1).clamp_min(1.0)
    row_degree = registers.sum(-1)
    active_rows = active.any(-1)[:, None]
    object_count = active_rows.float().sum(-1).clamp_min(1.0)
    row_mean = (
        row_degree * active_rows.float()
    ).sum(-1) / object_count
    centered = row_degree - row_mean[..., None]
    row_variance = (
        centered.square() * active_rows.float()
    ).sum(-1) / object_count
    pairwise_difference = (
        registers[:, :, None] - registers[:, None, :]
    ).abs()
    pairwise_difference = (
        pairwise_difference * active[:, None, None].float()
    ).sum(dim=(-1, -2)) / count[:, None]
    return torch.cat(
        (
            density,
            diagonal_density,
            row_mean / object_count,
            row_variance / object_count.square(),
            pairwise_difference.flatten(1),
        ),
        dim=-1,
    )


def _action_affordance_features(
    registers: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Invariant magnitude of every legal action's prospective state change."""

    batch = registers.shape[0]
    left = registers[:, :, None].expand(
        -1,
        -1,
        REGISTER_COUNT,
        -1,
        -1,
    )
    right = registers[:, None, :].expand(
        -1,
        REGISTER_COUNT,
        -1,
        -1,
        -1,
    )
    pair_active = active[:, None, None].expand(
        -1,
        REGISTER_COUNT,
        REGISTER_COUNT,
        -1,
        -1,
    )
    candidates = relation_algebra_candidates(
        left.reshape(-1, MAX_OBJECTS, MAX_OBJECTS),
        right.reshape(-1, MAX_OBJECTS, MAX_OBJECTS),
        pair_active.reshape(-1, MAX_OBJECTS, MAX_OBJECTS),
    )
    candidates = candidates.reshape(
        batch,
        REGISTER_COUNT,
        REGISTER_COUNT,
        OPERATION_COUNT,
        MAX_OBJECTS,
        MAX_OBJECTS,
    ).permute(0, 3, 1, 2, 4, 5)
    destination = registers[
        :,
        None,
        None,
        None,
        :,
    ]
    difference = (
        candidates[:, :, :, :, None] - destination
    ).abs()
    mask = active[:, None, None, None, None].float()
    normalizer = mask.sum(dim=(-1, -2)).clamp_min(1.0)
    magnitude = (difference * mask).sum(dim=(-1, -2)) / normalizer
    maximum = (difference * mask).amax(dim=(-1, -2))
    return torch.cat((magnitude.flatten(1), maximum.flatten(1)), dim=-1)


class EquivariantRelationRegisterMachine(nn.Module):
    """Learned controller with object-permutation-invariant decisions."""

    def __init__(
        self,
        *,
        controller_width: int = 512,
        controller_layers: int = 3,
        maximum_steps: int = MAX_STEPS,
    ) -> None:
        super().__init__()
        if (
            controller_width < 8
            or controller_layers < 1
            or not 1 <= maximum_steps <= MAX_STEPS
        ):
            raise RelationRegisterError("controller geometry differs")
        self.controller_width = int(controller_width)
        self.maximum_steps = int(maximum_steps)
        feature_width = (
            4 * REGISTER_COUNT
            + REGISTER_COUNT * REGISTER_COUNT
            + 2 * OPERATION_COUNT * REGISTER_COUNT**3
        )
        self.feature_projection = nn.Linear(
            feature_width + PHASE_COUNT,
            controller_width,
        )
        layers: list[nn.Module] = []
        for _ in range(controller_layers):
            layers.extend(
                (
                    nn.Linear(controller_width, controller_width),
                    nn.GELU(),
                    nn.LayerNorm(controller_width),
                )
            )
        self.controller = nn.Sequential(*layers)
        self.operation_head = nn.Linear(controller_width, OPERATION_COUNT)
        self.left_head = nn.Linear(controller_width, REGISTER_COUNT)
        self.right_head = nn.Linear(controller_width, REGISTER_COUNT)
        self.destination_head = nn.Linear(
            controller_width,
            WRITABLE_REGISTER_COUNT,
        )
        self.halt_head = nn.Linear(controller_width, 2)
        self.phase_head = nn.Linear(controller_width, PHASE_COUNT)
        with torch.no_grad():
            self.halt_head.bias.copy_(torch.tensor([4.0, -4.0]))

    @property
    def added_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _action(
        self,
        hidden: torch.Tensor,
        *,
        hard: bool,
    ) -> ControllerAction:
        operation_logits = self.operation_head(hidden)
        left_logits = self.left_head(hidden)
        right_logits = self.right_head(hidden)
        destination_logits = self.destination_head(hidden)
        halt_logits = self.halt_head(hidden)
        phase_logits = self.phase_head(hidden)
        writable_destination = _straight_through_one_hot(
            destination_logits,
            hard=hard,
        )
        destination = F.pad(
            writable_destination,
            (READ_ONLY_REGISTERS, 0),
        )
        return ControllerAction(
            operation=_straight_through_one_hot(
                operation_logits,
                hard=hard,
            ),
            left=_straight_through_one_hot(
                left_logits,
                hard=hard,
            ),
            right=_straight_through_one_hot(
                right_logits,
                hard=hard,
            ),
            destination=destination,
            halt=_straight_through_one_hot(
                halt_logits,
                hard=hard,
            ),
            phase=_straight_through_one_hot(
                phase_logits,
                hard=hard,
            ),
            operation_logits=operation_logits,
            left_logits=left_logits,
            right_logits=right_logits,
            destination_logits=destination_logits,
            halt_logits=halt_logits,
            phase_logits=phase_logits,
        )

    def forward(
        self,
        packet: DeletedRelationRegisterPacket,
        query: LateRelationRegisterQuery,
        *,
        hard: bool = False,
    ) -> RelationRegisterRollout:
        if query.register.shape != (packet.batch_size,):
            raise RelationRegisterError("register packet/query batch differs")
        if bool(query.position.ge(packet.cardinality.long()).any()):
            raise RelationRegisterError("late query leaves active cardinality")
        active = _active_square(packet.cardinality)
        registers = packet.registers.float()
        phase = torch.zeros(
            packet.batch_size,
            PHASE_COUNT,
            device=registers.device,
        )
        phase[:, 0] = 1.0
        alive = torch.ones(packet.batch_size, device=registers.device)
        halted = torch.zeros_like(registers)
        actions: list[ControllerAction] = []
        register_trajectory: list[torch.Tensor] = []
        alive_trajectory: list[torch.Tensor] = []
        halt_trajectory: list[torch.Tensor] = []

        for _ in range(self.maximum_steps):
            features = torch.cat(
                (
                    _invariant_features(registers, active),
                    _action_affordance_features(
                        registers,
                        active,
                    ).detach(),
                ),
                dim=-1,
            )
            hidden = self.controller(
                self.feature_projection(torch.cat((features, phase), dim=-1))
            )
            action = self._action(hidden, hard=hard)
            left = torch.einsum("br,brij->bij", action.left, registers)
            right = torch.einsum("br,brij->bij", action.right, registers)
            candidates = relation_algebra_candidates(left, right, active)
            selected = torch.einsum(
                "bo,boij->bij",
                action.operation,
                candidates,
            )
            destination = action.destination[..., None, None]
            proposed = (
                (1.0 - destination) * registers
                + destination * selected[:, None]
            )
            halt_probability = action.halt[:, HALT]
            continue_probability = action.halt[:, CONTINUE]
            halted = halted + (
                alive * halt_probability
            )[:, None, None, None] * registers
            alive = alive * continue_probability
            registers = proposed
            committed = halted + alive[:, None, None, None] * registers
            actions.append(action)
            register_trajectory.append(committed)
            alive_trajectory.append(alive)
            halt_trajectory.append(1.0 - alive)
            phase = action.phase
        final_registers = halted + alive[:, None, None, None] * registers
        batch = torch.arange(packet.batch_size, device=registers.device)
        answer = final_registers[
            batch,
            query.register,
            query.position,
        ]
        active_objects = (
            torch.arange(MAX_OBJECTS, device=registers.device)[None]
            < packet.cardinality.long()[:, None]
        )
        answer = answer * active_objects.float()
        return RelationRegisterRollout(
            final_registers=final_registers,
            answer=answer,
            actions=tuple(actions),
            register_trajectory=tuple(register_trajectory),
            alive_trajectory=tuple(alive_trajectory),
            halt_trajectory=tuple(halt_trajectory),
            halted_by_deadline=alive.le(1e-6),
        )


def controller_parameter_receipt(
    machine: EquivariantRelationRegisterMachine,
    *,
    base_parameters: int = 125_081_664,
    strict_cap: int = 200_000_000,
) -> dict[str, int]:
    added = machine.added_parameters
    complete = base_parameters + added
    if complete >= strict_cap:
        raise RelationRegisterError("relation register machine reaches cap")
    return {
        "base": int(base_parameters),
        "added": int(added),
        "complete_system": int(complete),
        "strict_cap": int(strict_cap),
        "headroom": int(strict_cap - complete),
    }
