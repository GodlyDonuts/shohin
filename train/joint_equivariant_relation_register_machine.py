"""Joint-action variant of the equivariant relation-register controller.

The factorized controller predicts operation, operands, destination, phase,
and HALT independently. This matched variant instead predicts one categorical
symbol from the complete set of legal transitions plus HALT. Both soft and
hard execution therefore operate on whole legal tuples and cannot synthesize
an illegal cross-head combination.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from equivariant_relation_register_machine import (
    CONTINUE,
    HALT,
    MAX_OBJECTS,
    MAX_STEPS,
    OPERATION_COUNT,
    PHASE_COUNT,
    READ_ONLY_REGISTERS,
    REGISTER_COUNT,
    WRITABLE_REGISTER_COUNT,
    DeletedRelationRegisterPacket,
    LateRelationRegisterQuery,
    RelationRegisterError,
    _action_affordance_features,
    _active_square,
    _invariant_features,
    relation_algebra_candidates,
)


LEGAL_TRANSITION_COUNT = (
    OPERATION_COUNT
    * REGISTER_COUNT
    * REGISTER_COUNT
    * WRITABLE_REGISTER_COUNT
    * PHASE_COUNT
)
HALT_ACTION_INDEX = LEGAL_TRANSITION_COUNT
JOINT_ACTION_COUNT = LEGAL_TRANSITION_COUNT + 1


@dataclass(frozen=True, slots=True)
class LegalRelationTransition:
    operation: int
    left: int
    right: int
    destination: int
    next_phase: int


def encode_legal_transition(
    operation: int,
    left: int,
    right: int,
    destination: int,
    next_phase: int,
) -> int:
    """Encode one legal transition into its joint categorical index."""

    values = (operation, left, right, destination, next_phase)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise RelationRegisterError("joint transition indices differ")
    if not 0 <= operation < OPERATION_COUNT:
        raise RelationRegisterError("joint operation leaves its domain")
    if not 0 <= left < REGISTER_COUNT or not 0 <= right < REGISTER_COUNT:
        raise RelationRegisterError("joint operand leaves its domain")
    if not READ_ONLY_REGISTERS <= destination < REGISTER_COUNT:
        raise RelationRegisterError("joint destination is not writable")
    if not 0 <= next_phase < PHASE_COUNT:
        raise RelationRegisterError("joint phase leaves its domain")
    writable_destination = destination - READ_ONLY_REGISTERS
    index = operation
    index = index * REGISTER_COUNT + left
    index = index * REGISTER_COUNT + right
    index = index * WRITABLE_REGISTER_COUNT + writable_destination
    return index * PHASE_COUNT + next_phase


def decode_legal_transition(index: int) -> LegalRelationTransition:
    """Decode one non-HALT joint index back into its legal transition."""

    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < LEGAL_TRANSITION_COUNT
    ):
        raise RelationRegisterError("joint transition index leaves its domain")
    remainder, next_phase = divmod(index, PHASE_COUNT)
    remainder, writable_destination = divmod(
        remainder,
        WRITABLE_REGISTER_COUNT,
    )
    remainder, right = divmod(remainder, REGISTER_COUNT)
    operation, left = divmod(remainder, REGISTER_COUNT)
    return LegalRelationTransition(
        operation=operation,
        left=left,
        right=right,
        destination=writable_destination + READ_ONLY_REGISTERS,
        next_phase=next_phase,
    )


def _straight_through_joint(
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


def expected_joint_controller_parameters(
    *,
    controller_width: int,
    controller_layers: int,
) -> int:
    """Return the exact trainable parameter count from architecture geometry."""

    if controller_width < 8 or controller_layers < 1:
        raise RelationRegisterError("controller geometry differs")
    feature_width = (
        4 * REGISTER_COUNT
        + REGISTER_COUNT * REGISTER_COUNT
        + 2 * OPERATION_COUNT * REGISTER_COUNT**3
    )
    projection = (feature_width + PHASE_COUNT) * controller_width
    projection += controller_width
    controller = controller_layers * (
        controller_width * controller_width + 3 * controller_width
    )
    joint_head = controller_width * JOINT_ACTION_COUNT + JOINT_ACTION_COUNT
    return projection + controller + joint_head


@dataclass(frozen=True, slots=True)
class JointControllerAction:
    joint: torch.Tensor
    transition: torch.Tensor
    operation: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    destination: torch.Tensor
    halt: torch.Tensor
    phase: torch.Tensor
    joint_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class JointRelationRegisterRollout:
    final_registers: torch.Tensor
    answer: torch.Tensor
    actions: tuple[JointControllerAction, ...]
    register_trajectory: tuple[torch.Tensor, ...]
    alive_trajectory: tuple[torch.Tensor, ...]
    halt_trajectory: tuple[torch.Tensor, ...]
    halted_by_deadline: torch.Tensor


class JointEquivariantRelationRegisterMachine(nn.Module):
    """Equivariant relation machine with one categorical legal-action head."""

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
        self.controller_layers = int(controller_layers)
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
        self.joint_head = nn.Linear(controller_width, JOINT_ACTION_COUNT)

        indices = torch.arange(LEGAL_TRANSITION_COUNT)
        phase = indices.remainder(PHASE_COUNT)
        quotient = indices.div(PHASE_COUNT, rounding_mode="floor")
        destination = quotient.remainder(WRITABLE_REGISTER_COUNT)
        quotient = quotient.div(
            WRITABLE_REGISTER_COUNT,
            rounding_mode="floor",
        )
        right = quotient.remainder(REGISTER_COUNT)
        quotient = quotient.div(REGISTER_COUNT, rounding_mode="floor")
        left = quotient.remainder(REGISTER_COUNT)
        operation = quotient.div(REGISTER_COUNT, rounding_mode="floor")
        self.register_buffer(
            "_operation_decoder",
            F.one_hot(operation, OPERATION_COUNT).float(),
            persistent=False,
        )
        self.register_buffer(
            "_left_decoder",
            F.one_hot(left, REGISTER_COUNT).float(),
            persistent=False,
        )
        self.register_buffer(
            "_right_decoder",
            F.one_hot(right, REGISTER_COUNT).float(),
            persistent=False,
        )
        self.register_buffer(
            "_destination_decoder",
            F.one_hot(destination, WRITABLE_REGISTER_COUNT).float(),
            persistent=False,
        )
        self.register_buffer(
            "_phase_decoder",
            F.one_hot(phase, PHASE_COUNT).float(),
            persistent=False,
        )
        with torch.no_grad():
            self.joint_head.bias.zero_()
            self.joint_head.bias[HALT_ACTION_INDEX] = -8.0

        if self.added_parameters != expected_joint_controller_parameters(
            controller_width=controller_width,
            controller_layers=controller_layers,
        ):
            raise RelationRegisterError("joint controller parameter receipt differs")

    @property
    def added_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _action(
        self,
        hidden: torch.Tensor,
        current_phase: torch.Tensor,
        *,
        hard: bool,
    ) -> JointControllerAction:
        joint_logits = self.joint_head(hidden)
        joint = _straight_through_joint(joint_logits, hard=hard)
        transitions = joint[:, :LEGAL_TRANSITION_COUNT]
        continue_probability = transitions.sum(-1)
        halt_probability = joint[:, HALT_ACTION_INDEX]
        denominator = continue_probability[:, None].clamp_min(
            torch.finfo(transitions.dtype).tiny
        )
        conditional = transitions / denominator
        has_transition = continue_probability.gt(0)[:, None]
        conditional = torch.where(
            has_transition,
            conditional,
            torch.zeros_like(conditional),
        )
        transition = conditional.reshape(
            -1,
            OPERATION_COUNT,
            REGISTER_COUNT,
            REGISTER_COUNT,
            WRITABLE_REGISTER_COUNT,
            PHASE_COUNT,
        )
        operation = conditional @ self._operation_decoder
        left = conditional @ self._left_decoder
        right = conditional @ self._right_decoder
        writable_destination = conditional @ self._destination_decoder
        destination = F.pad(
            writable_destination,
            (READ_ONLY_REGISTERS, 0),
        )
        next_phase = conditional @ self._phase_decoder
        next_phase = torch.where(
            has_transition,
            next_phase,
            current_phase,
        )
        halt = torch.stack(
            (continue_probability, halt_probability),
            dim=-1,
        )
        return JointControllerAction(
            joint=joint,
            transition=transition,
            operation=operation,
            left=left,
            right=right,
            destination=destination,
            halt=halt,
            phase=next_phase,
            joint_logits=joint_logits,
        )

    @staticmethod
    def _propose(
        registers: torch.Tensor,
        active: torch.Tensor,
        transition: torch.Tensor,
    ) -> torch.Tensor:
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
        state_weights = transition.sum(-1)
        destination_mass = state_weights.sum(dim=(1, 2, 3))
        selected_mass = torch.einsum(
            "bolrd,bolrij->bdij",
            state_weights,
            candidates,
        )
        writable = (
            (1.0 - destination_mass[..., None, None])
            * registers[:, READ_ONLY_REGISTERS:]
            + selected_mass
        )
        return torch.cat(
            (registers[:, :READ_ONLY_REGISTERS], writable),
            dim=1,
        )

    def forward(
        self,
        packet: DeletedRelationRegisterPacket,
        query: LateRelationRegisterQuery,
        *,
        hard: bool = False,
    ) -> JointRelationRegisterRollout:
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
        actions: list[JointControllerAction] = []
        register_trajectory: list[torch.Tensor] = []
        alive_trajectory: list[torch.Tensor] = []
        halt_trajectory: list[torch.Tensor] = []

        for _ in range(self.maximum_steps):
            features = torch.cat(
                (
                    _invariant_features(registers, active),
                    _action_affordance_features(registers, active).detach(),
                ),
                dim=-1,
            )
            hidden = self.controller(
                self.feature_projection(torch.cat((features, phase), dim=-1))
            )
            action = self._action(
                hidden,
                phase,
                hard=hard,
            )
            proposed = self._propose(
                registers,
                active,
                action.transition,
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
        return JointRelationRegisterRollout(
            final_registers=final_registers,
            answer=answer,
            actions=tuple(actions),
            register_trajectory=tuple(register_trajectory),
            alive_trajectory=tuple(alive_trajectory),
            halt_trajectory=tuple(halt_trajectory),
            halted_by_deadline=alive.le(1e-6),
        )


def joint_controller_parameter_receipt(
    machine: JointEquivariantRelationRegisterMachine,
    *,
    base_parameters: int = 125_081_664,
    strict_cap: int = 200_000_000,
) -> dict[str, int]:
    added = machine.added_parameters
    expected = expected_joint_controller_parameters(
        controller_width=machine.controller_width,
        controller_layers=machine.controller_layers,
    )
    if added != expected:
        raise RelationRegisterError("joint controller parameter receipt differs")
    complete = base_parameters + added
    if complete >= strict_cap:
        raise RelationRegisterError("joint relation register machine reaches cap")
    return {
        "base": int(base_parameters),
        "added": int(added),
        "complete_system": int(complete),
        "strict_cap": int(strict_cap),
        "headroom": int(strict_cap - complete),
    }
