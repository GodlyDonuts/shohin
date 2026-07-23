"""Source-deleted tensor machine for contextual recursive relation programs.

This is a structured execution substrate, not a language compiler. A packet
contains anonymous relation constants, physical program-node records, opaque
operation slots, argument links, and equation roots. A separate contextual
binder supplies a per-episode mapping from opaque slots to a tied primitive
bank.

Execution uses fixed synchronous tensor loops. No predicted operation, link,
fixed-point transition, or halt decision controls Python flow. The query is
applied only after the private terminal state is committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn

from equivariant_relation_register_machine import (
    MAX_OBJECTS,
    RelationRegisterError,
    _active_square,
    boolean_relation_compose,
)


MAX_PROGRAM_NODES = 96
MAX_PROGRAM_CONSTANTS = 12
MAX_OPERATION_SLOTS = 8
PROGRAM_VARIABLES = 2


class ProgramNodeKind(IntEnum):
    CONSTANT = 0
    VARIABLE = 1
    OPERATION = 2


class ProgramPrimitive(IntEnum):
    UNION = 0
    INTERSECTION = 1
    COMPOSE = 2
    CONVERSE = 3
    IDENTITY = 4


PROGRAM_PRIMITIVE_COUNT = len(ProgramPrimitive)
PRIMITIVE_ARITY = torch.tensor((2, 2, 2, 1, 0), dtype=torch.long)


class ContextualProgramError(ValueError):
    """Raised when a private relation-program packet violates its contract."""


def _require_shape(
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    label: str,
) -> None:
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
        raise ContextualProgramError(f"{label} geometry differs")


def _gather_relations(
    bank: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather relation matrices from a batched bank."""

    if (
        bank.ndim != 4
        or indices.ndim != 2
        or bank.shape[0] != indices.shape[0]
        or bank.shape[-2:] != (MAX_OBJECTS, MAX_OBJECTS)
    ):
        raise ContextualProgramError("relation gather geometry differs")
    gather = indices.clamp_min(0)[..., None, None].expand(
        -1,
        -1,
        MAX_OBJECTS,
        MAX_OBJECTS,
    )
    return bank.gather(1, gather)


def _program_depths(
    packet: DeletedContextualProgramPacket,
) -> tuple[int, ...]:
    """Return root depths while rejecting physical operation-graph cycles."""

    depths: list[int] = []
    for batch_index in range(packet.batch_size):
        memo: dict[int, int] = {}
        active: set[int] = set()

        def visit(node_index: int) -> int:
            if node_index in memo:
                return memo[node_index]
            if node_index in active:
                raise ContextualProgramError("program operation graph is cyclic")
            active.add(node_index)
            kind = int(packet.node_kind[batch_index, node_index])
            if kind != int(ProgramNodeKind.OPERATION):
                depth = 0
            else:
                slot = int(packet.operation_slot[batch_index, node_index])
                arity = int(packet.slot_arity[batch_index, slot])
                references = (
                    int(packet.left_index[batch_index, node_index]),
                    int(packet.right_index[batch_index, node_index]),
                )
                depth = 1 + max(
                    (visit(reference) for reference in references[:arity]),
                    default=-1,
                )
            active.remove(node_index)
            memo[node_index] = depth
            return depth

        depth = max(
            visit(int(root))
            for root in packet.equation_root[batch_index]
        )
        valid = {
            index
            for index in range(MAX_PROGRAM_NODES)
            if bool(packet.node_valid[batch_index, index])
        }
        if set(memo) != valid:
            raise ContextualProgramError(
                "program packet contains disconnected nodes"
            )
        depths.append(depth)
    return tuple(depths)


@dataclass(frozen=True, slots=True)
class DeletedContextualProgramPacket:
    """Private graph available after source and compiler-state deletion."""

    cardinality: torch.Tensor
    constants: torch.Tensor
    constant_valid: torch.Tensor
    node_valid: torch.Tensor
    node_kind: torch.Tensor
    constant_index: torch.Tensor
    variable_index: torch.Tensor
    operation_slot: torch.Tensor
    left_index: torch.Tensor
    right_index: torch.Tensor
    equation_root: torch.Tensor
    slot_arity: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.constants, torch.Tensor)
            or self.constants.ndim != 4
            or self.constants.shape[1:] != (
                MAX_PROGRAM_CONSTANTS,
                MAX_OBJECTS,
                MAX_OBJECTS,
            )
            or not self.constants.is_floating_point()
            or not bool(torch.isfinite(self.constants).all())
            or bool((self.constants < 0).any())
            or bool((self.constants > 1).any())
        ):
            raise ContextualProgramError("program constants differ")
        batch = self.constants.shape[0]
        _require_shape(self.cardinality, (batch,), "program cardinality")
        _require_shape(
            self.constant_valid,
            (batch, MAX_PROGRAM_CONSTANTS),
            "constant validity",
        )
        _require_shape(
            self.node_valid,
            (batch, MAX_PROGRAM_NODES),
            "node validity",
        )
        for tensor, label in (
            (self.node_kind, "node kind"),
            (self.constant_index, "node constant index"),
            (self.variable_index, "node variable index"),
            (self.operation_slot, "node operation slot"),
            (self.left_index, "left argument"),
            (self.right_index, "right argument"),
        ):
            _require_shape(tensor, (batch, MAX_PROGRAM_NODES), label)
            if tensor.dtype != torch.long:
                raise ContextualProgramError(f"{label} dtype differs")
        _require_shape(
            self.equation_root,
            (batch, PROGRAM_VARIABLES),
            "equation roots",
        )
        _require_shape(
            self.slot_arity,
            (batch, MAX_OPERATION_SLOTS),
            "operation-slot arity",
        )
        if (
            self.cardinality.dtype
            not in {torch.uint8, torch.int32, torch.int64}
            or self.constant_valid.dtype != torch.bool
            or self.node_valid.dtype != torch.bool
            or self.equation_root.dtype != torch.long
            or self.slot_arity.dtype != torch.long
        ):
            raise ContextualProgramError("program packet dtype differs")

        active = _active_square(self.cardinality)
        if bool(
            self.constants.masked_select(~active[:, None]).ne(0).any()
        ):
            raise ContextualProgramError("program constants contain outside state")
        if bool(
            self.constants.masked_select(
                ~self.constant_valid[..., None, None]
            ).ne(0).any()
        ):
            raise ContextualProgramError("invalid constants contain covert state")
        if bool(
            self.node_kind[self.node_valid].lt(0).any()
            or self.node_kind[self.node_valid].ge(len(ProgramNodeKind)).any()
        ):
            raise ContextualProgramError("valid node kind leaves its domain")
        if bool(
            self.equation_root.lt(0).any()
            or self.equation_root.ge(MAX_PROGRAM_NODES).any()
        ):
            raise ContextualProgramError("equation root leaves node domain")
        roots_valid = self.node_valid.gather(1, self.equation_root)
        if not bool(roots_valid.all()):
            raise ContextualProgramError("equation root references invalid node")
        if bool(
            self.slot_arity.lt(-1).any()
            or self.slot_arity.gt(2).any()
        ):
            raise ContextualProgramError("operation-slot arity leaves its domain")

        for batch_index in range(batch):
            valid_nodes = int(self.node_valid[batch_index].sum())
            if valid_nodes < PROGRAM_VARIABLES:
                raise ContextualProgramError("program has too few valid nodes")
            for node_index in range(MAX_PROGRAM_NODES):
                valid = bool(self.node_valid[batch_index, node_index])
                if not valid:
                    fields = (
                        self.node_kind[batch_index, node_index],
                        self.constant_index[batch_index, node_index],
                        self.variable_index[batch_index, node_index],
                        self.operation_slot[batch_index, node_index],
                        self.left_index[batch_index, node_index],
                        self.right_index[batch_index, node_index],
                    )
                    if any(int(value) != -1 for value in fields):
                        raise ContextualProgramError(
                            "invalid node contains covert state"
                        )
                    continue
                kind = int(self.node_kind[batch_index, node_index])
                if kind == ProgramNodeKind.CONSTANT:
                    constant = int(
                        self.constant_index[batch_index, node_index]
                    )
                    if (
                        not 0 <= constant < MAX_PROGRAM_CONSTANTS
                        or not bool(self.constant_valid[batch_index, constant])
                    ):
                        raise ContextualProgramError(
                            "constant node references invalid constant"
                        )
                    irrelevant = (
                        self.variable_index[batch_index, node_index],
                        self.operation_slot[batch_index, node_index],
                        self.left_index[batch_index, node_index],
                        self.right_index[batch_index, node_index],
                    )
                    if any(int(value) != -1 for value in irrelevant):
                        raise ContextualProgramError(
                            "constant node contains covert state"
                        )
                elif kind == ProgramNodeKind.VARIABLE:
                    variable = int(
                        self.variable_index[batch_index, node_index]
                    )
                    if not 0 <= variable < PROGRAM_VARIABLES:
                        raise ContextualProgramError(
                            "variable node leaves its domain"
                        )
                    irrelevant = (
                        self.constant_index[batch_index, node_index],
                        self.operation_slot[batch_index, node_index],
                        self.left_index[batch_index, node_index],
                        self.right_index[batch_index, node_index],
                    )
                    if any(int(value) != -1 for value in irrelevant):
                        raise ContextualProgramError(
                            "variable node contains covert state"
                        )
                else:
                    slot = int(self.operation_slot[batch_index, node_index])
                    if (
                        not 0 <= slot < MAX_OPERATION_SLOTS
                        or int(self.slot_arity[batch_index, slot]) < 0
                    ):
                        raise ContextualProgramError(
                            "operation node references invalid slot"
                        )
                    arity = int(self.slot_arity[batch_index, slot])
                    references = (
                        int(self.left_index[batch_index, node_index]),
                        int(self.right_index[batch_index, node_index]),
                    )
                    required = references[:arity]
                    if any(
                        reference < 0
                        or reference >= MAX_PROGRAM_NODES
                        or not bool(
                            self.node_valid[batch_index, reference]
                        )
                        for reference in required
                    ):
                        raise ContextualProgramError(
                            "operation argument references invalid node"
                        )
                    if (
                        int(self.constant_index[batch_index, node_index]) != -1
                        or int(self.variable_index[batch_index, node_index])
                        != -1
                        or any(
                            reference != -1
                            for reference in references[arity:]
                        )
                    ):
                        raise ContextualProgramError(
                            "operation node contains covert state"
                        )
            referenced_constants = {
                int(self.constant_index[batch_index, node_index])
                for node_index in range(MAX_PROGRAM_NODES)
                if bool(self.node_valid[batch_index, node_index])
                and int(self.node_kind[batch_index, node_index])
                == int(ProgramNodeKind.CONSTANT)
            }
            valid_constants = {
                index
                for index in range(MAX_PROGRAM_CONSTANTS)
                if bool(self.constant_valid[batch_index, index])
            }
            referenced_slots = {
                int(self.operation_slot[batch_index, node_index])
                for node_index in range(MAX_PROGRAM_NODES)
                if bool(self.node_valid[batch_index, node_index])
                and int(self.node_kind[batch_index, node_index])
                == int(ProgramNodeKind.OPERATION)
            }
            valid_slots = {
                index
                for index in range(MAX_OPERATION_SLOTS)
                if int(self.slot_arity[batch_index, index]) >= 0
            }
            if (
                referenced_constants != valid_constants
                or referenced_slots != valid_slots
            ):
                raise ContextualProgramError(
                    "program packet contains unused constants or slots"
                )
        _program_depths(self)

    @property
    def batch_size(self) -> int:
        return int(self.constants.shape[0])


@dataclass(frozen=True, slots=True)
class LateContextualProgramQuery:
    variable: torch.Tensor
    position: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.variable.ndim != 1
            or self.position.ndim != 1
            or self.variable.shape != self.position.shape
            or self.variable.dtype != torch.long
            or self.position.dtype != torch.long
            or (
                self.variable.numel() > 0
                and (
                    int(self.variable.min()) < 0
                    or int(self.variable.max()) >= PROGRAM_VARIABLES
                    or int(self.position.min()) < 0
                    or int(self.position.max()) >= MAX_OBJECTS
                )
            )
        ):
            raise ContextualProgramError("late contextual query differs")


@dataclass(frozen=True, slots=True)
class ContextualProgramRollout:
    terminal_variables: torch.Tensor
    answer: torch.Tensor
    final_nodes: torch.Tensor
    variable_trajectory: tuple[torch.Tensor, ...]
    converged: torch.Tensor
    convergence_step: torch.Tensor


def contextual_primitive_candidates(
    left: torch.Tensor,
    right: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the tied contextual primitive bank tensorially."""

    if (
        left.shape != right.shape
        or left.ndim != 4
        or left.shape[-2:] != (MAX_OBJECTS, MAX_OBJECTS)
        or active.shape != (
            left.shape[0],
            MAX_OBJECTS,
            MAX_OBJECTS,
        )
    ):
        raise ContextualProgramError("contextual primitive operands differ")
    union = 1.0 - (1.0 - left) * (1.0 - right)
    intersection = left * right
    compose = boolean_relation_compose(
        left.flatten(0, 1),
        right.flatten(0, 1),
    ).unflatten(0, left.shape[:2])
    converse = left.transpose(-1, -2)
    identity = torch.eye(
        MAX_OBJECTS,
        device=left.device,
        dtype=left.dtype,
    )[None, None].expand_as(left)
    candidates = torch.stack(
        (union, intersection, compose, converse, identity),
        dim=2,
    )
    return candidates * active[:, None, None].float()


class ContextualBekicGraphMachine(nn.Module):
    """Parameter-free private executor for bound recursive relation programs."""

    def __init__(
        self,
        *,
        expression_ticks: int = MAX_PROGRAM_NODES,
        fixed_point_steps: int = 2 * MAX_OBJECTS * MAX_OBJECTS + 2,
    ) -> None:
        super().__init__()
        if (
            not 1 <= expression_ticks <= MAX_PROGRAM_NODES
            or not 1 <= fixed_point_steps <= 2 * MAX_OBJECTS * MAX_OBJECTS + 2
        ):
            raise ContextualProgramError("contextual executor bounds differ")
        self.expression_ticks = int(expression_ticks)
        self.fixed_point_steps = int(fixed_point_steps)

    @property
    def added_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _validate_assignment(
        packet: DeletedContextualProgramPacket,
        primitive_assignment: torch.Tensor,
        *,
        require_discrete: bool,
    ) -> None:
        expected = (
            packet.batch_size,
            MAX_OPERATION_SLOTS,
            PROGRAM_PRIMITIVE_COUNT,
        )
        _require_shape(
            primitive_assignment,
            expected,
            "contextual primitive assignment",
        )
        if (
            not primitive_assignment.is_floating_point()
            or not bool(torch.isfinite(primitive_assignment).all())
            or bool((primitive_assignment < 0).any())
            or bool((primitive_assignment > 1).any())
        ):
            raise ContextualProgramError("primitive assignment differs")
        active_slots = packet.slot_arity.ge(0)
        sums = primitive_assignment.sum(-1)
        if not bool(
            torch.allclose(
                sums[active_slots],
                torch.ones_like(sums[active_slots]),
                atol=1e-5,
                rtol=0.0,
            )
        ):
            raise ContextualProgramError(
                "active primitive assignment is not normalized"
            )
        if bool(
            primitive_assignment.masked_select(
                ~active_slots[..., None]
            ).ne(0).any()
        ):
            raise ContextualProgramError(
                "inactive primitive assignment contains state"
            )
        arities = PRIMITIVE_ARITY.to(packet.slot_arity.device)
        legal = packet.slot_arity[..., None].eq(arities)
        if bool(
            primitive_assignment.masked_select(~legal).gt(1e-6).any()
        ):
            raise ContextualProgramError(
                "primitive assignment violates operation arity"
            )
        if require_discrete and not torch.equal(
            primitive_assignment,
            primitive_assignment.round(),
        ):
            raise ContextualProgramError(
                "hard primitive assignment is not discrete"
            )

    def _evaluate_expressions(
        self,
        packet: DeletedContextualProgramPacket,
        primitive_assignment: torch.Tensor,
        variables: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        batch = packet.batch_size
        nodes = torch.zeros(
            batch,
            MAX_PROGRAM_NODES,
            MAX_OBJECTS,
            MAX_OBJECTS,
            device=packet.constants.device,
            dtype=packet.constants.dtype,
        )
        kind = packet.node_kind
        valid = packet.node_valid[..., None, None].float()
        constant_mask = kind.eq(
            int(ProgramNodeKind.CONSTANT)
        )[..., None, None].float()
        variable_mask = kind.eq(
            int(ProgramNodeKind.VARIABLE)
        )[..., None, None].float()
        operation_mask = kind.eq(
            int(ProgramNodeKind.OPERATION)
        )[..., None, None].float()
        constants = _gather_relations(
            packet.constants,
            packet.constant_index,
        )
        variable_values = _gather_relations(
            variables,
            packet.variable_index,
        )
        slot_assignment = primitive_assignment.gather(
            1,
            packet.operation_slot.clamp_min(0)[..., None].expand(
                -1,
                -1,
                PROGRAM_PRIMITIVE_COUNT,
            ),
        )

        for _ in range(self.expression_ticks):
            left = _gather_relations(nodes, packet.left_index)
            right = _gather_relations(nodes, packet.right_index)
            candidates = contextual_primitive_candidates(
                left,
                right,
                active,
            )
            operations = torch.einsum(
                "bnp,bnpij->bnij",
                slot_assignment,
                candidates,
            )
            nodes = valid * (
                constant_mask * constants
                + variable_mask * variable_values
                + operation_mask * operations
            )
        return nodes

    def forward(
        self,
        packet: DeletedContextualProgramPacket,
        primitive_assignment: torch.Tensor,
        query: LateContextualProgramQuery,
        *,
        hard: bool = True,
    ) -> ContextualProgramRollout:
        if query.variable.shape != (packet.batch_size,):
            raise ContextualProgramError("program/query batch differs")
        if bool(query.position.ge(packet.cardinality.long()).any()):
            raise ContextualProgramError("late query leaves active cardinality")
        if max(_program_depths(packet)) + 1 > self.expression_ticks:
            raise ContextualProgramError(
                "expression tick budget is shorter than program depth"
            )
        self._validate_assignment(
            packet,
            primitive_assignment,
            require_discrete=hard,
        )
        active = _active_square(packet.cardinality)
        variables = torch.zeros(
            packet.batch_size,
            PROGRAM_VARIABLES,
            MAX_OBJECTS,
            MAX_OBJECTS,
            device=packet.constants.device,
            dtype=packet.constants.dtype,
        )
        trajectory: list[torch.Tensor] = []
        convergence_step = torch.full(
            (packet.batch_size,),
            -1,
            device=variables.device,
            dtype=torch.long,
        )
        final_nodes = torch.zeros(
            packet.batch_size,
            MAX_PROGRAM_NODES,
            MAX_OBJECTS,
            MAX_OBJECTS,
            device=variables.device,
            dtype=variables.dtype,
        )
        for step in range(self.fixed_point_steps):
            final_nodes = self._evaluate_expressions(
                packet,
                primitive_assignment,
                variables,
                active,
            )
            proposal = _gather_relations(
                final_nodes,
                packet.equation_root,
            )
            proposal = proposal * active[:, None].float()
            stable = proposal.eq(variables).flatten(1).all(-1)
            newly_stable = convergence_step.lt(0) & stable
            convergence_step = torch.where(
                newly_stable,
                torch.full_like(convergence_step, step),
                convergence_step,
            )
            variables = proposal
            trajectory.append(variables)

        converged = convergence_step.ge(0)
        batch = torch.arange(packet.batch_size, device=variables.device)
        answer = variables[
            batch,
            query.variable,
            query.position,
        ]
        active_objects = (
            torch.arange(MAX_OBJECTS, device=variables.device)[None]
            < packet.cardinality.long()[:, None]
        )
        answer = answer * active_objects.float()
        return ContextualProgramRollout(
            terminal_variables=variables,
            answer=answer,
            final_nodes=final_nodes,
            variable_trajectory=tuple(trajectory),
            converged=converged,
            convergence_step=convergence_step,
        )


def contextual_graph_parameter_receipt(
    machine: ContextualBekicGraphMachine,
    *,
    base_parameters: int = 125_081_664,
    strict_cap: int = 200_000_000,
) -> dict[str, int]:
    added = machine.added_parameters
    complete = base_parameters + added
    if complete >= strict_cap:
        raise RelationRegisterError("contextual graph system reaches cap")
    return {
        "base": int(base_parameters),
        "added": int(added),
        "complete_system": int(complete),
        "strict_cap": int(strict_cap),
        "headroom": int(strict_cap - complete),
    }
