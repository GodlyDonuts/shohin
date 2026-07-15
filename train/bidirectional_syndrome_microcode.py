"""State/goal-conditioned recurrent microcode with an executable syndrome.

This is the trainable tensor contract for R9c.  It deliberately contains no
language encoder: an upstream text compiler supplies one feature vector per
event, lexical numeric values, an initial two-register state, and a query
covector.  The forward channel sees only incoming state.  The backward channel
sees only the future goal propagated through the suffix.  Their only recurrent
cross-channel communication is the signed disagreement between complete affine
operator effects.

The module is a mechanism candidate, not a reasoning result.  Runtime switches
produce parameter-identical static, no-syndrome, fixed-replay, and adaptive-
replay controls.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from categorical_microcode import OPCODES


def _matrix(rows):
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def opcode_operator_bank(values: torch.Tensor) -> torch.Tensor:
    """Return all nine lawful affine operators for each lexical value.

    Args:
        values: ``[batch, events]`` numeric literals.

    Returns:
        Tensor ``[batch, events, opcodes, 3, 3]`` in ``OPCODES`` order.
    """
    if values.ndim != 2 or not torch.is_floating_point(values):
        raise ValueError("values must be a rank-2 floating tensor")
    z = torch.zeros_like(values)
    o = torch.ones_like(values)
    v = values
    operators = (
        _matrix(((o, z, v), (z, o, z), (z, z, o))),
        _matrix(((o, z, z), (z, o, v), (z, z, o))),
        _matrix(((o, z, -v), (z, o, z), (z, z, o))),
        _matrix(((o, z, z), (z, o, -v), (z, z, o))),
        _matrix(((o, z, -v), (z, o, v), (z, z, o))),
        _matrix(((o, z, v), (z, o, -v), (z, z, o))),
        _matrix(((o, z, z), (o, o, z), (z, z, o))),
        _matrix(((o, o, z), (z, o, z), (z, z, o))),
        _matrix(((z, o, z), (o, z, z), (z, z, o))),
    )
    if len(operators) != len(OPCODES):
        raise RuntimeError("operator bank differs from categorical vocabulary")
    return torch.stack(operators, dim=2)


def expected_operators(logits: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Differentiably project categorical hypotheses into affine operators."""
    if logits.ndim != 3 or logits.shape[:2] != values.shape:
        raise ValueError("logits must be [batch,events,opcodes]")
    if logits.shape[-1] != len(OPCODES):
        raise ValueError("logit vocabulary differs from OPCODES")
    probabilities = logits.float().softmax(dim=-1).to(values.dtype)
    return torch.einsum("btc,btcij->btij", probabilities, opcode_operator_bank(values))


def apply_operator(operator: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    if operator.shape[-2:] != (3, 3) or state.shape[-1] != 3:
        raise ValueError("operator/state shapes must end in [3,3] and [3]")
    return torch.einsum("...ij,...j->...i", operator, state)


def pull_back_goal(goal: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    if operator.shape[-2:] != (3, 3) or goal.shape[-1] != 3:
        raise ValueError("goal/operator shapes must end in [3] and [3,3]")
    return torch.einsum("...i,...ij->...j", goal, operator)


def homogeneous_state(initial_values: torch.Tensor) -> torch.Tensor:
    if initial_values.ndim != 2 or initial_values.shape[-1] != 2:
        raise ValueError("initial_values must have shape [batch,2]")
    return torch.cat((initial_values, torch.ones_like(initial_values[:, :1])), dim=-1)


def signed_log_coordinates(value: torch.Tensor) -> torch.Tensor:
    """Keep large carried values numerically usable without erasing sign."""
    return value.sign() * value.abs().log1p()


class DirectionalCompilerCell(nn.Module):
    """One direction's private recurrent evidence accumulator."""

    def __init__(self, event_dim: int, memory_dim: int):
        super().__init__()
        if int(event_dim) <= 0 or int(memory_dim) <= 0:
            raise ValueError("event_dim and memory_dim must be positive")
        self.event_dim = int(event_dim)
        self.memory_dim = int(memory_dim)
        self.event_encoder = nn.Sequential(
            nn.LayerNorm(event_dim),
            nn.Linear(event_dim, memory_dim, bias=False),
            nn.SiLU(),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(3, memory_dim, bias=False),
            nn.SiLU(),
        )
        self.syndrome_encoder = nn.Sequential(
            nn.Linear(9, memory_dim, bias=False),
            nn.Tanh(),
        )
        self.recurrent = nn.GRUCell(3 * memory_dim, memory_dim)
        self.operator_head = nn.Linear(memory_dim, len(OPCODES))

    def update(self, event, condition, syndrome, memory, active):
        if event.ndim != 2 or event.shape[-1] != self.event_dim:
            raise ValueError("event features have the wrong shape")
        if condition.shape != (event.shape[0], 3) or syndrome.shape != (event.shape[0], 9):
            raise ValueError("directional condition or syndrome has the wrong shape")
        if memory.shape != (event.shape[0], self.memory_dim):
            raise ValueError("directional memory has the wrong shape")
        if active.shape != (event.shape[0],) or active.dtype != torch.bool:
            raise ValueError("active must be one boolean per batch item")
        encoded = torch.cat((
            self.event_encoder(event),
            self.condition_encoder(signed_log_coordinates(condition)),
            self.syndrome_encoder(syndrome),
        ), dim=-1)
        proposal = self.recurrent(encoded, memory)
        updated = torch.where(active.unsqueeze(-1), proposal, memory)
        return updated, self.operator_head(updated).float()


@dataclass
class SyndromeMicrocodeRun:
    forward_logits: torch.Tensor
    backward_logits: torch.Tensor
    forward_operators: torch.Tensor
    backward_operators: torch.Tensor
    prefix_states: torch.Tensor
    suffix_goals: torch.Tensor
    syndrome: torch.Tensor
    syndrome_norm: torch.Tensor
    active_masks: tuple[torch.Tensor, ...]
    forward_logit_history: tuple[torch.Tensor, ...]
    backward_logit_history: tuple[torch.Tensor, ...]


class BidirectionalSyndromeMicrocode(nn.Module):
    """Independent directional compilers joined only by executable syndrome."""

    def __init__(self, event_dim: int, memory_dim: int = 96):
        super().__init__()
        self.event_dim = int(event_dim)
        self.memory_dim = int(memory_dim)
        self.forward_compiler = DirectionalCompilerCell(event_dim, memory_dim)
        self.backward_compiler = DirectionalCompilerCell(event_dim, memory_dim)

    def adapter_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        event_features: torch.Tensor,
        values: torch.Tensor,
        initial_values: torch.Tensor,
        query_goals: torch.Tensor,
        *,
        rounds: int = 2,
        conditioning: str = "directional",
        use_syndrome: bool = True,
        adaptive: bool = False,
        syndrome_threshold: float = 0.05,
    ) -> SyndromeMicrocodeRun:
        if event_features.ndim != 3 or event_features.shape[-1] != self.event_dim:
            raise ValueError("event_features must be [batch,events,event_dim]")
        batch, events, _ = event_features.shape
        if events <= 0 or values.shape != (batch, events):
            raise ValueError("values must match a nonempty event sequence")
        if initial_values.shape != (batch, 2) or query_goals.shape != (batch, 3):
            raise ValueError("initial state or query goal has the wrong shape")
        if int(rounds) <= 0 or conditioning not in {"directional", "static"}:
            raise ValueError("invalid replay schedule or conditioning mode")
        if not all(tensor.device == event_features.device for tensor in (
            values, initial_values, query_goals,
        )):
            raise ValueError("all inputs must share a device")
        dtype = event_features.dtype
        values = values.to(dtype=dtype)
        initial_values = initial_values.to(dtype=dtype)
        query_goals = query_goals.to(dtype=dtype)
        forward_memory = [event_features.new_zeros(batch, self.memory_dim) for _ in range(events)]
        backward_memory = [event_features.new_zeros(batch, self.memory_dim) for _ in range(events)]
        previous_syndrome = event_features.new_zeros(batch, events, 3, 3)
        previous_norm = event_features.new_full((batch, events), float("inf"))
        forward_history = []
        backward_history = []
        active_history = []

        for replay in range(int(rounds)):
            if replay == 0 or not adaptive:
                active = torch.ones((batch, events), dtype=torch.bool, device=event_features.device)
            else:
                active = previous_norm > float(syndrome_threshold)
            active_history.append(active)

            state = homogeneous_state(initial_values)
            prefix = [state]
            forward_logits = []
            forward_operators = []
            for event in range(events):
                condition = state if conditioning == "directional" else torch.zeros_like(state)
                residual = previous_syndrome[:, event].reshape(batch, 9)
                if not use_syndrome:
                    residual = torch.zeros_like(residual)
                memory, logits = self.forward_compiler.update(
                    event_features[:, event], condition, residual,
                    forward_memory[event], active[:, event],
                )
                forward_memory[event] = memory
                operator = expected_operators(
                    logits.unsqueeze(1), values[:, event:event + 1],
                )[:, 0]
                forward_logits.append(logits)
                forward_operators.append(operator)
                state = apply_operator(operator, state)
                prefix.append(state)

            goal = query_goals
            suffix = [None] * (events + 1)
            suffix[events] = goal
            backward_logits = [None] * events
            backward_operators = [None] * events
            for event in range(events - 1, -1, -1):
                condition = goal if conditioning == "directional" else torch.zeros_like(goal)
                residual = -previous_syndrome[:, event].reshape(batch, 9)
                if not use_syndrome:
                    residual = torch.zeros_like(residual)
                memory, logits = self.backward_compiler.update(
                    event_features[:, event], condition, residual,
                    backward_memory[event], active[:, event],
                )
                backward_memory[event] = memory
                operator = expected_operators(
                    logits.unsqueeze(1), values[:, event:event + 1],
                )[:, 0]
                backward_logits[event] = logits
                backward_operators[event] = operator
                goal = pull_back_goal(goal, operator)
                suffix[event] = goal

            forward_logits = torch.stack(forward_logits, dim=1)
            backward_logits = torch.stack(backward_logits, dim=1)
            forward_operators = torch.stack(forward_operators, dim=1)
            backward_operators = torch.stack(backward_operators, dim=1)
            previous_syndrome = forward_operators - backward_operators
            previous_norm = previous_syndrome.square().mean(dim=(-1, -2)).sqrt()
            forward_history.append(forward_logits)
            backward_history.append(backward_logits)

        return SyndromeMicrocodeRun(
            forward_logits=forward_logits,
            backward_logits=backward_logits,
            forward_operators=forward_operators,
            backward_operators=backward_operators,
            prefix_states=torch.stack(prefix, dim=1),
            suffix_goals=torch.stack(suffix, dim=1),
            syndrome=previous_syndrome,
            syndrome_norm=previous_norm,
            active_masks=tuple(active_history),
            forward_logit_history=tuple(forward_history),
            backward_logit_history=tuple(backward_history),
        )
