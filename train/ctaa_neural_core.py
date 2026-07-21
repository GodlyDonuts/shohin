"""Neural closure-tied categorical transition core.

Copy actions and states share the tuple space ``{0, ..., width-1}^width``.
The same learned binary operator therefore implements both action application
and action composition. This module does not compile actions from language.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


CTAA_WIDTH = 3
CTAA_ACTION_COUNT = 4
CTAA_MAX_STEPS = 41
CTAA_STOP_ID = CTAA_ACTION_COUNT


@dataclass(frozen=True)
class HardExecutionTrace:
    states: torch.Tensor
    halted: torch.Tensor


@dataclass(frozen=True)
class HardDualExecutionTrace:
    state_route: HardExecutionTrace
    composed_cards: torch.Tensor
    composed_states: torch.Tensor


def _validate_ids(values: torch.Tensor, width: int, name: str) -> None:
    if values.dtype != torch.long or values.shape[-1] != width:
        raise ValueError(f"CTAA {name} geometry differs")
    if values.numel() and (
        int(values.min()) < 0 or int(values.max()) >= width
    ):
        raise ValueError(f"CTAA {name} leaves the categorical domain")


def _validate_execution_packet(
    action_cards: torch.Tensor,
    schedule: torch.Tensor,
    initial: torch.Tensor,
    width: int,
) -> tuple[int, int]:
    if width != CTAA_WIDTH or action_cards.ndim != 3:
        raise ValueError("CTAA action-card tensor differs")
    batch, action_count, card_width = action_cards.shape
    if action_count != CTAA_ACTION_COUNT or card_width != width:
        raise ValueError("CTAA action-card geometry differs")
    if initial.shape != (batch, width):
        raise ValueError("CTAA initial-state batch differs")
    _validate_ids(action_cards, width, "action cards")
    _validate_ids(initial, width, "initial state")
    if schedule.shape != (batch, CTAA_MAX_STEPS) or schedule.dtype != torch.long:
        raise ValueError("CTAA schedule geometry differs")
    if schedule.numel() == 0 or int(schedule.min()) < 0 or int(schedule.max()) > action_count:
        raise ValueError("CTAA schedule leaves action/STOP domain")
    stop_mask = schedule.eq(action_count)
    if not bool(stop_mask.sum(1).eq(1).all()):
        raise ValueError("CTAA schedule requires exactly one STOP")
    stop_index = stop_mask.long().argmax(1)
    if not bool(((stop_index > 0) & (stop_index < CTAA_MAX_STEPS - 1)).all()):
        raise ValueError("CTAA STOP boundary differs")
    return batch, action_count


class ClosureTiedPointerCore(nn.Module):
    """One learned address law shared by composition and state transition."""

    def __init__(self, width: int = 3) -> None:
        super().__init__()
        if width < 2:
            raise ValueError("CTAA neural width differs")
        self.width = int(width)
        self.address_logits = nn.Parameter(torch.empty(width, width))
        nn.init.normal_(self.address_logits, mean=0.0, std=0.02)

    @property
    def unique_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward_distributions(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        """Return log probabilities for ``right[left[i]]`` at every slot.

        ``left`` and ``right`` are categorical distributions with shape
        ``[batch, width, width]``. The first tuple supplies addresses and the
        second supplies values. No type flag distinguishes action composition
        from action application.
        """
        expected_tail = (self.width, self.width)
        if left.ndim != 3 or left.shape[-2:] != expected_tail or right.shape != left.shape:
            raise ValueError("CTAA neural distribution geometry differs")
        address = self.address_logits.float().softmax(-1)
        selector = torch.einsum("biu,up->bip", left.float(), address)
        output = torch.einsum("bip,bpv->biv", selector, right.float())
        return output.clamp_min(1e-30).log()

    def forward(self, left_ids: torch.Tensor, right_ids: torch.Tensor) -> torch.Tensor:
        if left_ids.ndim != 2 or right_ids.ndim != 2:
            raise ValueError("CTAA neural tuple batches differ")
        _validate_ids(left_ids, self.width, "left tuple")
        _validate_ids(right_ids, self.width, "right tuple")
        if left_ids.shape != right_ids.shape:
            raise ValueError("CTAA neural tuple batches differ")
        left = F.one_hot(left_ids, self.width)
        right = F.one_hot(right_ids, self.width)
        return self.forward_distributions(left, right)

    def hard_step(self, left_ids: torch.Tensor, right_ids: torch.Tensor) -> torch.Tensor:
        return self(left_ids, right_ids).argmax(-1)

    def straight_through_commit(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-2:] != (self.width, self.width):
            raise ValueError("CTAA commit geometry differs")
        soft = logits.float().softmax(-1)
        hard = F.one_hot(soft.argmax(-1), self.width).to(soft.dtype)
        return hard + soft - soft.detach()

def execute_streamed_state_route(
    core: nn.Module,
    width: int,
    action_cards: torch.Tensor,
    schedule: torch.Tensor,
    initial: torch.Tensor,
) -> HardExecutionTrace:
    return _execute_streamed_state_route(core, width, action_cards, schedule, initial)


def _execute_streamed_state_route(
    core: nn.Module,
    width: int,
    action_cards: torch.Tensor,
    schedule: torch.Tensor,
    initial: torch.Tensor,
) -> HardExecutionTrace:
    """Host-stream events while exposing only ``(current_action, state)`` to the core."""
    batch, action_count = _validate_execution_packet(
        action_cards,
        schedule,
        initial,
        width,
    )
    state = initial
    halted = torch.zeros(batch, dtype=torch.bool, device=initial.device)
    states = [state]
    halt_history = [halted]
    batch_index = torch.arange(batch, device=initial.device)
    for event in schedule.unbind(1):
        is_stop = event.eq(action_count)
        active = ~(halted | is_stop)
        selected = action_cards[batch_index, event.clamp_max(action_count - 1)]
        candidate = core(selected, state).argmax(-1)
        state = torch.where(active[:, None], candidate, state)
        halted = halted | is_stop
        states.append(state)
        halt_history.append(halted)
    return HardExecutionTrace(
        states=torch.stack(states, dim=1),
        halted=torch.stack(halt_history, dim=1),
    )


def execute_streamed_dual(
    core: nn.Module,
    width: int,
    action_cards: torch.Tensor,
    schedule: torch.Tensor,
    initial: torch.Tensor,
) -> HardDualExecutionTrace:
    batch, action_count = _validate_execution_packet(
        action_cards,
        schedule,
        initial,
        width,
    )
    state_route = _execute_streamed_state_route(
        core, width, action_cards, schedule, initial
    )
    stop_id = action_count
    batch_index = torch.arange(batch, device=initial.device)
    composed = torch.arange(width, device=initial.device)[None].expand(batch, -1)
    halted = torch.zeros(batch, dtype=torch.bool, device=initial.device)
    composed_cards = [composed]
    composed_states = [initial]
    for event in schedule.unbind(1):
        is_stop = event.eq(stop_id)
        active = ~(halted | is_stop)
        selected = action_cards[batch_index, event.clamp_max(action_count - 1)]
        candidate = core(selected, composed).argmax(-1)
        composed = torch.where(active[:, None], candidate, composed)
        halted = halted | is_stop
        composed_cards.append(composed)
        from_composed = core(composed, initial).argmax(-1)
        composed_states.append(from_composed)
    return HardDualExecutionTrace(
        state_route=state_route,
        composed_cards=torch.stack(composed_cards, dim=1),
        composed_states=torch.stack(composed_states, dim=1),
    )


class ClosureFeatureTransitionCore(nn.Module):
    """Matched CTAA treatment using only composition-aligned monomials."""

    def __init__(self, width: int = 3, hidden: int = 2912) -> None:
        super().__init__()
        if width != 3 or hidden != 2912:
            raise ValueError("CTAA matched treatment geometry differs")
        self.width = int(width)
        self.hidden = int(hidden)
        self.network = nn.Sequential(
            nn.Linear(27, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 9),
        )

    @property
    def unique_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def features(self, left_ids: torch.Tensor, right_ids: torch.Tensor) -> torch.Tensor:
        _validate_matched_pair(left_ids, right_ids, self.width)
        left = F.one_hot(left_ids, self.width).float()
        right = F.one_hot(right_ids, self.width).float()
        return (left.unsqueeze(-1) * right.unsqueeze(1)).flatten(1)

    def forward(self, left_ids: torch.Tensor, right_ids: torch.Tensor) -> torch.Tensor:
        return self.network(self.features(left_ids, right_ids)).reshape(-1, 3, 3)

    @property
    def analytic_inference_flops(self) -> int:
        return 215_530


class OuterProductTransitionControl(nn.Module):
    """Parameter/FLOP-matched generic recurrence with a full outer product."""

    def __init__(self, width: int = 3, hidden: int = 1184) -> None:
        super().__init__()
        if width != 3 or hidden != 1184:
            raise ValueError("CTAA matched control geometry differs")
        self.width = int(width)
        self.hidden = int(hidden)
        self.network = nn.Sequential(
            nn.Linear(81, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 9),
        )

    @property
    def unique_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def features(self, left_ids: torch.Tensor, right_ids: torch.Tensor) -> torch.Tensor:
        _validate_matched_pair(left_ids, right_ids, self.width)
        left = F.one_hot(left_ids, self.width).flatten(1).float()
        right = F.one_hot(right_ids, self.width).flatten(1).float()
        return (left.unsqueeze(-1) * right.unsqueeze(1)).flatten(1)

    def forward(self, left_ids: torch.Tensor, right_ids: torch.Tensor) -> torch.Tensor:
        return self.network(self.features(left_ids, right_ids)).reshape(-1, 3, 3)

    @property
    def analytic_inference_flops(self) -> int:
        return 215_584


def _validate_matched_pair(
    left_ids: torch.Tensor,
    right_ids: torch.Tensor,
    width: int,
) -> None:
    if left_ids.ndim != 2 or right_ids.ndim != 2:
        raise ValueError("CTAA matched tuple batches differ")
    _validate_ids(left_ids, width, "matched left tuple")
    _validate_ids(right_ids, width, "matched right tuple")
    if left_ids.shape != right_ids.shape:
        raise ValueError("CTAA matched tuple batches differ")
