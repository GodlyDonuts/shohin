"""Pure runtime interventions for the frozen CTAA causal panel.

The functions in this module operate only on materialized categorical packets,
executor registers, and late queries. They never accept source text, targets,
or oracle rows. Every mutation is deterministic and returns the exact first
schedule slot at which its effect can become visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ctaa_neural_core import (
    CTAA_ACTION_COUNT,
    CTAA_MAX_STEPS,
    CTAA_WIDTH,
    HardExecutionTrace,
)
from ctaa_trunk_compiler import HardCTAAPacket, HardCTAAQuery


@dataclass(frozen=True)
class PacketIntervention:
    operation: str
    packet: HardCTAAPacket
    first_exposure_step: torch.Tensor

    def __post_init__(self) -> None:
        batch = self.packet.schedule.shape[0]
        if (
            not self.operation
            or self.first_exposure_step.dtype != torch.long
            or self.first_exposure_step.shape != (batch,)
            or bool(
                (
                    (self.first_exposure_step < 0)
                    | (self.first_exposure_step >= CTAA_MAX_STEPS)
                ).any()
            )
        ):
            raise ValueError("CTAA packet-intervention boundary differs")


def _stop_index(packet: HardCTAAPacket) -> torch.Tensor:
    return packet.schedule.eq(CTAA_ACTION_COUNT).long().argmax(1)


def _require_row_permutations(order: torch.Tensor, batch: int) -> torch.Tensor:
    if order.dtype != torch.long or order.shape != (batch, CTAA_ACTION_COUNT):
        raise ValueError("CTAA card-storage permutation geometry differs")
    expected = torch.arange(CTAA_ACTION_COUNT, device=order.device)[None].expand_as(
        order
    )
    if not torch.equal(order.sort(1).values, expected):
        raise ValueError("CTAA card-storage order is not a permutation")
    return order


def card_storage_reindex(
    packet: HardCTAAPacket,
    storage_order: torch.Tensor,
) -> PacketIntervention:
    """Reindex card storage and rebind every active opcode consistently.

    ``storage_order[b, new_slot]`` gives the old slot copied into ``new_slot``.
    STOP and suffix events remain untouched.
    """

    batch = packet.schedule.shape[0]
    order = _require_row_permutations(storage_order, batch).to(
        packet.action_cards.device
    )
    batch_index = torch.arange(batch, device=packet.action_cards.device)[:, None]
    cards = packet.action_cards[batch_index, order]
    inverse = torch.empty_like(order)
    inverse.scatter_(
        1,
        order,
        torch.arange(CTAA_ACTION_COUNT, device=order.device)[None].expand_as(order),
    )
    schedule = packet.schedule.long()
    active = schedule.lt(CTAA_ACTION_COUNT)
    rebound = inverse.gather(1, schedule.clamp_max(CTAA_ACTION_COUNT - 1))
    schedule = torch.where(active, rebound, schedule).to(torch.uint8)
    changed = order.ne(torch.arange(CTAA_ACTION_COUNT, device=order.device)[None]).any(
        1
    )
    if not bool(changed.all()):
        raise ValueError("CTAA card-storage reindex contains an identity row")
    return PacketIntervention(
        operation="card_storage_reindex",
        packet=HardCTAAPacket(cards, packet.initial_state.clone(), schedule),
        first_exposure_step=torch.zeros(batch, dtype=torch.long, device=order.device),
    )


def post_stop_poison(packet: HardCTAAPacket) -> PacketIntervention:
    """Change every packet event strictly after STOP without moving STOP."""

    schedule = packet.schedule.long().clone()
    stop = _stop_index(packet).to(schedule.device)
    positions = torch.arange(CTAA_MAX_STEPS, device=schedule.device)[None]
    suffix = positions.gt(stop[:, None])
    poisoned = (schedule.clamp_max(CTAA_ACTION_COUNT - 1) + 1) % CTAA_ACTION_COUNT
    schedule = torch.where(suffix, poisoned, schedule).to(torch.uint8)
    if not bool(schedule.long().ne(packet.schedule.long()).eq(suffix).all()):
        raise RuntimeError("CTAA post-STOP poison did not mutate the exact suffix")
    return PacketIntervention(
        operation="post_stop_poison",
        packet=HardCTAAPacket(
            packet.action_cards.clone(), packet.initial_state.clone(), schedule
        ),
        first_exposure_step=stop + 1,
    )


def packet_transplant(
    packet: HardCTAAPacket,
    donor: HardCTAAPacket,
) -> PacketIntervention:
    """Replace each literal 56-byte packet row with its precommitted donor."""

    if (
        donor.action_cards.shape != packet.action_cards.shape
        or donor.initial_state.shape != packet.initial_state.shape
        or donor.schedule.shape != packet.schedule.shape
    ):
        raise ValueError("CTAA packet-transplant geometry differs")
    same = (
        donor.action_cards.eq(packet.action_cards).flatten(1).all(1)
        & donor.initial_state.eq(packet.initial_state).all(1)
        & donor.schedule.eq(packet.schedule).all(1)
    )
    if bool(same.any()):
        raise ValueError("CTAA packet transplant contains an unchanged row")
    batch = packet.schedule.shape[0]
    return PacketIntervention(
        operation="packet_transplant",
        packet=HardCTAAPacket(
            donor.action_cards.clone(),
            donor.initial_state.clone(),
            donor.schedule.clone(),
        ),
        first_exposure_step=torch.zeros(
            batch, dtype=torch.long, device=packet.schedule.device
        ),
    )


def future_schedule_counterfactual(
    packet: HardCTAAPacket,
    first_exposure_step: torch.Tensor,
) -> PacketIntervention:
    """Mutate active future events from a frozen first-exposure slot to STOP."""

    batch = packet.schedule.shape[0]
    if first_exposure_step.dtype != torch.long or first_exposure_step.shape != (batch,):
        raise ValueError("CTAA future-mask boundary differs")
    schedule = packet.schedule.long().clone()
    stop = _stop_index(packet).to(schedule.device)
    boundary = first_exposure_step.to(schedule.device)
    if bool(((boundary <= 0) | (boundary >= stop)).any()):
        raise ValueError("CTAA future-mask boundary is not active")
    positions = torch.arange(CTAA_MAX_STEPS, device=schedule.device)[None]
    future = positions.ge(boundary[:, None]) & positions.lt(stop[:, None])
    changed = (schedule + 1) % CTAA_ACTION_COUNT
    schedule = torch.where(future, changed, schedule).to(torch.uint8)
    if not bool(schedule.long().ne(packet.schedule.long()).eq(future).all()):
        raise RuntimeError("CTAA future counterfactual changed the wrong slots")
    return PacketIntervention(
        operation="future_mask",
        packet=HardCTAAPacket(
            packet.action_cards.clone(), packet.initial_state.clone(), schedule
        ),
        first_exposure_step=boundary,
    )


def execute_with_midpoint_intervention(
    core: nn.Module,
    packet: HardCTAAPacket,
    *,
    operation: str,
    midpoint_step: torch.Tensor,
    donor_state: torch.Tensor | None = None,
    donor_action: torch.Tensor | None = None,
) -> HardExecutionTrace:
    """Inject immediately before committed event ``midpoint_step`` and execute."""

    if operation not in {"midpoint_donor_state", "midpoint_donor_action"}:
        raise ValueError("CTAA midpoint intervention differs")
    cards = packet.action_cards.long()
    schedule = packet.schedule.long()
    state = packet.initial_state.long()
    batch = schedule.shape[0]
    stop = _stop_index(packet).to(schedule.device)
    midpoint = midpoint_step.to(schedule.device)
    if midpoint_step.dtype != torch.long or midpoint.shape != (batch,):
        raise ValueError("CTAA midpoint geometry differs")
    if bool(((midpoint <= 0) | (midpoint >= stop)).any()):
        raise ValueError("CTAA midpoint is not an active schedule slot")
    if operation == "midpoint_donor_state":
        if (
            donor_state is None
            or donor_state.dtype != torch.long
            or donor_state.shape != (batch, CTAA_WIDTH)
            or donor_action is not None
        ):
            raise ValueError("CTAA midpoint donor-state geometry differs")
        if donor_state.numel() and bool(
            ((donor_state < 0) | (donor_state >= CTAA_WIDTH)).any()
        ):
            raise ValueError("CTAA midpoint donor state leaves the domain")
    else:
        if (
            donor_action is None
            or donor_action.dtype != torch.long
            or donor_action.shape != (batch, CTAA_WIDTH)
            or donor_state is not None
        ):
            raise ValueError("CTAA midpoint donor-action geometry differs")
        if donor_action.numel() and bool(
            ((donor_action < 0) | (donor_action >= CTAA_WIDTH)).any()
        ):
            raise ValueError("CTAA midpoint donor action leaves the domain")

    halted = torch.zeros(batch, dtype=torch.bool, device=schedule.device)
    states = [state]
    halt_history = [halted]
    batch_index = torch.arange(batch, device=schedule.device)
    for step, event in enumerate(schedule.unbind(1)):
        is_stop = event.eq(CTAA_ACTION_COUNT)
        active = ~(halted | is_stop)
        selected = cards[batch_index, event.clamp_max(CTAA_ACTION_COUNT - 1)]
        at_midpoint = midpoint.eq(step)
        if operation == "midpoint_donor_action":
            assert donor_action is not None
            if bool(selected[at_midpoint].eq(donor_action[at_midpoint]).all(1).any()):
                raise ValueError("CTAA midpoint donor action contains an unchanged row")
            selected = torch.where(at_midpoint[:, None], donor_action, selected)
        else:
            assert donor_state is not None
            if bool(state[at_midpoint].eq(donor_state[at_midpoint]).all(1).any()):
                raise ValueError("CTAA midpoint donor state contains an unchanged row")
            midpoint_index = batch_index[at_midpoint]
            state = state.clone()
            state[midpoint_index] = donor_state[at_midpoint]
        active_index = batch_index[active]
        if active_index.numel():
            candidate = core(selected[active], state[active]).argmax(-1)
            state = state.clone()
            state[active_index] = candidate
        halted = halted | is_stop
        states.append(state)
        halt_history.append(halted)
    return HardExecutionTrace(
        states=torch.stack(states, dim=1),
        halted=torch.stack(halt_history, dim=1),
    )


def prefix_before_exposure_equal(
    parent: HardExecutionTrace,
    child: HardExecutionTrace,
    first_exposure_step: torch.Tensor,
) -> torch.Tensor:
    """Compare states strictly before each mutated event is exposed."""

    if (
        parent.states.shape != child.states.shape
        or parent.halted.shape != child.halted.shape
    ):
        raise ValueError("CTAA prefix comparison trace geometry differs")
    batch, state_count, _ = parent.states.shape
    if (
        first_exposure_step.dtype != torch.long
        or first_exposure_step.shape != (batch,)
        or bool(
            ((first_exposure_step < 0) | (first_exposure_step >= state_count)).any()
        )
    ):
        raise ValueError("CTAA prefix comparison boundary differs")
    positions = torch.arange(state_count, device=parent.states.device)[None]
    included = positions.le(first_exposure_step.to(parent.states.device)[:, None])
    state_equal = parent.states.eq(child.states).all(-1)
    halt_equal = parent.halted.eq(child.halted)
    return ((state_equal & halt_equal) | ~included).all(1)


def late_query_swap(
    parent_query: HardCTAAQuery,
    donor_query: HardCTAAQuery,
    immutable_parent_trace: HardExecutionTrace,
) -> torch.Tensor:
    """Answer a donor late query from an already committed parent execution."""

    if donor_query.position.shape != parent_query.position.shape:
        raise ValueError("CTAA late-query donor geometry differs")
    if bool(donor_query.position.eq(parent_query.position).any()):
        raise ValueError("CTAA late-query swap contains an unchanged row")
    return donor_query.answer(immutable_parent_trace)
