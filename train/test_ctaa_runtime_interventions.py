from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ctaa_runtime_interventions import (
    card_storage_reindex,
    execute_with_midpoint_intervention,
    future_schedule_counterfactual,
    late_query_swap,
    packet_transplant,
    post_stop_poison,
    prefix_before_exposure_equal,
)
from ctaa_trunk_compiler import HardCTAAPacket, HardCTAAQuery


class ExactCopyCore(nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        del right
        return torch.nn.functional.one_hot(left, 3).float().mul(40).sub(20)


class RightCopyCore(nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        del left
        return torch.nn.functional.one_hot(right, 3).float().mul(40).sub(20)


def packet() -> HardCTAAPacket:
    return HardCTAAPacket(
        action_cards=torch.tensor(
            [
                [[1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0]],
                [[2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 1, 0]],
            ],
            dtype=torch.uint8,
        ),
        initial_state=torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.uint8),
        schedule=torch.tensor(
            [
                [0, 1, 2, 4, *([0] * 37)],
                [1, 2, 0, 3, 4, *([1] * 36)],
            ],
            dtype=torch.uint8,
        ),
    )


def test_card_storage_reindex_preserves_complete_execution() -> None:
    parent = packet()
    order = torch.tensor([[1, 2, 3, 0], [2, 3, 0, 1]])
    child = card_storage_reindex(parent, order)
    core = ExactCopyCore()
    parent_trace = parent.execute(core)
    child_trace = child.packet.execute(core)
    assert torch.equal(parent_trace.states, child_trace.states)
    assert torch.equal(parent_trace.halted, child_trace.halted)
    assert child.first_exposure_step.tolist() == [0, 0]


def test_card_storage_reindex_rejects_identity_and_nonpermutation() -> None:
    parent = packet()
    with pytest.raises(ValueError, match="identity"):
        card_storage_reindex(parent, torch.arange(4)[None].expand(2, -1))
    with pytest.raises(ValueError, match="not a permutation"):
        card_storage_reindex(parent, torch.zeros((2, 4), dtype=torch.long))


def test_post_stop_poison_changes_only_suffix_and_preserves_trace() -> None:
    parent = packet()
    child = post_stop_poison(parent)
    assert child.first_exposure_step.tolist() == [4, 5]
    assert torch.equal(
        parent.execute(ExactCopyCore()).states,
        child.packet.execute(ExactCopyCore()).states,
    )
    stop = parent.schedule.eq(4).long().argmax(1)
    positions = torch.arange(41)[None]
    assert torch.equal(
        parent.schedule.ne(child.packet.schedule), positions.gt(stop[:, None])
    )


def test_future_counterfactual_has_exact_prefix_boundary() -> None:
    parent = packet()
    child = future_schedule_counterfactual(parent, torch.tensor([2, 3]))
    parent_trace = parent.execute(ExactCopyCore())
    child_trace = child.packet.execute(ExactCopyCore())
    assert prefix_before_exposure_equal(
        parent_trace, child_trace, child.first_exposure_step
    ).tolist() == [True, True]
    assert not torch.equal(parent_trace.states[:, -1], child_trace.states[:, -1])
    with pytest.raises(ValueError, match="not active"):
        future_schedule_counterfactual(parent, torch.tensor([3, 4]))


def test_packet_transplant_is_literal_and_rejects_same_rows() -> None:
    parent = packet()
    donor = HardCTAAPacket(
        parent.action_cards.flip(0).clone(),
        parent.initial_state.flip(0).clone(),
        parent.schedule.flip(0).clone(),
    )
    child = packet_transplant(parent, donor)
    assert torch.equal(child.packet.action_cards, donor.action_cards)
    assert torch.equal(child.packet.schedule, donor.schedule)
    with pytest.raises(ValueError, match="unchanged row"):
        packet_transplant(parent, parent)


def test_midpoint_state_and_action_injections_take_effect_at_exact_step() -> None:
    parent = packet()
    midpoint = torch.tensor([1, 2])
    donor_state = torch.tensor([[2, 2, 2], [0, 0, 0]])
    state_trace = execute_with_midpoint_intervention(
        RightCopyCore(),
        parent,
        operation="midpoint_donor_state",
        midpoint_step=midpoint,
        donor_state=donor_state,
    )
    assert torch.equal(state_trace.states[torch.arange(2), midpoint + 1], donor_state)
    donor_action = torch.tensor([[0, 0, 0], [1, 1, 1]])
    action_trace = execute_with_midpoint_intervention(
        ExactCopyCore(),
        parent,
        operation="midpoint_donor_action",
        midpoint_step=midpoint,
        donor_action=donor_action,
    )
    assert torch.equal(action_trace.states[torch.arange(2), midpoint + 1], donor_action)


def test_midpoint_rejects_unchanged_donor_registers() -> None:
    parent = packet()
    midpoint = torch.tensor([1, 2])
    native_state = parent.initial_state.long()
    with pytest.raises(ValueError, match="state contains an unchanged row"):
        execute_with_midpoint_intervention(
            RightCopyCore(),
            parent,
            operation="midpoint_donor_state",
            midpoint_step=midpoint,
            donor_state=native_state,
        )
    native_action = parent.action_cards.long()[
        torch.arange(2), parent.schedule.long()[torch.arange(2), midpoint]
    ]
    with pytest.raises(ValueError, match="action contains an unchanged row"):
        execute_with_midpoint_intervention(
            ExactCopyCore(),
            parent,
            operation="midpoint_donor_action",
            midpoint_step=midpoint,
            donor_action=native_action,
        )


def test_midpoint_rejects_stop_boundary_and_mixed_donor_types() -> None:
    parent = packet()
    with pytest.raises(ValueError, match="not an active"):
        execute_with_midpoint_intervention(
            ExactCopyCore(),
            parent,
            operation="midpoint_donor_state",
            midpoint_step=torch.tensor([3, 4]),
            donor_state=torch.zeros((2, 3), dtype=torch.long),
        )
    with pytest.raises(ValueError, match="donor-state geometry"):
        execute_with_midpoint_intervention(
            ExactCopyCore(),
            parent,
            operation="midpoint_donor_state",
            midpoint_step=torch.tensor([1, 2]),
            donor_state=torch.zeros((2, 3), dtype=torch.long),
            donor_action=torch.zeros((2, 3), dtype=torch.long),
        )


def test_late_query_swap_uses_parent_trace_without_rerun() -> None:
    parent = packet()
    trace = parent.execute(ExactCopyCore())
    parent_query = HardCTAAQuery(torch.tensor([0, 1], dtype=torch.uint8))
    donor_query = HardCTAAQuery(torch.tensor([2, 0], dtype=torch.uint8))
    answer = late_query_swap(parent_query, donor_query, trace)
    expected = trace.states[:, -1].gather(1, donor_query.position.long()[:, None])
    assert torch.equal(answer, expected.squeeze(1))
    with pytest.raises(ValueError, match="unchanged row"):
        late_query_swap(parent_query, parent_query, trace)
