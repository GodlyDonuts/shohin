from __future__ import annotations

import torch

from er_relation_tensor_motor import hard_relation, rollout_relation_tensor


def _matrix(rows: tuple[int, ...], width: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(rows), width).float()


def test_relation_motor_supports_variable_cardinality_and_persistent_halt() -> None:
    width = 6
    active = torch.tensor(
        [[True, True, True, False, False, False], [True] * width]
    )
    initial = torch.stack(
        [
            _matrix((0, 1, 2, 3, 4, 5), width) * active[0, :, None],
            _matrix((5, 4, 3, 2, 1, 0), width),
        ]
    )
    cards = torch.stack(
        [
            torch.stack([_matrix((0, 0, 2, 3, 4, 5), width), _matrix((1, 2, 0, 3, 4, 5), width)]),
            torch.stack([_matrix((1, 1, 3, 3, 5, 5), width), _matrix((5, 4, 3, 2, 1, 0), width)]),
        ]
    )
    result = rollout_relation_tensor(
        initial,
        cards,
        event_card=torch.tensor([[0, 1, 1], [1, 0, 1]]),
        event_halt=torch.tensor([[0, 0, 1], [0, 1, 0]]),
        active=active,
    )
    expected0 = torch.mm(cards[0, 1], torch.mm(cards[0, 0], initial[0]))
    expected1 = torch.mm(cards[1, 1], initial[1])
    assert torch.equal(result.final_state[0], expected0 * active[0, :, None])
    assert torch.equal(result.final_state[1], expected1)
    assert result.alive_trajectory[-1].tolist() == [False, False]


def test_hard_relation_masks_inactive_rows_and_inputs() -> None:
    logits = torch.zeros((1, 1, 4, 4))
    logits[..., 3] = 100
    active = torch.tensor([[True, True, True, False]])
    relation = hard_relation(logits, active)
    assert relation.shape == logits.shape
    assert relation[0, 0, :3, :3].sum(-1).eq(1).all()
    assert relation[0, 0, 3].eq(0).all()
    assert relation[..., 3].eq(0).all()
