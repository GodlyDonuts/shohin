from __future__ import annotations

import torch

from contrastive_fixed_point_board import generate_rows
from equivariant_relation_register_machine import (
    EquivariantRelationRegisterMachine,
)
from train_fixed_point_register_machine import (
    batch_from_rows,
    evaluate,
    fixed_point_loss,
    teacher_action_loss,
    training_curriculum,
)


def test_batch_loss_metrics_and_gradient_contract() -> None:
    rows = generate_rows(split="train", count=6, seed=3901)
    machine = EquivariantRelationRegisterMachine(
        controller_width=32,
        controller_layers=1,
        maximum_steps=12,
    )
    packet, query, targets, answers = batch_from_rows(
        rows,
        device=torch.device("cpu"),
    )
    assert packet.registers.shape == (6, 6, 8, 8)
    assert query.position.shape == (6,)
    assert targets.shape == (6, 6, 8, 8)
    assert answers.shape == (6, 8)

    loss, components = fixed_point_loss(
        machine,
        rows,
        device=torch.device("cpu"),
        hard=True,
    )
    assert bool(torch.isfinite(loss))
    assert set(components) == {
        "state",
        "answer",
        "alive",
        "expected_runtime",
        "teacher",
    }
    loss.backward()
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum()) > 0
        for parameter in machine.parameters()
    )

    metrics = evaluate(
        machine,
        rows,
        batch_size=3,
        device=torch.device("cpu"),
    )
    assert metrics["rows"] == 6
    for name in (
        "work_accuracy",
        "answer_accuracy",
        "halt_accuracy",
        "joint_accuracy",
    ):
        assert 0.0 <= metrics[name] <= 1.0


def test_wrong_hard_actions_receive_teacher_gradients() -> None:
    row = generate_rows(split="train", count=1, seed=9017)[0]
    machine = EquivariantRelationRegisterMachine(
        controller_width=32,
        controller_layers=1,
        maximum_steps=12,
    )
    with torch.no_grad():
        for head in (
            machine.operation_head,
            machine.left_head,
            machine.right_head,
            machine.destination_head,
            machine.halt_head,
            machine.phase_head,
        ):
            head.weight.zero_()
            head.bias.copy_(
                torch.linspace(
                    3.0,
                    -3.0,
                    head.bias.numel(),
                )
            )
    packet, query, _, _ = batch_from_rows(
        [row],
        device=torch.device("cpu"),
    )
    result = machine(packet, query, hard=True)
    loss = teacher_action_loss(
        result.actions,
        [row],
        device=torch.device("cpu"),
    )
    loss.backward()
    assert float(loss.detach()) > 0.0
    for head in (
        machine.operation_head,
        machine.left_head,
        machine.right_head,
        machine.destination_head,
        machine.halt_head,
        machine.phase_head,
    ):
        assert head.bias.grad is not None
        assert bool(torch.isfinite(head.bias.grad).all())
        assert float(head.bias.grad.abs().sum()) > 0.0


def test_hard_curriculum_overlaps_teacher_supervision() -> None:
    updates = 1_000
    before = training_curriculum(899, updates)
    start = training_curriculum(900, updates)
    final = training_curriculum(999, updates)
    assert not before[0]
    assert start[0] and start[1] == 0.1
    assert final[0] and final[1] == 0.1
