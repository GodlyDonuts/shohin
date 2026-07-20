from __future__ import annotations

import ast
import itertools
from dataclasses import fields
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sd_cst
from model import GPT, GPTConfig
from sd_cst import (
    AMOUNT_COUNT,
    EVENT_KIND_COUNT,
    EVENT_STEPS,
    IDENTITY_COUNT,
    STATE_COUNT,
    CategoricalStateReader,
    DeletedProgramTape,
    HardLateQuery,
    HardProgramTape,
    LateQuery,
    SDCSTSystem,
    StateSwap,
    TiedCategoricalMotor,
    atomic_motor_loss,
    compiler_field_losses,
    late_query_loss,
    reader_loss,
    rollout_hard_categorical,
    swap_late_queries,
    swap_tape_suffix,
)


PERMUTATIONS = tuple(itertools.permutations(range(3)))
PERMUTATION_TO_INDEX = {value: index for index, value in enumerate(PERMUTATIONS)}


def tiny_base() -> GPT:
    return GPT(GPTConfig(
        vocab_size=48,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        d_model=32,
        d_ff=64,
        seq_len=32,
        tie_embeddings=True,
    ))


def tiny_system(*, motor_hidden: int = 48, reader_hidden: int = 32) -> SDCSTSystem:
    return SDCSTSystem(
        tiny_base(),
        compiler_layer=0,
        compiler_width=24,
        compiler_heads=4,
        compiler_layers=1,
        compiler_ff=48,
        motor_hidden=motor_hidden,
        reader_hidden=reader_hidden,
    )


def categorical_logits(labels: torch.Tensor, classes: int, scale: float = 20.0):
    return F.one_hot(labels, num_classes=classes).float() * scale


def transition_target(state: int, kind: int, identity: int, amount_index: int) -> int:
    if kind == 2:
        return state
    order = list(PERMUTATIONS[state])
    source = order.index(identity)
    distance = amount_index + 1
    destination = max(0, source - distance) if kind == 0 else min(2, source + distance)
    order.insert(destination, order.pop(source))
    return PERMUTATION_TO_INDEX[tuple(order)]


def atomic_motor_board():
    rows = [
        (state, kind, identity, amount)
        for state in range(STATE_COUNT)
        for kind in range(EVENT_KIND_COUNT)
        for identity in range(IDENTITY_COUNT)
        for amount in range(AMOUNT_COUNT)
    ]
    state_ids = torch.tensor([row[0] for row in rows])
    kind_ids = torch.tensor([row[1] for row in rows])
    identity_ids = torch.tensor([row[2] for row in rows])
    amount_ids = torch.tensor([row[3] for row in rows])
    targets = torch.tensor([transition_target(*row) for row in rows])
    return (
        F.one_hot(state_ids, STATE_COUNT).float(),
        F.one_hot(kind_ids, EVENT_KIND_COUNT).float(),
        F.one_hot(identity_ids, IDENTITY_COUNT).float(),
        F.one_hot(amount_ids, AMOUNT_COUNT).float(),
        targets,
    )


def fit_atomic_modules():
    torch.manual_seed(17)
    motor = TiedCategoricalMotor(hidden=96)
    motor_inputs = atomic_motor_board()
    optimizer = torch.optim.AdamW(motor.parameters(), lr=0.025, weight_decay=0.0)
    for _ in range(1200):
        optimizer.zero_grad(set_to_none=True)
        loss = atomic_motor_loss(
            motor,
            state=motor_inputs[0],
            event_kind=motor_inputs[1],
            event_identity=motor_inputs[2],
            amount=motor_inputs[3],
            next_state_targets=motor_inputs[4],
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            predicted = motor(*motor_inputs[:4]).argmax(dim=-1)
            if torch.equal(predicted, motor_inputs[4]) and loss.item() < 1e-3:
                break
    assert torch.equal(motor(*motor_inputs[:4]).argmax(dim=-1), motor_inputs[4])

    reader = CategoricalStateReader(hidden=48)
    state_ids = torch.arange(STATE_COUNT).repeat_interleave(3)
    query_ids = torch.arange(3).repeat(STATE_COUNT)
    states = F.one_hot(state_ids, STATE_COUNT).float()
    queries = F.one_hot(query_ids, 3).float()
    targets = torch.tensor([
        PERMUTATIONS[state][query]
        for state, query in zip(state_ids.tolist(), query_ids.tolist(), strict=True)
    ])
    optimizer = torch.optim.AdamW(reader.parameters(), lr=0.04, weight_decay=0.0)
    for _ in range(800):
        optimizer.zero_grad(set_to_none=True)
        loss = reader_loss(
            reader, state=states, query=queries, answer_targets=targets,
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            predicted = reader(states, queries).argmax(dim=-1)
            if torch.equal(predicted, targets) and loss.item() < 1e-3:
                break
    assert torch.equal(reader(states, queries).argmax(dim=-1), targets)
    return motor.eval(), reader.eval()


@pytest.fixture(scope="module")
def oracle_system():
    system = tiny_system(motor_hidden=96, reader_hidden=48)
    system.motor, system.reader = fit_atomic_modules()
    return system.eval()


def episode_tape() -> tuple[
    DeletedProgramTape, LateQuery, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    # Kinds: left=0, right=1, STOP=2.  Events after STOP are deliberately active-looking.
    kinds = torch.tensor([
        [1, 0, 2, 1, 0, 1, 0, 1],
        [0, 1, 2, 0, 1, 0, 1, 0],
    ])
    identities = torch.tensor([
        [0, 2, 0, 1, 2, 0, 1, 2],
        [1, 1, 2, 0, 2, 1, 0, 2],
    ])
    amounts = torch.tensor([
        [1, 0, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0, 1, 0],
    ])
    initial_states = torch.tensor([0, 4])
    queries = torch.tensor([0, 2])
    return DeletedProgramTape(
        categorical_logits(initial_states, STATE_COUNT),
        categorical_logits(kinds, EVENT_KIND_COUNT),
        categorical_logits(identities, IDENTITY_COUNT),
        categorical_logits(amounts, AMOUNT_COUNT),
    ), LateQuery(categorical_logits(queries, 3)), initial_states, kinds, identities, amounts


def manual_rollout(
    kinds: torch.Tensor,
    identities: torch.Tensor,
    amounts: torch.Tensor,
    initial_states: torch.Tensor,
    *,
    state_swap_after: int | None = None,
    state_swap_order: tuple[int, ...] | None = None,
) -> list[int]:
    states = initial_states.tolist()
    alive = [True for _ in states]
    for step in range(EVENT_STEPS):
        for row in range(len(states)):
            if alive[row] and int(kinds[row, step]) == 2:
                alive[row] = False
            elif alive[row]:
                states[row] = transition_target(
                    states[row],
                    int(kinds[row, step]),
                    int(identities[row, step]),
                    int(amounts[row, step]),
                )
        if state_swap_after == step:
            states = [states[index] for index in state_swap_order]
    return states


def test_deleted_tape_is_the_strict_source_boundary_and_static_code_is_clean():
    assert [field.name for field in fields(DeletedProgramTape)] == [
        "initial_state", "event_kind", "event_identity", "amount",
    ]
    tape, query, _, _, _, _ = episode_tape()
    assert not hasattr(tape, "__dict__")
    assert tape.initial_state.shape == (2, 6)
    assert tape.event_kind.shape == (2, 8, 3)
    assert tape.event_identity.shape == (2, 8, 3)
    assert tape.amount.shape == (2, 8, 2)
    assert query.logits.shape == (2, 3)
    hard_tape = tape.hard()
    hard_query = query.hard()
    assert isinstance(hard_tape, HardProgramTape)
    assert isinstance(hard_query, HardLateQuery)
    assert hard_tape.event_kind.dtype == torch.uint8
    assert hard_query.position.dtype == torch.uint8
    no_stop = torch.zeros_like(hard_tape.event_kind)
    with pytest.raises(ValueError, match="exactly one STOP"):
        HardProgramTape(
            hard_tape.initial_state,
            no_stop,
            hard_tape.event_identity,
            hard_tape.amount,
        )

    source = Path(sd_cst.__file__).read_text()
    tree = ast.parse(source)
    forbidden_imports = {"categorical_permutation_executor"}
    forbidden_calls = {
        "S3ClosedActionPermutationExecutor", "local_action_ids", "apply_op",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not ({alias.name for alias in node.names} & forbidden_imports)
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_imports
        elif isinstance(node, ast.Call):
            function = node.func
            name = function.id if isinstance(function, ast.Name) else (
                function.attr if isinstance(function, ast.Attribute) else None
            )
            assert name not in forbidden_calls

    rollout_node = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "rollout"
    )
    assert not any(isinstance(node, (ast.Break, ast.While)) for node in ast.walk(rollout_node))
    loops = [node for node in ast.walk(rollout_node) if isinstance(node, ast.For)]
    assert len(loops) == 1
    assert ast.unparse(loops[0].iter) == "range(EVENT_STEPS)"


def test_source_compiler_deletes_source_and_parameter_report_is_exact():
    system = tiny_system()
    ids = torch.randint(0, 48, (3, 11))
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[1, -2:] = False
    tape = system.compile_program(ids, valid)
    query = system.compile_late_query(ids[:, :4], valid[:, :4])
    assert isinstance(tape, DeletedProgramTape)
    assert isinstance(query, LateQuery)
    assert [field.name for field in fields(tape)] == [
        "initial_state", "event_kind", "event_identity", "amount",
    ]
    assert tape.initial_state.shape == (3, 6)
    assert tape.event_kind.shape == (3, 8, 3)
    assert tape.event_identity.shape == (3, 8, 3)
    assert tape.amount.shape == (3, 8, 2)
    assert query.logits.shape == (3, 3)
    assert not any(parameter.requires_grad for parameter in system.base_model.parameters())

    report = system.parameter_report()
    assert report["complete_system"] == sum(p.numel() for p in system.parameters())
    assert report["added"] == (
        report["compiler_added"] + report["motor"] + report["reader"]
    )
    assert report["complete_system"] == report["base"] + report["added"]
    assert report["trainable"] == report["added"]
    assert report["complete_system"] < report["strict_cap"] == 150_000_000
    assert report["headroom"] == report["strict_cap"] - report["complete_system"]


def test_rollout_calls_one_tied_motor_eight_times_and_stop_is_persistent():
    class ShiftMotor(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, state, event_kind, event_identity, amount):
            del event_kind, event_identity, amount
            self.calls += 1
            return torch.roll(state, shifts=1, dims=-1) * 20.0

    system = tiny_system()
    motor = ShiftMotor()
    system.motor = motor
    tape, query, initial_states, _, _, _ = episode_tape()
    result = system.rollout(tape, query, hard=True)
    assert motor.calls == EVENT_STEPS
    assert len(result.motor_logits) == EVENT_STEPS
    assert len(result.state_trajectory) == EVENT_STEPS
    assert torch.equal(result.state_trajectory[2], result.state_trajectory[-1])
    assert torch.equal(result.alive_trajectory[2], torch.zeros((2, 1)))

    frozen = system.rollout(tape, query, hard=True, control="freeze")
    expected_initial = F.one_hot(initial_states, STATE_COUNT).float()
    assert torch.equal(frozen.final_state, expected_initial)
    reset = system.rollout(tape, query, hard=True, control="reset")
    assert torch.equal(reset.final_state, result.state_trajectory[0])


def test_factorized_losses_and_recurrent_path_have_finite_gradients():
    torch.manual_seed(3)
    system = tiny_system()
    ids = torch.randint(0, 48, (2, 9))
    valid = torch.ones_like(ids, dtype=torch.bool)
    tape = system.compile_program(ids, valid)
    query = system.compile_late_query(ids[:, :4], valid[:, :4])
    kinds = torch.tensor([
        [0, 1, 2, 0, 1, 0, 1, 0],
        [1, 0, 2, 1, 0, 1, 0, 1],
    ])
    identities = torch.randint(0, 3, (2, EVENT_STEPS))
    amounts = torch.randint(0, 2, (2, EVENT_STEPS))
    queries = torch.tensor([0, 2])
    compiler_losses = compiler_field_losses(
        tape,
        initial_state_targets=torch.tensor([0, 4]),
        event_kind_targets=kinds,
        event_identity_targets=identities,
        amount_targets=amounts,
    )
    query_objective = late_query_loss(query, query_targets=queries)

    motor_inputs = atomic_motor_board()
    motor_objective = atomic_motor_loss(
        system.motor,
        state=motor_inputs[0],
        event_kind=motor_inputs[1],
        event_identity=motor_inputs[2],
        amount=motor_inputs[3],
        next_state_targets=motor_inputs[4],
    )
    state_ids = torch.tensor([0, 4, 5])
    query_ids = torch.tensor([2, 1, 0])
    reader_objective = reader_loss(
        system.reader,
        state=F.one_hot(state_ids, STATE_COUNT).float(),
        query=F.one_hot(query_ids, 3).float(),
        answer_targets=torch.tensor([
            PERMUTATIONS[state][query]
            for state, query in zip(state_ids.tolist(), query_ids.tolist(), strict=True)
        ]),
    )
    rollout = system.rollout(tape, query, hard=False)
    differentiability_probe = sum(
        logits.float().square().mean() for logits in rollout.motor_logits
    ) + rollout.answer_logits.float().square().mean()
    objective = (
        compiler_losses["total"] + query_objective + motor_objective + reader_objective
        + 1e-4 * differentiability_probe
    )
    objective.backward()

    trainable = [parameter for parameter in system.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)
    assert all(parameter.grad is None for parameter in system.base_model.parameters())


def test_oracle_fit_state_suffix_query_swaps_and_source_poison(oracle_system):
    tape, query, initial_states, kinds, identities, amounts = episode_tape()
    ordinary = oracle_system.rollout(tape, query, hard=True)
    expected = manual_rollout(kinds, identities, amounts, initial_states)
    assert ordinary.final_state.argmax(dim=-1).tolist() == expected

    permutation = torch.tensor([1, 0])
    swapped_state = oracle_system.rollout(
        tape,
        query,
        hard=True,
        state_swap=StateSwap(after_step=0, batch_permutation=permutation),
    )
    expected_state_swap = manual_rollout(
        kinds,
        identities,
        amounts,
        initial_states,
        state_swap_after=0,
        state_swap_order=(1, 0),
    )
    assert swapped_state.final_state.argmax(dim=-1).tolist() == expected_state_swap

    suffix = swap_tape_suffix(tape, permutation, start_step=1)
    suffix_result = oracle_system.rollout(suffix, query, hard=True)
    mixed_kinds = torch.cat((kinds[:, :1], kinds[permutation, 1:]), dim=1)
    mixed_identities = torch.cat((
        identities[:, :1], identities[permutation, 1:],
    ), dim=1)
    mixed_amounts = torch.cat((amounts[:, :1], amounts[permutation, 1:]), dim=1)
    assert suffix_result.final_state.argmax(dim=-1).tolist() == manual_rollout(
        mixed_kinds, mixed_identities, mixed_amounts, initial_states,
    )

    query_swapped = swap_late_queries(query, permutation)
    query_result = oracle_system.rollout(tape, query_swapped, hard=True)
    assert torch.equal(query_result.final_state, ordinary.final_state)
    expected_answers = [
        PERMUTATIONS[state][query]
        for state, query in zip(expected, [2, 0], strict=True)
    ]
    assert query_result.answer_logits.argmax(dim=-1).tolist() == expected_answers

    source_storage = torch.arange(32)
    poison = oracle_system.source_poison_invariance(
        tape, query, lambda: source_storage.fill_(-1), hard=True,
    )
    assert poison.bit_identical
    assert torch.equal(source_storage, torch.full_like(source_storage, -1))

    hard = oracle_system.rollout_hard(tape.hard(), query.hard())
    standalone = rollout_hard_categorical(
        oracle_system.motor, oracle_system.reader, tape.hard(), query.hard(),
    )
    assert hard.final_state.tolist() == expected
    assert torch.equal(standalone.final_state, hard.final_state)
    assert torch.equal(standalone.answer_logits, hard.answer_logits)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            standalone.state_trajectory, hard.state_trajectory, strict=True,
        )
    )
    assert all(state.dtype == torch.uint8 for state in hard.state_trajectory)
    assert all(alive.dtype == torch.bool for alive in hard.alive_trajectory)
