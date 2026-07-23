from __future__ import annotations

import dataclasses
from functools import lru_cache
import inspect

import pytest
import torch
from neural_tcrr_board import ExpectedTransitionRecord, packet_sha256
from neural_tcrr_committer import (
    CHILD_PRESENT,
    KIND_CONSTRUCTOR,
    KIND_VARIABLE,
    NODE_WRITE,
    commit_neural_tcrr_graph,
)
from neural_tcrr_motor import (
    NeuralTcrrGraphDelta,
    NeuralTcrrMotor,
    NeuralTcrrMotorConfig,
    NeuralTcrrMotorOutput,
)
from tensorize_neural_tcrr_packets import (
    A,
    N,
    NeuralTcrrPacketTensors,
    tensorize_neural_tcrr_packets,
)
from tensorize_neural_tcrr_training import (
    PATH_STOP,
    tensorize_neural_tcrr_training,
)
import train_neural_tcrr_motor as harness


@lru_cache(maxsize=1)
def _partitions() -> harness.ProceduralPartitions:
    return harness.load_procedural_partitions(
        source="pilot",
        seed=2026072302,
    )


def _tiny_motor(seed: int = 7) -> NeuralTcrrMotor:
    torch.manual_seed(seed)
    return NeuralTcrrMotor(
        NeuralTcrrMotorConfig(
            hidden_dim=8,
            entity_rounds=1,
            term_rounds=1,
            graph_rounds=1,
            max_arity=A,
        )
    )


def _selected_logits(
    base: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    output = torch.full_like(base, -20.0)
    output.scatter_(-1, indices.unsqueeze(-1), 20.0)
    return output


def _forced_graph_delta(
    base: NeuralTcrrGraphDelta,
    target: harness.GraphTransactionTarget,
) -> NeuralTcrrGraphDelta:
    operation = _selected_logits(
        base.node_operation_logits,
        target.node_operation.view(1, N),
    )
    root = _selected_logits(
        base.root_pointer_logits,
        torch.tensor([target.root_pointer]),
    )
    kind_indices = torch.zeros((1, N), dtype=torch.long)
    type_indices = torch.zeros((1, N), dtype=torch.long)
    constructor_indices = torch.zeros((1, N), dtype=torch.long)
    variable_indices = torch.zeros((1, N), dtype=torch.long)
    child_indices = torch.full((1, N, A), N, dtype=torch.long)
    presence_indices = torch.zeros((1, N, A), dtype=torch.long)
    for storage in range(N):
        if int(target.node_operation[storage]) != NODE_WRITE:
            continue
        kind = int(target.node_kind[storage])
        kind_indices[0, storage] = kind
        type_indices[0, storage] = int(target.node_type_pointer[storage])
        if kind == KIND_CONSTRUCTOR:
            constructor_indices[0, storage] = int(
                target.node_constructor_pointer[storage]
            )
        elif kind == KIND_VARIABLE:
            variable_indices[0, storage] = int(target.node_variable_pointer[storage])
        for argument in range(A):
            presence = int(target.child_presence[storage, argument])
            presence_indices[0, storage, argument] = presence
            if presence == CHILD_PRESENT:
                child_indices[0, storage, argument] = int(
                    target.child_pointer[storage, argument]
                )
    return NeuralTcrrGraphDelta(
        node_operation_logits=operation,
        node_operation_mask=base.node_operation_mask,
        root_pointer_logits=root,
        root_pointer_mask=base.root_pointer_mask,
        node_kind_logits=_selected_logits(base.node_kind_logits, kind_indices),
        node_kind_mask=base.node_kind_mask,
        node_type_pointer_logits=_selected_logits(
            base.node_type_pointer_logits,
            type_indices,
        ),
        node_type_pointer_mask=base.node_type_pointer_mask,
        node_constructor_pointer_logits=_selected_logits(
            base.node_constructor_pointer_logits,
            constructor_indices,
        ),
        node_constructor_pointer_mask=base.node_constructor_pointer_mask,
        node_variable_pointer_logits=_selected_logits(
            base.node_variable_pointer_logits,
            variable_indices,
        ),
        node_variable_pointer_mask=base.node_variable_pointer_mask,
        child_pointer_logits=_selected_logits(
            base.child_pointer_logits,
            child_indices,
        ),
        child_pointer_mask=base.child_pointer_mask,
        child_presence_logits=_selected_logits(
            base.child_presence_logits,
            presence_indices,
        ),
        child_presence_mask=base.child_presence_mask,
    )


def _forced_output(
    packet: object,
    record: ExpectedTransitionRecord,
    *,
    action_index: int | None,
) -> NeuralTcrrMotorOutput:
    packetized = tensorize_neural_tcrr_packets((packet,))
    model = _tiny_motor(seed=11)
    model.eval()
    with torch.inference_mode():
        base = model(packetized.tensors)
    offline = tensorize_neural_tcrr_training((packet,), (record,)).tensors
    target = harness.derive_graph_transaction_target(
        packetized.tensors,
        offline if action_index is not None else None,
        batch_index=0,
        action_index=action_index,
    )
    no_redex_logits = torch.tensor(
        [[-20.0, 20.0] if action_index is None else [20.0, -20.0]]
    )
    halt_logits = torch.tensor(
        [[-20.0, 20.0] if action_index is None else [20.0, -20.0]]
    )
    rule_logits = torch.full_like(base.rule_logits, -20.0)
    path_logits = torch.full_like(base.path_logits, -20.0)
    binding_logits = torch.full_like(base.binding_logits, -20.0)
    if action_index is None:
        rule_logits[base.rule_mask] = 0.0
        path_logits[base.path_mask] = 0.0
        binding_logits[base.binding_mask] = 0.0
    else:
        rule = int(
            torch.nonzero(
                offline.rule_pointer[0, action_index],
                as_tuple=False,
            )[0].item()
        )
        rule_logits[0, rule] = 20.0
        for depth in range(offline.path_tokens.shape[-1]):
            if not bool(offline.path_token_mask[0, action_index, depth]):
                break
            token = int(offline.path_tokens[0, action_index, depth])
            path_logits[0, depth, token] = 20.0
            if token == PATH_STOP:
                break
        for variable in (
            torch.nonzero(
                offline.variable_binding_mask[0, action_index],
                as_tuple=False,
            )
            .flatten()
            .tolist()
        ):
            storage = int(
                torch.nonzero(
                    offline.variable_binding[0, action_index, variable],
                    as_tuple=False,
                )[0].item()
            )
            binding_logits[0, rule, variable, storage] = 20.0
        binding_logits[base.binding_mask & (binding_logits == -20.0)] = 0.0
    return NeuralTcrrMotorOutput(
        no_redex_logits=no_redex_logits,
        halt_logits=halt_logits,
        rule_logits=rule_logits,
        rule_mask=base.rule_mask,
        path_logits=path_logits,
        path_mask=base.path_mask,
        binding_logits=binding_logits,
        binding_mask=base.binding_mask,
        graph_delta=_forced_graph_delta(base.graph_delta, target),
    )


class _StaticMotor(torch.nn.Module):
    def __init__(self, output: NeuralTcrrMotorOutput) -> None:
        super().__init__()
        self.output = output
        self.seen_packet_boundary = False

    def forward(self, packets: NeuralTcrrPacketTensors) -> NeuralTcrrMotorOutput:
        assert isinstance(packets, NeuralTcrrPacketTensors)
        self.seen_packet_boundary = True
        return self.output


def test_complete_legal_set_loss_is_label_permutation_invariant() -> None:
    partition = _partitions().train
    packet = partition.packets[0]
    record = partition.records[0]
    assert len(record.transitions) == 4
    reversed_record = dataclasses.replace(
        record,
        transitions=tuple(reversed(record.transitions)),
    )
    packets = tensorize_neural_tcrr_packets((packet,)).tensors
    original = tensorize_neural_tcrr_training((packet,), (record,)).tensors
    permuted = tensorize_neural_tcrr_training(
        (packet,),
        (reversed_record,),
    ).tensors
    model = _tiny_motor()
    output = model(packets)
    first = harness.complete_legal_set_loss(output, packets, original)
    second = harness.complete_legal_set_loss(output, packets, permuted)
    torch.testing.assert_close(first.loss, second.loss, rtol=0.0, atol=1e-6)
    assert first.complete_legal_actions == len(record.transitions)
    assert first.redex_examples == 1
    assert first.no_redex_examples == 0
    first.loss.backward()
    for parameter in (
        model.no_redex_head.layers[-1].weight,
        model.rule_score.weight,
        model.path_argument.layers[-1].weight,
        model.binding_key.weight,
        model.node_operation.weight,
    ):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert float(parameter.grad.abs().sum()) > 0.0


def test_derived_transactions_commit_to_every_exact_successor() -> None:
    partition = _partitions().train
    packet = partition.packets[0]
    record = partition.records[0]
    packetized = tensorize_neural_tcrr_packets((packet,))
    offline = tensorize_neural_tcrr_training((packet,), (record,)).tensors
    for action_index in range(len(record.transitions)):
        target = harness.derive_graph_transaction_target(
            packetized.tensors,
            offline,
            batch_index=0,
            action_index=action_index,
        )
        transaction = harness.graph_transaction_from_target(
            target,
            device=torch.device("cpu"),
        )
        committed = commit_neural_tcrr_graph(packetized.tensors, transaction)
        assert harness._committed_matches_successor(  # noqa: SLF001
            committed,
            offline,
            action_index=action_index,
        )

    no_redex_packet = partition.packets[7]
    no_redex_record = partition.records[7]
    assert not no_redex_record.transitions
    no_redex_tensors = tensorize_neural_tcrr_packets((no_redex_packet,)).tensors
    identity = harness.derive_graph_transaction_target(
        no_redex_tensors,
        None,
        batch_index=0,
        action_index=None,
    )
    committed = commit_neural_tcrr_graph(
        no_redex_tensors,
        harness.graph_transaction_from_target(
            identity,
            device=torch.device("cpu"),
        ),
    )
    assert harness._committed_matches_input(  # noqa: SLF001
        committed,
        no_redex_tensors,
    )


def test_tiny_no_redex_overfit_decreases_loss_and_hard_passes() -> None:
    partitions = _partitions()
    tiny = harness.subset_partition(partitions.train, (7,))
    assert not tiny.records[0].transitions
    model = _tiny_motor()
    report = harness.train_motor(
        model,
        tiny,
        config=harness.TrainingConfig(
            updates=8,
            batch_size=1,
            learning_rate=2e-2,
            weight_decay=0.0,
            gradient_clip=5.0,
            seed=7,
            log_interval=8,
        ),
        device=torch.device("cpu"),
    )
    assert report["loss_decreased"]
    assert report["final_calibration_loss"] < 0.1 * report["initial_calibration_loss"]
    expected_digest = packet_sha256(tiny.packets[0])
    assert report["optimizer_packet_digests"] == [expected_digest]
    development_digests = {
        item.packet_sha256 for item in partitions.development.metadata
    }
    assert not development_digests.intersection(report["optimizer_packet_digests"])
    evaluation = harness.evaluate_motor(
        model,
        tiny,
        device=torch.device("cpu"),
    )
    assert evaluation["exact_rate"] == 1.0
    assert evaluation["no_redex_exact_rate"] == 1.0
    assert evaluation["invalid_commit_reasons"] == {}


def test_hard_positive_decode_commits_once_and_matches_complete_set() -> None:
    partition = _partitions().train
    packet = partition.packets[0]
    record = partition.records[0]
    output = _forced_output(packet, record, action_index=2)
    model = _StaticMotor(output)
    result = harness.assess_one_example(
        model,
        packet,
        record,
        device=torch.device("cpu"),
    )
    assert model.seen_packet_boundary
    assert result.exact
    assert result.commit_valid
    assert result.matched_action_index == 2
    assert not result.predicted_no_redex
    assert not result.predicted_halt

    wrong_status = dataclasses.replace(
        output,
        no_redex_logits=torch.tensor([[-20.0, 20.0]]),
    )
    wrong = harness.assess_one_example(
        _StaticMotor(wrong_status),
        packet,
        record,
        device=torch.device("cpu"),
    )
    assert not wrong.exact
    assert wrong.commit_valid


def test_exact_no_redex_and_invalid_commit_reason_are_audited() -> None:
    partition = _partitions().train
    packet = partition.packets[7]
    record = partition.records[7]
    output = _forced_output(packet, record, action_index=None)
    result = harness.assess_one_example(
        _StaticMotor(output),
        packet,
        record,
        device=torch.device("cpu"),
    )
    assert result.exact
    assert result.no_redex_target

    invalid_root_logits = torch.full_like(
        output.graph_delta.root_pointer_logits,
        -20.0,
    )
    invalid_root_logits[0, N] = 20.0
    invalid_delta = dataclasses.replace(
        output.graph_delta,
        root_pointer_logits=invalid_root_logits,
    )
    invalid_output = dataclasses.replace(output, graph_delta=invalid_delta)
    invalid_result = harness.assess_one_example(
        _StaticMotor(invalid_output),
        packet,
        record,
        device=torch.device("cpu"),
    )
    assert not invalid_result.exact
    assert not invalid_result.commit_valid
    assert invalid_result.invalid_reason == "null_root_with_active_graph"


def test_optimizer_rejects_development_and_forward_has_no_label_channel() -> None:
    partitions = _partitions()
    with pytest.raises(
        harness.NeuralTcrrTrainingHarnessError,
        match="train partition",
    ):
        harness.train_motor(
            _tiny_motor(),
            partitions.development,
            config=harness.TrainingConfig(updates=1, batch_size=1),
            device=torch.device("cpu"),
        )
    singleton = harness.subset_partition(partitions.train, (0,))
    with pytest.raises(
        harness.NeuralTcrrTrainingHarnessError,
        match="unique training packets",
    ):
        harness.train_motor(
            _tiny_motor(),
            singleton,
            config=harness.TrainingConfig(updates=1, batch_size=2),
            device=torch.device("cpu"),
        )
    signature = inspect.signature(harness._forward_packet_only)  # noqa: SLF001
    assert tuple(signature.parameters) == ("model", "packets")
    source = inspect.getsource(harness._forward_packet_only).lower()  # noqa: SLF001
    for forbidden in ("label", "successor", "expected", "oracle", "record"):
        assert forbidden not in source
    train_signature = inspect.signature(harness.train_motor)
    assert "development" not in train_signature.parameters
    assert "evaluation" not in train_signature.parameters


def test_score_bearing_source_receipt_covers_generator_and_mechanics() -> None:
    pilot = harness._source_receipts(source="pilot")  # noqa: SLF001
    corpus = harness._source_receipts(source="corpus")  # noqa: SLF001
    required = {
        "trainer",
        "motor",
        "committer",
        "packet_tensorizer",
        "training_tensorizer",
        "packet_mechanics",
        "rewrite_mechanics",
        "procedural_generator",
    }
    assert set(pilot) == required
    assert set(corpus) == required
    assert all(len(value) == 64 for value in pilot.values())
    assert all(len(value) == 64 for value in corpus.values())
    assert pilot["procedural_generator"] != corpus["procedural_generator"]


def test_evaluation_report_has_crossed_cells_and_parameter_budget() -> None:
    partition = harness.subset_partition(_partitions().development, (0,))
    model = _tiny_motor()
    report = harness.evaluate_motor(
        model,
        partition,
        device=torch.device("cpu"),
    )
    assert report["examples"] == 1
    assert len(report["cells"]) == 1
    cell = report["cells"][0]
    assert set(("family", "depth", "renderer", "composition", "exact_rate")).issubset(
        cell
    )
    ledger = model.parameter_count()
    assert ledger.under_cap
    assert ledger.under_system_cap
