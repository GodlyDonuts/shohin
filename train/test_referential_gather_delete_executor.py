from types import SimpleNamespace

import torch

from referential_gather_delete_executor import (
    GatherDeletePermutationExecutor,
    SourceRetainedAnswerControl,
    execution_targets,
    executor_loss,
    gather_source_deleted_packet,
    normalized_sigmoid_weights,
    select_packet_operations,
    shuffle_operation_packet,
    shuffle_query_packet,
    sinkhorn,
)


def example(program=(("right", "aa", 1), ("left", "cc", 1)), query=2):
    positions = {
        "intro.entity0": (0,), "intro.entity1": (1,), "intro.entity2": (2,),
        "op0.kind": (3,), "op0.entity": (4,), "op0.literal": (5,),
        "op1.kind": (6,), "op1.entity": (7,), "op1.literal": (8,),
        "query.position": (9,),
    }
    return SimpleNamespace(
        initial_order=("aa", "bb", "cc"),
        program=program,
        query_position=query,
        target_positions=positions,
        kind_targets=(1, 0),
    )


def compiler_outputs(batch=2, length=10, width=8):
    torch.manual_seed(4)
    memory = torch.randn(batch, length, width)
    pointer_logits = {}
    labels = example().target_positions
    for label, positions in labels.items():
        logits = torch.full((batch, length), -20.0)
        logits[:, positions[0]] = 20.0
        pointer_logits[label] = logits
    kind_logits = torch.tensor([[[0.0, 20.0], [20.0, 0.0]]] * batch)
    return {
        "memory": memory,
        "lexical_memory": torch.randn(batch, length, width + 4),
        "pointer_logits": pointer_logits,
        "kind_logits": kind_logits,
    }


def test_normalized_sigmoid_weights_preserve_a_multi_token_span():
    logits = torch.tensor([[-30.0, 30.0, 30.0, -30.0]])
    weights = normalized_sigmoid_weights(logits, torch.ones(1, 4, dtype=torch.bool))
    assert torch.allclose(weights, torch.tensor([[0.0, 0.5, 0.5, 0.0]]), atol=1e-6)


def test_lexical_span_packet_separates_identity_and_context_channels():
    outputs = compiler_outputs(batch=1)
    valid = torch.ones(1, 10, dtype=torch.bool)
    packet = gather_source_deleted_packet(
        outputs, [example()], valid, packet_mode="lexical_sigmoid_span",
    )
    assert packet["initial_entities"].shape == (1, 3, 12)
    assert packet["operations"][0]["entity"].shape == (1, 12)
    assert packet["operations"][0]["kind_context"].shape == (1, 8)
    assert packet["query"].shape == (1, 8)
    assert packet["packet_mode"] == "lexical_sigmoid_span"


def test_split_width_executor_accepts_lexical_span_packet():
    outputs = compiler_outputs(batch=2)
    valid = torch.ones(2, 10, dtype=torch.bool)
    packet = gather_source_deleted_packet(
        outputs, [example(), example()], valid, packet_mode="lexical_sigmoid_span",
    )
    executor = GatherDeletePermutationExecutor(
        identity_width=12, context_width=8, width=16, tied=True,
    )
    result = executor(packet)
    assert result["answer_probabilities"].shape == (2, 3)


def test_execution_targets_are_destination_to_source_permutations():
    targets = execution_targets(example())
    assert targets.transition_sources == ((1, 0, 2), (0, 2, 1))
    assert targets.entity_locations == (0, 2)
    assert targets.amounts == (0, 0)
    assert targets.answer_identity == 0
    assert targets.final_identities == (1, 2, 0)


def test_atomic_target_restarts_from_initial_state():
    targets = execution_targets(example(), operation_indices=(1,))
    assert targets.transition_sources == ((0, 2, 1),)
    assert targets.entity_locations == (2,)
    assert targets.answer_identity == 1


def test_sinkhorn_is_doubly_stochastic():
    matrix = sinkhorn(torch.randn(7, 3, 3), iterations=20)
    assert torch.allclose(matrix.sum(-1), torch.ones(7, 3), atol=1e-4)
    assert torch.allclose(matrix.sum(-2), torch.ones(7, 3), atol=1e-4)


def test_packet_is_invariant_to_nonselected_source_changes():
    outputs = compiler_outputs(batch=1)
    valid = torch.ones(1, 10, dtype=torch.bool)
    packet = gather_source_deleted_packet(outputs, [example()], valid)
    changed = {
        **outputs,
        "memory": outputs["memory"].clone(),
    }
    # Every field is selected from positions 0..9, so append a valid distractor
    # whose logits are fixed near zero mass and perturb it arbitrarily.
    changed["memory"] = torch.cat((changed["memory"], torch.randn(1, 1, 8) * 1e6), dim=1)
    changed["pointer_logits"] = {
        label: torch.cat((logits, torch.full((1, 1), -1e9)), dim=1)
        for label, logits in changed["pointer_logits"].items()
    }
    changed_valid = torch.ones(1, 11, dtype=torch.bool)
    changed_packet = gather_source_deleted_packet(changed, [example()], changed_valid)
    assert torch.equal(packet["initial_entities"], changed_packet["initial_entities"])
    assert torch.equal(packet["query"], changed_packet["query"])
    for original, altered in zip(packet["operations"], changed_packet["operations"]):
        for field in original:
            assert torch.equal(original[field], altered[field])


def test_operation_shuffle_leaves_initial_and_query_packets_fixed():
    outputs = compiler_outputs(batch=2)
    valid = torch.ones(2, 10, dtype=torch.bool)
    packet = gather_source_deleted_packet(outputs, [example(), example()], valid)
    shuffled = shuffle_operation_packet(packet, [1, 0])
    assert shuffled["initial_entities"].data_ptr() == packet["initial_entities"].data_ptr()
    assert shuffled["query"].data_ptr() == packet["query"].data_ptr()
    assert torch.equal(shuffled["operations"][0]["entity"][0], packet["operations"][0]["entity"][1])


def test_query_shuffle_leaves_state_and_operation_packets_fixed():
    outputs = compiler_outputs(batch=2)
    valid = torch.ones(2, 10, dtype=torch.bool)
    packet = gather_source_deleted_packet(outputs, [example(), example()], valid)
    shuffled = shuffle_query_packet(packet, [1, 0])
    assert shuffled["initial_entities"].data_ptr() == packet["initial_entities"].data_ptr()
    assert shuffled["operations"] is packet["operations"]
    assert torch.equal(shuffled["query"][0], packet["query"][1])


def test_executor_accepts_only_bounded_packet_and_reuses_one_cell():
    outputs = compiler_outputs(batch=2)
    valid = torch.ones(2, 10, dtype=torch.bool)
    packet = gather_source_deleted_packet(outputs, [example(), example()], valid)
    executor = GatherDeletePermutationExecutor(packet_width=8, width=16, tied=True)
    result = executor(packet)
    assert len(executor.cells) == 1
    assert result["assignment"].shape == (2, 3, 3)
    assert result["answer_probabilities"].shape == (2, 3)
    assert torch.allclose(result["answer_probabilities"].sum(-1), torch.ones(2), atol=1e-4)
    atomic = select_packet_operations(packet, (1,))
    atomic_result = executor(atomic, cell_indices=(1,))
    assert len(atomic_result["transition_logits"]) == 1


def test_untied_executor_has_two_distinct_update_cells():
    executor = GatherDeletePermutationExecutor(packet_width=8, width=16, tied=False)
    assert len(executor.cells) == 2
    first = next(executor.cells[0].parameters())
    second = next(executor.cells[1].parameters())
    assert first.data_ptr() != second.data_ptr()


def test_executor_loss_has_finite_gradients():
    outputs = compiler_outputs(batch=2)
    valid = torch.ones(2, 10, dtype=torch.bool)
    examples = [example(), example()]
    packet = gather_source_deleted_packet(outputs, examples, valid)
    executor = GatherDeletePermutationExecutor(packet_width=8, width=16, tied=True)
    result = executor(packet)
    losses = executor_loss(result, [execution_targets(row) for row in examples])
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in executor.parameters()
    )


def test_source_retained_control_is_explicitly_separate():
    control = SourceRetainedAnswerControl(
        packet_width=8, width=16, heads=4, layers=1, ff=32,
    )
    result = control(torch.randn(2, 10, 8), torch.ones(2, 10, dtype=torch.bool))
    assert result["answer_logits"].shape == (2, 3)
