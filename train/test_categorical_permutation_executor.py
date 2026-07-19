import torch

from categorical_permutation_executor import (
    EquivariantCategoricalUpdateCell,
    PERMUTATIONS,
    S3CategoricalPermutationExecutor,
    S3ClosedActionPermutationExecutor,
    S3EquivariantPermutationExecutor,
    categorical_executor_loss,
    lexical_kind_predictions,
    permutation_matrices,
    local_action_ids,
    straight_through_permutation,
)
from referential_gather_delete_executor import ExecutionTargets


def test_permutation_matrices_are_the_s3_group_elements():
    matrices = permutation_matrices()
    assert matrices.shape == (6, 3, 3)
    assert torch.equal(matrices.sum(-1), torch.ones(6, 3))
    assert torch.equal(matrices.sum(-2), torch.ones(6, 3))
    assert len({tuple(matrix.argmax(-1).tolist()) for matrix in matrices}) == len(PERMUTATIONS)


def test_straight_through_forward_is_hard_and_backward_is_finite():
    logits = torch.randn(4, 6, requires_grad=True)
    matrix = straight_through_permutation(logits, permutation_matrices())
    assert torch.equal(matrix.detach().sum(-1), torch.ones(4, 3))
    assert torch.equal(matrix.detach().sum(-2), torch.ones(4, 3))
    matrix.square().sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_executor_keeps_exact_permutation_state_and_loss_backpropagates():
    executor = S3CategoricalPermutationExecutor(8, 6, width=12)
    packet = {
        "operations": tuple({
            "identity_probabilities": torch.eye(3)[torch.tensor([0, 1])],
            "kind_context": torch.randn(2, 6),
            "literal": torch.randn(2, 8),
            "kind_probabilities": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        } for _ in range(2)),
        "query": torch.randn(2, 6),
    }
    outputs = executor(packet)
    assert torch.equal(outputs["assignment"].detach().sum(-1), torch.ones(2, 3))
    assert torch.equal(outputs["assignment"].detach().sum(-2), torch.ones(2, 3))
    targets = [
        ExecutionTargets(
            transition_sources=((1, 0, 2), (1, 2, 0)),
            entity_locations=(0, 1),
            amounts=(0, 1),
            query_position=0,
            answer_identity=1,
            final_identities=(1, 2, 0),
        ),
        ExecutionTargets(
            transition_sources=((0, 2, 1), (2, 0, 1)),
            entity_locations=(1, 0),
            amounts=(1, 0),
            query_position=2,
            answer_identity=1,
            final_identities=(2, 0, 1),
        ),
    ]
    loss = categorical_executor_loss(outputs, targets)["total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in executor.parameters()
    )


def test_equivariant_cell_ignores_global_assignment_and_identity():
    cell = EquivariantCategoricalUpdateCell(width=8)
    location = torch.tensor([[0.0, 1.0, 0.0]])
    operation = torch.randn(1, 8)
    literal = torch.randn(1, 8)
    kind = torch.tensor([[1.0, 0.0]])
    first = cell(
        torch.eye(3).unsqueeze(0),
        torch.tensor([[1.0, 0.0, 0.0]]),
        location, operation, literal, kind,
    )
    second = cell(
        permutation_matrices()[4].unsqueeze(0),
        torch.tensor([[0.0, 0.0, 1.0]]),
        location, operation, literal, kind,
    )
    assert torch.equal(first, second)


def test_equivariant_executor_is_smaller_than_v1():
    v1 = S3CategoricalPermutationExecutor(576, 384, width=192)
    v11 = S3EquivariantPermutationExecutor(576, 384, width=192)
    assert v11.num_params() < v1.num_params()


def test_closed_local_action_table_matches_pop_insert_semantics():
    table = local_action_ids()
    matrices = permutation_matrices()
    for source in range(3):
        for kind in range(2):
            for amount_id in range(2):
                destination = (
                    max(0, source - amount_id - 1)
                    if kind == 0 else
                    min(2, source + amount_id + 1)
                )
                expected = list(range(3))
                expected.insert(destination, expected.pop(source))
                actual = matrices[table[source, kind, amount_id]].argmax(-1).tolist()
                assert actual == expected


def test_closed_action_composes_exactly_from_nonidentity_state():
    executor = S3ClosedActionPermutationExecutor(8, 6, width=12)
    with torch.no_grad():
        executor.amount_head.weight.zero_()
        executor.amount_head.bias[:] = torch.tensor([10.0, -10.0])
    packet = {
        "operations": (
            {
                "identity_probabilities": torch.tensor([[1.0, 0.0, 0.0]]),
                "kind_context": torch.randn(1, 6),
                "literal": torch.randn(1, 8),
                "kind_probabilities": torch.tensor([[0.0, 1.0]]),
            },
            {
                "identity_probabilities": torch.tensor([[0.0, 1.0, 0.0]]),
                "kind_context": torch.randn(1, 6),
                "literal": torch.randn(1, 8),
                "kind_probabilities": torch.tensor([[1.0, 0.0]]),
            },
        ),
        "query": torch.randn(1, 6),
    }
    outputs = executor(packet)
    assert outputs["assignment"].argmax(-1).tolist() == [[1, 0, 2]]
    assert [value.tolist() for value in outputs["kind_predictions"]] == [[1], [0]]


def test_lexical_kind_decoder_uses_pointer_mass_and_has_explicit_fallback():
    ids = torch.tensor([
        [9, 10, 11, 20, 21, 0],
        [9, 10, 11, 20, 21, 0],
    ])
    weights = torch.tensor([
        [0.01, 0.01, 0.01, 0.49, 0.49, 0.0],
        [0.10, 0.10, 0.10, 0.10, 0.10, 0.50],
    ])
    lexicon = {"patterns": [
        {"kind": "left", "token_ids": [9, 10, 11]},
        {"kind": "right", "token_ids": [20, 21]},
    ]}
    predictions, matched, scores = lexical_kind_predictions(
        ids, weights, lexicon, minimum_mass=0.5,
    )
    assert predictions.tolist() == [1, 0]
    assert matched.tolist() == [True, False]
    assert scores[0, 1] > scores[0, 0]
