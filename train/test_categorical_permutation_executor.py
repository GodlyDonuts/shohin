import torch

from categorical_permutation_executor import (
    PERMUTATIONS,
    S3CategoricalPermutationExecutor,
    categorical_executor_loss,
    permutation_matrices,
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
