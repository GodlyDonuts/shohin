import torch

from probe_referential_identity_packet import (
    gather,
    identity_predictions,
    normalized_sigmoid_weights,
)


def test_normalized_sigmoid_weights_ignore_padding_and_sum_to_one():
    logits = torch.tensor([[10.0, 10.0, -10.0, 20.0]])
    valid = torch.tensor([[True, True, True, False]])
    weights = normalized_sigmoid_weights(logits, valid)
    assert torch.allclose(weights.sum(-1), torch.ones(1))
    assert weights[0, 3] == 0
    assert torch.allclose(weights[0, 0], weights[0, 1])


def test_identity_prediction_is_invariant_to_shared_rotation():
    initial = torch.eye(3).unsqueeze(0)
    operations = [initial[:, 2], initial[:, 0]]
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    original = identity_predictions(initial, operations)
    rotated = identity_predictions(
        initial @ rotation,
        [operation @ rotation for operation in operations],
    )
    assert [value.tolist() for value in original] == [[2], [0]]
    assert [value.tolist() for value in rotated] == [[2], [0]]


def test_gather_preserves_weighted_identity():
    memory = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    weights = torch.tensor([[0.25, 0.75]])
    assert torch.allclose(gather(memory, weights), torch.tensor([[0.25, 0.75]]))
