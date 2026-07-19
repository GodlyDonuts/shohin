import torch

from probe_rgde_relational_identity import (
    ordered_sequence_scores,
    token_mass_cosine_scores,
)


def test_ordered_kernel_distinguishes_reversed_token_sequences():
    ids = torch.tensor([[4, 9, 12, 4, 12, 9]])
    left = torch.tensor([[1 / 3, 1 / 3, 1 / 3, 0, 0, 0]], dtype=torch.float32)
    # Reassign the second candidate to the reversed subsequence [4, 12, 9].
    ids = torch.tensor([[4, 9, 12, 4, 12, 9, 4, 9, 12]])
    left = torch.tensor([[1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0, 0, 0]])
    exact = torch.tensor([[0, 0, 0, 0, 0, 0, 1 / 3, 1 / 3, 1 / 3]])
    reversed_role = torch.tensor([[0, 0, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0]])
    scores = ordered_sequence_scores(ids, [exact, reversed_role], left)
    assert scores.argmax(-1).item() == 0
    assert scores[0, 0] > scores[0, 1]


def test_unordered_mass_treats_reversal_as_same_bag():
    ids = torch.tensor([[4, 9, 12, 4, 12, 9]])
    left = torch.tensor([[1 / 3, 1 / 3, 1 / 3, 0, 0, 0]])
    right = torch.tensor([[0, 0, 0, 1 / 3, 1 / 3, 1 / 3]])
    score = token_mass_cosine_scores(ids, [right], left, vocab_size=16)
    assert torch.allclose(score, torch.ones_like(score))


def test_ordered_kernel_is_translation_invariant():
    ids = torch.tensor([[1, 2, 3, 8, 8, 1, 2, 3]])
    left = torch.tensor([[0.2, 0.3, 0.5, 0, 0, 0, 0, 0]])
    right = torch.tensor([[0, 0, 0, 0, 0, 0.2, 0.3, 0.5]])
    score = ordered_sequence_scores(ids, [right], left)
    assert torch.allclose(score, torch.ones_like(score), atol=1e-6)
