#!/usr/bin/env python3
"""CPU contracts for the exact future-residual Jacobian estimator."""

import torch

from jacobian_workspace import jacobian_for_ids, transport, valid_position_mask
from model import GPT, GPTConfig


def main():
    torch.manual_seed(17)
    model = GPT(GPTConfig(
        vocab_size=64,
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        d_model=16,
        d_ff=32,
        seq_len=32,
        zloss=0.0,
    )).eval()
    model.requires_grad_(False)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]], dtype=torch.long)
    one, count_one = jacobian_for_ids(
        model, ids, [0, 2], target_layer=3, dim_batch=1, skip_first=1,
    )
    four, count_four = jacobian_for_ids(
        model, ids, [0, 2], target_layer=3, dim_batch=4, skip_first=1,
    )
    assert count_one == count_four == 6
    assert valid_position_mask(8, 1).tolist() == [False, True, True, True, True, True, True, False]
    for layer in (0, 2):
        assert one[layer].shape == (16, 16)
        assert torch.isfinite(one[layer]).all()
        assert torch.allclose(one[layer], four[layer], atol=2e-6, rtol=2e-5)
        projected = transport(one[layer], torch.randn(3, 16))
        assert projected.shape == (3, 16)
    assert all(parameter.grad is None for parameter in model.parameters())
    print("jacobian workspace tests passed")


if __name__ == "__main__":
    main()
