#!/usr/bin/env python3
"""CPU contracts for role-factorized microcode."""

import torch

from categorical_microcode import OPCODES, QUERIES
from model import GPT, GPTConfig
from role_equivariant_microcode import (
    IGNORE_ROLE,
    RoleEquivariantMicrocodeCompiler,
    compose_operation,
    compose_query,
    factor_operation,
    factor_query,
    permute_opcode,
    permute_query,
)


def main():
    for index, name in enumerate(OPCODES):
        kind, role = factor_operation(index)
        assert compose_operation(kind, role) == index, name
        twice = permute_opcode(permute_opcode(index))
        assert twice == index, name
        if name == "swap":
            assert role == IGNORE_ROLE and permute_opcode(index) == index
        else:
            assert factor_operation(permute_opcode(index))[1] == 1 - role
    for index, name in enumerate(QUERIES):
        kind, role = factor_query(index)
        assert compose_query(kind, role) == index, name
        assert permute_query(permute_query(index)) == index, name
        if name == "sum":
            assert role == IGNORE_ROLE and permute_query(index) == index
        else:
            assert factor_query(permute_query(index))[1] == 1 - role

    torch.manual_seed(3)
    model = GPT(GPTConfig(
        vocab_size=64, n_layer=3, n_head=4, n_kv_head=2,
        d_model=32, d_ff=64, seq_len=32, zloss=0.0,
    )).eval()
    compiler = RoleEquivariantMicrocodeCompiler(model, layer=1, hidden=24)
    ids = torch.randint(0, 64, (2, 12))
    hidden = compiler.encode(ids)
    batch = torch.tensor([0, 1])
    position = torch.tensor([6, 7])
    operation = compiler.classify_positions(hidden, batch, position, "operation")
    query = compiler.classify_positions(hidden, batch, position, "query")
    loss = operation.sum() + query.sum() + compiler.basis_loss()
    loss.backward()
    assert operation.shape == (2, len(OPCODES))
    assert query.shape == (2, len(QUERIES))
    assert any(parameter.grad is not None for parameter in compiler.adapter_parameters())
    assert all(parameter.grad is None for parameter in compiler.model.parameters())
    print("role-equivariant microcode tests passed")


if __name__ == "__main__":
    main()
