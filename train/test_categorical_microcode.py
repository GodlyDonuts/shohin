#!/usr/bin/env python3
"""CPU contracts for the Causal Microcode Bottleneck."""

from types import SimpleNamespace

import torch

from categorical_microcode import (
    CategoricalMicrocodeCompiler,
    OPCODE_TO_ID,
    QUERY_TO_ID,
    alu_basis_accuracy,
    compile_example,
    execute_program,
    lexical_initial_values,
    lexical_operation_value,
    transition_basis_targets,
)
from model import GPT, GPTConfig


class CharacterTokenizer:
    def encode(self, text):
        offsets, ids = [], []
        for index, character in enumerate(text):
            if not character.isspace():
                offsets.append((index, index + 1))
                ids.append(ord(character) % 127)
        return SimpleNamespace(ids=ids, offsets=offsets)


def row(heldout=False):
    if heldout:
        question = (
            "Task: A harbor inventory record marked H-harbor-000017 lists 37 items under crates and "
            "41 items under lanterns. The reference is not a quantity.\n"
            "Event 1: Relocate 12 items from lanterns into crates.\n"
            "Event 2: Exchange the values assigned to crates and lanterns.\n"
            "Request: After all updates, what is the final crates total?\nAnswer:"
        )
        initial = {"crates": 37, "lanterns": 41}
        keys = ["crates", "lanterns"]
        operations = [
            {"kind": "move", "source": "lanterns", "target": "crates", "value": 12},
            {"kind": "swap", "left": "crates", "right": "lanterns"},
        ]
        query = {"kind": "read", "key": "crates"}
        answer = "29"
    else:
        question = (
            "Question: In a workshop record copper has 12 parts and silver has 11 parts. "
            "The record label is not a quantity.\n"
            "Step 1: Add 3 parts to copper.\n"
            "Step 2: Move 2 parts from copper to silver.\n"
            "Question: How many more parts are in copper than in silver?\nAnswer:"
        )
        initial = {"copper": 12, "silver": 11}
        keys = ["copper", "silver"]
        operations = [
            {"kind": "add", "target": "copper", "value": 3},
            {"kind": "move", "source": "copper", "target": "silver", "value": 2},
        ]
        query = {"kind": "difference", "high": "copper", "low": "silver"}
        answer = "0"
    return {
        "question": question, "initial": initial, "keys": keys, "operations": operations,
        "query": query, "answer": answer, "reference": "r", "eval_regime": "fit_iid",
    }


def exact_table():
    target = transition_basis_targets()
    logits = torch.full((*target.shape, 20), -10.0)
    logits.scatter_(-1, target.unsqueeze(-1), 10.0)
    return logits


def main():
    tokenizer = CharacterTokenizer()
    core = compile_example(row(False), tokenizer)
    heldout = compile_example(row(True), tokenizer)
    assert core.initial_values == (12, 11)
    assert heldout.initial_values == (37, 41)
    assert core.operation_targets == (
        OPCODE_TO_ID["add_0"], OPCODE_TO_ID["move_0_1"],
    )
    assert heldout.operation_targets == (
        OPCODE_TO_ID["move_1_0"], OPCODE_TO_ID["swap"],
    )
    assert core.query_target == QUERY_TO_ID["difference_0_1"]
    assert heldout.query_target == QUERY_TO_ID["read_0"]
    assert lexical_initial_values(row(True)["question"]) == [37, 41]
    assert lexical_operation_value("Event 7: Relocate 12 things from a into b.") == 12

    table = exact_table()
    assert alu_basis_accuracy(table) == (400, 400)
    assert execute_program(
        core.initial_values, list(core.operation_targets), core.operation_values,
        core.query_target, table,
    ) == core.answer
    assert execute_program(
        heldout.initial_values, list(heldout.operation_targets), heldout.operation_values,
        heldout.query_target, table,
    ) == heldout.answer
    assert execute_program((999, 1), [OPCODE_TO_ID["add_0"]], [1], QUERY_TO_ID["read_0"], table) == 1000
    assert execute_program((1000, 1), [OPCODE_TO_ID["sub_0"]], [1], QUERY_TO_ID["read_0"], table) == 999

    torch.manual_seed(7)
    model = GPT(GPTConfig(
        vocab_size=128, n_layer=4, n_head=4, n_kv_head=2,
        d_model=32, d_ff=64, seq_len=512, zloss=0.0,
    )).eval()
    compiler = CategoricalMicrocodeCompiler(model, layer=2, hidden=24)
    ids = torch.tensor([core.ids], dtype=torch.long)
    hidden = compiler.encode(ids)
    operation = compiler.classify_positions(
        hidden, torch.tensor([0]), torch.tensor([core.operation_positions[0]]), "operation",
    )
    query = compiler.classify_positions(
        hidden, torch.tensor([0]), torch.tensor([core.query_position]), "query",
    )
    loss = operation.sum() + query.sum() + compiler.basis_loss()
    loss.backward()
    assert operation.shape == (1, 9) and query.shape == (1, 5)
    assert any(parameter.grad is not None for parameter in compiler.adapter_parameters())
    assert all(parameter.grad is None for parameter in compiler.model.parameters())

    print("categorical microcode tests passed")


if __name__ == "__main__":
    main()
