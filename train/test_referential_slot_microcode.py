#!/usr/bin/env python3
"""CPU contracts for binding-first referential slot compilation."""

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from referential_slot_microcode import (
    ReferentialSlotMicrocodeCompiler,
    attention_mass_loss,
    compile_referential_example,
)


def main():
    root = Path(__file__).resolve().parents[1]
    tokenizer = Tokenizer.from_file(str(root / "artifacts/shohin-tok-32k.json"))
    row = json.loads((root / "artifacts/evals/categorical_microcode_manual_v1.jsonl").open().readline())
    example = compile_referential_example(row, tokenizer)
    assert len(example.intro_slot_targets) == 2
    assert len(example.operation_spans) == len(row["operations"])
    assert example.operation_mention_targets[2] == ()
    assert example.query_mention_target == ()
    for targets in example.intro_slot_targets:
        assert set(targets).issubset(example.intro_positions)
    for span, targets in zip(example.operation_spans, example.operation_mention_targets):
        assert set(targets).issubset(span)

    torch.manual_seed(7)
    model = GPT(GPTConfig(
        vocab_size=tokenizer.get_vocab_size(), n_layer=3, n_head=4, n_kv_head=2,
        d_model=32, d_ff=64, seq_len=256, zloss=0.0,
    )).eval()
    pointer = ReferentialSlotMicrocodeCompiler(model, layer=1, hidden=24, role_mode="pointer")
    absolute = ReferentialSlotMicrocodeCompiler(model, layer=1, hidden=24, role_mode="absolute")
    assert pointer.adapter_num_params() == absolute.adapter_num_params()
    ids = torch.tensor([example.compiled.ids], dtype=torch.long)
    hidden, identity = pointer.encode(ids)
    result = pointer.classify_text(
        hidden[0], identity[0], example.intro_positions, example.operation_spans, example.query_span,
    )
    assert len(result["operations"]) == len(row["operations"])
    operation = result["operations"][0]
    composed = pointer.compose_operation_logits(
        operation["kind_logits"].unsqueeze(0), operation["role_logits"].unsqueeze(0),
    )
    query = pointer.compose_query_logits(
        result["query"]["kind_logits"].unsqueeze(0), result["query"]["role_logits"].unsqueeze(0),
    )
    assert composed.shape == (1, 9)
    assert query.shape == (1, 5)

    losses = []
    for slot in range(2):
        losses.append(attention_mass_loss(
            result["intro_weights"][:, slot], result["intro_positions"],
            example.intro_slot_targets[slot],
        ))
    for output, targets in zip(result["operations"], example.operation_mention_targets):
        losses.append(attention_mass_loss(output["target_weights"], output["positions"], targets))
    losses.append(attention_mass_loss(
        result["query"]["target_weights"], result["query"]["positions"],
        example.query_mention_target,
    ))
    loss = composed.sum() + query.sum() + torch.stack(losses).sum() + pointer.basis_loss()
    loss.backward()
    assert all(parameter.grad is None for parameter in pointer.model.parameters())
    assert any(parameter.grad is not None for parameter in pointer.adapter_parameters())

    # Pointer role logits are exactly equivariant to exchanging the two slots.
    target = torch.randn(pointer.identity_projection.out_features)
    slots = torch.randn(2, pointer.identity_projection.out_features)
    context = torch.randn(24)
    logits = pointer._role_logits(target, context, torch.nn.functional.normalize(slots, dim=-1), "operation")
    swapped = pointer._role_logits(
        target, context, torch.nn.functional.normalize(slots.flip(0), dim=-1), "operation",
    )
    assert torch.allclose(logits.flip(0), swapped)
    print("referential slot microcode tests passed")


if __name__ == "__main__":
    main()
