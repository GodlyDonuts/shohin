#!/usr/bin/env python3
"""Focused gradient and source-removal checks for continuous memory packets."""

import torch

from model import GPT, GPTConfig
from source_dropping_memory import SourceDroppingMemory


def tiny_model():
    return GPT(GPTConfig(
        vocab_size=64,
        n_layer=2,
        n_head=3,
        n_kv_head=1,
        d_model=48,
        d_ff=96,
        seq_len=48,
        zloss=0.0,
    ))


def main():
    torch.manual_seed(7)
    chunks = torch.tensor([
        [[2, 3, 4, 5], [6, 7, 8, 9]],
        [[10, 11, 12, 13], [14, 15, 16, 17]],
    ])
    query = torch.tensor([[18, 19, 20], [21, 22, 23]])
    answer = torch.tensor([[24, 25], [26, 27]])
    memory = SourceDroppingMemory(tiny_model(), slots=2, max_chunks=3)
    logits, loss, packet, targets = memory.supervised_loss(chunks, query, answer, eos_id=1)
    assert logits.shape[:2] == targets.shape
    assert packet.shape == (2, 2, 48)
    context = memory.answer_context(packet, query)
    assert context.shape[1] == 2 + query.shape[1]
    assert context.shape[1] < chunks.shape[1] * chunks.shape[2] + query.shape[1]
    loss.backward()
    assert torch.isfinite(loss)
    assert memory.initial_slots.grad is not None and memory.initial_slots.grad.abs().sum() > 0
    assert memory.write_slots.grad is not None and memory.write_slots.grad.abs().sum() > 0
    assert memory.model.tok.weight.grad is not None and memory.model.tok.weight.grad.abs().sum() > 0

    changed = chunks.clone()
    changed[:, 0, 0] += 1
    with torch.no_grad():
        assert not torch.allclose(packet, memory.encode(changed))

    memory.zero_grad(set_to_none=True)
    final_packet, trajectory = memory.encode(chunks, return_trace=True)
    assert len(trajectory) == chunks.shape[1]
    assert torch.allclose(final_packet, packet)
    assert torch.allclose(trajectory[-1], final_packet)
    trajectory[-1].square().mean().backward()
    assert memory.chunk_bias.grad is not None and memory.chunk_bias.grad.abs().sum() > 0

    # Real source records have a stable chunk-count but variable chunk widths.
    ragged_chunks = (chunks[:, 0, :], chunks[:, 1, :-1])
    ragged_logits, ragged_loss, ragged_packet, _ = memory.supervised_loss(
        ragged_chunks, query, answer, eos_id=1,
    )
    assert ragged_logits.shape[0] == query.shape[0]
    assert ragged_packet.shape == packet.shape
    assert torch.isfinite(ragged_loss)

    empty = SourceDroppingMemory(tiny_model(), slots=0, max_chunks=3)
    empty_packet = empty.encode(chunks)
    assert empty_packet.shape == (2, 0, 48)
    assert empty.answer_context(empty_packet, query).shape[1] == query.shape[1]
    empty_final, empty_trace = empty.encode(chunks, return_trace=True)
    assert len(empty_trace) == chunks.shape[1]
    assert empty_final.shape == (2, 0, 48)
    print("source-dropping memory tests passed")


if __name__ == "__main__":
    main()
