"""Fixed-slot continuous memory with source removal for isolated research.

Unlike the continuous-feedback control, this module does not keep the original
source prompt available while it thinks. Each source chunk is written into a
fixed number of continuous slots, and the answer decoder receives only those
slots plus a fresh query. This makes information flow and ablations explicit.

The wrapper deliberately leaves ``GPT.forward`` and the flagship path
unchanged. It is for small, verifier-backed research runs only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from latent_rollout import build_answer_targets


class SourceDroppingMemory(nn.Module):
    """Read chunks into continuous slots, then decode with source text absent."""

    def __init__(self, model, slots: int, max_chunks: int):
        super().__init__()
        if slots < 0:
            raise ValueError("slots must be non-negative")
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        self.model = model
        self.slots = int(slots)
        self.max_chunks = int(max_chunks)
        d_model = model.cfg.d_model
        if self.slots:
            scale = model.tok.weight.detach().float().square().mean().sqrt().item()
            self.initial_slots = nn.Parameter(torch.randn(1, self.slots, d_model) * scale)
            self.write_slots = nn.Parameter(torch.randn(1, self.slots, d_model) * scale)
            # The writer needs to distinguish chunk order even though RoPE
            # positions reset per chunk to keep each bounded segment compact.
            self.chunk_bias = nn.Parameter(torch.zeros(self.max_chunks, self.slots, d_model))
        else:
            self.register_parameter("initial_slots", None)
            self.register_parameter("write_slots", None)
            self.register_parameter("chunk_bias", None)

    def _input_scale(self, dtype, device):
        scale = self.model.tok.weight.detach().float().square().mean().sqrt()
        return scale.to(dtype=dtype, device=device)

    def _validate_chunks(self, chunks):
        if isinstance(chunks, torch.Tensor):
            if chunks.ndim != 3:
                raise ValueError("dense chunks must have shape [batch, chunks, tokens]")
            chunks = tuple(chunks[:, index, :] for index in range(chunks.shape[1]))
        elif isinstance(chunks, (list, tuple)):
            chunks = tuple(chunks)
        else:
            raise ValueError("chunks must be a dense tensor or a sequence of [batch, tokens] tensors")
        if not chunks:
            raise ValueError("chunks must include at least one non-empty chunk")
        batch = None
        for chunk in chunks:
            if not isinstance(chunk, torch.Tensor) or chunk.ndim != 2 or chunk.dtype != torch.long:
                raise ValueError("each chunk must have shape [batch, tokens] and dtype torch.long")
            if not chunk.shape[1]:
                raise ValueError("chunks must include at least one non-empty chunk")
            if batch is None:
                batch = chunk.shape[0]
            elif chunk.shape[0] != batch:
                raise ValueError("all chunks must have the same batch size")
            if self.slots * 2 + chunk.shape[1] > self.model.cfg.seq_len:
                raise ValueError("source chunk plus slots exceeds model sequence length")
        chunk_count = len(chunks)
        if chunk_count > self.max_chunks:
            raise ValueError("chunk count exceeds configured max_chunks")
        return chunks

    def encode(self, chunks):
        """Recursively write chunks into a fixed continuous memory packet."""
        chunks = self._validate_chunks(chunks)
        batch, chunk_count = chunks[0].shape[0], len(chunks)
        if not self.slots:
            return self.model.tok.weight.new_empty((batch, 0, self.model.cfg.d_model))
        state = self.initial_slots.to(
            dtype=self.model.tok.weight.dtype,
            device=chunks[0].device,
        ).expand(batch, -1, -1)
        for chunk_index in range(chunk_count):
            source = self.model.tok(chunks[chunk_index])
            write = self.write_slots.to(dtype=source.dtype, device=source.device).expand(batch, -1, -1)
            write = write + self.chunk_bias[chunk_index].to(dtype=source.dtype, device=source.device).unsqueeze(0)
            _, _, hidden = self.model.forward_embeds(torch.cat((state, source, write), dim=1), return_hidden=True)
            state = hidden[:, -self.slots:, :] * self._input_scale(hidden.dtype, hidden.device)
        return state

    def answer_context(self, memory, query_ids):
        """Return the only context the decoder may see: packet plus new query."""
        if query_ids.ndim != 2 or query_ids.dtype != torch.long:
            raise ValueError("query_ids must have shape [batch, tokens] and dtype torch.long")
        if memory.ndim != 3 or memory.shape[0] != query_ids.shape[0] or memory.shape[1] != self.slots:
            raise ValueError("memory must have matching batch and configured slot count")
        return torch.cat((memory, self.model.tok(query_ids)), dim=1)

    def supervised_loss(self, chunks, query_ids, answer_ids, eos_id: int):
        """Answer-only loss after source removal, with gradients through all writes."""
        if answer_ids.ndim != 2 or answer_ids.dtype != torch.long:
            raise ValueError("answer_ids must have shape [batch, tokens] and dtype torch.long")
        if answer_ids.shape[0] != query_ids.shape[0]:
            raise ValueError("query and answer batch sizes differ")
        memory = self.encode(chunks)
        prefix = self.answer_context(memory, query_ids)
        if prefix.shape[1] + answer_ids.shape[1] > self.model.cfg.seq_len:
            raise ValueError("memory packet plus query and answer exceeds model sequence length")
        full_embeds = torch.cat((prefix, self.model.tok(answer_ids)), dim=1)
        targets = build_answer_targets(answer_ids, prefix.shape[1], 0, eos_id)
        logits, loss = self.model.forward_embeds(full_embeds, targets=targets)
        return logits, loss, memory, targets

    @torch.no_grad()
    def generate(self, chunks, query_ids, eos_id: int, max_new: int):
        """Greedy answer decoding with the source absent after ``encode``."""
        if max_new <= 0:
            raise ValueError("max_new must be positive")
        if query_ids.shape[0] != 1:
            raise ValueError("source-dropping generation currently accepts one example")
        context = self.answer_context(self.encode(chunks), query_ids)
        generated = []
        for _ in range(max_new):
            logits, _ = self.model.forward_embeds(context)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token = int(next_id.item())
            if token == int(eos_id):
                break
            generated.append(token)
            if context.shape[1] >= self.model.cfg.seq_len:
                break
            context = torch.cat((context, self.model.tok(next_id)), dim=1)
        return torch.tensor(generated, dtype=torch.long, device=query_ids.device)
