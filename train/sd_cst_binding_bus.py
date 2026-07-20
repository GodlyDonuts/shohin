"""Content-addressable binding bus for the byte-addressed SD-CST compiler."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sd_cst import DeletedProgramTape, LateQuery
from sd_cst_byte_addressed import ByteAddressedCompiler


BIGRAM_VOCAB = 256 * 256 + 1
BIGRAM_PAD = 256 * 256
PERMUTATIONS = tuple(itertools.permutations(range(3)))


@dataclass(frozen=True, slots=True)
class BindingBusOutput:
    tape: DeletedProgramTape
    line_pointer_logits: torch.Tensor
    binding_pointer_logits: torch.Tensor
    initial_entity_pointer_logits: torch.Tensor
    event_entity_pointer_logits: torch.Tensor


class BindingBusCompiler(ByteAddressedCompiler):
    """Resolve arbitrary names by learned addressing plus shared content fingerprints."""

    def __init__(self, *, fingerprint_width: int = 96, **kwargs: int) -> None:
        super().__init__(**kwargs)
        if fingerprint_width <= 0:
            raise ValueError("fingerprint width must be positive")
        self.fingerprint_width = int(fingerprint_width)
        self.binding_queries = nn.Parameter(torch.empty(3, self.width))
        self.initial_entity_queries = nn.Parameter(torch.empty(3, self.width))
        self.event_entity_queries = nn.Parameter(torch.empty(8, self.width))
        nn.init.normal_(self.binding_queries, std=0.02)
        nn.init.normal_(self.initial_entity_queries, std=0.02)
        nn.init.normal_(self.event_entity_queries, std=0.02)
        self.bigram_embedding = nn.Embedding(
            BIGRAM_VOCAB, fingerprint_width, padding_idx=BIGRAM_PAD,
        )
        self.fingerprint_projection = nn.Linear(
            fingerprint_width, fingerprint_width, bias=False,
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        permutation = torch.tensor(PERMUTATIONS, dtype=torch.long)
        self.register_buffer("permutations", permutation, persistent=True)

    def _pointer_logits(
        self,
        memory: torch.Tensor,
        valid_mask: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.einsum(
            "sw,blw->bsl",
            self.query_projection(queries),
            self.key_projection(memory),
        ) / math.sqrt(self.width)
        return logits.masked_fill(
            ~valid_mask[:, None], torch.finfo(logits.dtype).min,
        ).float()

    @staticmethod
    def _bigram_ids(ids: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        following = torch.full_like(ids, 256)
        following[:, :-1] = ids[:, 1:].clamp_max(255)
        first = ids.clamp_max(255)
        bigrams = first * 256 + following
        pair_valid = valid_mask & torch.cat(
            (valid_mask[:, 1:], torch.zeros_like(valid_mask[:, :1])), dim=1,
        )
        return torch.where(pair_valid, bigrams, torch.full_like(bigrams, BIGRAM_PAD))

    def _fingerprints(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
        pointer_logits: torch.Tensor,
    ) -> torch.Tensor:
        weights = pointer_logits.softmax(-1)
        content = self.bigram_embedding(self._bigram_ids(ids, valid_mask))
        pair_weights = weights * torch.cat(
            (weights[:, :, 1:], torch.zeros_like(weights[:, :, :1])), dim=-1,
        )
        pair_weights = pair_weights / pair_weights.sum(-1, keepdim=True).clamp_min(1e-12)
        pooled = torch.einsum("bsl,blf->bsf", pair_weights, content)
        return F.normalize(self.fingerprint_projection(pooled), dim=-1)

    def compile_program(
        self, ids: torch.Tensor, valid_mask: torch.Tensor,
    ) -> BindingBusOutput:
        memory = self._encode(ids, valid_mask)
        line_slots, line_pointer_logits = self._address(
            memory, valid_mask, self.program_queries,
        )
        line_slots = self.slot_norm(self.slot_encoder(line_slots))
        events = line_slots[:, 1:]

        binding_pointer_logits = self._pointer_logits(
            memory, valid_mask, self.binding_queries,
        )
        initial_pointer_logits = self._pointer_logits(
            memory, valid_mask, self.initial_entity_queries,
        )
        event_pointer_logits = self._pointer_logits(
            memory, valid_mask, self.event_entity_queries,
        )
        bindings = self._fingerprints(
            ids, valid_mask, binding_pointer_logits,
        )
        initial_entities = self._fingerprints(
            ids, valid_mask, initial_pointer_logits,
        )
        event_entities = self._fingerprints(
            ids, valid_mask, event_pointer_logits,
        )
        scale = self.logit_scale.exp().clamp(max=100.0)
        initial_matches = scale * torch.einsum(
            "bpf,brf->bpr", initial_entities, bindings,
        )
        event_matches = scale * torch.einsum(
            "bef,brf->ber", event_entities, bindings,
        )
        state_logits = torch.stack([
            sum(initial_matches[:, position, role] for position, role in enumerate(perm))
            for perm in PERMUTATIONS
        ], dim=-1)
        tape = DeletedProgramTape(
            initial_state=state_logits.float(),
            event_kind=self.kind_head(events).float(),
            event_identity=event_matches.float(),
            amount=self.amount_head(events).float(),
        )
        return BindingBusOutput(
            tape=tape,
            line_pointer_logits=line_pointer_logits,
            binding_pointer_logits=binding_pointer_logits,
            initial_entity_pointer_logits=initial_pointer_logits,
            event_entity_pointer_logits=event_pointer_logits,
        )

    def compile_query(self, ids: torch.Tensor, valid_mask: torch.Tensor) -> LateQuery:
        return super().compile_query(ids, valid_mask)


class HierarchicalBindingBusCompiler(BindingBusCompiler):
    """Bind event names only inside each model-addressed semantic event line."""

    @staticmethod
    def _selected_line_mask(
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
        pointer_logits: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)[None, None]
        anchors = pointer_logits.argmax(-1)[..., None]
        newlines = (ids.eq(10) & valid_mask)[:, None]
        before = torch.where(
            newlines & positions.lt(anchors), positions, torch.full_like(positions, -1),
        ).amax(-1, keepdim=True)
        lengths = valid_mask.sum(-1)[:, None, None]
        after = torch.where(
            newlines & positions.ge(anchors), positions + 1, lengths,
        ).amin(-1, keepdim=True)
        return valid_mask[:, None] & positions.ge(before + 1) & positions.lt(after)

    def compile_program(
        self, ids: torch.Tensor, valid_mask: torch.Tensor,
    ) -> BindingBusOutput:
        memory = self._encode(ids, valid_mask)
        line_slots, line_pointer_logits = self._address(
            memory, valid_mask, self.program_queries,
        )
        line_slots = self.slot_norm(self.slot_encoder(line_slots))
        events = line_slots[:, 1:]

        binding_pointer_logits = self._pointer_logits(
            memory, valid_mask, self.binding_queries,
        )
        initial_pointer_logits = self._pointer_logits(
            memory, valid_mask, self.initial_entity_queries,
        )
        event_pointer_logits = self._pointer_logits(
            memory, valid_mask, self.event_entity_queries,
        )
        line_mask = self._selected_line_mask(ids, valid_mask, line_pointer_logits)
        event_pointer_logits = event_pointer_logits.masked_fill(
            ~line_mask[:, 1:], torch.finfo(event_pointer_logits.dtype).min,
        )

        bindings = self._fingerprints(ids, valid_mask, binding_pointer_logits)
        initial_entities = self._fingerprints(
            ids, valid_mask, initial_pointer_logits,
        )
        event_entities = self._fingerprints(
            ids, valid_mask, event_pointer_logits,
        )
        scale = self.logit_scale.exp().clamp(max=100.0)
        initial_matches = scale * torch.einsum(
            "bpf,brf->bpr", initial_entities, bindings,
        )
        event_matches = scale * torch.einsum(
            "bef,brf->ber", event_entities, bindings,
        )
        state_logits = torch.stack([
            sum(initial_matches[:, position, role] for position, role in enumerate(perm))
            for perm in PERMUTATIONS
        ], dim=-1)
        tape = DeletedProgramTape(
            initial_state=state_logits.float(),
            event_kind=self.kind_head(events).float(),
            event_identity=event_matches.float(),
            amount=self.amount_head(events).float(),
        )
        return BindingBusOutput(
            tape=tape,
            line_pointer_logits=line_pointer_logits,
            binding_pointer_logits=binding_pointer_logits,
            initial_entity_pointer_logits=initial_pointer_logits,
            event_entity_pointer_logits=event_pointer_logits,
        )
