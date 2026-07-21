"""Occurrence-addressed routing for marginal identity transport."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from er_dual_stream_relation_adapter import DualStreamRelationCompiler
from er_relation_tensor_adapter import MAX_CARDINALITY, TT_RECORDS


MAX_RECORD_SYMBOLS = 1 + 2 * MAX_CARDINALITY


class AddressedMarginalRelationCompiler(DualStreamRelationCompiler):
    """Factor opaque identity from learned ordinal and record-size addresses."""

    def __init__(self, **kwargs: int) -> None:
        super().__init__(**kwargs)
        address_count = MAX_RECORD_SYMBOLS + 1
        self.er_am_candidate_ordinal_embedding = nn.Embedding(
            address_count, self.record_width
        )
        self.er_am_candidate_count_embedding = nn.Embedding(
            address_count, self.record_width
        )
        nn.init.normal_(self.er_am_candidate_ordinal_embedding.weight, std=0.02)
        nn.init.normal_(self.er_am_candidate_count_embedding.weight, std=0.02)

    @staticmethod
    def candidate_addresses(
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return within-record ordinal and record symbol count for each token."""
        if candidates.ndim != 3 or candidates.dtype is not torch.bool:
            raise ValueError("addressed marginal candidates differ")
        count = candidates.sum(-1).clamp(min=1, max=MAX_RECORD_SYMBOLS).long()
        ordinal = (candidates.long().cumsum(-1) - 1).clamp(
            min=0, max=MAX_RECORD_SYMBOLS
        )
        return ordinal, count

    def _routed_symbol_logits(
        self,
        records: torch.Tensor,
        token_memory: torch.Tensor,
        local_valid: torch.Tensor,
        source_indices: torch.Tensor,
        starts: torch.Tensor,
        assignment: torch.Tensor,
        semantic_roles: slice,
        queries: torch.Tensor,
        source_width: int,
    ) -> torch.Tensor:
        """Route over an identity-free occurrence-address channel."""
        records = records.detach()
        token_memory = token_memory.detach()
        candidates = self._local_candidates(starts, source_indices, local_valid)
        ordinal, count = self.candidate_addresses(candidates)
        count_address = self.er_am_candidate_count_embedding(count)
        key_address = self.er_am_candidate_ordinal_embedding(ordinal)
        key_address = key_address + count_address[:, :, None]

        query = self.er_ds_router_norm(records + count_address)[:, :, None]
        query = self.er_ds_router_query(query + queries[None, None])
        keys = self.er_ds_router_key(token_memory + key_address)
        local_logits = torch.einsum("bpow,bpkw->bpok", query, keys)
        local_logits = local_logits / math.sqrt(self.record_width)
        local_logits = local_logits.masked_fill(
            ~candidates[:, :, None], torch.finfo(local_logits.dtype).min
        )
        local_probabilities = local_logits.float().softmax(-1)
        physical = torch.zeros(
            records.shape[0],
            TT_RECORDS,
            queries.shape[0],
            source_width,
            device=records.device,
        )
        physical.scatter_add_(
            -1,
            source_indices[:, :, None].expand(-1, -1, queries.shape[0], -1),
            local_probabilities * local_valid[:, :, None],
        )
        semantic_assignment = assignment[:, :, semantic_roles].float()
        probabilities = torch.einsum(
            "bpr,bpol->brol", semantic_assignment, physical
        )
        return probabilities.clamp_min(1e-30).log()
