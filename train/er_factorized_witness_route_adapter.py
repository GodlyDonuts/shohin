"""Factorized residual address bias for ER-TT witness routing."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from er_dual_stream_relation_adapter import DualStreamRelationCompiler
from er_relation_tensor_adapter import (
    DECLARATION_OCCURRENCES,
    MAX_CARDINALITY,
    MAX_RULES,
    TT_RECORDS,
)


MAX_RECORD_SYMBOLS = 1 + 2 * MAX_CARDINALITY


class FactorizedWitnessRouteCompiler(DualStreamRelationCompiler):
    """Add a learned count/role/ordinal residual only to witness routes."""

    def __init__(self, **kwargs: int) -> None:
        super().__init__(**kwargs)
        address_count = MAX_RECORD_SYMBOLS + 1
        self.er_fw_witness_address_bias = nn.Parameter(
            torch.empty(
                address_count,
                DECLARATION_OCCURRENCES,
                address_count,
            )
        )
        self.er_fw_witness_gate = nn.Parameter(torch.zeros(DECLARATION_OCCURRENCES))
        nn.init.normal_(self.er_fw_witness_address_bias, std=0.02)
        self.er_fw_route_mode = "treatment"

    @staticmethod
    def candidate_addresses(
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidates.ndim != 3 or candidates.dtype is not torch.bool:
            raise ValueError("factorized witness candidates differ")
        count = candidates.sum(-1).clamp(min=1, max=MAX_RECORD_SYMBOLS).long()
        ordinal = (candidates.long().cumsum(-1) - 1).clamp(
            min=0, max=MAX_RECORD_SYMBOLS
        )
        return ordinal, count

    @staticmethod
    def _is_witness_route(semantic_roles: slice, query_count: int) -> bool:
        return (
            semantic_roles.start == 1
            and semantic_roles.stop == 1 + MAX_RULES
            and semantic_roles.step is None
            and query_count == DECLARATION_OCCURRENCES
        )

    def set_route_mode(self, mode: str) -> None:
        if mode not in {
            "treatment",
            "baseline",
            "structural_only",
            "shuffled_address",
        }:
            raise ValueError(f"unknown factorized witness route mode: {mode}")
        self.er_fw_route_mode = mode

    def _address_residual(
        self,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        ordinal, count = self.candidate_addresses(candidates)
        if self.er_fw_route_mode == "shuffled_address":
            physical = torch.arange(candidates.shape[1], device=candidates.device)[
                None, :, None
            ]
            ordinal = (ordinal + physical) % count[:, :, None]
        count_table = self.er_fw_witness_address_bias[count]
        raw = count_table.gather(
            -1,
            ordinal[:, :, None].expand(-1, -1, DECLARATION_OCCURRENCES, -1),
        ).tanh()
        valid = candidates[:, :, None].to(raw.dtype)
        mean = (raw * valid).sum(-1, keepdim=True)
        mean = mean / valid.sum(-1, keepdim=True).clamp_min(1.0)
        centered = (raw - mean) * valid
        gate = 4.0 * self.er_fw_witness_gate.tanh()
        return centered * gate[None, None, :, None]

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
        """Preserve structural logits and add a witness-only address residual."""
        records = records.detach()
        token_memory = token_memory.detach()
        candidates = self._local_candidates(starts, source_indices, local_valid)
        query = self.er_ds_router_norm(records)[:, :, None] + queries[None, None]
        query = self.er_ds_router_query(query)
        keys = self.er_ds_router_key(token_memory)
        local_logits = torch.einsum("bpow,bpkw->bpok", query, keys)
        local_logits = local_logits / math.sqrt(self.record_width)

        if self._is_witness_route(semantic_roles, queries.shape[0]):
            if self.er_fw_route_mode == "structural_only":
                local_logits = torch.zeros_like(local_logits)
            if self.er_fw_route_mode != "baseline":
                local_logits = local_logits + self._address_residual(candidates)

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
        probabilities = torch.einsum("bpr,bpol->brol", semantic_assignment, physical)
        return probabilities.clamp_min(1e-30).log()
