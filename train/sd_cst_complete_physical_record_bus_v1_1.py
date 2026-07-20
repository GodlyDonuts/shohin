"""Declaration-key repair for the complete SD-CST physical-record compiler."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sd_cst_complete_physical_record_bus import (
    CompletePhysicalRecordBusCompiler,
    _freeze_to_declared,
)


class CompletePhysicalRecordBusCompilerV1_1(CompletePhysicalRecordBusCompiler):
    """Give declaration occurrences their own trainable address geometry."""

    def __init__(self, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self.local_declaration_key_projection = nn.Linear(
            self.record_width,
            self.record_width,
            bias=False,
        )

    def _global_declaration_logits(
        self,
        records: torch.Tensor,
        token_memory: torch.Tensor,
        local_valid: torch.Tensor,
        source_indices: torch.Tensor,
        assignment: torch.Tensor,
        source_width: int,
    ) -> torch.Tensor:
        queries = records[:, :, None] + self.local_declaration_queries[None, None]
        queries = self.local_declaration_query_projection(queries)
        keys = self.local_declaration_key_projection(token_memory)
        logits = torch.einsum("bpsw,bpkw->bpsk", queries, keys)
        logits = logits / math.sqrt(self.record_width)
        logits = logits.masked_fill(
            ~local_valid[:, :, None],
            torch.finfo(logits.dtype).min,
        )
        local_probabilities = logits.float().softmax(-1)
        physical_probabilities = torch.zeros(
            records.shape[0],
            token_memory.shape[1],
            self.local_declaration_queries.shape[0],
            source_width,
            device=records.device,
        )
        physical_probabilities.scatter_add_(
            -1,
            source_indices[:, :, None].expand(
                -1,
                -1,
                self.local_declaration_queries.shape[0],
                -1,
            ),
            local_probabilities * local_valid[:, :, None],
        )
        declaration_assignment = assignment[:, :, 0].float()
        semantic_probabilities = torch.einsum(
            "bp,bpsl->bsl",
            declaration_assignment,
            physical_probabilities,
        )
        return semantic_probabilities.clamp_min(1e-30).log()


def declaration_repair_trainable_names(
    model: CompletePhysicalRecordBusCompilerV1_1,
) -> frozenset[str]:
    expected = frozenset(
        {
            "local_declaration_queries",
            "local_declaration_query_projection.weight",
            "local_declaration_key_projection.weight",
        }
    )
    actual = frozenset(name for name, _ in model.named_parameters() if name in expected)
    if actual != expected:
        raise ValueError("declaration-repair parameter contract differs")
    return actual


def freeze_to_declaration_repair(
    model: CompletePhysicalRecordBusCompilerV1_1,
) -> tuple[str, ...]:
    return _freeze_to_declared(model, declaration_repair_trainable_names(model))
