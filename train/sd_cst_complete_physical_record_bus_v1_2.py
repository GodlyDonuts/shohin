"""Nonlinear local occurrence head for the complete physical-record compiler."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sd_cst_complete_physical_record_bus import (
    CompletePhysicalRecordBusCompiler,
    _freeze_to_declared,
)


class CompletePhysicalRecordBusCompilerV1_2(CompletePhysicalRecordBusCompiler):
    """Classify six declaration occurrence roles at every local byte."""

    def __init__(self, *, occurrence_ff: int = 1536, **kwargs: int) -> None:
        super().__init__(**kwargs)
        if occurrence_ff <= 0:
            raise ValueError("occurrence feed-forward width must be positive")
        self.occurrence_ff = int(occurrence_ff)
        self.local_occurrence_norm = nn.LayerNorm(self.record_width)
        self.local_occurrence_hidden = nn.Linear(
            self.record_width,
            self.occurrence_ff,
        )
        self.local_occurrence_head = nn.Linear(self.occurrence_ff, 6)

    def _global_declaration_logits(
        self,
        records: torch.Tensor,
        token_memory: torch.Tensor,
        local_valid: torch.Tensor,
        source_indices: torch.Tensor,
        assignment: torch.Tensor,
        source_width: int,
    ) -> torch.Tensor:
        del records
        hidden = self.local_occurrence_norm(token_memory)
        hidden = F.gelu(self.local_occurrence_hidden(hidden))
        local_logits = self.local_occurrence_head(hidden).permute(0, 1, 3, 2)
        local_logits = local_logits.masked_fill(
            ~local_valid[:, :, None],
            torch.finfo(local_logits.dtype).min,
        )
        local_probabilities = local_logits.float().softmax(-1)
        physical_probabilities = torch.zeros(
            token_memory.shape[0],
            token_memory.shape[1],
            6,
            source_width,
            device=token_memory.device,
        )
        physical_probabilities.scatter_add_(
            -1,
            source_indices[:, :, None].expand(-1, -1, 6, -1),
            local_probabilities * local_valid[:, :, None],
        )
        declaration_assignment = assignment[:, :, 0].float()
        semantic_probabilities = torch.einsum(
            "bp,bpsl->bsl",
            declaration_assignment,
            physical_probabilities,
        )
        return semantic_probabilities.clamp_min(1e-30).log()


def occurrence_head_trainable_names(
    model: CompletePhysicalRecordBusCompilerV1_2,
) -> frozenset[str]:
    names = frozenset(
        name for name, _ in model.named_parameters() if name.startswith("local_occurrence_")
    )
    if len(names) != 6:
        raise ValueError("occurrence-head parameter contract differs")
    return names


def freeze_to_occurrence_head(
    model: CompletePhysicalRecordBusCompilerV1_2,
) -> tuple[str, ...]:
    return _freeze_to_declared(model, occurrence_head_trainable_names(model))
