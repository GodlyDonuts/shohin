"""Delimiter-indexed physical records for categorical program compilation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sd_cst import DeletedProgramTape
from sd_cst_binding_bus import BindingBusOutput
from sd_cst_byte_addressed import BYTE_PAD, BYTE_VOCAB, PROGRAM_SLOTS
from sd_cst_renderer_native_program import RendererNativeProgramCompiler


class PhysicalRecordBusCompiler(RendererNativeProgramCompiler):
    """Compile local physical records before model-owned categorical writes."""

    def __init__(
        self,
        *,
        record_width: int = 384,
        record_heads: int = 6,
        record_layers: int = 4,
        record_set_layers: int = 2,
        record_ff: int = 1536,
        max_line_bytes: int = 144,
        sinkhorn_steps: int = 8,
        constrained_assignment: bool = True,
        **kwargs: int,
    ) -> None:
        super().__init__(**kwargs)
        if record_width <= 0 or record_heads <= 0:
            raise ValueError("record dimensions must be positive")
        if record_width % record_heads:
            raise ValueError("record width must be divisible by heads")
        if record_layers <= 0 or record_set_layers <= 0 or record_ff <= 0:
            raise ValueError("record depths and feed-forward width must be positive")
        if max_line_bytes <= 0 or max_line_bytes > self.max_bytes:
            raise ValueError("record line window is invalid")
        if sinkhorn_steps <= 0:
            raise ValueError("record Sinkhorn steps must be positive")
        self.record_width = int(record_width)
        self.max_line_bytes = int(max_line_bytes)
        self.sinkhorn_steps = int(sinkhorn_steps)
        self.constrained_assignment = bool(constrained_assignment)

        self.record_byte_embedding = nn.Embedding(
            BYTE_VOCAB,
            record_width,
            padding_idx=BYTE_PAD,
        )
        self.record_position_embedding = nn.Embedding(
            max_line_bytes + 1,
            record_width,
        )
        self.record_query = nn.Parameter(torch.empty(record_width))
        line_layer = nn.TransformerEncoderLayer(
            d_model=record_width,
            nhead=record_heads,
            dim_feedforward=record_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.record_line_encoder = nn.TransformerEncoder(
            line_layer,
            num_layers=record_layers,
            enable_nested_tensor=False,
        )
        set_layer = nn.TransformerEncoderLayer(
            d_model=record_width,
            nhead=record_heads,
            dim_feedforward=record_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.record_set_encoder = nn.TransformerEncoder(
            set_layer,
            num_layers=record_set_layers,
            enable_nested_tensor=False,
        )
        self.record_line_norm = nn.LayerNorm(record_width)
        self.record_set_norm = nn.LayerNorm(record_width)
        self.record_role_head = nn.Linear(record_width, PROGRAM_SLOTS)
        self.record_role_embedding = nn.Parameter(
            torch.empty(PROGRAM_SLOTS, record_width)
        )
        self.record_kind_head = nn.Linear(record_width, 3)
        self.record_amount_head = nn.Linear(record_width, 2)
        self.record_entity_query = nn.Linear(record_width, record_width, bias=False)
        self.record_entity_key = nn.Linear(record_width, record_width, bias=False)
        nn.init.normal_(self.record_query, std=0.02)
        nn.init.normal_(self.record_role_embedding, std=0.02)

    def _physical_line_masks(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_orbit_input(ids, valid_mask)
        newline = ids.eq(10) & valid_mask
        if not bool(newline.sum(-1).eq(PROGRAM_SLOTS - 1).all()):
            raise ValueError("record compiler requires exactly nine physical lines")
        line_index = newline.long().cumsum(-1) - newline.long()
        roles = torch.arange(PROGRAM_SLOTS, device=ids.device)
        masks = line_index[:, None].eq(roles[None, :, None])
        masks = masks & valid_mask[:, None]
        counts = masks.sum(-1)
        if bool(counts.eq(0).any()) or bool(counts.gt(self.max_line_bytes).any()):
            raise ValueError("physical line is empty or exceeds record window")
        return masks

    def _pack_physical_lines(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = self._physical_line_masks(ids, valid_mask)
        width = ids.shape[1]
        positions = torch.arange(width, device=ids.device)
        candidates = torch.where(
            masks,
            positions[None, None],
            torch.full((), width, device=ids.device),
        )
        indices = candidates.topk(
            self.max_line_bytes,
            dim=-1,
            largest=False,
            sorted=True,
        ).values
        local_valid = indices.lt(width)
        safe_indices = indices.clamp_max(width - 1)
        expanded = ids[:, None].expand(-1, PROGRAM_SLOTS, -1)
        local_ids = expanded.gather(-1, safe_indices)
        local_ids = torch.where(
            local_valid,
            local_ids,
            torch.full_like(local_ids, BYTE_PAD),
        )
        return local_ids, local_valid, safe_indices, masks

    def _encode_records(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        local_ids, local_valid, source_indices, line_masks = (
            self._pack_physical_lines(ids, valid_mask)
        )
        batch = ids.shape[0]
        flat_ids = local_ids.reshape(batch * PROGRAM_SLOTS, self.max_line_bytes)
        flat_valid = local_valid.reshape(
            batch * PROGRAM_SLOTS,
            self.max_line_bytes,
        )
        positions = torch.arange(self.max_line_bytes + 1, device=ids.device)
        token_hidden = self.record_byte_embedding(flat_ids)
        token_hidden = token_hidden + self.record_position_embedding(
            positions[1:]
        )[None]
        query = self.record_query[None, None].expand(
            batch * PROGRAM_SLOTS,
            1,
            -1,
        )
        query = query + self.record_position_embedding(positions[:1])[None]
        hidden = torch.cat([query, token_hidden], dim=1)
        hidden_valid = torch.cat(
            [
                torch.ones(
                    batch * PROGRAM_SLOTS,
                    1,
                    dtype=torch.bool,
                    device=ids.device,
                ),
                flat_valid,
            ],
            dim=1,
        )
        hidden = self.record_line_encoder(
            hidden,
            src_key_padding_mask=~hidden_valid,
        )
        hidden = self.record_line_norm(hidden)
        records = hidden[:, 0].reshape(batch, PROGRAM_SLOTS, self.record_width)
        records = self.record_set_norm(self.record_set_encoder(records))
        token_memory = hidden[:, 1:].reshape(
            batch,
            PROGRAM_SLOTS,
            self.max_line_bytes,
            self.record_width,
        )
        return records, token_memory, local_valid, source_indices, line_masks

    def _soft_assignment(self, role_logits: torch.Tensor) -> torch.Tensor:
        if not self.constrained_assignment:
            return role_logits.softmax(1)
        log_assignment = role_logits.float()
        for _ in range(self.sinkhorn_steps):
            log_assignment = log_assignment - log_assignment.logsumexp(
                -1,
                keepdim=True,
            )
            log_assignment = log_assignment - log_assignment.logsumexp(
                -2,
                keepdim=True,
            )
        return log_assignment.exp().to(role_logits.dtype)

    @staticmethod
    def _greedy_one_to_one(role_logits: torch.Tensor) -> torch.Tensor:
        if role_logits.ndim != 3 or role_logits.shape[1:] != (
            PROGRAM_SLOTS,
            PROGRAM_SLOTS,
        ):
            raise ValueError("record role logits have invalid shape")
        batch = role_logits.shape[0]
        remaining = role_logits.float().clone()
        assignment = torch.zeros_like(remaining)
        floor = torch.finfo(remaining.dtype).min
        for _ in range(PROGRAM_SLOTS):
            flat = remaining.reshape(batch, -1).argmax(-1)
            physical = torch.div(flat, PROGRAM_SLOTS, rounding_mode="floor")
            semantic = flat.remainder(PROGRAM_SLOTS)
            assignment[
                torch.arange(batch, device=role_logits.device),
                physical,
                semantic,
            ] = 1.0
            physical_mask = torch.nn.functional.one_hot(
                physical,
                PROGRAM_SLOTS,
            ).bool()
            semantic_mask = torch.nn.functional.one_hot(
                semantic,
                PROGRAM_SLOTS,
            ).bool()
            remaining = remaining.masked_fill(physical_mask[:, :, None], floor)
            remaining = remaining.masked_fill(semantic_mask[:, None, :], floor)
        return assignment.to(role_logits.dtype)

    def _assignment(self, role_logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._soft_assignment(role_logits)
        if self.constrained_assignment:
            return self._greedy_one_to_one(role_logits)
        selected = role_logits.argmax(1)
        return torch.nn.functional.one_hot(
            selected,
            PROGRAM_SLOTS,
        ).transpose(1, 2).to(role_logits.dtype)

    def _global_line_logits(
        self,
        assignment: torch.Tensor,
        line_masks: torch.Tensor,
    ) -> torch.Tensor:
        line_distribution = line_masks.float()
        line_distribution = line_distribution / line_distribution.sum(
            -1,
            keepdim=True,
        )
        probabilities = torch.einsum(
            "bps,bpl->bsl",
            assignment.float(),
            line_distribution,
        )
        return probabilities.clamp_min(1e-30).log()

    def _global_event_logits(
        self,
        records: torch.Tensor,
        token_memory: torch.Tensor,
        local_valid: torch.Tensor,
        source_indices: torch.Tensor,
        assignment: torch.Tensor,
        source_width: int,
    ) -> torch.Tensor:
        queries = self.record_entity_query(records)
        keys = self.record_entity_key(token_memory)
        logits = torch.einsum("bpw,bpkw->bpk", queries, keys)
        logits = logits / math.sqrt(self.record_width)
        logits = logits.masked_fill(
            ~local_valid,
            torch.finfo(logits.dtype).min,
        )
        local_probabilities = logits.float().softmax(-1)
        physical_probabilities = torch.zeros(
            records.shape[0],
            PROGRAM_SLOTS,
            source_width,
            device=records.device,
        )
        physical_probabilities.scatter_add_(
            -1,
            source_indices,
            local_probabilities * local_valid,
        )
        semantic_probabilities = torch.einsum(
            "bpe,bpl->bel",
            assignment[:, :, 1:].float(),
            physical_probabilities,
        )
        return semantic_probabilities.clamp_min(1e-30).log()

    def compile_program(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> BindingBusOutput:
        parent = super().compile_program(ids, valid_mask)
        (
            records,
            token_memory,
            local_valid,
            source_indices,
            line_masks,
        ) = self._encode_records(ids, valid_mask)

        role_logits = self.record_role_head(records)
        assignment = self._assignment(role_logits)
        semantic_records = torch.einsum(
            "bps,bpw->bsw",
            assignment,
            records,
        )
        semantic_records = semantic_records + self.record_role_embedding[None]
        events = semantic_records[:, 1:]
        line_pointer_logits = self._global_line_logits(assignment, line_masks)
        event_pointer_logits = self._global_event_logits(
            records,
            token_memory,
            local_valid,
            source_indices,
            assignment,
            ids.shape[1],
        )
        bindings = self._fingerprints(
            ids,
            valid_mask,
            parent.binding_pointer_logits,
        )
        event_entities = self._fingerprints(
            ids,
            valid_mask,
            event_pointer_logits,
        )
        event_matches = self.logit_scale.exp().clamp(max=100.0) * torch.einsum(
            "bef,brf->ber",
            event_entities,
            bindings,
        )
        return BindingBusOutput(
            tape=DeletedProgramTape(
                initial_state=parent.tape.initial_state,
                event_kind=self.record_kind_head(events).float(),
                event_identity=event_matches.float(),
                amount=self.record_amount_head(events).float(),
            ),
            line_pointer_logits=line_pointer_logits,
            binding_pointer_logits=parent.binding_pointer_logits,
            initial_entity_pointer_logits=parent.initial_entity_pointer_logits,
            event_entity_pointer_logits=event_pointer_logits,
        )


def physical_record_trainable_names(
    model: PhysicalRecordBusCompiler,
) -> frozenset[str]:
    return frozenset(
        name for name, _ in model.named_parameters() if name.startswith("record_")
    )


def freeze_to_physical_record_bus(
    model: PhysicalRecordBusCompiler,
) -> tuple[str, ...]:
    declared = physical_record_trainable_names(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in declared)
    actual = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if actual != declared:
        raise ValueError("physical-record trainable parameter contract mismatch")
    return tuple(sorted(actual))
