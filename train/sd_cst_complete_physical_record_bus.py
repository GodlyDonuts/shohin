"""Complete local front end for the SD-CST physical-record compiler."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sd_cst import DeletedProgramTape, LateQuery
from sd_cst_binding_bus import BindingBusOutput, PERMUTATIONS
from sd_cst_byte_addressed import BYTE_PAD
from sd_cst_physical_record_bus import (
    PhysicalRecordBusCompiler,
    physical_record_trainable_names,
)
from sd_cst_renderer_orbit_frontend import OrbitLateQueryOutput


class CompletePhysicalRecordBusCompiler(PhysicalRecordBusCompiler):
    """Compile every source-facing field through bounded local evidence."""

    def __init__(self, **kwargs: int) -> None:
        kwargs.setdefault("constrained_assignment", False)
        super().__init__(**kwargs)
        self.local_declaration_queries = nn.Parameter(
            torch.empty(6, self.record_width)
        )
        self.local_declaration_query_projection = nn.Linear(
            self.record_width,
            self.record_width,
            bias=False,
        )
        self.local_query_selector = nn.Parameter(torch.empty(self.record_width))
        self.local_query_query_projection = nn.Linear(
            self.record_width,
            self.record_width,
            bias=False,
        )
        self.local_query_key_projection = nn.Linear(
            self.record_width,
            self.record_width,
            bias=False,
        )
        self.local_query_value_projection = nn.Linear(
            self.record_width,
            self.record_width,
            bias=False,
        )
        self.local_query_norm = nn.LayerNorm(self.record_width)
        self.local_query_head = nn.Linear(self.record_width, 3)
        nn.init.normal_(self.local_declaration_queries, std=0.02)
        nn.init.normal_(self.local_query_selector, std=0.02)

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
        keys = self.record_entity_key(token_memory)
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

    def _encode_query_record(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_orbit_input(ids, valid_mask)
        if ids.shape[1] > self.max_line_bytes:
            raise ValueError("local query exceeds record window")
        batch, width = ids.shape
        padded_ids = torch.full(
            (batch, self.max_line_bytes),
            BYTE_PAD,
            dtype=torch.long,
            device=ids.device,
        )
        padded_valid = torch.zeros(
            (batch, self.max_line_bytes),
            dtype=torch.bool,
            device=ids.device,
        )
        padded_ids[:, :width] = ids
        padded_valid[:, :width] = valid_mask
        positions = torch.arange(self.max_line_bytes + 1, device=ids.device)
        token_hidden = self.record_byte_embedding(padded_ids)
        token_hidden = token_hidden + self.record_position_embedding(
            positions[1:]
        )[None]
        query = self.record_query[None, None].expand(batch, 1, -1)
        query = query + self.record_position_embedding(positions[:1])[None]
        hidden = torch.cat([query, token_hidden], dim=1)
        hidden_valid = torch.cat(
            [
                torch.ones(batch, 1, dtype=torch.bool, device=ids.device),
                padded_valid,
            ],
            dim=1,
        )
        hidden = self.record_line_encoder(
            hidden,
            src_key_padding_mask=~hidden_valid,
        )
        hidden = self.record_line_norm(hidden)
        return hidden[:, 0], hidden[:, 1 : width + 1]

    def compile_query_with_evidence(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> OrbitLateQueryOutput:
        _, memory = self._encode_query_record(ids, valid_mask)
        query = self.local_query_query_projection(self.local_query_selector)
        keys = self.local_query_key_projection(memory)
        pointer_logits = torch.einsum("w,blw->bl", query, keys)
        pointer_logits = pointer_logits / math.sqrt(self.record_width)
        pointer_logits = pointer_logits.masked_fill(
            ~valid_mask,
            torch.finfo(pointer_logits.dtype).min,
        ).float()
        weights = pointer_logits.softmax(-1).to(memory.dtype)
        values = self.local_query_value_projection(self.record_byte_embedding(ids))
        selected = torch.einsum("bl,blw->bw", weights, values)
        selected = self.local_query_norm(selected)
        return OrbitLateQueryOutput(
            query=LateQuery(self.local_query_head(selected).float()),
            pointer_logits=pointer_logits,
        )

    def compile_query(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> LateQuery:
        return self.compile_query_with_evidence(ids, valid_mask).query

    def compile_program(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> BindingBusOutput:
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
        declaration_logits = self._global_declaration_logits(
            records,
            token_memory,
            local_valid,
            source_indices,
            assignment,
            ids.shape[1],
        )
        binding_pointer_logits = declaration_logits[:, :3]
        initial_pointer_logits = declaration_logits[:, 3:]
        bindings = self._fingerprints(ids, valid_mask, binding_pointer_logits)
        initial_entities = self._fingerprints(
            ids,
            valid_mask,
            initial_pointer_logits,
        )
        event_entities = self._fingerprints(
            ids,
            valid_mask,
            event_pointer_logits,
        )
        scale = self.logit_scale.exp().clamp(max=100.0)
        initial_matches = scale * torch.einsum(
            "bpf,brf->bpr",
            initial_entities,
            bindings,
        )
        event_matches = scale * torch.einsum(
            "bef,brf->ber",
            event_entities,
            bindings,
        )
        state_logits = torch.stack(
            [
                sum(
                    initial_matches[:, position, role]
                    for position, role in enumerate(permutation)
                )
                for permutation in PERMUTATIONS
            ],
            dim=-1,
        )
        return BindingBusOutput(
            tape=DeletedProgramTape(
                initial_state=state_logits.float(),
                event_kind=self.record_kind_head(events).float(),
                event_identity=event_matches.float(),
                amount=self.record_amount_head(events).float(),
            ),
            line_pointer_logits=line_pointer_logits,
            binding_pointer_logits=binding_pointer_logits,
            initial_entity_pointer_logits=initial_pointer_logits,
            event_entity_pointer_logits=event_pointer_logits,
        )


def local_completion_trainable_names(
    model: CompletePhysicalRecordBusCompiler,
) -> frozenset[str]:
    return frozenset(
        name for name, _ in model.named_parameters() if name.startswith("local_")
    )


def complete_record_trainable_names(
    model: CompletePhysicalRecordBusCompiler,
) -> frozenset[str]:
    return physical_record_trainable_names(model) | local_completion_trainable_names(
        model
    )


def _freeze_to_declared(
    model: CompletePhysicalRecordBusCompiler,
    declared: frozenset[str],
) -> tuple[str, ...]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in declared)
    actual = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if actual != declared:
        raise ValueError("complete physical-record trainable contract mismatch")
    return tuple(sorted(actual))


def freeze_to_local_completion(
    model: CompletePhysicalRecordBusCompiler,
) -> tuple[str, ...]:
    return _freeze_to_declared(model, local_completion_trainable_names(model))


def freeze_to_complete_record_bus(
    model: CompletePhysicalRecordBusCompiler,
) -> tuple[str, ...]:
    return _freeze_to_declared(model, complete_record_trainable_names(model))
