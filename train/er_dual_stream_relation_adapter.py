"""Alpha-invariant routing plus whole-symbol identity for relation transport."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from er_relation_tensor_adapter import (
    DECLARATION_OCCURRENCES,
    MAX_CARDINALITY,
    MAX_RULES,
    SHOHIN_BASE_PARAMETERS,
    STRICT_PARAMETER_CAP,
    TT_RECORDS,
    EpisodicRelationTensorCompiler,
    RelationTensorCompilerOutput,
    RelationTensorProgram,
    RelationTensorQuery,
)


OPAQUE_SYMBOL_BYTES = 6
OPAQUE_CANONICAL_BYTE = ord("x")


class DualStreamRelationCompiler(EpisodicRelationTensorCompiler):
    """Keep syntactic routing alpha-invariant and identity transport separate."""

    def __init__(self, **kwargs: int) -> None:
        super().__init__(**kwargs)
        width = self.record_width

        del self.er_tt_occurrence_head
        del self.er_tt_witness_side_embedding
        del self.er_tt_witness_position_embedding
        del self.er_event_card_query_projection
        del self.er_rule_card_key_projection
        del self.er_equality_projection
        del self.er_equality_scale
        del self.er_witness_norm
        del self.er_witness_query_projection
        del self.er_witness_key_projection
        del self.local_occurrence_norm
        del self.local_occurrence_hidden
        del self.bigram_embedding
        del self.fingerprint_projection
        del self.logit_scale

        self.er_ds_router_norm = nn.LayerNorm(width)
        self.er_ds_router_query = nn.Linear(width, width, bias=False)
        self.er_ds_router_key = nn.Linear(width, width, bias=False)
        self.er_ds_declaration_queries = nn.Parameter(
            torch.empty(DECLARATION_OCCURRENCES, width)
        )
        self.er_ds_witness_queries = nn.Parameter(
            torch.empty(DECLARATION_OCCURRENCES, width)
        )
        self.er_ds_rule_opcode_query = nn.Parameter(torch.empty(1, width))
        self.er_ds_event_opcode_query = nn.Parameter(torch.empty(1, width))
        for parameter in (
            self.er_ds_declaration_queries,
            self.er_ds_witness_queries,
            self.er_ds_rule_opcode_query,
            self.er_ds_event_opcode_query,
        ):
            nn.init.normal_(parameter, std=0.02)

    @staticmethod
    def opaque_symbol_starts(
        ids: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Find six-byte lowercase/base36 whitespace tokens without reading identity."""
        if ids.shape != valid_mask.shape or ids.ndim != 2:
            raise ValueError("dual-stream source shape differs")
        batch, width = ids.shape
        whitespace = ids.eq(32) | ids.eq(10)
        previous_boundary = torch.ones(
            batch, width, dtype=torch.bool, device=ids.device
        )
        previous_boundary[:, 1:] = whitespace[:, :-1] | ~valid_mask[:, :-1]
        starts = valid_mask & previous_boundary & ~whitespace
        exact = starts.clone()
        for offset in range(OPAQUE_SYMBOL_BYTES):
            present = torch.zeros_like(starts)
            if offset < width:
                value = ids[:, offset:]
                value_valid = valid_mask[:, offset:]
                alnum = value.ge(ord("0")) & value.le(ord("9"))
                alnum |= value.ge(ord("a")) & value.le(ord("z"))
                present[:, : width - offset] = value_valid & alnum
            exact &= present
        boundary = torch.ones_like(starts)
        if OPAQUE_SYMBOL_BYTES < width:
            trailing = ids[:, OPAQUE_SYMBOL_BYTES:]
            trailing_valid = valid_mask[:, OPAQUE_SYMBOL_BYTES:]
            boundary[:, : width - OPAQUE_SYMBOL_BYTES] = (
                ~trailing_valid | trailing.eq(32) | trailing.eq(10)
            )
        return exact & boundary

    @classmethod
    def structural_view(
        cls, ids: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonicalize opaque payloads while preserving every byte position."""
        starts = cls.opaque_symbol_starts(ids, valid_mask)
        membership = torch.zeros_like(starts)
        width = ids.shape[1]
        for offset in range(OPAQUE_SYMBOL_BYTES):
            membership[:, offset:] |= starts[:, : width - offset]
        structural = torch.where(
            membership,
            torch.full_like(ids, OPAQUE_CANONICAL_BYTE),
            ids,
        )
        return structural, starts

    @staticmethod
    def _local_candidates(
        starts: torch.Tensor,
        source_indices: torch.Tensor,
        local_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, records, width = source_indices.shape
        gathered = starts[:, None].expand(-1, records, -1).gather(
            -1, source_indices.clamp_max(starts.shape[1] - 1)
        )
        candidates = gathered & local_valid
        missing = ~candidates.any(-1, keepdim=True)
        fallback = torch.zeros_like(candidates)
        fallback[..., 0] = local_valid[..., 0]
        return candidates | (missing & fallback)

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
        """Route semantic slots to opaque candidates using structural memory only."""
        records = records.detach()
        token_memory = token_memory.detach()
        candidates = self._local_candidates(starts, source_indices, local_valid)
        query = self.er_ds_router_norm(records)[:, :, None] + queries[None, None]
        query = self.er_ds_router_query(query)
        keys = self.er_ds_router_key(token_memory)
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
        probabilities = torch.einsum("bpr,bpol->brol", semantic_assignment, physical)
        return probabilities.clamp_min(1e-30).log()

    @staticmethod
    def _all_symbol_bytes(ids: torch.Tensor) -> torch.Tensor:
        """Materialize one six-byte candidate at every source position."""
        if ids.ndim != 2:
            raise ValueError("dual-stream source ids must be rank two")
        batch, width = ids.shape
        positions = torch.arange(width, device=ids.device)
        offsets = torch.arange(OPAQUE_SYMBOL_BYTES, device=ids.device)
        indices = (positions[:, None] + offsets[None]).clamp_max(width - 1)
        return ids[:, indices]

    def _marginal_identity_equality(
        self,
        ids: torch.Tensor,
        starts: torch.Tensor,
        left_logits: torch.Tensor,
        right_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Marginalize exact symbol equality over all model-owned routes."""
        if left_logits.shape[0] != ids.shape[0] or right_logits.shape[0] != ids.shape[0]:
            raise ValueError("dual-stream marginal equality batch differs")
        if left_logits.shape[-1] != ids.shape[1] or right_logits.shape[-1] != ids.shape[1]:
            raise ValueError("dual-stream marginal equality source width differs")
        negative = torch.finfo(left_logits.dtype).min
        left = left_logits.masked_fill(~starts[(slice(None),) + (None,) * (left_logits.ndim - 2)], negative)
        right = right_logits.masked_fill(~starts[(slice(None),) + (None,) * (right_logits.ndim - 2)], negative)
        left_probability = left.float().softmax(-1)
        right_probability = right.float().softmax(-1)
        symbols = self._all_symbol_bytes(ids)
        exact = symbols[:, :, None].eq(symbols[:, None]).all(-1)
        exact &= starts[:, :, None] & starts[:, None, :]
        equality = exact.to(left_probability.dtype)
        if left_logits.ndim == 3 and right_logits.ndim == 3:
            probability = torch.einsum(
                "bil,blm,bjm->bij",
                left_probability,
                equality,
                right_probability,
            )
        elif (
            left_logits.ndim == 4
            and right_logits.ndim == 4
            and left_logits.shape[1] == right_logits.shape[1]
        ):
            probability = torch.einsum(
                "bgil,blm,bgjm->bgij",
                left_probability,
                equality,
                right_probability,
            )
        else:
            raise ValueError("dual-stream marginal equality leading dimensions differ")
        return probability.clamp_min(1e-30).log()

    def compile_relation_program(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
        query_ids: torch.Tensor,
        query_valid_mask: torch.Tensor,
    ) -> RelationTensorCompilerOutput:
        structural_ids, starts = self.structural_view(ids, valid_mask)
        records, token_memory, local_valid, source_indices, line_masks = (
            self._er_encode_records(structural_ids, valid_mask)
        )
        semantic_role_logits = self.er_tt_record_role_head(records)
        semantic_assignment = self._er_assignment(semantic_role_logits)
        # Routing supervision updates role assignment, but cannot disturb the
        # shared structural encoder. Both assignments are numerically identical.
        routing_role_logits = self.er_tt_record_role_head(records.detach())
        routing_assignment = self._er_assignment(routing_role_logits)
        semantic_records = torch.einsum(
            "bps,bpw->bsw", semantic_assignment, records
        )
        semantic_records = semantic_records + self.er_tt_record_role_embedding[None]
        declaration = semantic_records[:, 0]
        rules = self.er_rule_norm(semantic_records[:, 1 : 1 + MAX_RULES])
        events = self.er_event_norm(semantic_records[:, 1 + MAX_RULES :])

        declaration_logits = self._routed_symbol_logits(
            records,
            token_memory,
            local_valid,
            source_indices,
            starts,
            routing_assignment,
            slice(0, 1),
            self.er_ds_declaration_queries,
            ids.shape[1],
        )[:, 0]
        binding_pointer_logits = declaration_logits[:, :MAX_CARDINALITY]
        initial_pointer_logits = declaration_logits[:, MAX_CARDINALITY:]
        initial_equality = self._marginal_identity_equality(
            ids,
            starts,
            initial_pointer_logits,
            binding_pointer_logits,
        )

        witness_pointer_logits = self._routed_symbol_logits(
            records,
            token_memory,
            local_valid,
            source_indices,
            starts,
            routing_assignment,
            slice(1, 1 + MAX_RULES),
            self.er_ds_witness_queries,
            ids.shape[1],
        )
        relation_equality = self._marginal_identity_equality(
            ids,
            starts,
            witness_pointer_logits[..., MAX_CARDINALITY:, :],
            witness_pointer_logits[..., :MAX_CARDINALITY, :],
        )

        rule_opcode_logits = self._routed_symbol_logits(
            records,
            token_memory,
            local_valid,
            source_indices,
            starts,
            routing_assignment,
            slice(1, 1 + MAX_RULES),
            self.er_ds_rule_opcode_query,
            ids.shape[1],
        ).squeeze(2)
        event_opcode_logits = self._routed_symbol_logits(
            records,
            token_memory,
            local_valid,
            source_indices,
            starts,
            routing_assignment,
            slice(1 + MAX_RULES, TT_RECORDS),
            self.er_ds_event_opcode_query,
            ids.shape[1],
        ).squeeze(2)
        event_card_logits = self._marginal_identity_equality(
            ids,
            starts,
            event_opcode_logits,
            rule_opcode_logits,
        )

        line_distribution = line_masks.float()
        line_distribution = line_distribution / line_distribution.sum(-1, keepdim=True)
        line_pointer_logits = torch.einsum(
            "bps,bpl->bsl", routing_assignment.float(), line_distribution
        ).clamp_min(1e-30).log()
        query: RelationTensorQuery = self.compile_relation_query(
            query_ids, query_valid_mask
        )
        return RelationTensorCompilerOutput(
            program=RelationTensorProgram(
                cardinality=self.er_tt_cardinality_head(declaration).float(),
                initial_state=initial_equality.float(),
                rule_cards=relation_equality.float(),
                rule_active=self.er_tt_rule_active_head(rules).float(),
                event_card=event_card_logits.float(),
                event_halt=self.er_event_halt_head(events).float(),
            ),
            query=query,
            line_pointer_logits=line_pointer_logits,
            binding_pointer_logits=binding_pointer_logits,
            initial_entity_pointer_logits=initial_pointer_logits,
            witness_pointer_logits=witness_pointer_logits,
            initial_equality_logits=initial_equality.float(),
            relation_equality_logits=relation_equality.float(),
        )


def dual_stream_trainable_names(
    model: DualStreamRelationCompiler,
) -> frozenset[str]:
    forbidden = (
        "er_tt_occurrence_head",
        "er_tt_witness_side_embedding",
        "er_tt_witness_position_embedding",
        "er_event_card_query_projection",
        "er_rule_card_key_projection",
    )
    if any(hasattr(model, name) for name in forbidden):
        raise ValueError("dual-stream compiler retains a removed v1 path")
    er_names = frozenset(
        name for name, _ in model.named_parameters() if name.startswith("er_")
    )
    required = {
        "er_ds_declaration_queries",
        "er_ds_witness_queries",
        "er_ds_rule_opcode_query",
        "er_ds_event_opcode_query",
        "er_ds_router_norm.weight",
        "er_ds_router_query.weight",
        "er_ds_router_key.weight",
    }
    if not required.issubset(er_names):
        raise ValueError("dual-stream parameter contract differs")
    shared_prefixes = (
        "record_byte_embedding.",
        "record_position_embedding.",
        "record_query",
        "record_line_encoder.",
        "record_line_norm.",
        "record_set_encoder.",
        "record_set_norm.",
    )
    return er_names | frozenset(
        name
        for name, _ in model.named_parameters()
        if name.startswith(shared_prefixes)
    )


def freeze_to_dual_stream(
    model: DualStreamRelationCompiler,
) -> tuple[str, ...]:
    declared = dual_stream_trainable_names(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in declared)
    actual = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if actual != declared:
        raise ValueError("dual-stream trainable parameter contract differs")
    return tuple(sorted(actual))


def dual_stream_parameter_report(
    model: DualStreamRelationCompiler,
    *, base_parameters: int = SHOHIN_BASE_PARAMETERS,
) -> dict[str, int]:
    compiler = int(model.parameter_count())
    complete = int(base_parameters + compiler)
    report = {
        "base": int(base_parameters),
        "compiler": compiler,
        "motor": 0,
        "reader": 0,
        "complete_system": complete,
        "headroom_below_200m": STRICT_PARAMETER_CAP - complete,
        "trainable": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    if complete >= STRICT_PARAMETER_CAP or report["headroom_below_200m"] <= 0:
        raise ValueError("dual-stream complete system reaches 200M")
    return report
