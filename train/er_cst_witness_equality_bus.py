"""Dedicated witness-equality compilation for ER-CST v1.1."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from er_cst_rule_card_adapter import (
    ER_RECORDS,
    RULE_COUNT,
    EpisodicRuleCardCompiler,
    RuleCardCompilerOutput,
    RuleCardProgram,
)
from sd_cst_binding_bus import PERMUTATIONS


WITNESS_POSITIONS = 6


@dataclass(frozen=True, slots=True)
class WitnessEqualityCompilerOutput(RuleCardCompilerOutput):
    witness_pointer_logits: torch.Tensor
    equality_logits: torch.Tensor


class WitnessEqualityBusCompiler(EpisodicRuleCardCompiler):
    """Infer episodic permutation cards from learned witness equality."""

    def __init__(self, *, equality_width: int | None = None, **kwargs: int) -> None:
        super().__init__(**kwargs)
        del self.er_rule_permutation_head
        width = self.record_width
        equality_width = self.fingerprint_width if equality_width is None else equality_width
        if equality_width <= 0:
            raise ValueError("ER-CST equality width must be positive")
        self.equality_width = int(equality_width)
        self.er_witness_queries = nn.Parameter(torch.empty(WITNESS_POSITIONS, width))
        self.er_witness_norm = nn.LayerNorm(width)
        self.er_witness_query_projection = nn.Linear(width, width, bias=False)
        self.er_witness_key_projection = nn.Linear(width, width, bias=False)
        self.er_equality_projection = nn.Linear(
            self.fingerprint_width,
            self.equality_width,
            bias=False,
        )
        self.er_equality_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        nn.init.normal_(self.er_witness_queries, std=0.02)

    def _global_witness_logits(
        self,
        records: torch.Tensor,
        token_memory: torch.Tensor,
        local_valid: torch.Tensor,
        source_indices: torch.Tensor,
        assignment: torch.Tensor,
        source_width: int,
    ) -> torch.Tensor:
        """Return model-owned pointers for before0..2 and after0..2 per rule."""
        # Card supervision must not rewrite the already-successful declaration path.
        records = records.detach()
        token_memory = token_memory.detach()
        assignment = assignment.detach()
        queries = self.er_witness_norm(records)[:, :, None]
        queries = queries + self.er_witness_queries[None, None]
        queries = self.er_witness_query_projection(queries)
        keys = self.er_witness_key_projection(token_memory)
        local_logits = torch.einsum("bpow,bpkw->bpok", queries, keys)
        local_logits = local_logits / math.sqrt(self.record_width)
        local_logits = local_logits.masked_fill(
            ~local_valid[:, :, None],
            torch.finfo(local_logits.dtype).min,
        )
        local_probabilities = local_logits.float().softmax(-1)
        physical_probabilities = torch.zeros(
            records.shape[0],
            ER_RECORDS,
            WITNESS_POSITIONS,
            source_width,
            device=records.device,
        )
        physical_probabilities.scatter_add_(
            -1,
            source_indices[:, :, None].expand(
                -1, -1, WITNESS_POSITIONS, -1
            ),
            local_probabilities * local_valid[:, :, None],
        )
        rule_assignment = assignment[:, :, 1 : 1 + RULE_COUNT].float()
        semantic_probabilities = torch.einsum(
            "bpr,bpol->brol",
            rule_assignment,
            physical_probabilities,
        )
        return semantic_probabilities.clamp_min(1e-30).log()

    def _equality_card_logits(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
        witness_pointer_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, rules, occurrences, source_width = witness_pointer_logits.shape
        if (
            rules != RULE_COUNT
            or occurrences != WITNESS_POSITIONS
            or source_width != ids.shape[1]
        ):
            raise ValueError("ER-CST witness pointer shape differs")
        fingerprints = self._fingerprints(
            ids,
            valid_mask,
            witness_pointer_logits.reshape(batch, rules * occurrences, source_width),
        ).reshape(batch, rules, occurrences, self.fingerprint_width)
        fingerprints = F.normalize(
            self.er_equality_projection(fingerprints),
            dim=-1,
        )
        before = fingerprints[:, :, :3]
        after = fingerprints[:, :, 3:]
        equality = self.er_equality_scale.exp().clamp(max=100.0) * torch.einsum(
            "brif,brjf->brij",
            after,
            before,
        )
        cards = torch.stack(
            [
                sum(
                    equality[:, :, after_position, before_position]
                    for after_position, before_position in enumerate(permutation)
                )
                for permutation in PERMUTATIONS
            ],
            dim=-1,
        )
        return cards.float(), equality.float()

    def compile_rule_program(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
        query_ids: torch.Tensor,
        query_valid_mask: torch.Tensor,
    ) -> WitnessEqualityCompilerOutput:
        records, token_memory, local_valid, source_indices, line_masks = (
            self._er_encode_records(ids, valid_mask)
        )
        role_logits = self.er_record_role_head(records)
        assignment = self._er_assignment(role_logits)
        semantic_records = torch.einsum("bps,bpw->bsw", assignment, records)
        semantic_records = semantic_records + self.er_record_role_embedding[None]
        rules = self.er_rule_norm(semantic_records[:, 1 : 1 + RULE_COUNT])
        events = self.er_event_norm(semantic_records[:, 1 + RULE_COUNT :])

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
        initial_entities = self._fingerprints(ids, valid_mask, initial_pointer_logits)
        initial_matches = self.logit_scale.exp().clamp(max=100.0) * torch.einsum(
            "bpf,brf->bpr", initial_entities, bindings
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

        event_queries = self.er_event_card_query_projection(events)
        rule_keys = self.er_rule_card_key_projection(rules)
        event_card_logits = torch.einsum(
            "bew,brw->ber", event_queries, rule_keys
        ) / math.sqrt(self.record_width)
        witness_pointer_logits = self._global_witness_logits(
            records,
            token_memory,
            local_valid,
            source_indices,
            assignment,
            ids.shape[1],
        )
        card_logits, equality_logits = self._equality_card_logits(
            ids,
            valid_mask,
            witness_pointer_logits,
        )

        line_distribution = line_masks.float()
        line_distribution = line_distribution / line_distribution.sum(-1, keepdim=True)
        line_pointer_logits = torch.einsum(
            "bps,bpl->bsl", assignment.float(), line_distribution
        ).clamp_min(1e-30).log()
        query_output = self.compile_query_with_evidence(query_ids, query_valid_mask)
        return WitnessEqualityCompilerOutput(
            program=RuleCardProgram(
                initial_state=state_logits.float(),
                rule_cards=card_logits,
                event_card=event_card_logits.float(),
                event_halt=self.er_event_halt_head(events).float(),
            ),
            query=query_output.query,
            line_pointer_logits=line_pointer_logits,
            binding_pointer_logits=binding_pointer_logits,
            initial_entity_pointer_logits=initial_pointer_logits,
            query_pointer_logits=query_output.pointer_logits,
            witness_pointer_logits=witness_pointer_logits,
            equality_logits=equality_logits,
        )


def witness_equality_adaptive_parameter_names(
    model: WitnessEqualityBusCompiler,
) -> frozenset[str]:
    if hasattr(model, "er_rule_permutation_head"):
        raise ValueError("direct ER-CST card classifier remains installed")
    er_names = frozenset(
        name for name, _ in model.named_parameters() if name.startswith("er_")
    )
    required = {
        "er_witness_queries",
        "er_witness_norm.weight",
        "er_witness_norm.bias",
        "er_witness_query_projection.weight",
        "er_witness_key_projection.weight",
        "er_equality_projection.weight",
        "er_equality_scale",
    }
    if not required.issubset(er_names):
        raise ValueError("witness-equality parameter contract differs")
    shared_prefixes = (
        "record_byte_embedding.",
        "record_position_embedding.",
        "record_query",
        "record_line_encoder.",
        "record_line_norm.",
        "record_set_encoder.",
        "record_set_norm.",
        "local_occurrence_",
    )
    return er_names | frozenset(
        name
        for name, _ in model.named_parameters()
        if name.startswith(shared_prefixes)
    )


def freeze_to_witness_equality_adaptive(
    model: WitnessEqualityBusCompiler,
) -> tuple[str, ...]:
    declared = witness_equality_adaptive_parameter_names(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in declared)
    actual = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if actual != declared:
        raise ValueError("witness-equality trainable contract differs")
    return tuple(sorted(actual))
