"""Variable-cardinality neural compiler for Episodic Relation Tensor Transport."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from er_cst_witness_equality_bus import WitnessEqualityBusCompiler
from er_relation_tensor_motor import hard_relation, rollout_relation_tensor
from sd_cst_byte_addressed import BYTE_PAD


MIN_CARDINALITY = 3
MAX_CARDINALITY = 6
MAX_RULES = 4
EVENT_SLOTS = 13
TT_RECORDS = 1 + MAX_RULES + EVENT_SLOTS
DECLARATION_OCCURRENCES = 2 * MAX_CARDINALITY
WITNESS_SIDES = 2
STRICT_PARAMETER_CAP = 200_000_000
SHOHIN_BASE_PARAMETERS = 125_081_664


@dataclass(frozen=True, slots=True)
class RelationTensorQuery:
    logits: torch.Tensor
    pointer_logits: torch.Tensor

    def __post_init__(self) -> None:
        batch = self.logits.shape[0]
        if self.logits.shape != (batch, MAX_CARDINALITY):
            raise ValueError("ER-TT query logits differ")
        if self.pointer_logits.ndim != 2 or self.pointer_logits.shape[0] != batch:
            raise ValueError("ER-TT query pointer logits differ")


@dataclass(frozen=True, slots=True)
class HardRelationTensorProgram:
    cardinality: torch.Tensor
    active: torch.Tensor
    initial_state: torch.Tensor
    rule_cards: torch.Tensor
    rule_active: torch.Tensor
    event_card: torch.Tensor
    event_halt: torch.Tensor

    def __post_init__(self) -> None:
        batch = self.cardinality.shape[0]
        expected = {
            "cardinality": (batch,),
            "active": (batch, MAX_CARDINALITY),
            "initial_state": (batch, MAX_CARDINALITY, MAX_CARDINALITY),
            "rule_cards": (batch, MAX_RULES, MAX_CARDINALITY, MAX_CARDINALITY),
            "rule_active": (batch, MAX_RULES),
            "event_card": (batch, EVENT_SLOTS),
            "event_halt": (batch, EVENT_SLOTS),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"hard ER-TT field differs: {name}")
        if self.cardinality.dtype != torch.long or self.event_card.dtype != torch.long:
            raise ValueError("hard ER-TT categorical fields must be long")
        if any(
            value.dtype != torch.bool
            for value in (self.active, self.rule_active, self.event_halt)
        ):
            raise ValueError("hard ER-TT masks must be boolean")
        if bool(self.cardinality.lt(MIN_CARDINALITY).any()) or bool(
            self.cardinality.gt(MAX_CARDINALITY).any()
        ):
            raise ValueError("hard ER-TT cardinality is invalid")

    @property
    def batch_size(self) -> int:
        return int(self.cardinality.shape[0])

    def validate_references(self) -> None:
        if bool(self.event_card.lt(0).any()) or bool(self.event_card.ge(MAX_RULES).any()):
            raise ValueError("ER-TT event references an invalid rule slot")
        selected_active = self.rule_active.gather(1, self.event_card)
        alive = torch.ones(self.batch_size, dtype=torch.bool, device=self.active.device)
        for slot in range(EVENT_SLOTS):
            apply = alive & ~self.event_halt[:, slot]
            if bool((apply & ~selected_active[:, slot]).any()):
                raise ValueError("ER-TT live event references an inactive rule")
            alive = apply

    def rollout(self):
        self.validate_references()
        return rollout_relation_tensor(
            self.initial_state,
            self.rule_cards,
            self.event_card,
            self.event_halt,
            self.active,
        )


@dataclass(frozen=True, slots=True)
class RelationTensorProgram:
    cardinality: torch.Tensor
    initial_state: torch.Tensor
    rule_cards: torch.Tensor
    rule_active: torch.Tensor
    event_card: torch.Tensor
    event_halt: torch.Tensor

    def __post_init__(self) -> None:
        batch = self.cardinality.shape[0]
        expected = {
            "cardinality": (batch, MAX_CARDINALITY - MIN_CARDINALITY + 1),
            "initial_state": (batch, MAX_CARDINALITY, MAX_CARDINALITY),
            "rule_cards": (batch, MAX_RULES, MAX_CARDINALITY, MAX_CARDINALITY),
            "rule_active": (batch, MAX_RULES, 2),
            "event_card": (batch, EVENT_SLOTS, MAX_RULES),
            "event_halt": (batch, EVENT_SLOTS, 2),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"ER-TT program field differs: {name}")

    @property
    def batch_size(self) -> int:
        return int(self.cardinality.shape[0])

    def hard(self) -> HardRelationTensorProgram:
        cardinality = self.cardinality.argmax(-1) + MIN_CARDINALITY
        positions = torch.arange(MAX_CARDINALITY, device=self.cardinality.device)
        active = positions[None] < cardinality[:, None]
        return HardRelationTensorProgram(
            cardinality=cardinality,
            active=active,
            initial_state=hard_relation(self.initial_state[:, None], active)[:, 0],
            rule_cards=hard_relation(self.rule_cards, active),
            rule_active=self.rule_active.argmax(-1).bool(),
            event_card=self.event_card.argmax(-1),
            event_halt=self.event_halt.argmax(-1).bool(),
        )


@dataclass(frozen=True, slots=True)
class RelationTensorCompilerOutput:
    program: RelationTensorProgram
    query: RelationTensorQuery
    line_pointer_logits: torch.Tensor
    binding_pointer_logits: torch.Tensor
    initial_entity_pointer_logits: torch.Tensor
    witness_pointer_logits: torch.Tensor
    initial_equality_logits: torch.Tensor
    relation_equality_logits: torch.Tensor


def hard_relation_answer(
    final_state: torch.Tensor,
    query_logits: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Read the entity role at one model-selected terminal position."""
    if final_state.ndim != 3 or final_state.shape[-2:] != (
        MAX_CARDINALITY,
        MAX_CARDINALITY,
    ):
        raise ValueError("ER-TT final state differs")
    if query_logits.shape != (final_state.shape[0], MAX_CARDINALITY):
        raise ValueError("ER-TT query logits differ from final state")
    if active.shape != query_logits.shape:
        raise ValueError("ER-TT query mask differs")
    masked = query_logits.masked_fill(~active, torch.finfo(query_logits.dtype).min)
    query = F.one_hot(masked.argmax(-1), MAX_CARDINALITY).to(final_state.dtype)
    return torch.bmm(query[:, None], final_state).squeeze(1)


class EpisodicRelationTensorCompiler(WitnessEqualityBusCompiler):
    """Compile opaque witnesses directly into variable finite relations."""

    def __init__(self, **kwargs: int) -> None:
        super().__init__(**kwargs)
        width = self.record_width

        del self.er_record_role_head
        del self.er_record_role_embedding
        del self.er_witness_queries
        del self.local_occurrence_head
        del self.local_query_head
        del self.permutations

        self.er_tt_record_role_head = nn.Linear(width, TT_RECORDS)
        self.er_tt_record_role_embedding = nn.Parameter(torch.empty(TT_RECORDS, width))
        self.er_tt_witness_side_embedding = nn.Parameter(
            torch.empty(WITNESS_SIDES, width)
        )
        self.er_tt_witness_position_embedding = nn.Parameter(
            torch.empty(MAX_CARDINALITY, width)
        )
        self.er_tt_occurrence_head = nn.Linear(
            self.occurrence_ff,
            DECLARATION_OCCURRENCES,
        )
        self.er_tt_query_head = nn.Linear(width, MAX_CARDINALITY)
        self.er_tt_cardinality_head = nn.Linear(
            width,
            MAX_CARDINALITY - MIN_CARDINALITY + 1,
        )
        self.er_tt_rule_active_head = nn.Linear(width, 2)
        for parameter in (
            self.er_tt_record_role_embedding,
            self.er_tt_witness_side_embedding,
            self.er_tt_witness_position_embedding,
        ):
            nn.init.normal_(parameter, std=0.02)

    def _er_physical_line_masks(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_orbit_input(ids, valid_mask)
        newline = ids.eq(10) & valid_mask
        if not bool(newline.sum(-1).eq(TT_RECORDS - 1).all()):
            raise ValueError("ER-TT compiler requires exactly eighteen physical lines")
        line_index = newline.long().cumsum(-1) - newline.long()
        roles = torch.arange(TT_RECORDS, device=ids.device)
        masks = line_index[:, None].eq(roles[None, :, None]) & valid_mask[:, None]
        counts = masks.sum(-1)
        if bool(counts.eq(0).any()) or bool(counts.gt(self.max_line_bytes).any()):
            raise ValueError("ER-TT line is empty or exceeds the record window")
        return masks

    def _er_encode_records(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = self._er_physical_line_masks(ids, valid_mask)
        batch, width = ids.shape
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
        expanded = ids[:, None].expand(-1, TT_RECORDS, -1)
        local_ids = expanded.gather(-1, safe_indices)
        local_ids = torch.where(
            local_valid,
            local_ids,
            torch.full_like(local_ids, BYTE_PAD),
        )
        flat_ids = local_ids.reshape(batch * TT_RECORDS, self.max_line_bytes)
        flat_valid = local_valid.reshape(batch * TT_RECORDS, self.max_line_bytes)
        local_positions = torch.arange(self.max_line_bytes + 1, device=ids.device)
        token_hidden = self.record_byte_embedding(flat_ids)
        token_hidden = token_hidden + self.record_position_embedding(local_positions[1:])[None]
        query = self.record_query[None, None].expand(batch * TT_RECORDS, 1, -1)
        query = query + self.record_position_embedding(local_positions[:1])[None]
        hidden = torch.cat([query, token_hidden], dim=1)
        hidden_valid = torch.cat(
            [
                torch.ones(
                    batch * TT_RECORDS,
                    1,
                    dtype=torch.bool,
                    device=ids.device,
                ),
                flat_valid,
            ],
            dim=1,
        )
        hidden = self.record_line_encoder(hidden, src_key_padding_mask=~hidden_valid)
        hidden = self.record_line_norm(hidden)
        records = hidden[:, 0].reshape(batch, TT_RECORDS, self.record_width)
        records = self.record_set_norm(self.record_set_encoder(records))
        token_memory = hidden[:, 1:].reshape(
            batch,
            TT_RECORDS,
            self.max_line_bytes,
            self.record_width,
        )
        return records, token_memory, local_valid, safe_indices, masks

    def _er_assignment(self, role_logits: torch.Tensor) -> torch.Tensor:
        if role_logits.shape[1:] != (TT_RECORDS, TT_RECORDS):
            raise ValueError("ER-TT role logits differ")
        if self.training:
            return role_logits.softmax(1)
        selected = role_logits.argmax(1)
        return F.one_hot(selected, TT_RECORDS).transpose(1, 2).to(role_logits.dtype)

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
        local_logits = self.er_tt_occurrence_head(hidden).permute(0, 1, 3, 2)
        local_logits = local_logits.masked_fill(
            ~local_valid[:, :, None],
            torch.finfo(local_logits.dtype).min,
        )
        local_probabilities = local_logits.float().softmax(-1)
        physical_probabilities = torch.zeros(
            token_memory.shape[0],
            token_memory.shape[1],
            DECLARATION_OCCURRENCES,
            source_width,
            device=token_memory.device,
        )
        physical_probabilities.scatter_add_(
            -1,
            source_indices[:, :, None].expand(
                -1,
                -1,
                DECLARATION_OCCURRENCES,
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

    def _global_witness_logits(
        self,
        records: torch.Tensor,
        token_memory: torch.Tensor,
        local_valid: torch.Tensor,
        source_indices: torch.Tensor,
        assignment: torch.Tensor,
        source_width: int,
    ) -> torch.Tensor:
        records = records.detach()
        token_memory = token_memory.detach()
        assignment = assignment.detach()
        coordinate = (
            self.er_tt_witness_side_embedding[:, None]
            + self.er_tt_witness_position_embedding[None]
        ).reshape(DECLARATION_OCCURRENCES, self.record_width)
        queries = self.er_witness_norm(records)[:, :, None] + coordinate[None, None]
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
            TT_RECORDS,
            DECLARATION_OCCURRENCES,
            source_width,
            device=records.device,
        )
        physical_probabilities.scatter_add_(
            -1,
            source_indices[:, :, None].expand(
                -1,
                -1,
                DECLARATION_OCCURRENCES,
                -1,
            ),
            local_probabilities * local_valid[:, :, None],
        )
        rule_assignment = assignment[:, :, 1 : 1 + MAX_RULES].float()
        semantic_probabilities = torch.einsum(
            "bpr,bpol->brol",
            rule_assignment,
            physical_probabilities,
        )
        return semantic_probabilities.clamp_min(1e-30).log()

    def _equality_logits(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
        pointer_logits: torch.Tensor,
    ) -> torch.Tensor:
        leading = pointer_logits.shape[:-2]
        if pointer_logits.shape[-2:] != (DECLARATION_OCCURRENCES, ids.shape[1]):
            raise ValueError("ER-TT equality pointers differ")
        flattened = pointer_logits.reshape(
            pointer_logits.shape[0],
            -1,
            pointer_logits.shape[-1],
        )
        fingerprints = self._fingerprints(ids, valid_mask, flattened)
        fingerprints = fingerprints.reshape(
            *leading,
            DECLARATION_OCCURRENCES,
            self.fingerprint_width,
        )
        fingerprints = F.normalize(self.er_equality_projection(fingerprints), dim=-1)
        before = fingerprints[..., :MAX_CARDINALITY, :]
        after = fingerprints[..., MAX_CARDINALITY:, :]
        return self.er_equality_scale.exp().clamp(max=100.0) * torch.einsum(
            "...if,...jf->...ij",
            after,
            before,
        )

    def compile_relation_query(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> RelationTensorQuery:
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
        selected = self.local_query_norm(torch.einsum("bl,blw->bw", weights, values))
        return RelationTensorQuery(
            logits=self.er_tt_query_head(selected).float(),
            pointer_logits=pointer_logits,
        )

    def compile_relation_program(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
        query_ids: torch.Tensor,
        query_valid_mask: torch.Tensor,
    ) -> RelationTensorCompilerOutput:
        records, token_memory, local_valid, source_indices, line_masks = (
            self._er_encode_records(ids, valid_mask)
        )
        role_logits = self.er_tt_record_role_head(records)
        assignment = self._er_assignment(role_logits)
        semantic_records = torch.einsum("bps,bpw->bsw", assignment, records)
        semantic_records = semantic_records + self.er_tt_record_role_embedding[None]
        declaration = semantic_records[:, 0]
        rules = self.er_rule_norm(semantic_records[:, 1 : 1 + MAX_RULES])
        events = self.er_event_norm(semantic_records[:, 1 + MAX_RULES :])

        declaration_logits = self._global_declaration_logits(
            records,
            token_memory,
            local_valid,
            source_indices,
            assignment,
            ids.shape[1],
        )
        binding_pointer_logits = declaration_logits[:, :MAX_CARDINALITY]
        initial_pointer_logits = declaration_logits[:, MAX_CARDINALITY:]
        bindings = self._fingerprints(ids, valid_mask, binding_pointer_logits)
        initial_entities = self._fingerprints(ids, valid_mask, initial_pointer_logits)
        initial_equality = self.logit_scale.exp().clamp(max=100.0) * torch.einsum(
            "bpf,brf->bpr",
            initial_entities,
            bindings,
        )

        witness_pointer_logits = self._global_witness_logits(
            records,
            token_memory,
            local_valid,
            source_indices,
            assignment,
            ids.shape[1],
        )
        relation_equality = self._equality_logits(
            ids,
            valid_mask,
            witness_pointer_logits,
        )
        event_queries = self.er_event_card_query_projection(events)
        rule_keys = self.er_rule_card_key_projection(rules)
        event_card_logits = torch.einsum(
            "bew,brw->ber",
            event_queries,
            rule_keys,
        ) / math.sqrt(self.record_width)

        line_distribution = line_masks.float()
        line_distribution = line_distribution / line_distribution.sum(-1, keepdim=True)
        line_pointer_logits = torch.einsum(
            "bps,bpl->bsl",
            assignment.float(),
            line_distribution,
        ).clamp_min(1e-30).log()
        query = self.compile_relation_query(query_ids, query_valid_mask)
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


def relation_tensor_adaptive_parameter_names(
    model: EpisodicRelationTensorCompiler,
) -> frozenset[str]:
    forbidden = {
        "er_record_role_head",
        "er_record_role_embedding",
        "er_witness_queries",
        "er_rule_permutation_head",
        "local_occurrence_head",
        "local_query_head",
        "permutations",
    }
    if any(hasattr(model, name) for name in forbidden):
        raise ValueError("ER-TT retains a fixed-ontology component")
    er_names = frozenset(
        name for name, _ in model.named_parameters() if name.startswith("er_")
    )
    required_prefixes = {
        "er_tt_record_role_head.",
        "er_tt_record_role_embedding",
        "er_tt_witness_side_embedding",
        "er_tt_witness_position_embedding",
        "er_tt_occurrence_head.",
        "er_tt_query_head.",
        "er_tt_cardinality_head.",
        "er_tt_rule_active_head.",
    }
    if any(not any(name.startswith(prefix) for name in er_names) for prefix in required_prefixes):
        raise ValueError("ER-TT new-parameter contract differs")
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


def freeze_to_relation_tensor_adaptive(
    model: EpisodicRelationTensorCompiler,
) -> tuple[str, ...]:
    declared = relation_tensor_adaptive_parameter_names(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in declared)
    actual = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if actual != declared:
        raise ValueError("ER-TT trainable parameter contract differs")
    return tuple(sorted(actual))


def relation_tensor_parameter_report(
    model: EpisodicRelationTensorCompiler,
    *,
    base_parameters: int = SHOHIN_BASE_PARAMETERS,
) -> dict[str, int]:
    compiler = model.parameter_count()
    complete = int(base_parameters + compiler)
    report = {
        "base": int(base_parameters),
        "compiler": int(compiler),
        "motor": 0,
        "reader": 0,
        "complete_system": complete,
        "headroom_below_200m": STRICT_PARAMETER_CAP - complete,
        "trainable": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    if complete >= STRICT_PARAMETER_CAP or report["headroom_below_200m"] <= 0:
        raise ValueError("ER-TT complete system reaches the strict 200M ceiling")
    return report
