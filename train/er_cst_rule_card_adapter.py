"""Neural rule-card compiler and tied motor for ER-CST."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sd_cst_binding_bus import PERMUTATIONS
from sd_cst_byte_addressed import BYTE_PAD
from sd_cst_complete_physical_record_bus_v1_2 import (
    CompletePhysicalRecordBusCompilerV1_2,
)


RULE_COUNT = 3
RULE_CARD_COUNT = len(PERMUTATIONS)
EVENT_SLOTS = 8
ER_RECORDS = 1 + RULE_COUNT + EVENT_SLOTS
HALT_CLASS = 1


@dataclass(frozen=True, slots=True)
class RuleCardProgram:
    initial_state: torch.Tensor
    rule_cards: torch.Tensor
    event_card: torch.Tensor
    event_halt: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.initial_state.shape[0])

    def hard(self) -> HardRuleCardProgram:
        return HardRuleCardProgram(
            initial_state=self.initial_state.argmax(-1),
            rule_cards=self.rule_cards.argmax(-1),
            event_card=self.event_card.argmax(-1),
            event_halt=self.event_halt.argmax(-1),
        )


@dataclass(frozen=True, slots=True)
class HardRuleCardProgram:
    initial_state: torch.Tensor
    rule_cards: torch.Tensor
    event_card: torch.Tensor
    event_halt: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.initial_state.shape[0])

    def __post_init__(self) -> None:
        batch = self.initial_state.shape[0]
        expected = {
            "initial_state": (batch,),
            "rule_cards": (batch, RULE_COUNT),
            "event_card": (batch, EVENT_SLOTS),
            "event_halt": (batch, EVENT_SLOTS),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != torch.long:
                raise ValueError(f"hard rule-card field differs: {name}")


@dataclass(frozen=True, slots=True)
class RuleCardCompilerOutput:
    program: RuleCardProgram
    line_pointer_logits: torch.Tensor
    binding_pointer_logits: torch.Tensor
    initial_entity_pointer_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class RuleCardRollout:
    final_state: torch.Tensor
    state_trajectory: tuple[torch.Tensor, ...]
    alive_trajectory: tuple[torch.Tensor, ...]


class EpisodicRuleCardCompiler(CompletePhysicalRecordBusCompilerV1_2):
    """Compile twelve physical records into cards, references, and state."""

    def __init__(self, **kwargs: int) -> None:
        super().__init__(**kwargs)
        width = self.record_width
        self.er_record_role_head = nn.Linear(width, ER_RECORDS)
        self.er_record_role_embedding = nn.Parameter(
            torch.empty(ER_RECORDS, width)
        )
        self.er_rule_norm = nn.LayerNorm(width)
        self.er_event_norm = nn.LayerNorm(width)
        self.er_rule_permutation_head = nn.Linear(width, RULE_CARD_COUNT)
        self.er_event_halt_head = nn.Linear(width, 2)
        self.er_event_card_query_projection = nn.Linear(width, width, bias=False)
        self.er_rule_card_key_projection = nn.Linear(width, width, bias=False)
        nn.init.normal_(self.er_record_role_embedding, std=0.02)

    def _er_physical_line_masks(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_orbit_input(ids, valid_mask)
        newline = ids.eq(10) & valid_mask
        if not bool(newline.sum(-1).eq(ER_RECORDS - 1).all()):
            raise ValueError("ER-CST compiler requires exactly twelve physical lines")
        line_index = newline.long().cumsum(-1) - newline.long()
        roles = torch.arange(ER_RECORDS, device=ids.device)
        masks = line_index[:, None].eq(roles[None, :, None])
        masks = masks & valid_mask[:, None]
        counts = masks.sum(-1)
        if bool(counts.eq(0).any()) or bool(counts.gt(self.max_line_bytes).any()):
            raise ValueError("ER-CST line is empty or exceeds the record window")
        return masks

    def _er_encode_records(
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
        expanded = ids[:, None].expand(-1, ER_RECORDS, -1)
        local_ids = expanded.gather(-1, safe_indices)
        local_ids = torch.where(
            local_valid,
            local_ids,
            torch.full_like(local_ids, BYTE_PAD),
        )

        flat_ids = local_ids.reshape(batch * ER_RECORDS, self.max_line_bytes)
        flat_valid = local_valid.reshape(batch * ER_RECORDS, self.max_line_bytes)
        local_positions = torch.arange(self.max_line_bytes + 1, device=ids.device)
        token_hidden = self.record_byte_embedding(flat_ids)
        token_hidden = token_hidden + self.record_position_embedding(
            local_positions[1:]
        )[None]
        query = self.record_query[None, None].expand(
            batch * ER_RECORDS, 1, -1
        )
        query = query + self.record_position_embedding(local_positions[:1])[None]
        hidden = torch.cat([query, token_hidden], dim=1)
        hidden_valid = torch.cat(
            [
                torch.ones(
                    batch * ER_RECORDS,
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
        records = hidden[:, 0].reshape(batch, ER_RECORDS, self.record_width)
        records = self.record_set_norm(self.record_set_encoder(records))
        token_memory = hidden[:, 1:].reshape(
            batch,
            ER_RECORDS,
            self.max_line_bytes,
            self.record_width,
        )
        return records, token_memory, local_valid, safe_indices, masks

    def _er_assignment(self, role_logits: torch.Tensor) -> torch.Tensor:
        if role_logits.shape[1:] != (ER_RECORDS, ER_RECORDS):
            raise ValueError("ER-CST role logits differ")
        if self.training:
            return role_logits.softmax(1)
        selected = role_logits.argmax(1)
        return F.one_hot(selected, ER_RECORDS).transpose(1, 2).to(
            role_logits.dtype
        )

    def compile_rule_program(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> RuleCardCompilerOutput:
        records, token_memory, local_valid, source_indices, line_masks = (
            self._er_encode_records(ids, valid_mask)
        )
        role_logits = self.er_record_role_head(records)
        assignment = self._er_assignment(role_logits)
        semantic_records = torch.einsum("bps,bpw->bsw", assignment, records)
        semantic_records = semantic_records + self.er_record_role_embedding[None]
        declaration = assignment[:, :, 0]
        rules = self.er_rule_norm(
            semantic_records[:, 1 : 1 + RULE_COUNT]
        )
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
        initial_entities = self._fingerprints(
            ids, valid_mask, initial_pointer_logits
        )
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

        line_distribution = line_masks.float()
        line_distribution = line_distribution / line_distribution.sum(
            -1, keepdim=True
        )
        line_pointer_logits = torch.einsum(
            "bps,bpl->bsl", assignment.float(), line_distribution
        ).clamp_min(1e-30).log()
        if declaration.shape != (ids.shape[0], ER_RECORDS):
            raise RuntimeError("ER-CST declaration assignment differs")
        return RuleCardCompilerOutput(
            program=RuleCardProgram(
                initial_state=state_logits.float(),
                rule_cards=self.er_rule_permutation_head(rules).float(),
                event_card=event_card_logits.float(),
                event_halt=self.er_event_halt_head(events).float(),
            ),
            line_pointer_logits=line_pointer_logits,
            binding_pointer_logits=binding_pointer_logits,
            initial_entity_pointer_logits=initial_pointer_logits,
        )


class TiedRuleCardMotor(nn.Module):
    """Apply one learned state/card transition at every recurrent step."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("rule-card motor width must be positive")
        self.hidden = nn.Linear(RULE_CARD_COUNT * 2, hidden)
        self.output = nn.Linear(hidden, RULE_CARD_COUNT)

    def forward(
        self, state: torch.Tensor, card: torch.Tensor
    ) -> torch.Tensor:
        if state.shape != card.shape or state.shape[-1] != RULE_CARD_COUNT:
            raise ValueError("rule-card motor inputs differ")
        return self.output(F.gelu(self.hidden(torch.cat([state, card], dim=-1))))


def rule_motor_certificate(
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_ids = []
    card_ids = []
    target_ids = []
    state_index = {tuple(value): index for index, value in enumerate(PERMUTATIONS)}
    for state_id, state in enumerate(PERMUTATIONS):
        for card_id, card in enumerate(PERMUTATIONS):
            target = tuple(state[position] for position in card)
            state_ids.append(state_id)
            card_ids.append(card_id)
            target_ids.append(state_index[target])
    return (
        torch.tensor(state_ids, dtype=torch.long, device=device),
        torch.tensor(card_ids, dtype=torch.long, device=device),
        torch.tensor(target_ids, dtype=torch.long, device=device),
    )


def rollout_rule_cards(
    program: HardRuleCardProgram,
    motor: TiedRuleCardMotor,
) -> RuleCardRollout:
    device = program.initial_state.device
    if any(
        value.device != device
        for value in (
            program.rule_cards,
            program.event_card,
            program.event_halt,
        )
    ):
        raise ValueError("hard rule-card fields must share one device")
    state_ids = program.initial_state
    alive = torch.ones(program.batch_size, dtype=torch.bool, device=device)
    states = [state_ids.clone()]
    alive_states = [alive.clone()]
    rows = torch.arange(program.batch_size, device=device)
    for step in range(EVENT_SLOTS):
        stop = program.event_halt[:, step].eq(HALT_CLASS)
        active = alive & ~stop
        card_slot = program.event_card[:, step]
        if bool(card_slot.lt(0).any()) or bool(card_slot.ge(RULE_COUNT).any()):
            raise ValueError("event references an invalid rule-card slot")
        card_ids = program.rule_cards[rows, card_slot]
        logits = motor(
            F.one_hot(state_ids, RULE_CARD_COUNT).float(),
            F.one_hot(card_ids, RULE_CARD_COUNT).float(),
        )
        proposal = logits.argmax(-1)
        state_ids = torch.where(active, proposal, state_ids)
        alive = active
        states.append(state_ids.clone())
        alive_states.append(alive.clone())
    return RuleCardRollout(
        final_state=state_ids,
        state_trajectory=tuple(states),
        alive_trajectory=tuple(alive_states),
    )


def er_new_parameter_names(
    model: EpisodicRuleCardCompiler,
) -> frozenset[str]:
    names = frozenset(
        name for name, _ in model.named_parameters() if name.startswith("er_")
    )
    if len(names) != 13:
        raise ValueError("ER-CST new-parameter contract differs")
    return names


def er_adaptive_parameter_names(
    model: EpisodicRuleCardCompiler,
) -> frozenset[str]:
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
    return er_new_parameter_names(model) | frozenset(
        name
        for name, _ in model.named_parameters()
        if name.startswith(shared_prefixes)
    )


def freeze_to_er_adaptive(
    model: EpisodicRuleCardCompiler,
) -> tuple[str, ...]:
    declared = er_adaptive_parameter_names(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in declared)
    actual = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if actual != declared:
        raise ValueError("ER-CST trainable parameter contract differs")
    return tuple(sorted(actual))


def rule_card_parameter_report(
    model: EpisodicRuleCardCompiler,
    motor: TiedRuleCardMotor,
    *,
    base_parameters: int,
    reader_parameters: int,
) -> dict[str, int]:
    compiler = model.parameter_count()
    motor_count = sum(parameter.numel() for parameter in motor.parameters())
    complete = base_parameters + compiler + motor_count + reader_parameters
    trainable = sum(
        parameter.numel()
        for parameter in itertools.chain(model.parameters(), motor.parameters())
        if parameter.requires_grad
    )
    return {
        "base": int(base_parameters),
        "compiler": int(compiler),
        "motor": int(motor_count),
        "reader": int(reader_parameters),
        "complete_system": int(complete),
        "headroom_below_200m": int(200_000_000 - complete),
        "trainable": int(trainable),
    }
