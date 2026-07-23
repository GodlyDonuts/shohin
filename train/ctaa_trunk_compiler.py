"""Raw-trunk compiler for Closure-Tied Action Algebra packets.

Shohin reads a program once and emits a private categorical packet. Program
source is deleted before recurrence. A separate query source is disclosed only
after the terminal state has committed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import torch
import torch.nn as nn

from ctaa_neural_core import (
    CTAA_ACTION_COUNT,
    CTAA_MAX_STEPS,
    CTAA_WIDTH,
    ClosureTiedPointerCore,
    HardDualExecutionTrace,
    HardExecutionTrace,
    execute_streamed_dual,
    execute_streamed_state_route,
)


CTAA_H19_BLOCK_INDEX = 19
CTAA_H29_BLOCK_INDEX = 29


@dataclass(frozen=True)
class TrunkResidualBundle:
    early: torch.Tensor
    late: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class CTAAProgramLogits:
    action_cards: torch.Tensor
    initial_state: torch.Tensor
    schedule: torch.Tensor


@dataclass(frozen=True)
class HardCTAAPacket:
    """Source-deleted categorical executor input."""

    action_cards: torch.Tensor
    initial_state: torch.Tensor
    schedule: torch.Tensor

    def __post_init__(self) -> None:
        if {field.name for field in fields(self)} != {
            "action_cards",
            "initial_state",
            "schedule",
        }:
            raise ValueError("CTAA hard-packet schema differs")
        if any(value.dtype != torch.uint8 for value in self.__dict__.values()):
            raise ValueError("CTAA hard packet must use materialized bytes")
        batch = self.action_cards.shape[0] if self.action_cards.ndim == 3 else -1
        if self.action_cards.shape != (batch, CTAA_ACTION_COUNT, CTAA_WIDTH):
            raise ValueError("CTAA hard action-card geometry differs")
        if self.initial_state.shape != (batch, CTAA_WIDTH):
            raise ValueError("CTAA hard initial-state geometry differs")
        if self.schedule.shape != (batch, CTAA_MAX_STEPS):
            raise ValueError("CTAA hard schedule geometry differs")
        if self.action_cards.numel() and int(self.action_cards.max()) >= CTAA_WIDTH:
            raise ValueError("CTAA hard action card leaves categorical domain")
        if self.initial_state.numel() and int(self.initial_state.max()) >= CTAA_WIDTH:
            raise ValueError("CTAA hard initial state leaves categorical domain")
        if self.schedule.numel() == 0 or int(self.schedule.max()) > CTAA_ACTION_COUNT:
            raise ValueError("CTAA hard schedule leaves event domain")
        stop_mask = self.schedule.eq(CTAA_ACTION_COUNT)
        if not bool(stop_mask.sum(1).eq(1).all()):
            raise ValueError("CTAA hard schedule requires exactly one STOP")
        stop_index = stop_mask.long().argmax(1)
        if not bool(((stop_index > 0) & (stop_index < CTAA_MAX_STEPS - 1)).all()):
            raise ValueError("CTAA hard STOP boundary differs")

    @property
    def bytes_per_row(self) -> int:
        if self.action_cards.ndim != 3 or self.initial_state.ndim != 2:
            raise ValueError("CTAA hard-packet geometry differs")
        if self.schedule.ndim != 2:
            raise ValueError("CTAA hard schedule geometry differs")
        size = (
            self.action_cards.shape[1] * self.action_cards.shape[2]
            + self.initial_state.shape[1]
            + self.schedule.shape[1]
        )
        if size != 56:
            raise ValueError("CTAA hard-packet byte contract differs")
        return size

    def execute(self, core: ClosureTiedPointerCore) -> HardExecutionTrace:
        return execute_streamed_state_route(
            core,
            CTAA_WIDTH,
            self.action_cards.long(),
            self.schedule.long(),
            self.initial_state.long(),
        )

    def execute_dual(self, core: ClosureTiedPointerCore) -> HardDualExecutionTrace:
        return execute_streamed_dual(
            core,
            CTAA_WIDTH,
            self.action_cards.long(),
            self.schedule.long(),
            self.initial_state.long(),
        )


@dataclass(frozen=True)
class HardCTAAQuery:
    position: torch.Tensor

    def __post_init__(self) -> None:
        if self.position.ndim != 1 or self.position.dtype != torch.uint8:
            raise ValueError("CTAA hard query must be one byte per row")
        if self.position.numel() == 0 or int(self.position.max()) >= CTAA_WIDTH:
            raise ValueError("CTAA hard query leaves position domain")

    def answer(self, trace: HardExecutionTrace) -> torch.Tensor:
        terminal = trace.states[:, -1]
        if terminal.shape[0] != self.position.shape[0]:
            raise ValueError("CTAA trace and query batches differ")
        return terminal.gather(1, self.position.long()[:, None]).squeeze(1)


class TrunkCausalCTAACompiler(nn.Module):
    """Compile source-only program and later query into categorical fields.

    The decoder receives no source spans, parser metadata, targets, executor
    output, or verifier feedback. Residual interventions are evaluation-only.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        width: int = 3,
        action_count: int = 4,
        max_steps: int = 41,
        compiler_width: int = 384,
        heads: int = 8,
        encoder_layers: int = 5,
        encoder_feedforward: int = 1408,
        decoder_layers: int = 2,
        decoder_feedforward: int = 1024,
        early_layer: int = CTAA_H19_BLOCK_INDEX,
        late_layer: int = CTAA_H29_BLOCK_INDEX,
        padding_id: int = 1,
    ) -> None:
        super().__init__()
        if model.cfg.n_loop != 1:
            raise ValueError("CTAA compiler requires the raw one-pass trunk")
        if (width, action_count, max_steps) != (
            CTAA_WIDTH,
            CTAA_ACTION_COUNT,
            CTAA_MAX_STEPS,
        ):
            raise ValueError("CTAA compiler packet geometry differs")
        if compiler_width < 1 or compiler_width % heads:
            raise ValueError("CTAA compiler decoder geometry differs")
        if not 0 <= early_layer < late_layer < len(model.blocks):
            raise ValueError("CTAA trunk intervention layers differ")
        if padding_id < 0 or padding_id >= model.cfg.vocab_size:
            raise ValueError("CTAA padding id differs")
        self.model = model
        self.width = int(width)
        self.action_count = int(action_count)
        self.max_steps = int(max_steps)
        self.compiler_width = int(compiler_width)
        self.early_layer = int(early_layer)
        self.late_layer = int(late_layer)
        self.padding_id = int(padding_id)
        self.model.requires_grad_(False)

        self.action_slot_count = self.action_count * self.width
        self.initial_slot_start = self.action_slot_count
        self.schedule_slot_start = self.initial_slot_start + self.width
        self.program_slot_count = self.schedule_slot_start + self.max_steps

        self.early_memory_norm = nn.LayerNorm(model.cfg.d_model)
        self.early_memory_projection = nn.Linear(
            model.cfg.d_model,
            self.compiler_width,
            bias=False,
        )
        self.late_memory_norm = nn.LayerNorm(model.cfg.d_model)
        self.late_memory_projection = nn.Linear(
            model.cfg.d_model,
            self.compiler_width,
            bias=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.compiler_width,
            nhead=heads,
            dim_feedforward=encoder_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            enable_nested_tensor=False,
        )
        self.program_queries = nn.Parameter(
            torch.empty(self.program_slot_count, self.compiler_width)
        )
        self.query_query = nn.Parameter(torch.empty(1, self.compiler_width))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.compiler_width,
            nhead=heads,
            dim_feedforward=decoder_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
        )
        self.decoder_norm = nn.LayerNorm(self.compiler_width)
        self.tuple_head = nn.Linear(self.compiler_width, self.width)
        self.event_head = nn.Linear(
            self.compiler_width,
            self.action_count + 1,
        )
        self.query_head = nn.Linear(self.compiler_width, self.width)
        nn.init.normal_(self.program_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.query_query, mean=0.0, std=0.02)

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("model."):
                yield parameter

    @property
    def adapter_num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    @property
    def complete_num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initialize_qualified_memory(
        self,
        qualified_state: Mapping[str, torch.Tensor],
    ) -> tuple[str, ...]:
        """Warm-start only the hash-qualified layer-19 memory backbone."""
        mapping = {
            "memory_norm.": "early_memory_norm.",
            "memory_projection.": "early_memory_projection.",
            "memory_encoder.": "memory_encoder.",
        }
        own = self.state_dict()
        loaded = []
        with torch.no_grad():
            for source_name, value in qualified_state.items():
                for source_prefix, target_prefix in mapping.items():
                    if not source_name.startswith(source_prefix):
                        continue
                    target_name = target_prefix + source_name[len(source_prefix) :]
                    if target_name not in own or own[target_name].shape != value.shape:
                        raise ValueError("CTAA qualified memory geometry differs")
                    own[target_name].copy_(value)
                    loaded.append(target_name)
                    break
            expected = {
                name
                for name in own
                if name.startswith(
                    (
                        "early_memory_norm.",
                        "early_memory_projection.",
                        "memory_encoder.",
                    )
                )
            }
            if set(loaded) != expected:
                raise ValueError("CTAA qualified memory initialization incomplete")
            self.late_memory_norm.weight.copy_(self.early_memory_norm.weight)
            self.late_memory_norm.bias.copy_(self.early_memory_norm.bias)
            self.late_memory_projection.weight.zero_()
        return tuple(sorted(loaded))

    def encode_source(self, ids: torch.Tensor) -> TrunkResidualBundle:
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("CTAA source ids differ")
        if ids.shape[1] > self.model.cfg.seq_len:
            raise ValueError("CTAA source exceeds trunk context")
        valid = ids.ne(self.padding_id)
        if bool(~valid.any(-1).all()):
            raise ValueError("CTAA source contains an empty row")
        if ids.shape[1] > 1 and bool(((~valid[:, :-1]) & valid[:, 1:]).any()):
            raise ValueError("CTAA source must use monotonic right padding")
        with torch.no_grad():
            hidden = self.model.tok(ids)
            cos = self.model.cos[: ids.shape[1]].to(hidden.device)
            sin = self.model.sin[: ids.shape[1]].to(hidden.device)
            early = None
            late = None
            for layer_index, block in enumerate(self.model.blocks):
                hidden, _ = block(hidden, cos, sin)
                if layer_index == self.early_layer:
                    early = hidden.detach()
                if layer_index == self.late_layer:
                    late = hidden.detach()
            if early is None or late is None:
                raise RuntimeError("CTAA intervention residual was not captured")
            # h29 is the raw residual after zero-based block index 29 and
            # before the trunk's final normalization, as preregistered.
        return TrunkResidualBundle(early=early, late=late, valid=valid)

    def memory_from_residuals(
        self,
        bundle: TrunkResidualBundle,
        *,
        intervention: str = "native",
        donor: TrunkResidualBundle | None = None,
        batch_rotation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        early, late, valid = bundle.early, bundle.late, bundle.valid
        if early.shape != late.shape or early.shape[:2] != bundle.valid.shape:
            raise ValueError("CTAA residual bundle geometry differs")
        aliases = {
            "h19_zero": "zero_early",
            "h29_zero": "zero_late",
            "h19_batch_rotate": "batch_rotate_early",
            "h19_donor_transplant": "donor_early",
            "h29_batch_rotate": "batch_rotate_late",
            "h29_donor_transplant": "donor_late",
        }
        intervention = aliases.get(intervention, intervention)
        if intervention == "native":
            if donor is not None or batch_rotation is not None:
                raise ValueError("CTAA native residual received intervention data")
        elif intervention == "zero_early":
            if donor is not None or batch_rotation is not None:
                raise ValueError("CTAA zero residual received intervention data")
            early = torch.zeros_like(early)
        elif intervention == "zero_late":
            if donor is not None or batch_rotation is not None:
                raise ValueError("CTAA zero residual received intervention data")
            late = torch.zeros_like(late)
        elif intervention == "zero_both":
            if donor is not None or batch_rotation is not None:
                raise ValueError("CTAA zero residual received intervention data")
            early, late = torch.zeros_like(early), torch.zeros_like(late)
        elif intervention in {
            "batch_rotate",
            "batch_rotate_early",
            "batch_rotate_late",
        }:
            batch = valid.shape[0]
            if (
                donor is not None
                or batch_rotation is None
                or batch_rotation.dtype != torch.long
                or batch_rotation.shape != (batch,)
            ):
                raise ValueError("CTAA batch residual rotation differs")
            rotation = batch_rotation.to(valid.device)
            expected = torch.arange(batch, device=valid.device)
            if not torch.equal(rotation.sort().values, expected) or bool(
                rotation.eq(expected).any()
            ):
                raise ValueError("CTAA batch residual rotation is not a derangement")
            rotated_valid = valid.index_select(0, rotation)
            if not torch.equal(rotated_valid, valid):
                raise ValueError("CTAA rotated residual padding geometry differs")
            if intervention in {"batch_rotate", "batch_rotate_early"}:
                early = early.index_select(0, rotation)
            if intervention in {"batch_rotate", "batch_rotate_late"}:
                late = late.index_select(0, rotation)
        elif intervention in {"donor", "donor_early", "donor_late"}:
            if (
                donor is None
                or batch_rotation is not None
                or donor.early.shape != early.shape
                or donor.late.shape != late.shape
                or donor.valid.shape != valid.shape
                or not torch.equal(donor.valid.to(valid.device), valid)
            ):
                raise ValueError("CTAA donor residual geometry differs")
            if intervention in {"donor", "donor_early"}:
                early = donor.early.to(device=early.device, dtype=early.dtype)
            if intervention in {"donor", "donor_late"}:
                late = donor.late.to(device=late.device, dtype=late.dtype)
        else:
            raise ValueError("CTAA residual intervention differs")
        memory = self.early_memory_projection(self.early_memory_norm(early))
        memory = memory + self.late_memory_projection(self.late_memory_norm(late))
        memory = self.memory_encoder(memory, src_key_padding_mask=~valid)
        return memory, valid

    def _decode(
        self,
        memory: torch.Tensor,
        valid: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        slots = queries[None].expand(memory.shape[0], -1, -1)
        decoded = self.decoder(
            slots,
            memory,
            memory_key_padding_mask=~valid,
        )
        return self.decoder_norm(decoded)

    def compile_program_from_residuals(
        self,
        bundle: TrunkResidualBundle,
        *,
        intervention: str = "native",
        donor: TrunkResidualBundle | None = None,
        batch_rotation: torch.Tensor | None = None,
    ) -> CTAAProgramLogits:
        memory, valid = self.memory_from_residuals(
            bundle,
            intervention=intervention,
            donor=donor,
            batch_rotation=batch_rotation,
        )
        slots = self._decode(memory, valid, self.program_queries)
        tuple_logits = self.tuple_head(slots[:, : self.schedule_slot_start]).float()
        return CTAAProgramLogits(
            action_cards=tuple_logits[:, : self.action_slot_count].reshape(
                memory.shape[0],
                self.action_count,
                self.width,
                self.width,
            ),
            initial_state=tuple_logits[
                :,
                self.initial_slot_start : self.schedule_slot_start,
            ],
            schedule=self.event_head(slots[:, self.schedule_slot_start :]).float(),
        )

    def compile_program(
        self,
        ids: torch.Tensor,
        *,
        intervention: str = "native",
        donor: TrunkResidualBundle | None = None,
        batch_rotation: torch.Tensor | None = None,
    ) -> CTAAProgramLogits:
        return self.compile_program_from_residuals(
            self.encode_source(ids),
            intervention=intervention,
            donor=donor,
            batch_rotation=batch_rotation,
        )

    def compile_query(
        self,
        query_ids: torch.Tensor,
        *,
        intervention: str = "native",
        donor: TrunkResidualBundle | None = None,
        batch_rotation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bundle = self.encode_source(query_ids)
        memory, valid = self.memory_from_residuals(
            bundle,
            intervention=intervention,
            donor=donor,
            batch_rotation=batch_rotation,
        )
        decoded = self._decode(memory, valid, self.query_query)
        return self.query_head(decoded[:, 0]).float()

    @staticmethod
    def materialize_program(output: CTAAProgramLogits) -> HardCTAAPacket:
        return HardCTAAPacket(
            action_cards=output.action_cards.argmax(-1).to(torch.uint8),
            initial_state=output.initial_state.argmax(-1).to(torch.uint8),
            schedule=output.schedule.argmax(-1).to(torch.uint8),
        )

    @staticmethod
    def materialize_query(output: torch.Tensor) -> HardCTAAQuery:
        if output.ndim != 2:
            raise ValueError("CTAA query logits geometry differs")
        return HardCTAAQuery(position=output.argmax(-1).to(torch.uint8))
