"""Renderer-native program decoder over the frozen renderer-orbit memory."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sd_cst import DeletedProgramTape
from sd_cst_binding_bus import BindingBusOutput, PERMUTATIONS
from sd_cst_renderer_orbit_frontend import RendererOrbitGroundedCompiler


class RendererNativeProgramCompiler(RendererOrbitGroundedCompiler):
    """Decode program structure directly instead of translating into frozen heads."""

    def __init__(
        self,
        *,
        native_slot_layers: int = 2,
        native_slot_heads: int = 8,
        native_slot_ff: int = 2048,
        **kwargs: int,
    ) -> None:
        super().__init__(**kwargs)
        if native_slot_layers <= 0 or native_slot_heads <= 0:
            raise ValueError("native program depth and heads must be positive")
        if self.orbit_width % native_slot_heads or native_slot_ff <= 0:
            raise ValueError("native program dimensions are invalid")
        self.native_program_queries = nn.Parameter(torch.empty(9, self.orbit_width))
        self.native_event_queries = nn.Parameter(torch.empty(8, self.orbit_width))
        self.native_program_query_projection = nn.Linear(
            self.orbit_width,
            self.orbit_width,
            bias=False,
        )
        self.native_event_query_projection = nn.Linear(
            self.orbit_width,
            self.orbit_width,
            bias=False,
        )
        self.native_source_key_projection = nn.Linear(
            self.orbit_width,
            self.orbit_width,
            bias=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.orbit_width,
            nhead=native_slot_heads,
            dim_feedforward=native_slot_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.native_slot_encoder = nn.TransformerEncoder(
            layer,
            num_layers=native_slot_layers,
            enable_nested_tensor=False,
        )
        self.native_slot_norm = nn.LayerNorm(self.orbit_width)
        self.native_kind_head = nn.Linear(self.orbit_width, 3)
        self.native_amount_head = nn.Linear(self.orbit_width, 2)
        nn.init.normal_(self.native_program_queries, std=0.02)
        nn.init.normal_(self.native_event_queries, std=0.02)

    def _native_pointer_logits(
        self,
        memory: torch.Tensor,
        valid_mask: torch.Tensor,
        queries: torch.Tensor,
        query_projection: nn.Linear,
    ) -> torch.Tensor:
        logits = torch.einsum(
            "sw,blw->bsl",
            query_projection(queries),
            self.native_source_key_projection(memory),
        ) / math.sqrt(self.orbit_width)
        return logits.masked_fill(
            ~valid_mask[:, None],
            torch.finfo(logits.dtype).min,
        ).float()

    def compile_program(
        self,
        ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> BindingBusOutput:
        parent_memory, orbit_memory = self._encode_components(ids, valid_mask)
        line_pointer_logits = self._native_pointer_logits(
            orbit_memory,
            valid_mask,
            self.native_program_queries,
            self.native_program_query_projection,
        )
        line_weights = line_pointer_logits.softmax(-1).to(orbit_memory.dtype)
        slots = torch.einsum("bsl,blw->bsw", line_weights, orbit_memory)
        slots = self.native_slot_norm(self.native_slot_encoder(slots))
        events = slots[:, 1:]

        binding_pointer_logits = self._binding_pointer_logits(
            parent_memory,
            valid_mask,
            self.binding_queries,
        )
        initial_pointer_logits = self._binding_pointer_logits(
            parent_memory,
            valid_mask,
            self.initial_entity_queries,
        )
        event_pointer_logits = self._native_pointer_logits(
            orbit_memory,
            valid_mask,
            self.native_event_queries,
            self.native_event_query_projection,
        )
        line_mask = self._selected_line_mask(ids, valid_mask, line_pointer_logits)
        event_pointer_logits = event_pointer_logits.masked_fill(
            ~line_mask[:, 1:],
            torch.finfo(event_pointer_logits.dtype).min,
        )

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
                event_kind=self.native_kind_head(events).float(),
                event_identity=event_matches.float(),
                amount=self.native_amount_head(events).float(),
            ),
            line_pointer_logits=line_pointer_logits,
            binding_pointer_logits=binding_pointer_logits,
            initial_entity_pointer_logits=initial_pointer_logits,
            event_entity_pointer_logits=event_pointer_logits,
        )


def renderer_native_program_trainable_names(
    model: RendererNativeProgramCompiler,
) -> frozenset[str]:
    return frozenset(
        name for name, _ in model.named_parameters() if name.startswith("native_")
    )


def renderer_native_joint_trainable_names(
    model: RendererNativeProgramCompiler,
) -> frozenset[str]:
    shared_prefixes = (
        "orbit_byte_embedding.",
        "orbit_position_embedding.",
        "orbit_encoder.",
        "orbit_norm.",
    )
    return renderer_native_program_trainable_names(model) | frozenset(
        name for name, _ in model.named_parameters() if name.startswith(shared_prefixes)
    )


def _freeze_to_declared(
    model: RendererNativeProgramCompiler,
    declared: frozenset[str],
) -> tuple[str, ...]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in declared)
    actual = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if actual != declared:
        raise ValueError("renderer-native trainable parameter contract mismatch")
    return tuple(sorted(actual))


def freeze_to_renderer_native_program(
    model: RendererNativeProgramCompiler,
) -> tuple[str, ...]:
    declared = renderer_native_program_trainable_names(model)
    return _freeze_to_declared(model, declared)


def freeze_to_renderer_native_joint(
    model: RendererNativeProgramCompiler,
) -> tuple[str, ...]:
    declared = renderer_native_joint_trainable_names(model)
    return _freeze_to_declared(model, declared)
