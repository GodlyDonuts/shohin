"""Bounded prototype of an Autocatalytic Hysteretic Relation Field.

The module consumes only source-deleted structural tensors and opaque
relation-card witnesses. It has no primitive labels, operation codes,
execution schedule, target relation, named arithmetic, or host convergence
test. A fixed recurrence count is a safety envelope; a learned event detector
owns the absorbing halt latch and records the first model-selected halt step.

This is an architecture prototype, not evidence of general reasoning. Its
claims are limited to the mechanics tested in the adjacent unit-test module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


PROTECTED_BASE_PARAMETERS = 125_081_664
SYSTEM_PARAMETER_CAP = 200_000_000
CARD_ARGUMENT_ROLES = 2
GRAPH_EDGE_ROLES = 3
FEEDBACK_ROLE = 2


class AHRFError(ValueError):
    """Raised when an AHRF input or configuration violates its contract."""


@dataclass(frozen=True, slots=True)
class SourceDeletedRelationGraph:
    """Typed graph and contextual witnesses after source/target deletion.

    ``node_features`` may encode structural node types, but must not contain
    primitive identities. ``argument_edges[b, parent, child, role]`` carries
    typed graph links, while ``node_card_mask`` associates a node with at most
    one opaque witness card. ``root_mask`` is a structural graph-root marker,
    not a target relation.
    """

    node_features: torch.Tensor
    node_mask: torch.Tensor
    argument_edges: torch.Tensor
    node_card_mask: torch.Tensor
    root_mask: torch.Tensor
    seed_facts: torch.Tensor
    witness_left: torch.Tensor
    witness_right: torch.Tensor
    witness_output: torch.Tensor
    witness_mask: torch.Tensor
    argument_mask: torch.Tensor
    object_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class AHRFParameterReceipt:
    """Exact parameter accounting against the protected flagship."""

    protected_base: int
    ahrf_added: int
    complete_system: int
    system_cap: int
    headroom: int
    components: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class AHRFRollout:
    """Terminal semantic state and internal recurrence diagnostics."""

    terminal_facts: torch.Tensor
    terminal_readout: torch.Tensor
    terminal_membrane: torch.Tensor
    terminal_evidence: torch.Tensor
    halt_step: torch.Tensor
    learned_halted: torch.Tensor
    safety_exhausted: torch.Tensor
    halt_logits: torch.Tensor
    halt_probabilities: torch.Tensor
    write_probabilities: torch.Tensor
    fact_history: torch.Tensor | None
    membrane_history: torch.Tensor | None
    evidence_history: torch.Tensor | None
    halted_history: torch.Tensor | None


def _is_binary(value: torch.Tensor) -> bool:
    return bool(value.detach().eq(value.detach().round()).all())


class _SigmoidStraightThroughEvent(torch.autograd.Function):
    """Exact binary forward with an explicitly saved sigmoid derivative."""

    @staticmethod
    def forward(
        context: object,
        logit: torch.Tensor,
    ) -> torch.Tensor:
        probability = logit.sigmoid()
        context.save_for_backward(probability)
        return logit.ge(0).to(logit.dtype)

    @staticmethod
    def backward(
        context: object,
        output_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        (probability,) = context.saved_tensors
        return (
            output_gradient * probability * (1.0 - probability),
        )


def _straight_through_event(logit: torch.Tensor) -> torch.Tensor:
    return _SigmoidStraightThroughEvent.apply(logit)


def _masked_max(
    value: torch.Tensor,
    mask: torch.Tensor,
    dimensions: tuple[int, ...],
) -> torch.Tensor:
    minimum = torch.finfo(value.dtype).min
    return value.masked_fill(~mask, minimum).amax(dim=dimensions)


class _ObjectPairRound(nn.Module):
    """Object-equivariant row, column, transpose, and triadic messages."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.triad_left = nn.Linear(width, width, bias=False)
        self.triad_right = nn.Linear(width, width, bias=False)
        self.update = nn.Sequential(
            nn.Linear(5 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(
        self,
        state: torch.Tensor,
        full_mask: torch.Tensor,
        object_count: torch.Tensor,
    ) -> torch.Tensor:
        weight = full_mask.to(state.dtype)
        denominator = object_count[:, None, None, None, None, None]
        row = (state * weight).sum(4, keepdim=True) / denominator
        column = (state * weight).sum(3, keepdim=True) / denominator
        row = row.expand_as(state)
        column = column.expand_as(state)
        left = self.triad_left(state)
        right = self.triad_right(state)
        triad = torch.einsum(
            "bswikd,bswkjd->bswijd",
            left,
            right,
        ) / denominator
        proposal = self.update(
            torch.cat(
                (state, row, column, state.transpose(3, 4), triad),
                dim=-1,
            )
        )
        return self.norm(state + proposal) * weight


class _OpaqueCardFieldEncoder(nn.Module):
    """Encode witness cards without evaluating named primitive candidates."""

    def __init__(self, width: int, rounds: int) -> None:
        super().__init__()
        self.pair_input = nn.Linear(10, width)
        self.pair_rounds = nn.ModuleList(
            _ObjectPairRound(width) for _ in range(rounds)
        )
        self.witness_encoder = nn.Sequential(
            nn.Linear(2 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
            nn.LayerNorm(width),
        )
        self.slot_encoder = nn.Sequential(
            nn.Linear(2 * width + 3, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
            nn.LayerNorm(width),
        )
        self.pair_output = nn.Linear(width, width)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        output: torch.Tensor,
        witness_mask: torch.Tensor,
        argument_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, slots, witnesses, objects, _ = left.shape
        diagonal = torch.eye(
            objects,
            dtype=left.dtype,
            device=left.device,
        )[None, None, None].expand(batch, slots, witnesses, -1, -1)
        arity = argument_mask.long().sum(-1)
        arity_features = torch.nn.functional.one_hot(
            arity,
            3,
        ).to(left.dtype)
        card_arity = torch.where(
            witness_mask,
            arity,
            torch.full_like(arity, -1),
        ).amax(2).clamp_min(0)
        card_arity_features = torch.nn.functional.one_hot(
            card_arity,
            3,
        ).to(left.dtype)
        arity_pair = arity_features[
            ...,
            None,
            None,
            :,
        ].expand(-1, -1, -1, objects, objects, -1)
        pair_features = torch.cat(
            (
                left[..., None],
                right[..., None],
                output[..., None],
                left.transpose(-1, -2)[..., None],
                right.transpose(-1, -2)[..., None],
                output.transpose(-1, -2)[..., None],
                diagonal[..., None],
                arity_pair,
            ),
            dim=-1,
        )
        pair_mask = object_mask[:, :, None] & object_mask[:, None, :]
        full_mask = (
            witness_mask[..., None, None, None]
            & pair_mask[:, None, None, :, :, None]
        )
        state = self.pair_input(pair_features) * full_mask.to(left.dtype)
        object_count = object_mask.sum(-1).clamp_min(1).to(left.dtype)
        for pair_round in self.pair_rounds:
            state = pair_round(state, full_mask, object_count)

        pair_weight = full_mask.to(left.dtype)
        pair_count = pair_weight.sum((3, 4)).clamp_min(1.0)
        pair_mean = (state * pair_weight).sum((3, 4)) / pair_count
        pair_max = _masked_max(state, full_mask, (3, 4))
        pair_max = torch.where(
            witness_mask[..., None],
            pair_max,
            torch.zeros_like(pair_max),
        )
        witness_state = self.witness_encoder(
            torch.cat((pair_mean, pair_max), dim=-1)
        )
        witness_weight = witness_mask[..., None].to(left.dtype)
        witness_count = witness_weight.sum(2).clamp_min(1.0)
        witness_mean = (
            witness_state * witness_weight
        ).sum(2) / witness_count
        witness_max = _masked_max(
            witness_state,
            witness_mask[..., None],
            (2,),
        )
        active_slot = witness_mask.any(-1)
        witness_max = torch.where(
            active_slot[..., None],
            witness_max,
            torch.zeros_like(witness_max),
        )
        slot_state = self.slot_encoder(
            torch.cat(
                (
                    witness_mean,
                    witness_max,
                    card_arity_features,
                ),
                dim=-1,
            )
        )
        slot_state = slot_state * active_slot[..., None].to(left.dtype)

        slot_pair_count = witness_weight[
            ...,
            None,
            None,
        ].sum(2).clamp_min(1.0)
        slot_pair = (
            state * witness_weight[..., None, None]
        ).sum(2) / slot_pair_count
        slot_pair = self.pair_output(slot_pair)
        slot_pair = (
            slot_pair
            * active_slot[..., None, None, None].to(left.dtype)
            * pair_mask[:, None, :, :, None].to(left.dtype)
        )
        return slot_state, slot_pair


class AutocatalyticHystereticRelationField(nn.Module):
    """Learned recurrent relation field with exact absorbing latches.

    The model performs a fixed number of generic neural message-passing
    rounds. It never calls a named relation primitive and never compares two
    states for convergence. Hard fact/evidence events use straight-through
    estimators: their forward values are exact bits and their backward values
    retain local gradients.
    """

    def __init__(
        self,
        *,
        node_feature_dim: int,
        hidden_dim: int = 96,
        card_rounds: int = 2,
        max_steps: int = 16,
        hysteresis: bool = True,
        use_card_conditioning: bool = True,
    ) -> None:
        super().__init__()
        if (
            node_feature_dim < 1
            or hidden_dim < 8
            or card_rounds < 1
            or max_steps < 1
        ):
            raise AHRFError("AHRF geometry differs")
        self.node_feature_dim = int(node_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.card_rounds = int(card_rounds)
        self.max_steps = int(max_steps)
        self.hysteresis = bool(hysteresis)
        self.use_card_conditioning = bool(use_card_conditioning)

        self.card_encoder = _OpaqueCardFieldEncoder(
            self.hidden_dim,
            self.card_rounds,
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.seed_encoder = nn.Linear(1, self.hidden_dim)
        self.static_mixer = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.initial_membrane = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.edge_message = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            for _ in range(GRAPH_EDGE_ROLES)
        )
        drive_width = 3 * self.hidden_dim + 2 + GRAPH_EDGE_ROLES
        self.membrane_gate = nn.Linear(drive_width, self.hidden_dim)
        self.membrane_candidate = nn.Sequential(
            nn.Linear(drive_width, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.membrane_norm = nn.LayerNorm(self.hidden_dim)
        event_width = self.hidden_dim + 2 + GRAPH_EDGE_ROLES
        self.evidence_head = nn.Linear(event_width, 1)
        self.write_head = nn.Linear(
            self.hidden_dim + 3 + GRAPH_EDGE_ROLES,
            1,
        )
        self.halt_head = nn.Linear(self.hidden_dim + 4, 1)
        self.reset_parameters()
        self.parameter_receipt()

    @property
    def write_incoming_offset(self) -> int:
        """First incoming-role column in ``write_head`` for audit fixtures."""

        return self.hidden_dim + 3

    @property
    def added_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_receipt(
        self,
        *,
        protected_base: int = PROTECTED_BASE_PARAMETERS,
        system_cap: int = SYSTEM_PARAMETER_CAP,
    ) -> AHRFParameterReceipt:
        components = tuple(
            (
                name,
                sum(parameter.numel() for parameter in child.parameters()),
            )
            for name, child in self.named_children()
        )
        added = self.added_parameters
        if sum(count for _, count in components) != added:
            raise AHRFError("AHRF component receipt is incomplete")
        complete = int(protected_base) + added
        if complete >= int(system_cap):
            raise AHRFError("AHRF complete system reaches parameter cap")
        return AHRFParameterReceipt(
            protected_base=int(protected_base),
            ahrf_added=added,
            complete_system=complete,
            system_cap=int(system_cap),
            headroom=int(system_cap) - complete,
            components=components,
        )

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate(
        self,
        graph: SourceDeletedRelationGraph,
    ) -> tuple[int, int, int, int, torch.Tensor, torch.Tensor]:
        tensors = tuple(
            getattr(graph, field)
            for field in SourceDeletedRelationGraph.__dataclass_fields__
        )
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise AHRFError("AHRF input is not entirely tensorized")
        device = graph.node_features.device
        if any(value.device != device for value in tensors):
            raise AHRFError("AHRF input spans devices")
        if (
            graph.node_features.ndim != 3
            or graph.node_features.shape[-1] != self.node_feature_dim
            or not graph.node_features.is_floating_point()
        ):
            raise AHRFError("typed node features differ")
        batch, nodes, _ = graph.node_features.shape
        if (
            graph.node_mask.shape != (batch, nodes)
            or graph.node_mask.dtype != torch.bool
            or graph.root_mask.shape != (batch, nodes)
            or graph.root_mask.dtype != torch.bool
        ):
            raise AHRFError("node masks differ")
        if (
            graph.argument_edges.shape
            != (batch, nodes, nodes, GRAPH_EDGE_ROLES)
            or graph.argument_edges.dtype != torch.bool
        ):
            raise AHRFError("typed argument links differ")
        if graph.seed_facts.ndim != 4 or graph.seed_facts.shape[:2] != (
            batch,
            nodes,
        ):
            raise AHRFError("seed fact field differs")
        objects = graph.seed_facts.shape[-1]
        if graph.seed_facts.shape[-2] != objects:
            raise AHRFError("seed fact field is not square")
        if (
            graph.object_mask.shape != (batch, objects)
            or graph.object_mask.dtype != torch.bool
        ):
            raise AHRFError("object mask differs")
        if graph.witness_left.ndim != 5:
            raise AHRFError("card witness geometry differs")
        witness_shape = graph.witness_left.shape
        if (
            graph.witness_right.shape != witness_shape
            or graph.witness_output.shape != witness_shape
            or witness_shape[0] != batch
            or witness_shape[-2:] != (objects, objects)
        ):
            raise AHRFError("card witness geometry differs")
        slots, witnesses = witness_shape[1:3]
        if (
            graph.witness_mask.shape != (batch, slots, witnesses)
            or graph.witness_mask.dtype != torch.bool
            or graph.argument_mask.shape
            != (batch, slots, witnesses, CARD_ARGUMENT_ROLES)
            or graph.argument_mask.dtype != torch.bool
            or graph.node_card_mask.shape != (batch, nodes, slots)
            or graph.node_card_mask.dtype != torch.bool
        ):
            raise AHRFError("card masks differ")
        floating = (
            graph.node_features,
            graph.seed_facts,
            graph.witness_left,
            graph.witness_right,
            graph.witness_output,
        )
        if any(
            not value.is_floating_point()
            or value.dtype != graph.node_features.dtype
            or not bool(torch.isfinite(value.detach()).all())
            for value in floating
        ):
            raise AHRFError("AHRF floating values differ")
        if any(
            float(value.detach().amin()) < 0.0
            or float(value.detach().amax()) > 1.0
            or not _is_binary(value)
            for value in floating
        ):
            raise AHRFError("AHRF typed/relation values are not binary")
        if (
            not bool(graph.node_mask.any(-1).all())
            or not bool(graph.object_mask.any(-1).all())
            or not bool(graph.root_mask.any(-1).all())
            or bool((graph.root_mask & ~graph.node_mask).any())
        ):
            raise AHRFError("active node/object/root support differs")
        if bool(
            graph.node_features.masked_select(
                ~graph.node_mask[..., None]
            ).ne(0).any()
        ):
            raise AHRFError("masked node features contain covert state")
        valid_links = (
            graph.node_mask[:, :, None, None]
            & graph.node_mask[:, None, :, None]
        )
        if bool((graph.argument_edges & ~valid_links).any()):
            raise AHRFError("argument link touches a masked node")
        if bool(graph.argument_edges.sum(2).gt(1).any()):
            raise AHRFError("typed role has multiple children")
        if bool(
            graph.node_card_mask.masked_select(
                ~graph.node_mask[..., None]
            ).any()
            or graph.node_card_mask.sum(-1).gt(1).any()
        ):
            raise AHRFError("node/card attachment differs")
        active_pair = (
            graph.object_mask[:, :, None] & graph.object_mask[:, None, :]
        )
        node_pair = graph.node_mask[..., None, None] & active_pair[:, None]
        if bool(
            graph.seed_facts.masked_select(~node_pair).ne(0).any()
        ):
            raise AHRFError("masked seed facts contain covert state")
        witness_pair = (
            graph.witness_mask[..., None, None]
            & active_pair[:, None, None]
        )
        if any(
            bool(value.masked_select(~witness_pair).ne(0).any())
            for value in (
                graph.witness_left,
                graph.witness_right,
                graph.witness_output,
            )
        ):
            raise AHRFError("masked witnesses contain covert state")
        if bool(
            graph.argument_mask.masked_select(
                ~graph.witness_mask[..., None]
            ).any()
            or (
                graph.argument_mask[..., 1]
                & ~graph.argument_mask[..., 0]
            ).any()
        ):
            raise AHRFError("witness argument masks differ")
        if bool(
            graph.witness_left.masked_select(
                ~graph.argument_mask[..., 0, None, None]
            ).ne(0).any()
            or graph.witness_right.masked_select(
                ~graph.argument_mask[..., 1, None, None]
            ).ne(0).any()
        ):
            raise AHRFError("unused witness arguments contain covert state")
        witness_arity = graph.argument_mask.long().sum(-1)
        minimum = torch.where(
            graph.witness_mask,
            witness_arity,
            torch.full_like(witness_arity, 3),
        ).amin(-1)
        maximum = torch.where(
            graph.witness_mask,
            witness_arity,
            torch.full_like(witness_arity, -1),
        ).amax(-1)
        active_slot = graph.witness_mask.any(-1)
        if bool((active_slot & minimum.ne(maximum)).any()):
            raise AHRFError("opaque card arity changes by witness")
        if bool(
            graph.node_card_mask
            .masked_select(~active_slot[:, None, :])
            .any()
        ):
            raise AHRFError("node references an inactive opaque card")
        if not torch.equal(
            graph.node_card_mask.any(1),
            active_slot,
        ):
            raise AHRFError("opaque card is unused")
        reachable = graph.root_mask.clone()
        for _ in range(nodes):
            child_reachable = (
                graph.argument_edges
                & reachable[:, :, None, None]
            ).any((1, 3))
            reachable = reachable | child_reachable
        if bool((graph.node_mask & ~reachable).any()):
            raise AHRFError("active graph node is disconnected from every root")
        return batch, nodes, slots, objects, active_pair, node_pair

    @staticmethod
    def _gather_child_state(
        state: torch.Tensor,
        role_edge: torch.Tensor,
    ) -> torch.Tensor:
        child = role_edge.long().argmax(-1)
        trailing = state.shape[2:]
        indices = child.reshape(
            child.shape[0],
            child.shape[1],
            *(1 for _ in trailing),
        ).expand(-1, -1, *trailing)
        gathered = state.gather(1, indices)
        has_edge = role_edge.any(-1).reshape(
            role_edge.shape[0],
            role_edge.shape[1],
            *(1 for _ in trailing),
        )
        return gathered * has_edge.to(state.dtype)

    @classmethod
    def _incoming_fields(
        cls,
        facts: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            tuple(
                cls._gather_child_state(
                    facts,
                    edges[..., role],
                )
                for role in range(GRAPH_EDGE_ROLES)
            ),
            dim=-1,
        )

    def _incoming_membrane(
        self,
        membrane: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        total = torch.zeros_like(membrane)
        for role, projection in enumerate(self.edge_message):
            role_edge = edges[..., role]
            pooled = self._gather_child_state(
                membrane,
                role_edge,
            )
            total = total + projection(pooled)
        return total

    @staticmethod
    def _select_batch_state(
        active: torch.Tensor,
        proposal: torch.Tensor,
        previous: torch.Tensor,
    ) -> torch.Tensor:
        shape = (active.shape[0],) + (1,) * (proposal.ndim - 1)
        return torch.where(active.reshape(shape), proposal, previous)

    def _halt_features(
        self,
        facts: torch.Tensor,
        membrane: torch.Tensor,
        evidence: torch.Tensor,
        root_mask: torch.Tensor,
        active_pair: torch.Tensor,
    ) -> torch.Tensor:
        root_pair = root_mask[..., None, None] & active_pair[:, None]
        root_weight = root_pair.to(facts.dtype)
        count = root_weight.sum((1, 2, 3)).clamp_min(1.0)
        fact_mean = (facts * root_weight).sum((1, 2, 3)) / count
        evidence_mean = (evidence * root_weight).sum((1, 2, 3)) / count
        fact_max = _masked_max(facts, root_pair, (1, 2, 3))
        evidence_max = _masked_max(evidence, root_pair, (1, 2, 3))
        membrane_weight = root_weight[..., None]
        membrane_mean = (
            membrane * membrane_weight
        ).sum((1, 2, 3)) / count[:, None]
        return torch.cat(
            (
                fact_mean[:, None],
                fact_max[:, None],
                evidence_mean[:, None],
                evidence_max[:, None],
                membrane_mean,
            ),
            dim=-1,
        )

    def forward(
        self,
        graph: SourceDeletedRelationGraph,
        *,
        hard_events: bool = True,
        enable_halt: bool = True,
        return_history: bool = False,
    ) -> AHRFRollout:
        (
            batch,
            nodes,
            _slots,
            _objects,
            active_pair,
            node_pair,
        ) = self._validate(graph)
        dtype = graph.node_features.dtype
        pair_weight = node_pair[..., None].to(dtype)

        slot_state, slot_pair = self.card_encoder(
            graph.witness_left,
            graph.witness_right,
            graph.witness_output,
            graph.witness_mask,
            graph.argument_mask,
            graph.object_mask,
        )
        if not self.use_card_conditioning:
            slot_state = slot_state * 0.0
            slot_pair = slot_pair * 0.0
        node_card = graph.node_card_mask.to(dtype)
        card_node_state = torch.einsum(
            "bns,bsh->bnh",
            node_card,
            slot_state,
        )
        card_pair_state = torch.einsum(
            "bns,bsijh->bnijh",
            node_card,
            slot_pair,
        )
        typed_node_state = self.node_encoder(graph.node_features)
        typed_node_state = (
            typed_node_state + card_node_state
        )[:, :, None, None].expand(
            -1,
            -1,
            graph.seed_facts.shape[-2],
            graph.seed_facts.shape[-1],
            -1,
        )
        seed_state = self.seed_encoder(graph.seed_facts[..., None])
        static_state = self.static_mixer(
            torch.cat(
                (typed_node_state, card_pair_state, seed_state),
                dim=-1,
            )
        ) * pair_weight

        facts = graph.seed_facts.clone()
        evidence = graph.seed_facts.clone()
        membrane = self.initial_membrane(static_state) * pair_weight
        halted = torch.zeros(batch, dtype=torch.bool, device=facts.device)
        halt_step = torch.full(
            (batch,),
            -1,
            dtype=torch.long,
            device=facts.device,
        )

        fact_history = [facts] if return_history else None
        membrane_history = [membrane] if return_history else None
        evidence_history = [evidence] if return_history else None
        halted_history = [halted] if return_history else None
        halt_logits: list[torch.Tensor] = []
        halt_probabilities: list[torch.Tensor] = []
        write_probabilities: list[torch.Tensor] = []

        for step in range(self.max_steps):
            active_batch = ~halted
            incoming_facts = self._incoming_fields(
                facts,
                graph.argument_edges,
            )
            incoming_membrane = self._incoming_membrane(
                membrane,
                graph.argument_edges,
            )
            drive = torch.cat(
                (
                    membrane,
                    static_state,
                    incoming_membrane,
                    facts[..., None],
                    evidence[..., None],
                    incoming_facts,
                ),
                dim=-1,
            )
            membrane_gate = self.membrane_gate(drive).sigmoid()
            membrane_delta = torch.tanh(
                self.membrane_candidate(drive)
            )
            membrane_proposal = self.membrane_norm(
                membrane + membrane_gate * membrane_delta
            ) * pair_weight

            evidence_input = torch.cat(
                (
                    membrane_proposal,
                    evidence[..., None],
                    facts[..., None],
                    incoming_facts,
                ),
                dim=-1,
            )
            evidence_logit = self.evidence_head(
                evidence_input
            ).squeeze(-1)
            evidence_event = (
                _straight_through_event(evidence_logit)
                if hard_events
                else evidence_logit.sigmoid()
            )
            if self.hysteresis:
                evidence_proposal = (
                    evidence + (1.0 - evidence) * evidence_event
                )
            else:
                evidence_proposal = evidence_event
            evidence_proposal = evidence_proposal * node_pair.to(dtype)

            write_input = torch.cat(
                (
                    membrane_proposal,
                    evidence_proposal[..., None],
                    facts[..., None],
                    graph.seed_facts[..., None],
                    incoming_facts,
                ),
                dim=-1,
            )
            write_logit = self.write_head(write_input).squeeze(-1)
            write_probability = write_logit.sigmoid()
            write_event = (
                _straight_through_event(write_logit)
                if hard_events
                else write_probability
            )
            if self.hysteresis:
                fact_proposal = facts + (1.0 - facts) * write_event
            else:
                fact_proposal = torch.maximum(
                    graph.seed_facts,
                    write_event,
                )
            fact_proposal = fact_proposal * node_pair.to(dtype)

            membrane = self._select_batch_state(
                active_batch,
                membrane_proposal,
                membrane,
            )
            evidence = self._select_batch_state(
                active_batch,
                evidence_proposal,
                evidence,
            )
            facts = self._select_batch_state(
                active_batch,
                fact_proposal,
                facts,
            )
            halt_feature = self._halt_features(
                facts,
                membrane,
                evidence,
                graph.root_mask,
                active_pair,
            )
            halt_logit = self.halt_head(halt_feature).squeeze(-1)
            model_event = (
                halt_logit.ge(0)
                if hard_events and enable_halt
                else torch.zeros_like(active_batch)
            )
            newly_halted = active_batch & model_event
            halt_step = torch.where(
                newly_halted,
                torch.full_like(halt_step, step + 1),
                halt_step,
            )
            halted = halted | newly_halted

            halt_logits.append(halt_logit)
            halt_probabilities.append(halt_logit.sigmoid())
            write_probabilities.append(write_probability)
            if return_history:
                assert fact_history is not None
                assert membrane_history is not None
                assert evidence_history is not None
                assert halted_history is not None
                fact_history.append(facts)
                membrane_history.append(membrane)
                evidence_history.append(evidence)
                halted_history.append(halted)

        root_weight = graph.root_mask[..., None, None].to(dtype)
        terminal_readout = (
            facts * root_weight
        ).amax(1) * active_pair.to(dtype)
        return AHRFRollout(
            terminal_facts=facts,
            terminal_readout=terminal_readout,
            terminal_membrane=membrane,
            terminal_evidence=evidence,
            halt_step=halt_step,
            learned_halted=halted,
            safety_exhausted=~halted,
            halt_logits=torch.stack(halt_logits, dim=1),
            halt_probabilities=torch.stack(halt_probabilities, dim=1),
            write_probabilities=torch.stack(write_probabilities, dim=1),
            fact_history=(
                torch.stack(fact_history, dim=1)
                if fact_history is not None
                else None
            ),
            membrane_history=(
                torch.stack(membrane_history, dim=1)
                if membrane_history is not None
                else None
            ),
            evidence_history=(
                torch.stack(evidence_history, dim=1)
                if evidence_history is not None
                else None
            ),
            halted_history=(
                torch.stack(halted_history, dim=1)
                if halted_history is not None
                else None
            ),
        )
