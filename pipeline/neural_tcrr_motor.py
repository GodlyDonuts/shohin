"""Permutation-equivariant one-step motor for neural TCRR packets.

The module is intentionally limited to the model-visible packet tensor
boundary. It emits a factorized proposal and structural masks; it does not
execute rewrites or reconstruct symbolic records.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import sqrt

import torch
from tensorize_neural_tcrr_packets import NeuralTcrrPacketTensors
from torch import Tensor, nn


MASKED_LOGIT = -10_000.0
NODE_KEEP = 0
NODE_WRITE = 1
NODE_CLEAR = 2
NODE_OPERATION_COUNT = 3
GRAPH_KIND_COUNT = 3
TERM_KIND_COUNT = 2


class NeuralTcrrMotorError(ValueError):
    """Raised when a packet batch violates the motor tensor contract."""


@dataclass(frozen=True)
class NeuralTcrrMotorConfig:
    """Frozen architecture choices independent of local entity counts."""

    hidden_dim: int = 128
    entity_rounds: int = 2
    term_rounds: int = 2
    graph_rounds: int = 3
    max_arity: int = 3
    path_depth: int = 8
    parameter_cap: int = 16_000_000

    def __post_init__(self) -> None:
        for name in (
            "hidden_dim",
            "entity_rounds",
            "term_rounds",
            "graph_rounds",
            "max_arity",
            "path_depth",
            "parameter_cap",
        ):
            if getattr(self, name) <= 0:
                raise NeuralTcrrMotorError(f"{name} must be positive")


@dataclass(frozen=True)
class MotorParameterCount:
    """Auditable parameter budget for the standalone motor."""

    total: int
    trainable: int
    cap: int

    @property
    def under_cap(self) -> bool:
        return self.total < self.cap


@dataclass(frozen=True)
class NeuralTcrrGraphDelta:
    """Index-free graph mutation proposal with explicit validity masks."""

    node_operation_logits: Tensor
    node_operation_mask: Tensor
    root_pointer_logits: Tensor
    root_pointer_mask: Tensor
    node_kind_logits: Tensor
    node_kind_mask: Tensor
    node_type_pointer_logits: Tensor
    node_type_pointer_mask: Tensor
    node_constructor_pointer_logits: Tensor
    node_constructor_pointer_mask: Tensor
    node_variable_pointer_logits: Tensor
    node_variable_pointer_mask: Tensor
    child_pointer_logits: Tensor
    child_pointer_mask: Tensor
    child_presence_logits: Tensor
    child_presence_mask: Tensor


@dataclass(frozen=True)
class NeuralTcrrMotorOutput:
    """Factorized one-step proposal over anonymous local coordinates."""

    no_redex_logits: Tensor
    halt_logits: Tensor
    rule_logits: Tensor
    rule_mask: Tensor
    path_logits: Tensor
    path_mask: Tensor
    binding_logits: Tensor
    binding_mask: Tensor
    graph_delta: NeuralTcrrGraphDelta


class _SharedMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


def _masked_mean(value: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(value.dtype).unsqueeze(-1)
    numerator = (value * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


def _masked_logits(value: Tensor, mask: Tensor) -> Tensor:
    return torch.where(mask, value, torch.full_like(value, MASKED_LOGIT))


def _masked_softmax(value: Tensor, mask: Tensor, dim: int) -> Tensor:
    probabilities = torch.softmax(_masked_logits(value, mask), dim=dim)
    probabilities = probabilities * mask.to(value.dtype)
    denominator = probabilities.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return probabilities / denominator


class NeuralTcrrMotor(nn.Module):
    """Compact relational motor over one source-deleted local board."""

    def __init__(self, config: NeuralTcrrMotorConfig | None = None) -> None:
        super().__init__()
        self.config = config or NeuralTcrrMotorConfig()
        hidden = self.config.hidden_dim
        arity = self.config.max_arity

        self.type_seed = nn.Parameter(torch.empty(hidden))
        self.variable_seed = nn.Parameter(torch.empty(hidden))
        self.argument_embedding = nn.Parameter(torch.empty(arity, hidden))
        self.depth_embedding = nn.Parameter(
            torch.empty(self.config.path_depth + 1, hidden)
        )
        self.side_embedding = nn.Parameter(torch.empty(2, hidden))
        self.binding_null = nn.Parameter(torch.empty(hidden))
        self.root_null = nn.Parameter(torch.empty(hidden))
        self.child_null = nn.Parameter(torch.empty(hidden))

        self.constructor_argument = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(arity)
        )
        self.constructor_input = _SharedMlp(2 * hidden + 1, hidden, hidden)

        self.term_kind = nn.Linear(TERM_KIND_COUNT, hidden, bias=False)
        self.term_input = _SharedMlp(4 * hidden + 1, hidden, hidden)
        self.term_child = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(arity)
        )
        self.term_parent = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(arity)
        )
        self.term_update = _SharedMlp(3 * hidden, hidden, hidden)
        self.term_norm = nn.LayerNorm(hidden)

        self.type_constructor_result = nn.Linear(hidden, hidden, bias=False)
        self.type_constructor_argument = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(arity)
        )
        self.type_term = nn.Linear(hidden, hidden, bias=False)
        self.type_update = _SharedMlp(2 * hidden, hidden, hidden)
        self.type_norm = nn.LayerNorm(hidden)

        self.variable_term = nn.Linear(hidden, hidden, bias=False)
        self.variable_update = _SharedMlp(2 * hidden, hidden, hidden)
        self.variable_norm = nn.LayerNorm(hidden)

        self.rule_input = _SharedMlp(2 * hidden + 1, hidden, hidden)

        self.graph_kind = nn.Linear(GRAPH_KIND_COUNT, hidden, bias=False)
        self.graph_child_type = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(arity)
        )
        self.graph_input = _SharedMlp(5 * hidden + 3, hidden, hidden)
        self.graph_child = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(arity)
        )
        self.graph_parent = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(arity)
        )
        self.graph_update = _SharedMlp(3 * hidden + 1, hidden, hidden)
        self.graph_norm = nn.LayerNorm(hidden)

        self.global_input = _SharedMlp(6 * hidden, hidden, hidden)
        self.rule_context = nn.Linear(hidden, hidden, bias=False)
        self.rule_score = nn.Linear(hidden, 1)
        self.no_redex_head = _SharedMlp(hidden, hidden, 2)
        self.halt_head = _SharedMlp(hidden, hidden, 2)

        self.path_initial = _SharedMlp(3 * hidden, hidden, hidden)
        self.path_argument = _SharedMlp(4 * hidden, hidden, 1)
        self.path_stop = _SharedMlp(2 * hidden, hidden, 1)
        self.path_step = _SharedMlp(5 * hidden, hidden, hidden)
        self.path_recurrence = nn.GRUCell(hidden, hidden)

        self.binding_query = _SharedMlp(3 * hidden, hidden, hidden)
        self.binding_key = nn.Linear(hidden, hidden, bias=False)

        self.transaction_context = _SharedMlp(3 * hidden, hidden, hidden)
        self.transaction_node = _SharedMlp(2 * hidden, hidden, hidden)
        self.node_operation = nn.Linear(hidden, NODE_OPERATION_COUNT)
        self.root_query = nn.Linear(hidden, hidden, bias=False)
        self.root_key = nn.Linear(hidden, hidden, bias=False)
        self.node_kind = nn.Linear(hidden, GRAPH_KIND_COUNT)
        self.node_type_query = nn.Linear(hidden, hidden, bias=False)
        self.node_type_key = nn.Linear(hidden, hidden, bias=False)
        self.node_constructor_query = nn.Linear(hidden, hidden, bias=False)
        self.node_constructor_key = nn.Linear(hidden, hidden, bias=False)
        self.node_variable_query = nn.Linear(hidden, hidden, bias=False)
        self.node_variable_key = nn.Linear(hidden, hidden, bias=False)
        self.child_query = _SharedMlp(2 * hidden, hidden, hidden)
        self.child_key = nn.Linear(hidden, hidden, bias=False)
        self.child_presence = nn.Linear(hidden, 2)

        self._reset_parameters()
        count = self.parameter_count()
        if not count.under_cap:
            raise NeuralTcrrMotorError(
                f"motor has {count.total} parameters, cap is {count.cap}"
            )

    def _reset_parameters(self) -> None:
        for value in (
            self.type_seed,
            self.variable_seed,
            self.argument_embedding,
            self.depth_embedding,
            self.side_embedding,
            self.binding_null,
            self.root_null,
            self.child_null,
        ):
            nn.init.normal_(value, mean=0.0, std=0.02)

    def parameter_count(self) -> MotorParameterCount:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return MotorParameterCount(
            total=total,
            trainable=trainable,
            cap=self.config.parameter_cap,
        )

    def _validate_packets(
        self,
        packets: NeuralTcrrPacketTensors,
    ) -> tuple[int, int, int, int, int, int, int]:
        if not isinstance(packets, NeuralTcrrPacketTensors):
            raise NeuralTcrrMotorError(
                "forward requires NeuralTcrrPacketTensors directly"
            )
        batch_size = packets.constructor_active.shape[0]
        device = packets.constructor_active.device
        for field in fields(NeuralTcrrPacketTensors):
            value = getattr(packets, field.name)
            if not isinstance(value, Tensor):
                raise NeuralTcrrMotorError(f"{field.name} is not a tensor")
            if value.dtype is not torch.bool:
                raise NeuralTcrrMotorError(f"{field.name} must be boolean")
            if value.device != device:
                raise NeuralTcrrMotorError("all packet tensors must share a device")
            if value.shape[0] != batch_size:
                raise NeuralTcrrMotorError("all packet tensors must share a batch")

        constructors = packets.constructor_active.shape[1]
        types = packets.type_active.shape[1]
        rules = packets.rule_active.shape[1]
        variables = packets.variable_active.shape[1]
        storage = packets.storage_active.shape[1]
        side_nodes = packets.lhs_active.shape[2]
        arity = packets.constructor_argument_mask.shape[2]
        if arity != self.config.max_arity:
            raise NeuralTcrrMotorError(
                f"packet arity {arity} != configured arity {self.config.max_arity}"
            )
        expected = {
            "constructor_result_type": (batch_size, constructors, types),
            "constructor_argument_type": (
                batch_size,
                constructors,
                arity,
                types,
            ),
            "rule_delete": (batch_size, rules),
            "graph_root": (batch_size, storage + 1),
            "graph_kind": (batch_size, storage, GRAPH_KIND_COUNT),
            "graph_type": (batch_size, storage, types),
            "graph_constructor": (batch_size, storage, constructors),
            "graph_variable": (batch_size, storage, variables),
            "graph_children": (
                batch_size,
                storage,
                arity,
                storage + 1,
            ),
            "lhs_kind": (
                batch_size,
                rules,
                side_nodes,
                TERM_KIND_COUNT,
            ),
            "lhs_type": (batch_size, rules, side_nodes, types),
            "lhs_constructor": (
                batch_size,
                rules,
                side_nodes,
                constructors,
            ),
            "lhs_variable": (batch_size, rules, side_nodes, variables),
            "lhs_parent_child": (
                batch_size,
                rules,
                side_nodes,
                arity,
                side_nodes,
            ),
            "rhs_kind": (
                batch_size,
                rules,
                side_nodes,
                TERM_KIND_COUNT,
            ),
            "rhs_type": (batch_size, rules, side_nodes, types),
            "rhs_constructor": (
                batch_size,
                rules,
                side_nodes,
                constructors,
            ),
            "rhs_variable": (batch_size, rules, side_nodes, variables),
            "rhs_parent_child": (
                batch_size,
                rules,
                side_nodes,
                arity,
                side_nodes,
            ),
        }
        for name, shape in expected.items():
            if getattr(packets, name).shape != shape:
                raise NeuralTcrrMotorError(
                    f"{name} shape {tuple(getattr(packets, name).shape)} != {shape}"
                )
        if torch.any(packets.graph_active & ~packets.graph_capacity):
            raise NeuralTcrrMotorError("active graph nodes must be inside capacity")
        if torch.any(packets.graph_root.sum(dim=-1) != 1):
            raise NeuralTcrrMotorError("graph root must be one-hot including null")
        return (
            batch_size,
            constructors,
            types,
            rules,
            variables,
            storage,
            side_nodes,
        )

    def _constructor_states(
        self,
        packets: NeuralTcrrPacketTensors,
        type_state: Tensor,
    ) -> Tensor:
        result = torch.einsum(
            "bcy,byh->bch",
            packets.constructor_result_type.to(type_state.dtype),
            type_state,
        )
        arguments = torch.einsum(
            "bcay,byh->bcah",
            packets.constructor_argument_type.to(type_state.dtype),
            type_state,
        )
        argument_message = torch.zeros_like(result)
        for argument, layer in enumerate(self.constructor_argument):
            mask = packets.constructor_argument_mask[:, :, argument].unsqueeze(-1)
            argument_message = (
                argument_message + layer(arguments[:, :, argument]) * mask
            )
        active = packets.constructor_active.unsqueeze(-1)
        state = self.constructor_input(
            torch.cat(
                (result, argument_message, active.to(result.dtype)),
                dim=-1,
            )
        )
        return state * active

    def _term_states(
        self,
        *,
        active: Tensor,
        kind: Tensor,
        type_ref: Tensor,
        constructor_ref: Tensor,
        variable_ref: Tensor,
        parent_child: Tensor,
        type_state: Tensor,
        constructor_state: Tensor,
        variable_state: Tensor,
        side: int,
    ) -> Tensor:
        kind_state = self.term_kind(kind.to(type_state.dtype))
        type_context = torch.einsum(
            "brpy,byh->brph",
            type_ref.to(type_state.dtype),
            type_state,
        )
        constructor_context = torch.einsum(
            "brpc,bch->brph",
            constructor_ref.to(type_state.dtype),
            constructor_state,
        )
        variable_context = torch.einsum(
            "brpv,bvh->brph",
            variable_ref.to(type_state.dtype),
            variable_state,
        )
        active_float = active.unsqueeze(-1).to(type_state.dtype)
        state = self.term_input(
            torch.cat(
                (
                    kind_state,
                    type_context,
                    constructor_context,
                    variable_context,
                    active_float,
                ),
                dim=-1,
            )
        )
        state = (state + self.side_embedding[side]) * active_float
        for _ in range(self.config.term_rounds):
            child_message = torch.zeros_like(state)
            parent_message = torch.zeros_like(state)
            for argument in range(self.config.max_arity):
                relation = parent_child[:, :, :, argument, :].to(state.dtype)
                child = torch.einsum("brpq,brqh->brph", relation, state)
                parent = torch.einsum("brpq,brph->brqh", relation, state)
                child_message = child_message + self.term_child[argument](child)
                parent_message = parent_message + self.term_parent[argument](parent)
            update = self.term_update(
                torch.cat((state, child_message, parent_message), dim=-1)
            )
            state = self.term_norm(state + update) * active_float
        return state

    def _entity_states(
        self,
        packets: NeuralTcrrPacketTensors,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        dtype = self.type_seed.dtype
        type_mask = packets.type_active.unsqueeze(-1)
        variable_mask = packets.variable_active.unsqueeze(-1)
        type_state = self.type_seed.view(1, 1, -1) * type_mask
        variable_state = self.variable_seed.view(1, 1, -1) * variable_mask
        constructor_state = self._constructor_states(packets, type_state)

        for _ in range(self.config.entity_rounds):
            lhs = self._term_states(
                active=packets.lhs_active,
                kind=packets.lhs_kind,
                type_ref=packets.lhs_type,
                constructor_ref=packets.lhs_constructor,
                variable_ref=packets.lhs_variable,
                parent_child=packets.lhs_parent_child,
                type_state=type_state,
                constructor_state=constructor_state,
                variable_state=variable_state,
                side=0,
            )
            rhs = self._term_states(
                active=packets.rhs_active,
                kind=packets.rhs_kind,
                type_ref=packets.rhs_type,
                constructor_ref=packets.rhs_constructor,
                variable_ref=packets.rhs_variable,
                parent_child=packets.rhs_parent_child,
                type_state=type_state,
                constructor_state=constructor_state,
                variable_state=variable_state,
                side=1,
            )

            type_message = self.type_constructor_result(
                torch.einsum(
                    "bcy,bch->byh",
                    packets.constructor_result_type.to(dtype),
                    constructor_state,
                )
            )
            for argument, layer in enumerate(self.type_constructor_argument):
                type_message = type_message + layer(
                    torch.einsum(
                        "bcy,bch->byh",
                        packets.constructor_argument_type[:, :, argument, :].to(dtype),
                        constructor_state,
                    )
                )
            type_message = type_message + self.type_term(
                torch.einsum(
                    "brpy,brph->byh",
                    packets.lhs_type.to(dtype),
                    lhs,
                )
                + torch.einsum(
                    "brpy,brph->byh",
                    packets.rhs_type.to(dtype),
                    rhs,
                )
            )
            type_state = (
                self.type_norm(
                    type_state
                    + self.type_update(torch.cat((type_state, type_message), dim=-1))
                )
                * type_mask
            )

            variable_message = self.variable_term(
                torch.einsum(
                    "brpv,brph->bvh",
                    packets.lhs_variable.to(dtype),
                    lhs,
                )
                + torch.einsum(
                    "brpv,brph->bvh",
                    packets.rhs_variable.to(dtype),
                    rhs,
                )
            )
            variable_state = (
                self.variable_norm(
                    variable_state
                    + self.variable_update(
                        torch.cat((variable_state, variable_message), dim=-1)
                    )
                )
                * variable_mask
            )
            constructor_state = self._constructor_states(packets, type_state)

        lhs = self._term_states(
            active=packets.lhs_active,
            kind=packets.lhs_kind,
            type_ref=packets.lhs_type,
            constructor_ref=packets.lhs_constructor,
            variable_ref=packets.lhs_variable,
            parent_child=packets.lhs_parent_child,
            type_state=type_state,
            constructor_state=constructor_state,
            variable_state=variable_state,
            side=0,
        )
        rhs = self._term_states(
            active=packets.rhs_active,
            kind=packets.rhs_kind,
            type_ref=packets.rhs_type,
            constructor_ref=packets.rhs_constructor,
            variable_ref=packets.rhs_variable,
            parent_child=packets.rhs_parent_child,
            type_state=type_state,
            constructor_state=constructor_state,
            variable_state=variable_state,
            side=1,
        )
        return type_state, constructor_state, variable_state, lhs, rhs

    def _rule_states(
        self,
        packets: NeuralTcrrPacketTensors,
        lhs: Tensor,
        rhs: Tensor,
    ) -> Tensor:
        lhs_summary = _masked_mean(lhs, packets.lhs_active, dim=2)
        rhs_summary = _masked_mean(rhs, packets.rhs_active, dim=2)
        delete = packets.rule_delete.unsqueeze(-1).to(lhs.dtype)
        state = self.rule_input(torch.cat((lhs_summary, rhs_summary, delete), dim=-1))
        return state * packets.rule_active.unsqueeze(-1)

    def _graph_states(
        self,
        packets: NeuralTcrrPacketTensors,
        type_state: Tensor,
        constructor_state: Tensor,
        variable_state: Tensor,
    ) -> Tensor:
        dtype = type_state.dtype
        kind = self.graph_kind(packets.graph_kind.to(dtype))
        type_context = torch.einsum(
            "bny,byh->bnh",
            packets.graph_type.to(dtype),
            type_state,
        )
        constructor_context = torch.einsum(
            "bnc,bch->bnh",
            packets.graph_constructor.to(dtype),
            constructor_state,
        )
        variable_context = torch.einsum(
            "bnv,bvh->bnh",
            packets.graph_variable.to(dtype),
            variable_state,
        )
        child_type_context = torch.einsum(
            "bnay,byh->bnah",
            packets.graph_child_type.to(dtype),
            type_state,
        )
        child_type_message = torch.zeros_like(type_context)
        for argument, layer in enumerate(self.graph_child_type):
            mask = packets.graph_child_mask[:, :, argument].unsqueeze(-1)
            child_type_message = (
                child_type_message + layer(child_type_context[:, :, argument]) * mask
            )
        root_flag = packets.graph_root[:, :-1].unsqueeze(-1).to(dtype)
        scalar = torch.stack(
            (
                packets.graph_active.to(dtype),
                packets.graph_capacity.to(dtype),
                packets.graph_root[:, :-1].to(dtype),
            ),
            dim=-1,
        )
        state = self.graph_input(
            torch.cat(
                (
                    kind,
                    type_context,
                    constructor_context,
                    variable_context,
                    child_type_message,
                    scalar,
                ),
                dim=-1,
            )
        )
        capacity = packets.graph_capacity.unsqueeze(-1)
        state = state * capacity
        storage = state.shape[1]
        for _ in range(self.config.graph_rounds):
            child_message = torch.zeros_like(state)
            parent_message = torch.zeros_like(state)
            for argument in range(self.config.max_arity):
                relation = packets.graph_children[:, :, argument, :storage].to(dtype)
                child = torch.einsum("bnm,bmh->bnh", relation, state)
                parent = torch.einsum("bnm,bnh->bmh", relation, state)
                child_message = child_message + self.graph_child[argument](child)
                parent_message = parent_message + self.graph_parent[argument](parent)
            update = self.graph_update(
                torch.cat((state, child_message, parent_message, root_flag), dim=-1)
            )
            state = self.graph_norm(state + update) * capacity
        return state

    def _global_state(
        self,
        packets: NeuralTcrrPacketTensors,
        type_state: Tensor,
        constructor_state: Tensor,
        variable_state: Tensor,
        rule_state: Tensor,
        node_state: Tensor,
    ) -> Tensor:
        root = torch.einsum(
            "bn,bnh->bh",
            packets.graph_root[:, :-1].to(node_state.dtype),
            node_state,
        )
        return self.global_input(
            torch.cat(
                (
                    _masked_mean(
                        type_state,
                        packets.type_active,
                        dim=1,
                    ),
                    _masked_mean(
                        constructor_state,
                        packets.constructor_active,
                        dim=1,
                    ),
                    _masked_mean(
                        variable_state,
                        packets.variable_active,
                        dim=1,
                    ),
                    _masked_mean(
                        rule_state,
                        packets.rule_active,
                        dim=1,
                    ),
                    _masked_mean(
                        node_state,
                        packets.graph_capacity,
                        dim=1,
                    ),
                    root,
                ),
                dim=-1,
            )
        )

    def _path_head(
        self,
        packets: NeuralTcrrPacketTensors,
        node_state: Tensor,
        global_state: Tensor,
        selected_rule: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dtype = node_state.dtype
        storage = node_state.shape[1]
        adjacency = packets.graph_children[:, :, :, :storage].to(dtype)
        position = packets.graph_root[:, :storage].to(dtype) * packets.graph_active.to(
            dtype
        )
        root_context = torch.einsum("bn,bnh->bh", position, node_state)
        state = self.path_initial(
            torch.cat((global_state, selected_rule, root_context), dim=-1)
        )
        logits: list[Tensor] = []
        masks: list[Tensor] = []
        for depth in range(self.config.path_depth + 1):
            current = torch.einsum("bn,bnh->bh", position, node_state)
            next_by_argument = torch.einsum(
                "bi,biak,bkh->bah",
                position,
                adjacency,
                node_state,
            )
            reachable_mass = torch.einsum(
                "bi,biak->ba",
                position,
                adjacency,
            )
            argument_state = torch.cat(
                (
                    state[:, None, :].expand(-1, self.config.max_arity, -1),
                    current[:, None, :].expand(-1, self.config.max_arity, -1),
                    next_by_argument,
                    (self.argument_embedding + self.depth_embedding[depth].unsqueeze(0))
                    .unsqueeze(0)
                    .expand(state.shape[0], -1, -1),
                ),
                dim=-1,
            )
            argument_logits = self.path_argument(argument_state).squeeze(-1)
            argument_mask = reachable_mass > 0
            if depth == self.config.path_depth:
                argument_mask = torch.zeros_like(argument_mask)
            stop_logit = self.path_stop(torch.cat((state, current), dim=-1)).squeeze(-1)
            step_mask = torch.cat(
                (
                    argument_mask,
                    torch.ones(
                        (state.shape[0], 1),
                        dtype=torch.bool,
                        device=state.device,
                    ),
                ),
                dim=-1,
            )
            step_logits = torch.cat(
                (argument_logits, stop_logit.unsqueeze(-1)),
                dim=-1,
            )
            logits.append(_masked_logits(step_logits, step_mask))
            masks.append(step_mask)

            argument_probabilities = _masked_softmax(
                argument_logits,
                argument_mask,
                dim=-1,
            )
            next_position = torch.einsum(
                "ba,bi,biak->bk",
                argument_probabilities,
                position,
                adjacency,
            )
            position = next_position / next_position.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0)
            chosen_child = torch.einsum(
                "ba,bah->bh",
                argument_probabilities,
                next_by_argument,
            )
            step_input = self.path_step(
                torch.cat(
                    (
                        chosen_child,
                        current,
                        global_state,
                        selected_rule,
                        self.depth_embedding[depth]
                        .unsqueeze(0)
                        .expand(state.shape[0], -1),
                    ),
                    dim=-1,
                )
            )
            state = self.path_recurrence(step_input, state)
        return torch.stack(logits, dim=1), torch.stack(masks, dim=1), state

    def _binding_head(
        self,
        packets: NeuralTcrrPacketTensors,
        variable_state: Tensor,
        rule_state: Tensor,
        node_state: Tensor,
        global_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden = node_state.shape[-1]
        query = self.binding_query(
            torch.cat(
                (
                    variable_state[:, None, :, :].expand(
                        -1,
                        rule_state.shape[1],
                        -1,
                        -1,
                    ),
                    rule_state[:, :, None, :].expand(
                        -1,
                        -1,
                        variable_state.shape[1],
                        -1,
                    ),
                    global_state[:, None, None, :].expand(
                        -1,
                        rule_state.shape[1],
                        variable_state.shape[1],
                        -1,
                    ),
                ),
                dim=-1,
            )
        )
        key = self.binding_key(node_state)
        node_logits = torch.einsum("brvh,bnh->brvn", query, key) / sqrt(hidden)
        null_logit = torch.einsum(
            "brvh,h->brv",
            query,
            self.binding_null,
        ) / sqrt(hidden)
        logits = torch.cat((node_logits, null_logit.unsqueeze(-1)), dim=-1)

        required = (
            packets.lhs_variable.any(dim=2)
            & packets.rule_active[:, :, None]
            & packets.variable_active[:, None, :]
        )
        node_mask = required.unsqueeze(-1) & packets.graph_active[:, None, None, :]
        null_mask = ~required
        mask = torch.cat((node_mask, null_mask.unsqueeze(-1)), dim=-1)
        return _masked_logits(logits, mask), mask

    def _graph_delta(
        self,
        packets: NeuralTcrrPacketTensors,
        type_state: Tensor,
        constructor_state: Tensor,
        variable_state: Tensor,
        node_state: Tensor,
        global_state: Tensor,
        selected_rule: Tensor,
        path_state: Tensor,
    ) -> NeuralTcrrGraphDelta:
        hidden = node_state.shape[-1]
        batch_size, storage, _ = node_state.shape
        context = self.transaction_context(
            torch.cat((global_state, selected_rule, path_state), dim=-1)
        )
        node = self.transaction_node(
            torch.cat(
                (
                    node_state,
                    context[:, None, :].expand(-1, storage, -1),
                ),
                dim=-1,
            )
        )
        capacity = packets.graph_capacity

        operation_logits = self.node_operation(node)
        operation_mask = torch.stack(
            (
                torch.ones_like(capacity),
                capacity,
                capacity,
            ),
            dim=-1,
        )

        root_query = self.root_query(context)
        root_key = self.root_key(node_state)
        root_nodes = torch.einsum("bh,bnh->bn", root_query, root_key) / sqrt(hidden)
        root_null = torch.einsum("bh,h->b", root_query, self.root_null) / sqrt(hidden)
        root_logits = torch.cat((root_nodes, root_null.unsqueeze(-1)), dim=-1)
        root_mask = torch.cat(
            (
                capacity,
                torch.ones(
                    (batch_size, 1),
                    dtype=torch.bool,
                    device=node.device,
                ),
            ),
            dim=-1,
        )

        kind_logits = self.node_kind(node)
        kind_mask = capacity.unsqueeze(-1).expand(-1, -1, GRAPH_KIND_COUNT).clone()
        kind_mask[:, :, -1] |= ~capacity

        type_query = self.node_type_query(node)
        type_key = self.node_type_key(type_state)
        type_logits = torch.einsum("bnh,byh->bny", type_query, type_key) / sqrt(hidden)
        type_mask = capacity[:, :, None] & packets.type_active[:, None, :]

        constructor_query = self.node_constructor_query(node)
        constructor_key = self.node_constructor_key(constructor_state)
        constructor_logits = torch.einsum(
            "bnh,bch->bnc",
            constructor_query,
            constructor_key,
        ) / sqrt(hidden)
        constructor_mask = capacity[:, :, None] & packets.constructor_active[:, None, :]

        variable_query = self.node_variable_query(node)
        variable_key = self.node_variable_key(variable_state)
        variable_logits = torch.einsum(
            "bnh,bvh->bnv",
            variable_query,
            variable_key,
        ) / sqrt(hidden)
        variable_mask = capacity[:, :, None] & packets.variable_active[:, None, :]

        child_query = self.child_query(
            torch.cat(
                (
                    node[:, :, None, :].expand(
                        -1,
                        -1,
                        self.config.max_arity,
                        -1,
                    ),
                    self.argument_embedding[None, None, :, :].expand(
                        batch_size,
                        storage,
                        -1,
                        -1,
                    ),
                ),
                dim=-1,
            )
        )
        child_key = self.child_key(node_state)
        child_nodes = torch.einsum(
            "bnah,bmh->bnam",
            child_query,
            child_key,
        ) / sqrt(hidden)
        child_null = torch.einsum(
            "bnah,h->bna",
            child_query,
            self.child_null,
        ) / sqrt(hidden)
        child_logits = torch.cat((child_nodes, child_null.unsqueeze(-1)), dim=-1)
        child_choices = torch.cat(
            (
                capacity,
                torch.ones(
                    (batch_size, 1),
                    dtype=torch.bool,
                    device=node.device,
                ),
            ),
            dim=-1,
        )
        child_mask = (
            (capacity[:, :, None, None] & child_choices[:, None, None, :])
            .expand(-1, -1, self.config.max_arity, -1)
            .clone()
        )
        child_mask[:, :, :, -1] |= ~capacity[:, :, None]

        presence_logits = self.child_presence(child_query)
        presence_mask = (
            capacity[:, :, None, None]
            .expand(
                -1,
                -1,
                self.config.max_arity,
                2,
            )
            .clone()
        )
        presence_mask[:, :, :, 0] |= ~capacity[:, :, None]

        return NeuralTcrrGraphDelta(
            node_operation_logits=_masked_logits(
                operation_logits,
                operation_mask,
            ),
            node_operation_mask=operation_mask,
            root_pointer_logits=_masked_logits(root_logits, root_mask),
            root_pointer_mask=root_mask,
            node_kind_logits=_masked_logits(kind_logits, kind_mask),
            node_kind_mask=kind_mask,
            node_type_pointer_logits=_masked_logits(type_logits, type_mask),
            node_type_pointer_mask=type_mask,
            node_constructor_pointer_logits=_masked_logits(
                constructor_logits,
                constructor_mask,
            ),
            node_constructor_pointer_mask=constructor_mask,
            node_variable_pointer_logits=_masked_logits(
                variable_logits,
                variable_mask,
            ),
            node_variable_pointer_mask=variable_mask,
            child_pointer_logits=_masked_logits(child_logits, child_mask),
            child_pointer_mask=child_mask,
            child_presence_logits=_masked_logits(
                presence_logits,
                presence_mask,
            ),
            child_presence_mask=presence_mask,
        )

    def forward(
        self,
        packets: NeuralTcrrPacketTensors,
    ) -> NeuralTcrrMotorOutput:
        self._validate_packets(packets)
        (
            type_state,
            constructor_state,
            variable_state,
            lhs,
            rhs,
        ) = self._entity_states(packets)
        rule_state = self._rule_states(packets, lhs, rhs)
        node_state = self._graph_states(
            packets,
            type_state,
            constructor_state,
            variable_state,
        )
        global_state = self._global_state(
            packets,
            type_state,
            constructor_state,
            variable_state,
            rule_state,
            node_state,
        )

        rule_mask = packets.rule_active
        rule_logits = self.rule_score(
            rule_state + self.rule_context(global_state)[:, None, :]
        ).squeeze(-1)
        rule_logits = _masked_logits(rule_logits, rule_mask)
        rule_probabilities = _masked_softmax(rule_logits, rule_mask, dim=-1)
        selected_rule = torch.einsum(
            "br,brh->bh",
            rule_probabilities,
            rule_state,
        )
        path_logits, path_mask, path_state = self._path_head(
            packets,
            node_state,
            global_state,
            selected_rule,
        )
        binding_logits, binding_mask = self._binding_head(
            packets,
            variable_state,
            rule_state,
            node_state,
            global_state,
        )
        graph_delta = self._graph_delta(
            packets,
            type_state,
            constructor_state,
            variable_state,
            node_state,
            global_state,
            selected_rule,
            path_state,
        )
        return NeuralTcrrMotorOutput(
            no_redex_logits=self.no_redex_head(global_state),
            halt_logits=self.halt_head(global_state),
            rule_logits=rule_logits,
            rule_mask=rule_mask,
            path_logits=path_logits,
            path_mask=path_mask,
            binding_logits=binding_logits,
            binding_mask=binding_mask,
            graph_delta=graph_delta,
        )


__all__ = [
    "GRAPH_KIND_COUNT",
    "MASKED_LOGIT",
    "MotorParameterCount",
    "NODE_CLEAR",
    "NODE_KEEP",
    "NODE_OPERATION_COUNT",
    "NODE_WRITE",
    "NeuralTcrrGraphDelta",
    "NeuralTcrrMotor",
    "NeuralTcrrMotorConfig",
    "NeuralTcrrMotorError",
    "NeuralTcrrMotorOutput",
]
