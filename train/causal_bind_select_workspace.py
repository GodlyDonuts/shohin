"""Minimal causal bind-select workspace for Shohin.

This is the first architecture-native EPISODE slice. The workspace consumes
only transformer residuals derived from raw tokens. It receives no spans,
records, action roles, transition tensors, schedules, candidate systems, or
oracle products.

The staged API deliberately separates world compilation from late query
execution. A sealed workspace contains slots plus a token-position scalar, not
source tokens, residuals, KV cache, or compiler scratch state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, _supervised_lm_loss


PROTECTED_BASE_PARAMETERS = 125_081_664
SYSTEM_PARAMETER_CAP = 200_000_000


class WorkspaceContractError(ValueError):
    """A causal, shape, custody, or control contract failed."""


class _StraightThroughOneHot(torch.autograd.Function):
    """Emit an exact one-hot tensor while differentiating through probabilities."""

    @staticmethod
    def forward(
        ctx: object,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        indices = probabilities.argmax(dim=-1, keepdim=True)
        return torch.zeros_like(probabilities).scatter_(-1, indices, 1.0)

    @staticmethod
    def backward(
        ctx: object,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        del ctx
        return (gradient,)


@dataclass(frozen=True)
class CausalWorkspaceConfig:
    d_model: int = 576
    slot_width: int = 256
    num_slots: int = 4
    num_operators: int = 4
    operator_rank: int = 32
    stage_after_block: int = 19
    parameter_cap: int = SYSTEM_PARAMETER_CAP

    def validate(self, *, n_layer: int | None = None) -> None:
        fields = (
            self.d_model,
            self.slot_width,
            self.num_slots,
            self.num_operators,
            self.operator_rank,
        )
        if any(value <= 0 for value in fields):
            raise WorkspaceContractError("all workspace dimensions must be positive")
        if not 0 <= self.stage_after_block:
            raise WorkspaceContractError("stage_after_block must be nonnegative")
        if n_layer is not None and self.stage_after_block >= n_layer - 1:
            raise WorkspaceContractError(
                "workspace stage must leave at least one decoder block"
            )
        if self.parameter_cap > SYSTEM_PARAMETER_CAP:
            raise WorkspaceContractError("parameter cap exceeds the system maximum")


@dataclass(frozen=True)
class WorkspaceState:
    """Allowlisted source-deleted state passed to late query execution."""

    slots: torch.Tensor
    token_position: int
    sealed: bool = False


@dataclass(frozen=True)
class WorkspaceExecutionState:
    """Residual-level streaming state; this is not a transformer KV cache."""

    workspace: WorkspaceState
    cursor: torch.Tensor
    query_tokens: int


@dataclass(frozen=True)
class WorkspaceDiagnostics:
    bindings: torch.Tensor
    operator_probabilities: torch.Tensor
    cursor_states: torch.Tensor


@dataclass(frozen=True)
class WorkspaceParameterReceipt:
    workspace_parameters: int
    protected_base_parameters: int
    complete_system_parameters: int
    remaining_under_cap: int
    parameter_cap: int


@dataclass(frozen=True)
class WorkspaceControls:
    """Assessment-only causal ablations that preserve parameter geometry."""

    zero_workspace: bool = False
    uniform_binding: bool = False
    uniform_operator: bool = False
    binding_permutation: tuple[int, ...] | None = None
    operator_permutation: tuple[int, ...] | None = None
    scramble_selected_slot: bool = False
    scramble_discarded_slots: bool = False


class CausalBindSelectWorkspace(nn.Module):
    """Four-slot causal recurrent memory with independent operator selection."""

    def __init__(self, config: CausalWorkspaceConfig):
        super().__init__()
        config.validate()
        self.config = config
        d_model = config.d_model
        width = config.slot_width
        slots = config.num_slots
        operators = config.num_operators
        rank = config.operator_rank

        self.initial_slots = nn.Parameter(torch.zeros(slots, width))
        self.slot_addresses = nn.Parameter(torch.empty(slots, width))
        self.key_projection = nn.Linear(d_model, width, bias=False)
        self.value_projection = nn.Linear(d_model, width, bias=False)
        self.recurrent_update = nn.Linear(2 * width, 3 * width)
        self.operator_selector = nn.Linear(d_model + width, operators)
        self.operator_down = nn.Parameter(torch.empty(operators, width, rank))
        self.operator_up = nn.Parameter(torch.empty(operators, rank, width))
        self.read_projection = nn.Linear(width, d_model, bias=False)
        self.read_gate = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.initial_slots, std=0.02)
        nn.init.normal_(self.slot_addresses, std=0.02)
        nn.init.normal_(self.key_projection.weight, std=0.02)
        nn.init.normal_(self.value_projection.weight, std=0.02)
        nn.init.xavier_uniform_(self.recurrent_update.weight)
        nn.init.zeros_(self.recurrent_update.bias)
        nn.init.xavier_uniform_(self.operator_selector.weight)
        nn.init.zeros_(self.operator_selector.bias)
        nn.init.normal_(self.operator_down, std=0.02)
        nn.init.normal_(self.operator_up, std=0.02)
        nn.init.normal_(self.read_projection.weight, std=0.02)
        nn.init.zeros_(self.read_gate)

    def parameter_receipt(
        self,
        *,
        protected_base_parameters: int = PROTECTED_BASE_PARAMETERS,
    ) -> WorkspaceParameterReceipt:
        workspace = sum(parameter.numel() for parameter in self.parameters())
        complete = protected_base_parameters + workspace
        if complete > self.config.parameter_cap:
            raise WorkspaceContractError("complete system exceeds parameter cap")
        return WorkspaceParameterReceipt(
            workspace_parameters=workspace,
            protected_base_parameters=protected_base_parameters,
            complete_system_parameters=complete,
            remaining_under_cap=self.config.parameter_cap - complete,
            parameter_cap=self.config.parameter_cap,
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        token_position: int = 0,
    ) -> WorkspaceState:
        if batch_size <= 0:
            raise WorkspaceContractError("batch_size must be positive")
        if token_position < 0:
            raise WorkspaceContractError("token_position must be nonnegative")
        slots = self.initial_slots.to(device=device, dtype=dtype)
        slots = slots.unsqueeze(0).expand(batch_size, -1, -1).clone()
        return WorkspaceState(slots=slots, token_position=token_position)

    def compile_world(
        self,
        hidden: torch.Tensor,
        *,
        state: WorkspaceState | None = None,
        attention_mask: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> tuple[WorkspaceState, torch.Tensor | None]:
        """Causally compile raw-token residuals into source-deleted slots."""

        batch, tokens, _ = self._validate_hidden(hidden)
        if state is None:
            state = self.initial_state(
                batch,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        self._validate_workspace_state(state, batch=batch, hidden=hidden)
        mask = self._validated_mask(attention_mask, batch, tokens, hidden.device)

        slots = state.slots
        trace: list[torch.Tensor] = []
        for index in range(tokens):
            previous = slots
            token_hidden = hidden[:, index]
            binding = self._binding_probabilities(token_hidden)
            value = self.value_projection(token_hidden)
            write = binding.unsqueeze(-1) * value.unsqueeze(1)
            proposal = self._recurrent_step(previous, write)
            active = mask[:, index].view(batch, 1, 1)
            slots = torch.where(active, proposal, previous)
            if return_trace:
                trace.append(slots)

        next_position = state.token_position + int(mask[0].sum().item())
        if not torch.equal(mask.sum(dim=1), mask[0].sum().expand(batch)):
            raise WorkspaceContractError(
                "batched staged compilation requires equal active lengths"
            )
        result = WorkspaceState(slots=slots, token_position=next_position)
        if return_trace:
            return result, torch.stack(trace, dim=1)
        return result, None

    def seal_state(self, state: WorkspaceState) -> WorkspaceState:
        """Detach and clone the complete allowlisted execution state."""

        self._validate_workspace_state(state)
        return WorkspaceState(
            slots=state.slots.detach().clone(),
            token_position=state.token_position,
            sealed=True,
        )

    def start_execution(
        self,
        workspace: WorkspaceState,
        *,
        allow_unsealed_for_mechanism_fit: bool = False,
    ) -> WorkspaceExecutionState:
        self._validate_workspace_state(workspace)
        if not workspace.sealed and not allow_unsealed_for_mechanism_fit:
            raise WorkspaceContractError(
                "query execution requires a source-deleted sealed workspace"
            )
        cursor = workspace.slots.new_zeros(
            workspace.slots.shape[0],
            self.config.slot_width,
        )
        return WorkspaceExecutionState(
            workspace=workspace,
            cursor=cursor,
            query_tokens=0,
        )

    def execute_query(
        self,
        hidden: torch.Tensor,
        *,
        state: WorkspaceExecutionState | None = None,
        workspace: WorkspaceState | None = None,
        controls: WorkspaceControls | None = None,
        attention_mask: torch.Tensor | None = None,
        allow_unsealed_for_mechanism_fit: bool = False,
        assessment_only: bool = False,
    ) -> tuple[torch.Tensor, WorkspaceExecutionState, WorkspaceDiagnostics]:
        """Execute a late query using only a sealed workspace and query residuals."""

        batch, tokens, _ = self._validate_hidden(hidden)
        if (state is None) == (workspace is None):
            raise WorkspaceContractError(
                "provide exactly one of execution state or workspace"
            )
        if state is None:
            state = self.start_execution(
                workspace,
                allow_unsealed_for_mechanism_fit=allow_unsealed_for_mechanism_fit,
            )
        elif not state.workspace.sealed and not allow_unsealed_for_mechanism_fit:
            raise WorkspaceContractError(
                "query execution requires a source-deleted sealed workspace"
            )
        self._validate_execution_state(state, batch=batch, hidden=hidden)
        controls = controls or WorkspaceControls()
        self._validate_controls(controls)
        controls_active = self._controls_active(controls)
        if controls_active and not assessment_only:
            raise WorkspaceContractError("workspace interventions are assessment-only")
        if assessment_only and torch.is_grad_enabled():
            raise WorkspaceContractError(
                "assessment-only execution requires gradients to be disabled"
            )
        mask = self._validated_mask(attention_mask, batch, tokens, hidden.device)

        slots = state.workspace.slots
        if controls.zero_workspace:
            slots = torch.zeros_like(slots)
        cursor = state.cursor
        augmented: list[torch.Tensor] = []
        bindings: list[torch.Tensor] = []
        operators: list[torch.Tensor] = []
        cursors: list[torch.Tensor] = []
        gate = torch.tanh(self.read_gate).to(dtype=hidden.dtype)

        for index in range(tokens):
            token_hidden = hidden[:, index]
            binding = self._binding_probabilities(token_hidden)
            binding = self._controlled_distribution(
                binding,
                uniform=controls.uniform_binding,
                permutation=controls.binding_permutation,
            )
            intervened_slots = self._intervene_slots(
                slots,
                binding,
                scramble_selected=controls.scramble_selected_slot,
                scramble_discarded=controls.scramble_discarded_slots,
            )
            selected = torch.einsum("bs,bsw->bw", binding, intervened_slots)
            value = self.value_projection(token_hidden)
            proposal_cursor = cursor + selected + value

            operator_logits = self.operator_selector(
                torch.cat((token_hidden, selected), dim=-1)
            )
            operator_probability = _StraightThroughOneHot.apply(
                operator_logits.softmax(dim=-1)
            )
            operator_probability = self._controlled_distribution(
                operator_probability,
                uniform=controls.uniform_operator,
                permutation=controls.operator_permutation,
            )
            transformed = self._operator_candidates(proposal_cursor)
            proposal_cursor = torch.einsum(
                "bo,bow->bw",
                operator_probability,
                transformed,
            )

            active = mask[:, index].view(batch, 1)
            cursor = torch.where(active, proposal_cursor, cursor)
            delta = self.read_projection(cursor) * gate
            token_output = torch.where(active, token_hidden + delta, token_hidden)
            augmented.append(token_output)
            bindings.append(binding)
            operators.append(operator_probability)
            cursors.append(cursor)

        active_tokens = int(mask[0].sum().item())
        if not torch.equal(mask.sum(dim=1), mask[0].sum().expand(batch)):
            raise WorkspaceContractError(
                "batched staged execution requires equal active lengths"
            )
        result_state = WorkspaceExecutionState(
            workspace=state.workspace,
            cursor=cursor,
            query_tokens=state.query_tokens + active_tokens,
        )
        diagnostics = WorkspaceDiagnostics(
            bindings=torch.stack(bindings, dim=1),
            operator_probabilities=torch.stack(operators, dim=1),
            cursor_states=torch.stack(cursors, dim=1),
        )
        return torch.stack(augmented, dim=1), result_state, diagnostics

    def _binding_probabilities(self, hidden: torch.Tensor) -> torch.Tensor:
        key = self.key_projection(hidden)
        addresses = self.slot_addresses.to(device=hidden.device, dtype=hidden.dtype)
        logits = torch.einsum("bw,sw->bs", key, addresses)
        probabilities = (logits / math.sqrt(self.config.slot_width)).softmax(dim=-1)
        return _StraightThroughOneHot.apply(probabilities)

    def _recurrent_step(
        self,
        slots: torch.Tensor,
        write: torch.Tensor,
    ) -> torch.Tensor:
        reset, update, candidate = self.recurrent_update(
            torch.cat((slots, write), dim=-1)
        ).chunk(3, dim=-1)
        reset = reset.sigmoid()
        update = update.sigmoid()
        candidate = torch.tanh(candidate + reset * slots)
        return torch.lerp(slots, candidate, update)

    def _operator_candidates(self, cursor: torch.Tensor) -> torch.Tensor:
        down = self.operator_down.to(device=cursor.device, dtype=cursor.dtype)
        up = self.operator_up.to(device=cursor.device, dtype=cursor.dtype)
        latent = torch.einsum("bw,owr->bor", cursor, down)
        delta = torch.einsum("bor,orw->bow", F.silu(latent), up)
        return cursor.unsqueeze(1) + delta

    @staticmethod
    def _intervene_slots(
        slots: torch.Tensor,
        binding: torch.Tensor,
        *,
        scramble_selected: bool,
        scramble_discarded: bool,
    ) -> torch.Tensor:
        if not scramble_selected and not scramble_discarded:
            return slots
        scrambled = torch.roll(slots, shifts=1, dims=-1)
        selected = binding.detach().to(dtype=torch.bool).unsqueeze(-1)
        intervention_mask = selected if scramble_selected else ~selected
        return torch.where(intervention_mask, scrambled, slots)

    @staticmethod
    def _controlled_distribution(
        probabilities: torch.Tensor,
        *,
        uniform: bool,
        permutation: tuple[int, ...] | None,
    ) -> torch.Tensor:
        result = probabilities
        if uniform:
            result = torch.full_like(result, 1.0 / result.shape[-1])
        if permutation is not None:
            index = torch.tensor(permutation, device=result.device)
            result = result.index_select(-1, index)
        return result

    def _validate_hidden(self, hidden: torch.Tensor) -> tuple[int, int, int]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.d_model:
            raise WorkspaceContractError(
                "hidden must have shape [batch, tokens, d_model]"
            )
        if hidden.shape[0] <= 0 or hidden.shape[1] <= 0:
            raise WorkspaceContractError("hidden batch and token axes must be nonempty")
        if not hidden.is_floating_point():
            raise WorkspaceContractError("hidden states must be floating point")
        return tuple(hidden.shape)

    def _validate_workspace_state(
        self,
        state: WorkspaceState,
        *,
        batch: int | None = None,
        hidden: torch.Tensor | None = None,
    ) -> None:
        expected_tail = (self.config.num_slots, self.config.slot_width)
        if state.slots.ndim != 3 or tuple(state.slots.shape[1:]) != expected_tail:
            raise WorkspaceContractError("workspace slots have the wrong shape")
        if batch is not None and state.slots.shape[0] != batch:
            raise WorkspaceContractError("workspace batch does not match hidden batch")
        if state.token_position < 0:
            raise WorkspaceContractError("workspace token position is negative")
        if not isinstance(state.sealed, bool):
            raise WorkspaceContractError("workspace sealed flag must be boolean")
        if hidden is not None and (
            state.slots.device != hidden.device or state.slots.dtype != hidden.dtype
        ):
            raise WorkspaceContractError(
                "workspace device and dtype must match hidden states"
            )

    def _validate_execution_state(
        self,
        state: WorkspaceExecutionState,
        *,
        batch: int,
        hidden: torch.Tensor,
    ) -> None:
        self._validate_workspace_state(state.workspace, batch=batch, hidden=hidden)
        if state.cursor.shape != (batch, self.config.slot_width):
            raise WorkspaceContractError("execution cursor has the wrong shape")
        if state.cursor.device != hidden.device or state.cursor.dtype != hidden.dtype:
            raise WorkspaceContractError(
                "execution cursor device and dtype must match hidden states"
            )
        if state.query_tokens < 0:
            raise WorkspaceContractError("query token count is negative")

    def _validate_controls(self, controls: WorkspaceControls) -> None:
        boolean_controls = (
            controls.zero_workspace,
            controls.uniform_binding,
            controls.uniform_operator,
            controls.scramble_selected_slot,
            controls.scramble_discarded_slots,
        )
        if not all(isinstance(control, bool) for control in boolean_controls):
            raise WorkspaceContractError("workspace Boolean controls must be bool")
        if controls.scramble_selected_slot and controls.scramble_discarded_slots:
            raise WorkspaceContractError(
                "selected and discarded slot scrambles are mutually exclusive"
            )
        if controls.uniform_binding and (
            controls.scramble_selected_slot or controls.scramble_discarded_slots
        ):
            raise WorkspaceContractError(
                "slot scrambles require a hard selected binding"
            )
        self._validate_permutation(
            controls.binding_permutation,
            self.config.num_slots,
            "binding",
        )
        self._validate_permutation(
            controls.operator_permutation,
            self.config.num_operators,
            "operator",
        )

    @staticmethod
    def _controls_active(controls: WorkspaceControls) -> bool:
        return any(
            (
                controls.zero_workspace,
                controls.uniform_binding,
                controls.uniform_operator,
                controls.binding_permutation is not None,
                controls.operator_permutation is not None,
                controls.scramble_selected_slot,
                controls.scramble_discarded_slots,
            )
        )

    @staticmethod
    def _validate_permutation(
        permutation: tuple[int, ...] | None,
        size: int,
        label: str,
    ) -> None:
        if permutation is None:
            return
        if tuple(sorted(permutation)) != tuple(range(size)):
            raise WorkspaceContractError(f"{label} control is not a permutation")

    @staticmethod
    def _validated_mask(
        mask: torch.Tensor | None,
        batch: int,
        tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(batch, tokens, device=device, dtype=torch.bool)
        if mask.shape != (batch, tokens):
            raise WorkspaceContractError("attention_mask has the wrong shape")
        if mask.dtype != torch.bool:
            if not torch.all((mask == 0) | (mask == 1)):
                raise WorkspaceContractError("attention_mask must be binary")
            mask = mask.bool()
        if torch.any(mask[:, 1:].to(torch.int8) > mask[:, :-1].to(torch.int8)):
            raise WorkspaceContractError("attention_mask must be right padded")
        return mask.to(device=device)


class CausalWorkspaceGPT(nn.Module):
    """Shohin wrapper whose scored treatment accepts raw tokens only."""

    def __init__(self, base: GPT, workspace_config: CausalWorkspaceConfig):
        super().__init__()
        workspace_config.validate(n_layer=base.cfg.n_layer)
        if workspace_config.d_model != base.cfg.d_model:
            raise WorkspaceContractError(
                "workspace d_model must match the protected base"
            )
        if base.cfg.n_loop != 1:
            raise WorkspaceContractError("workspace reference requires n_loop=1")
        self.base = base
        self.workspace = CausalBindSelectWorkspace(workspace_config)
        self.workspace_config = workspace_config

    def compile_world_state(
        self,
        world_idx: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> WorkspaceState:
        """Compile and seal a world without accepting any query input."""

        return self._compile_world_state(
            world_idx,
            seal=True,
            attention_mask=attention_mask,
        )

    def execute_workspace_state(
        self,
        workspace: WorkspaceState,
        query_idx: torch.Tensor,
        *,
        controls: WorkspaceControls | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, WorkspaceDiagnostics]:
        """Execute a sealed source-deleted state without accepting world input."""

        assessment_only = controls is not None and self.workspace._controls_active(
            controls
        )
        if assessment_only and torch.is_grad_enabled():
            raise WorkspaceContractError(
                "assessment-only execution requires gradients to be disabled"
            )
        logits, loss, _, diagnostics = self._execute_query_from_state(
            workspace,
            query_idx,
            targets=None,
            controls=controls,
            attention_mask=attention_mask,
            allow_unsealed_for_mechanism_fit=False,
            assessment_only=assessment_only,
        )
        if loss is not None:
            raise WorkspaceContractError(
                "source-deleted execution unexpectedly produced a loss"
            )
        return logits, diagnostics

    def _compile_world_state(
        self,
        world_idx: torch.Tensor,
        *,
        seal: bool = True,
        attention_mask: torch.Tensor | None = None,
    ) -> WorkspaceState:
        hidden = self._encode_to_workspace(world_idx, pos=0)
        state, _ = self.workspace.compile_world(
            hidden,
            attention_mask=attention_mask,
        )
        return self.workspace.seal_state(state) if seal else state

    def _execute_query_from_state(
        self,
        workspace: WorkspaceState,
        query_idx: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        controls: WorkspaceControls | None = None,
        attention_mask: torch.Tensor | None = None,
        allow_unsealed_for_mechanism_fit: bool = False,
        assessment_only: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        WorkspaceExecutionState,
        WorkspaceDiagnostics,
    ]:
        hidden = self._encode_to_workspace(
            query_idx,
            pos=workspace.token_position,
        )
        hidden, execution, diagnostics = self.workspace.execute_query(
            hidden,
            workspace=workspace,
            controls=controls,
            attention_mask=attention_mask,
            allow_unsealed_for_mechanism_fit=allow_unsealed_for_mechanism_fit,
            assessment_only=assessment_only,
        )
        hidden = self._decode_from_workspace(
            hidden,
            pos=workspace.token_position,
        )
        logits = self.base.head(self.base.norm(hidden))
        loss = None
        if targets is not None:
            loss = _supervised_lm_loss(logits, targets, self.base.cfg.zloss)
        return logits, loss, execution, diagnostics

    def forward_staged(
        self,
        world_idx: torch.Tensor,
        query_idx: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        world_attention_mask: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the treatment from raw tokens through an internally sealed state."""

        logits, loss, _, _ = self._forward_raw_tokens(
            world_idx,
            query_idx,
            targets=targets,
            world_attention_mask=world_attention_mask,
            query_attention_mask=query_attention_mask,
            controls=None,
            seal_world=True,
            allow_unsealed_for_mechanism_fit=False,
        )
        return logits, loss

    def forward_staged_assessment(
        self,
        world_idx: torch.Tensor,
        query_idx: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        world_attention_mask: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        WorkspaceState,
        WorkspaceDiagnostics,
    ]:
        """Expose state and diagnostics for assessment, never treatment scoring."""

        return self._forward_raw_tokens(
            world_idx,
            query_idx,
            targets=targets,
            world_attention_mask=world_attention_mask,
            query_attention_mask=query_attention_mask,
            controls=None,
            seal_world=True,
            allow_unsealed_for_mechanism_fit=False,
        )

    def forward_staged_controlled(
        self,
        world_idx: torch.Tensor,
        query_idx: torch.Tensor,
        *,
        controls: WorkspaceControls,
        targets: torch.Tensor | None = None,
        world_attention_mask: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        WorkspaceState,
        WorkspaceDiagnostics,
    ]:
        """Run an explicit assessment-only intervention from raw tokens."""

        with torch.no_grad():
            return self._forward_raw_tokens(
                world_idx,
                query_idx,
                targets=targets,
                world_attention_mask=world_attention_mask,
                query_attention_mask=query_attention_mask,
                controls=controls,
                seal_world=True,
                allow_unsealed_for_mechanism_fit=False,
                assessment_only=True,
            )

    def forward_mechanism_fit(
        self,
        world_idx: torch.Tensor,
        query_idx: torch.Tensor,
        *,
        targets: torch.Tensor,
        world_attention_mask: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        WorkspaceState,
        WorkspaceDiagnostics,
    ]:
        """Explicit differentiable raw-token path for isolated mechanism fitting."""

        logits, loss, workspace, diagnostics = self._forward_raw_tokens(
            world_idx,
            query_idx,
            targets=targets,
            world_attention_mask=world_attention_mask,
            query_attention_mask=query_attention_mask,
            controls=None,
            seal_world=False,
            allow_unsealed_for_mechanism_fit=True,
        )
        if loss is None:
            raise WorkspaceContractError(
                "mechanism fitting requires supervised targets"
            )
        return logits, loss, workspace, diagnostics

    def _forward_raw_tokens(
        self,
        world_idx: torch.Tensor,
        query_idx: torch.Tensor,
        *,
        targets: torch.Tensor | None,
        world_attention_mask: torch.Tensor | None,
        query_attention_mask: torch.Tensor | None,
        controls: WorkspaceControls | None,
        seal_world: bool,
        allow_unsealed_for_mechanism_fit: bool,
        assessment_only: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        WorkspaceState,
        WorkspaceDiagnostics,
    ]:
        workspace = self._compile_world_state(
            world_idx,
            seal=seal_world,
            attention_mask=world_attention_mask,
        )
        logits, loss, _, diagnostics = self._execute_query_from_state(
            workspace,
            query_idx,
            targets=targets,
            controls=controls,
            attention_mask=query_attention_mask,
            allow_unsealed_for_mechanism_fit=allow_unsealed_for_mechanism_fit,
            assessment_only=assessment_only,
        )
        return logits, loss, workspace, diagnostics

    def _encode_to_workspace(self, idx: torch.Tensor, *, pos: int) -> torch.Tensor:
        self._validate_tokens(idx, pos=pos)
        hidden = self.base.tok(idx)
        cos = self.base.cos[pos : pos + idx.shape[1]].to(hidden.device)
        sin = self.base.sin[pos : pos + idx.shape[1]].to(hidden.device)
        for block in self.base.blocks[: self.workspace_config.stage_after_block + 1]:
            hidden, _ = block(hidden, cos, sin)
        return hidden

    def _decode_from_workspace(
        self,
        hidden: torch.Tensor,
        *,
        pos: int,
    ) -> torch.Tensor:
        cos = self.base.cos[pos : pos + hidden.shape[1]].to(hidden.device)
        sin = self.base.sin[pos : pos + hidden.shape[1]].to(hidden.device)
        for block in self.base.blocks[self.workspace_config.stage_after_block + 1 :]:
            hidden, _ = block(hidden, cos, sin)
        return hidden

    def _validate_tokens(self, idx: torch.Tensor, *, pos: int) -> None:
        if idx.ndim != 2 or idx.shape[0] <= 0 or idx.shape[1] <= 0:
            raise WorkspaceContractError(
                "token IDs must have shape [batch, nonempty tokens]"
            )
        if idx.dtype != torch.long:
            raise WorkspaceContractError("token IDs must have dtype torch.long")
        if pos < 0 or pos + idx.shape[1] > self.base.cfg.seq_len:
            raise WorkspaceContractError("token positions exceed configured sequence")

    def parameter_receipt(self) -> WorkspaceParameterReceipt:
        protected = self.base.num_params()
        if protected != PROTECTED_BASE_PARAMETERS:
            raise WorkspaceContractError(
                "base parameter count does not match protected Shohin"
            )
        return self.workspace.parameter_receipt(
            protected_base_parameters=protected,
        )


def freeze_protected_base(model: CausalWorkspaceGPT) -> None:
    """Freeze only protected base tensors; workspace parameters remain trainable."""

    for parameter in model.base.parameters():
        parameter.requires_grad_(False)
    for parameter in model.workspace.parameters():
        parameter.requires_grad_(True)


def trainable_workspace_parameters(
    model: CausalWorkspaceGPT,
) -> Sequence[nn.Parameter]:
    """Return the exact optimizer partition for the first isolated pilot."""

    base_trainable = [
        name for name, p in model.base.named_parameters() if p.requires_grad
    ]
    if base_trainable:
        raise WorkspaceContractError(
            "protected base contains trainable parameters: " + ",".join(base_trainable)
        )
    parameters = tuple(
        parameter
        for parameter in model.workspace.parameters()
        if parameter.requires_grad
    )
    if not parameters:
        raise WorkspaceContractError("workspace optimizer partition is empty")
    return parameters
