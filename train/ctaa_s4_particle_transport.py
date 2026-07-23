"""S4-tied particle transport component mechanics for CTAA workspaces.

The workspace carries a probability distribution over all 24 opcode-to-card
bindings. Rebinding cues update that distribution by convolution in the group
algebra of S4. Because S4 is non-abelian, the state can preserve cue order.
The matched control uses circular convolution on Z24 with identical
state geometry, trainable kernel count, and dense transport cost.

This module contains score-free component mechanics. It does not implement the
required byte-source compiler, source/KV destruction, or independent late
reader, and therefore cannot authorize a board seed, confirmation access, GPU
job, or reasoning claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ctaa_binding_completion import (
    ACTION_COUNT,
    BINDINGS,
    BINDING_TO_INDEX,
    COMPILER_WIDTH,
    FACTORIZED_MACS,
    READOUT_PARAMETERS,
    RELATION_SLOT_COUNT,
    BindingCompletionError,
    FactorizedBindingReadout,
)


PARTICLE_COUNT = len(BINDINGS)
REBINDING_CUE_COUNT = 6
CTAA_BASE_SYSTEM_PARAMETERS = 137_989_944
STRICT_SYSTEM_PARAMETER_LIMIT = 199_999_999
TRANSPORT_PARAMETERS = REBINDING_CUE_COUNT * PARTICLE_COUNT
DENSE_TRANSPORT_PARAMETERS = (
    REBINDING_CUE_COUNT * PARTICLE_COUNT * PARTICLE_COUNT
)
S4_TPT_WORKSPACE_PARAMETERS = READOUT_PARAMETERS + TRANSPORT_PARAMETERS
DENSE_CONTROL_PARAMETERS = READOUT_PARAMETERS + DENSE_TRANSPORT_PARAMETERS
S4_TPT_COMPLETE_SYSTEM_PARAMETERS = (
    CTAA_BASE_SYSTEM_PARAMETERS + S4_TPT_WORKSPACE_PARAMETERS
)
DENSE_CONTROL_COMPLETE_SYSTEM_PARAMETERS = (
    CTAA_BASE_SYSTEM_PARAMETERS + DENSE_CONTROL_PARAMETERS
)
TRANSPORT_MACS_PER_CUE = PARTICLE_COUNT * PARTICLE_COUNT
STOP_ID = ACTION_COUNT
CUE_EVENT = 0
ACTION_EVENT = 1
STOP_EVENT = 2
STATE_VALUE_COUNT = 3
FULL_STATE_COUNT = STATE_VALUE_COUNT**3
FULL_STATES = tuple(
    itertools.product(range(STATE_VALUE_COUNT), repeat=ACTION_COUNT - 1)
)
FULL_STATE_TENSOR = torch.tensor(FULL_STATES, dtype=torch.long)


def checked_permutation(value: Sequence[int]) -> tuple[int, int, int, int]:
    permutation = tuple(int(item) for item in value)
    if len(permutation) != ACTION_COUNT or sorted(permutation) != list(
        range(ACTION_COUNT)
    ):
        raise BindingCompletionError("CTAA S4 transport element leaves S4")
    return permutation  # type: ignore[return-value]


def compose_permutations(
    left: Sequence[int],
    right: Sequence[int],
) -> tuple[int, int, int, int]:
    """Return function composition ``left after right``."""

    left_checked = checked_permutation(left)
    right_checked = checked_permutation(right)
    return tuple(left_checked[right_checked[index]] for index in range(ACTION_COUNT))  # type: ignore[return-value]


def invert_permutation(value: Sequence[int]) -> tuple[int, int, int, int]:
    permutation = checked_permutation(value)
    inverse = [0] * ACTION_COUNT
    for source, destination in enumerate(permutation):
        inverse[destination] = source
    return tuple(inverse)  # type: ignore[return-value]


IDENTITY = tuple(range(ACTION_COUNT))
IDENTITY_INDEX = BINDING_TO_INDEX[IDENTITY]
S4_GENERATORS = tuple(
    tuple(
        right if index == left else left if index == right else index
        for index in range(ACTION_COUNT)
    )
    for left, right in itertools.combinations(range(ACTION_COUNT), 2)
)
S4_GENERATOR_INDICES = tuple(BINDING_TO_INDEX[item] for item in S4_GENERATORS)


def _s4_delta_table() -> torch.Tensor:
    rows = []
    for source in BINDINGS:
        inverse = invert_permutation(source)
        rows.append(
            [
                BINDING_TO_INDEX[compose_permutations(inverse, destination)]
                for destination in BINDINGS
            ]
        )
    return torch.tensor(rows, dtype=torch.long)


def _z24_delta_table() -> torch.Tensor:
    return torch.tensor(
        [
            [
                (destination - source) % PARTICLE_COUNT
                for destination in range(PARTICLE_COUNT)
            ]
            for source in range(PARTICLE_COUNT)
        ],
        dtype=torch.long,
    )


S4_DELTA_INDEX = _s4_delta_table()
Z24_DELTA_INDEX = _z24_delta_table()


def group_convolve(
    state: torch.Tensor,
    kernel: torch.Tensor,
    delta_index: torch.Tensor,
) -> torch.Tensor:
    """Convolve two batched distributions using a frozen multiplication table."""

    if (
        state.ndim != 2
        or state.shape[1] != PARTICLE_COUNT
        or kernel.shape != state.shape
        or delta_index.shape != (PARTICLE_COUNT, PARTICLE_COUNT)
        or delta_index.dtype != torch.long
    ):
        raise BindingCompletionError("CTAA S4 convolution geometry differs")
    if delta_index.device != state.device:
        delta_index = delta_index.to(state.device)
    selected = kernel[:, delta_index]
    return torch.einsum("bh,bhg->bg", state.float(), selected.float())


def apply_kernel_sequence(
    initial: torch.Tensor,
    kernels: torch.Tensor,
    delta_index: torch.Tensor,
) -> torch.Tensor:
    if (
        kernels.ndim != 3
        or kernels.shape[0] != initial.shape[0]
        or kernels.shape[2] != PARTICLE_COUNT
    ):
        raise BindingCompletionError("CTAA S4 sequence geometry differs")
    state = initial.float()
    for kernel in kernels.unbind(1):
        state = group_convolve(state, kernel, delta_index)
    return state


def one_hot_group_element(
    element_index: int,
    *,
    batch: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if not 0 <= int(element_index) < PARTICLE_COUNT or batch < 1:
        raise BindingCompletionError("CTAA S4 one-hot element differs")
    values = torch.zeros(batch, PARTICLE_COUNT, device=device)
    values[:, int(element_index)] = 1.0
    return values


def binding_particle_scores(pair_logits: torch.Tensor) -> torch.Tensor:
    if pair_logits.ndim != 3 or pair_logits.shape[1:] != (
        ACTION_COUNT,
        ACTION_COUNT,
    ):
        raise BindingCompletionError("CTAA S4 pair logits differ")
    candidates = torch.tensor(
        BINDINGS,
        dtype=torch.long,
        device=pair_logits.device,
    )
    rows = torch.arange(ACTION_COUNT, device=pair_logits.device)
    return pair_logits[:, rows, candidates].sum(-1)


def transform_binding_coordinates(
    binding: Sequence[int],
    opcode_order: Sequence[int],
    card_order: Sequence[int],
) -> tuple[int, int, int, int]:
    """Express a binding after independently reindexing opcode and card slots."""

    binding_checked = checked_permutation(binding)
    opcode_checked = checked_permutation(opcode_order)
    card_checked = checked_permutation(card_order)
    card_inverse = invert_permutation(card_checked)
    return tuple(
        card_inverse[binding_checked[opcode_checked[index]]]
        for index in range(ACTION_COUNT)
    )  # type: ignore[return-value]


def conjugate_rebinding_element(
    element: Sequence[int],
    opcode_order: Sequence[int],
) -> tuple[int, int, int, int]:
    """Transform a right-acting rebinding cue under opcode reindexing."""

    element_checked = checked_permutation(element)
    opcode_checked = checked_permutation(opcode_order)
    return compose_permutations(
        invert_permutation(opcode_checked),
        compose_permutations(element_checked, opcode_checked),
    )


def reindex_kernel_probabilities(
    probabilities: torch.Tensor,
    opcode_order: Sequence[int],
) -> torch.Tensor:
    """Conjugate an S4 cue kernel into reindexed opcode coordinates."""

    if probabilities.ndim != 2 or probabilities.shape[1] != PARTICLE_COUNT:
        raise BindingCompletionError("CTAA cue-kernel geometry differs")
    output = torch.empty_like(probabilities)
    for old_index, element in enumerate(BINDINGS):
        transformed = conjugate_rebinding_element(element, opcode_order)
        output[:, BINDING_TO_INDEX[transformed]] = probabilities[:, old_index]
    return output


def reindex_particle_probabilities(
    probabilities: torch.Tensor,
    opcode_order: Sequence[int],
    card_order: Sequence[int],
) -> torch.Tensor:
    if probabilities.ndim != 2 or probabilities.shape[1] != PARTICLE_COUNT:
        raise BindingCompletionError("CTAA S4 particle geometry differs")
    output = torch.empty_like(probabilities)
    for old_index, binding in enumerate(BINDINGS):
        transformed = transform_binding_coordinates(
            binding,
            opcode_order,
            card_order,
        )
        output[:, BINDING_TO_INDEX[transformed]] = probabilities[:, old_index]
    return output


class _LearnedGroupTransport(nn.Module):
    """Shared learned cue kernels over a fixed 24-element multiplication table."""

    def __init__(self, delta_index: torch.Tensor, cue_count: int) -> None:
        super().__init__()
        if (
            cue_count < 1
            or delta_index.shape != (PARTICLE_COUNT, PARTICLE_COUNT)
            or delta_index.dtype != torch.long
        ):
            raise BindingCompletionError("CTAA S4 transport contract differs")
        self.cue_count = int(cue_count)
        self.register_buffer("delta_index", delta_index.clone(), persistent=False)
        self.kernel_logits = nn.Parameter(
            torch.empty(self.cue_count, PARTICLE_COUNT)
        )
        nn.init.normal_(self.kernel_logits, mean=0.0, std=0.02)

    def forward(
        self,
        initial: torch.Tensor,
        cue_indices: torch.Tensor,
    ) -> torch.Tensor:
        if (
            initial.ndim != 2
            or initial.shape[1] != PARTICLE_COUNT
            or cue_indices.ndim != 2
            or cue_indices.shape[0] != initial.shape[0]
            or cue_indices.dtype != torch.long
        ):
            raise BindingCompletionError("CTAA S4 cue sequence differs")
        if cue_indices.numel() and (
            int(cue_indices.min()) < 0
            or int(cue_indices.max()) >= self.cue_count
        ):
            raise BindingCompletionError("CTAA S4 cue sequence differs")
        transitions = self.transition_matrices(cue_indices)
        state = initial.float()
        for transition in transitions.unbind(1):
            state = torch.einsum("bh,bhg->bg", state, transition)
        return state

    def transition_matrices(self, cue_indices: torch.Tensor) -> torch.Tensor:
        if cue_indices.ndim != 2 or cue_indices.dtype != torch.long:
            raise BindingCompletionError("CTAA S4 cue batch differs")
        if cue_indices.numel() and (
            int(cue_indices.min()) < 0
            or int(cue_indices.max()) >= self.cue_count
        ):
            raise BindingCompletionError("CTAA S4 cue batch differs")
        kernels = self.kernel_logits[cue_indices].float().softmax(-1)
        delta_index = self.delta_index
        if delta_index.device != kernels.device:
            delta_index = delta_index.to(kernels.device)
        return kernels[:, :, delta_index]

    @property
    def unique_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class S4TiedTransport(_LearnedGroupTransport):
    """Treatment: cue composition in the true non-abelian S4 group algebra."""

    def __init__(self, cue_count: int = REBINDING_CUE_COUNT) -> None:
        super().__init__(S4_DELTA_INDEX, cue_count)


class Z24CircularTransportControl(_LearnedGroupTransport):
    """Matched control: equal-size transport in the abelian group Z24."""

    def __init__(self, cue_count: int = REBINDING_CUE_COUNT) -> None:
        super().__init__(Z24_DELTA_INDEX, cue_count)


class Dense24TransportControl(nn.Module):
    """Favorable control: an unconstrained learned operator for every cue.

    This control has more parameters than either group-convolution arm and its
    hypothesis class contains every positive S4 transport kernel exactly.
    """

    def __init__(self, cue_count: int = REBINDING_CUE_COUNT) -> None:
        super().__init__()
        if cue_count < 1:
            raise BindingCompletionError("CTAA dense transport cue count differs")
        self.cue_count = int(cue_count)
        self.transition_logits = nn.Parameter(
            torch.empty(self.cue_count, PARTICLE_COUNT, PARTICLE_COUNT)
        )
        nn.init.normal_(self.transition_logits, mean=0.0, std=0.02)

    def forward(
        self,
        initial: torch.Tensor,
        cue_indices: torch.Tensor,
    ) -> torch.Tensor:
        if (
            initial.ndim != 2
            or initial.shape[1] != PARTICLE_COUNT
            or cue_indices.ndim != 2
            or cue_indices.shape[0] != initial.shape[0]
            or cue_indices.dtype != torch.long
        ):
            raise BindingCompletionError("CTAA dense transport sequence differs")
        if cue_indices.numel() and (
            int(cue_indices.min()) < 0
            or int(cue_indices.max()) >= self.cue_count
        ):
            raise BindingCompletionError("CTAA dense transport sequence differs")
        transitions = self.transition_matrices(cue_indices)
        state = initial.float()
        for transition in transitions.unbind(1):
            state = torch.einsum("bh,bhg->bg", state, transition)
        return state

    def transition_matrices(self, cue_indices: torch.Tensor) -> torch.Tensor:
        if cue_indices.ndim != 2 or cue_indices.dtype != torch.long:
            raise BindingCompletionError("CTAA dense cue batch differs")
        if cue_indices.numel() and (
            int(cue_indices.min()) < 0
            or int(cue_indices.max()) >= self.cue_count
        ):
            raise BindingCompletionError("CTAA dense cue batch differs")
        return self.transition_logits[cue_indices].float().softmax(-1)

    @property
    def unique_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def lift_group_kernel_logits_to_dense(
    kernel_logits: torch.Tensor,
    delta_index: torch.Tensor = S4_DELTA_INDEX,
) -> torch.Tensor:
    """Embed a group-convolution kernel in the favorable dense control."""

    if (
        kernel_logits.ndim != 2
        or kernel_logits.shape[1] != PARTICLE_COUNT
        or delta_index.shape != (PARTICLE_COUNT, PARTICLE_COUNT)
    ):
        raise BindingCompletionError("CTAA dense lift geometry differs")
    if delta_index.device != kernel_logits.device:
        delta_index = delta_index.to(kernel_logits.device)
    return kernel_logits[:, delta_index]


@dataclass(frozen=True)
class S4TransportWorkspaceOutput:
    pair_logits: torch.Tensor
    initial_particles: torch.Tensor
    transported_particles: torch.Tensor


class _S4TransportBindingWorkspace(nn.Module):
    def __init__(self, transport: nn.Module) -> None:
        super().__init__()
        self.readout = FactorizedBindingReadout()
        self.transport = transport

    def forward(
        self,
        slots: torch.Tensor,
        cue_indices: torch.Tensor,
    ) -> S4TransportWorkspaceOutput:
        if slots.ndim != 3 or slots.shape[1:] != (
            RELATION_SLOT_COUNT,
            COMPILER_WIDTH,
        ):
            raise BindingCompletionError("CTAA S4 slot geometry differs")
        pair_logits = self.readout(slots)
        initial = binding_particle_scores(pair_logits).softmax(-1)
        transported = self.transport(initial, cue_indices)
        return S4TransportWorkspaceOutput(pair_logits, initial, transported)


class S4TiedBindingWorkspace(_S4TransportBindingWorkspace):
    def __init__(self) -> None:
        super().__init__(S4TiedTransport())


class Z24BindingTransportControl(_S4TransportBindingWorkspace):
    def __init__(self) -> None:
        super().__init__(Z24CircularTransportControl())


class Dense24BindingTransportControl(_S4TransportBindingWorkspace):
    def __init__(self) -> None:
        super().__init__(Dense24TransportControl())


@dataclass(frozen=True)
class ParticleExecution:
    final_states: torch.Tensor
    final_state_marginals: torch.Tensor
    halted: torch.Tensor


@dataclass(frozen=True)
class InterleavedParticleExecution:
    final_joint: torch.Tensor
    binding_marginals: torch.Tensor
    full_state_marginals: torch.Tensor
    query_distribution: torch.Tensor
    binding_trajectory: torch.Tensor
    full_state_trajectory: torch.Tensor
    halted: torch.Tensor


def _validate_interleaved_inputs(
    transport: nn.Module,
    particle_probabilities: torch.Tensor,
    action_cards: torch.Tensor,
    initial_state: torch.Tensor,
    event_kinds: torch.Tensor,
    event_values: torch.Tensor,
    late_query: torch.Tensor,
) -> None:
    batch = particle_probabilities.shape[0]
    if (
        batch < 1
        or particle_probabilities.ndim != 2
        or particle_probabilities.shape[1] != PARTICLE_COUNT
        or action_cards.shape != (batch, ACTION_COUNT, ACTION_COUNT - 1)
        or initial_state.shape != (batch, ACTION_COUNT - 1)
        or event_kinds.ndim != 2
        or event_kinds.shape[0] != batch
        or event_values.shape != event_kinds.shape
        or late_query.shape != (batch,)
        or action_cards.dtype != torch.long
        or initial_state.dtype != torch.long
        or event_kinds.dtype != torch.long
        or event_values.dtype != torch.long
        or late_query.dtype != torch.long
        or event_kinds.shape[1] < 1
        or not hasattr(transport, "transition_matrices")
        or not hasattr(transport, "cue_count")
    ):
        raise BindingCompletionError("CTAA interleaved workspace geometry differs")
    cue_mask = event_kinds.eq(CUE_EVENT)
    action_mask = event_kinds.eq(ACTION_EVENT)
    stop_mask = event_kinds.eq(STOP_EVENT)
    cue_values = event_values[cue_mask]
    action_values = event_values[action_mask]
    if (
        not torch.isfinite(particle_probabilities).all()
        or bool((particle_probabilities < 0).any())
        or not torch.allclose(
            particle_probabilities.sum(-1),
            torch.ones(batch, device=particle_probabilities.device),
            rtol=1e-5,
            atol=1e-6,
        )
        or int(action_cards.min()) < 0
        or int(action_cards.max()) >= ACTION_COUNT - 1
        or int(initial_state.min()) < 0
        or int(initial_state.max()) >= STATE_VALUE_COUNT
        or int(event_kinds.min()) < CUE_EVENT
        or int(event_kinds.max()) > STOP_EVENT
        or not bool(stop_mask.sum(1).eq(1).all())
        or bool(event_values[stop_mask].ne(0).any())
        or (
            cue_values.numel() > 0
            and (
                int(cue_values.min()) < 0
                or int(cue_values.max()) >= int(transport.cue_count)
            )
        )
        or (
            action_values.numel() > 0
            and (
                int(action_values.min()) < 0
                or int(action_values.max()) >= ACTION_COUNT
            )
        )
        or int(late_query.min()) < 0
        or int(late_query.max()) >= ACTION_COUNT - 1
    ):
        raise BindingCompletionError("CTAA interleaved workspace values differ")


def execute_interleaved_particle_ctaa(
    transport: nn.Module,
    particle_probabilities: torch.Tensor,
    action_cards: torch.Tensor,
    initial_state: torch.Tensor,
    event_kinds: torch.Tensor,
    event_values: torch.Tensor,
    late_query: torch.Tensor,
) -> InterleavedParticleExecution:
    """Run cue and opcode events over a joint binding/state distribution.

    This is a differentiable tensor reference component. It does not compile
    byte sources or prove that source residuals and KV were destroyed.
    """

    _validate_interleaved_inputs(
        transport,
        particle_probabilities,
        action_cards,
        initial_state,
        event_kinds,
        event_values,
        late_query,
    )
    batch = particle_probabilities.shape[0]
    device = particle_probabilities.device
    initial_state_index = (
        initial_state[:, 0] * (STATE_VALUE_COUNT**2)
        + initial_state[:, 1] * STATE_VALUE_COUNT
        + initial_state[:, 2]
    )
    state_one_hot = F.one_hot(initial_state_index, FULL_STATE_COUNT).float()
    joint = particle_probabilities.float()[:, :, None] * state_one_hot[:, None, :]
    halted = torch.zeros(batch, dtype=torch.bool, device=device)
    candidates = torch.tensor(BINDINGS, dtype=torch.long, device=device)
    full_states = FULL_STATE_TENSOR.to(device)
    batch_index = torch.arange(batch, device=device)
    binding_trajectory = [joint.sum(-1)]
    full_state_trajectory = [joint.sum(1)]

    for kinds, values in zip(
        event_kinds.unbind(1),
        event_values.unbind(1),
        strict=True,
    ):
        active = ~halted
        cue_mask = active & kinds.eq(CUE_EVENT)
        action_mask = active & kinds.eq(ACTION_EVENT)
        stop_mask = active & kinds.eq(STOP_EVENT)

        if bool(cue_mask.any()):
            safe_cue = values.clamp(0, int(transport.cue_count) - 1)
            transition = transport.transition_matrices(safe_cue[:, None])[:, 0]
            cue_joint = torch.einsum("bhs,bhg->bgs", joint, transition)
            joint = torch.where(cue_mask[:, None, None], cue_joint, joint)

        if bool(action_mask.any()):
            safe_opcode = values.clamp(0, ACTION_COUNT - 1)
            card_index = candidates[:, safe_opcode].transpose(0, 1)
            selected_cards = action_cards[batch_index[:, None], card_index]
            expanded_states = full_states[None, None].expand(
                batch,
                PARTICLE_COUNT,
                -1,
                -1,
            )
            gather_index = selected_cards[:, :, None].expand(
                -1,
                -1,
                FULL_STATE_COUNT,
                -1,
            )
            next_states = expanded_states.gather(3, gather_index)
            next_state_index = (
                next_states[..., 0] * (STATE_VALUE_COUNT**2)
                + next_states[..., 1] * STATE_VALUE_COUNT
                + next_states[..., 2]
            )
            action_joint = torch.zeros_like(joint).scatter_add(
                2,
                next_state_index,
                joint,
            )
            joint = torch.where(action_mask[:, None, None], action_joint, joint)
        halted = halted | stop_mask
        binding_trajectory.append(joint.sum(-1))
        full_state_trajectory.append(joint.sum(1))

    full_state_marginals = joint.sum(1)
    query_values = full_states[:, late_query].transpose(0, 1)
    query_selector = F.one_hot(query_values, STATE_VALUE_COUNT).float()
    query_distribution = torch.einsum(
        "bs,bsv->bv",
        full_state_marginals,
        query_selector,
    )
    return InterleavedParticleExecution(
        final_joint=joint,
        binding_marginals=joint.sum(-1),
        full_state_marginals=full_state_marginals,
        query_distribution=query_distribution,
        binding_trajectory=torch.stack(binding_trajectory, dim=1),
        full_state_trajectory=torch.stack(full_state_trajectory, dim=1),
        halted=halted,
    )


class InterleavedParticleWorkspace(nn.Module):
    """Thin module wrapper around the joint event-stream tensor executor."""

    def __init__(self, transport: nn.Module) -> None:
        super().__init__()
        self.transport = transport

    def forward(
        self,
        particle_probabilities: torch.Tensor,
        action_cards: torch.Tensor,
        initial_state: torch.Tensor,
        event_kinds: torch.Tensor,
        event_values: torch.Tensor,
        late_query: torch.Tensor,
    ) -> InterleavedParticleExecution:
        return execute_interleaved_particle_ctaa(
            self.transport,
            particle_probabilities,
            action_cards,
            initial_state,
            event_kinds,
            event_values,
            late_query,
        )


class S4InterleavedParticleWorkspace(InterleavedParticleWorkspace):
    def __init__(self) -> None:
        super().__init__(S4TiedTransport())


class Z24InterleavedParticleControl(InterleavedParticleWorkspace):
    def __init__(self) -> None:
        super().__init__(Z24CircularTransportControl())


class Dense24InterleavedParticleControl(InterleavedParticleWorkspace):
    def __init__(self) -> None:
        super().__init__(Dense24TransportControl())


def execute_particle_ctaa(
    particle_probabilities: torch.Tensor,
    action_cards: torch.Tensor,
    initial_state: torch.Tensor,
    opcode_schedule: torch.Tensor,
) -> ParticleExecution:
    """Static-binding reference executor retained for backward mechanics tests."""

    if (
        particle_probabilities.ndim != 2
        or particle_probabilities.shape[1] != PARTICLE_COUNT
        or action_cards.ndim != 3
        or action_cards.shape[1:] != (ACTION_COUNT, 3)
        or initial_state.shape != (particle_probabilities.shape[0], 3)
        or opcode_schedule.ndim != 2
        or opcode_schedule.shape[0] != particle_probabilities.shape[0]
        or action_cards.dtype != torch.long
        or initial_state.dtype != torch.long
        or opcode_schedule.dtype != torch.long
    ):
        raise BindingCompletionError("CTAA S4 execution geometry differs")
    if (
        not torch.isfinite(particle_probabilities).all()
        or bool((particle_probabilities < 0).any())
        or not torch.allclose(
            particle_probabilities.sum(-1),
            torch.ones(
                particle_probabilities.shape[0],
                device=particle_probabilities.device,
            ),
            rtol=1e-5,
            atol=1e-6,
        )
        or action_cards.numel() == 0
        or int(action_cards.min()) < 0
        or int(action_cards.max()) >= 3
        or int(initial_state.min()) < 0
        or int(initial_state.max()) >= 3
        or int(opcode_schedule.min()) < 0
        or int(opcode_schedule.max()) > STOP_ID
        or not bool(opcode_schedule.eq(STOP_ID).sum(1).eq(1).all())
    ):
        raise BindingCompletionError("CTAA S4 execution values differ")

    batch = particle_probabilities.shape[0]
    candidates = torch.tensor(
        BINDINGS,
        dtype=torch.long,
        device=particle_probabilities.device,
    )
    batch_index = torch.arange(batch, device=particle_probabilities.device)
    state = initial_state[:, None].expand(-1, PARTICLE_COUNT, -1).clone()
    halted = torch.zeros(batch, dtype=torch.bool, device=state.device)
    for opcode in opcode_schedule.unbind(1):
        stop = opcode.eq(STOP_ID)
        active = ~(halted | stop)
        safe_opcode = opcode.clamp_max(ACTION_COUNT - 1)
        card_index = candidates[:, safe_opcode].transpose(0, 1)
        selected = action_cards[batch_index[:, None], card_index]
        candidate_state = state.gather(2, selected)
        state = torch.where(active[:, None, None], candidate_state, state)
        halted = halted | stop

    state_one_hot = F.one_hot(state, 3).float()
    marginals = torch.einsum(
        "bp,bpiv->biv",
        particle_probabilities.float(),
        state_one_hot,
    )
    return ParticleExecution(state, marginals, halted)


def s4_transport_resource_receipt() -> dict[str, int | str]:
    treatment = S4TiedBindingWorkspace()
    abelian_control = Z24BindingTransportControl()
    dense_control = Dense24BindingTransportControl()
    treatment_parameters = sum(
        parameter.numel() for parameter in treatment.parameters()
    )
    abelian_control_parameters = sum(
        parameter.numel() for parameter in abelian_control.parameters()
    )
    dense_control_parameters = sum(
        parameter.numel() for parameter in dense_control.parameters()
    )
    if (
        treatment_parameters != S4_TPT_WORKSPACE_PARAMETERS
        or abelian_control_parameters != S4_TPT_WORKSPACE_PARAMETERS
        or dense_control_parameters != DENSE_CONTROL_PARAMETERS
        or S4_TPT_COMPLETE_SYSTEM_PARAMETERS >= STRICT_SYSTEM_PARAMETER_LIMIT
        or DENSE_CONTROL_COMPLETE_SYSTEM_PARAMETERS
        >= STRICT_SYSTEM_PARAMETER_LIMIT
    ):
        raise AssertionError("CTAA S4 transport resource ledger differs")
    return {
        "schema": "r12_ctaa_s4_tied_particle_transport_resource_v2",
        "treatment_parameters": treatment_parameters,
        "abelian_control_parameters": abelian_control_parameters,
        "mechanistic_parameter_gap": 0,
        "dense_favorable_control_parameters": dense_control_parameters,
        "dense_favorable_control_extra_parameters": (
            dense_control_parameters - treatment_parameters
        ),
        "pair_readout_macs": FACTORIZED_MACS,
        "transport_macs_per_cue": TRANSPORT_MACS_PER_CUE,
        "treatment_transport_macs_per_cue": TRANSPORT_MACS_PER_CUE,
        "abelian_control_transport_macs_per_cue": TRANSPORT_MACS_PER_CUE,
        "dense_control_transport_macs_per_cue": TRANSPORT_MACS_PER_CUE,
        "complete_system_parameters": S4_TPT_COMPLETE_SYSTEM_PARAMETERS,
        "dense_control_complete_system_parameters": (
            DENSE_CONTROL_COMPLETE_SYSTEM_PARAMETERS
        ),
        "strict_system_parameter_limit": STRICT_SYSTEM_PARAMETER_LIMIT,
        "headroom": (
            STRICT_SYSTEM_PARAMETER_LIMIT - S4_TPT_COMPLETE_SYSTEM_PARAMETERS
        ),
        "dense_control_headroom": (
            STRICT_SYSTEM_PARAMETER_LIMIT
            - DENSE_CONTROL_COMPLETE_SYSTEM_PARAMETERS
        ),
    }
