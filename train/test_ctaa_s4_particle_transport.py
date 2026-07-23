from __future__ import annotations

import itertools

import pytest
import torch

from ctaa_binding_completion import (
    ACTION_COUNT,
    BINDINGS,
    BINDING_TO_INDEX,
    COMPILER_WIDTH,
    RELATION_SLOT_COUNT,
    BindingCompletionError,
)
from ctaa_s4_particle_transport import (
    ACTION_EVENT,
    CUE_EVENT,
    DENSE_CONTROL_COMPLETE_SYSTEM_PARAMETERS,
    DENSE_CONTROL_PARAMETERS,
    S4_TPT_COMPLETE_SYSTEM_PARAMETERS,
    S4_TPT_WORKSPACE_PARAMETERS,
    IDENTITY_INDEX,
    PARTICLE_COUNT,
    S4_DELTA_INDEX,
    S4_GENERATOR_INDICES,
    STOP_EVENT,
    STRICT_SYSTEM_PARAMETER_LIMIT,
    Z24_DELTA_INDEX,
    Dense24BindingTransportControl,
    Dense24TransportControl,
    S4TiedBindingWorkspace,
    S4TiedTransport,
    Z24CircularTransportControl,
    Z24BindingTransportControl,
    apply_kernel_sequence,
    binding_particle_scores,
    conjugate_rebinding_element,
    compose_permutations,
    execute_interleaved_particle_ctaa,
    execute_particle_ctaa,
    group_convolve,
    s4_transport_resource_receipt,
    invert_permutation,
    lift_group_kernel_logits_to_dense,
    one_hot_group_element,
    reindex_kernel_probabilities,
    reindex_particle_probabilities,
    transform_binding_coordinates,
)


def test_s4_group_tables_are_closed_and_invertible() -> None:
    identity = tuple(range(ACTION_COUNT))
    for left in BINDINGS:
        inverse = invert_permutation(left)
        assert compose_permutations(left, inverse) == identity
        assert compose_permutations(inverse, left) == identity
        for right in BINDINGS:
            assert compose_permutations(left, right) in BINDING_TO_INDEX
    assert S4_DELTA_INDEX.shape == Z24_DELTA_INDEX.shape == (
        PARTICLE_COUNT,
        PARTICLE_COUNT,
    )
    assert all(
        sorted(row.tolist()) == list(range(PARTICLE_COUNT))
        for row in S4_DELTA_INDEX
    )


def test_group_convolution_preserves_probability_mass() -> None:
    state = torch.rand(5, PARTICLE_COUNT)
    state /= state.sum(-1, keepdim=True)
    kernel = torch.rand(5, PARTICLE_COUNT)
    kernel /= kernel.sum(-1, keepdim=True)
    for table in (S4_DELTA_INDEX, Z24_DELTA_INDEX):
        output = group_convolve(state, kernel, table)
        torch.testing.assert_close(
            output.sum(-1),
            torch.ones(5),
            rtol=1e-6,
            atol=1e-6,
        )
        assert bool((output >= 0).all())


def test_nonabelian_transport_retains_order_while_z24_control_cannot() -> None:
    first = S4_GENERATOR_INDICES[0]
    second = S4_GENERATOR_INDICES[3]
    left = compose_permutations(BINDINGS[first], BINDINGS[second])
    right = compose_permutations(BINDINGS[second], BINDINGS[first])
    assert left != right

    initial = one_hot_group_element(IDENTITY_INDEX)
    first_kernel = one_hot_group_element(first)
    second_kernel = one_hot_group_element(second)
    forward = torch.stack((first_kernel, second_kernel), dim=1)
    reverse = torch.stack((second_kernel, first_kernel), dim=1)

    s4_forward = apply_kernel_sequence(initial, forward, S4_DELTA_INDEX)
    s4_reverse = apply_kernel_sequence(initial, reverse, S4_DELTA_INDEX)
    assert int(s4_forward.argmax(-1)) == BINDING_TO_INDEX[left]
    assert int(s4_reverse.argmax(-1)) == BINDING_TO_INDEX[right]
    assert not torch.equal(s4_forward, s4_reverse)

    z24_forward = apply_kernel_sequence(initial, forward, Z24_DELTA_INDEX)
    z24_reverse = apply_kernel_sequence(initial, reverse, Z24_DELTA_INDEX)
    torch.testing.assert_close(z24_forward, z24_reverse, rtol=0, atol=0)


def test_learned_treatment_and_control_are_exactly_resource_matched() -> None:
    treatment = S4TiedTransport()
    control = Z24CircularTransportControl()
    assert treatment.unique_parameters == control.unique_parameters == 6 * 24
    receipt = s4_transport_resource_receipt()
    assert receipt["treatment_parameters"] == S4_TPT_WORKSPACE_PARAMETERS
    assert receipt["abelian_control_parameters"] == S4_TPT_WORKSPACE_PARAMETERS
    assert receipt["mechanistic_parameter_gap"] == 0
    assert receipt["dense_favorable_control_parameters"] == DENSE_CONTROL_PARAMETERS
    assert (
        receipt["dense_control_complete_system_parameters"]
        == DENSE_CONTROL_COMPLETE_SYSTEM_PARAMETERS
        < STRICT_SYSTEM_PARAMETER_LIMIT
    )
    assert (
        receipt["complete_system_parameters"]
        == S4_TPT_COMPLETE_SYSTEM_PARAMETERS
        < STRICT_SYSTEM_PARAMETER_LIMIT
    )
    assert receipt["headroom"] == (
        STRICT_SYSTEM_PARAMETER_LIMIT - S4_TPT_COMPLETE_SYSTEM_PARAMETERS
    )


def test_dense_favorable_control_exactly_contains_s4_transport() -> None:
    treatment = S4TiedTransport()
    dense = Dense24TransportControl()
    with torch.no_grad():
        dense.transition_logits.copy_(
            lift_group_kernel_logits_to_dense(treatment.kernel_logits)
        )
    initial = torch.rand(4, PARTICLE_COUNT)
    initial /= initial.sum(-1, keepdim=True)
    cues = torch.tensor(
        [[0, 1, 2, 3], [3, 2, 1, 0], [5, 4, 3, 2], [1, 1, 4, 4]]
    )
    torch.testing.assert_close(
        dense(initial, cues),
        treatment(initial, cues),
        rtol=1e-6,
        atol=1e-7,
    )
    assert dense.unique_parameters > treatment.unique_parameters


def test_workspace_arms_share_geometry_and_gradients() -> None:
    treatment = S4TiedBindingWorkspace()
    control = Z24BindingTransportControl()
    original_control_table = control.transport.delta_index.clone()
    control.load_state_dict(treatment.state_dict(), strict=True)
    torch.testing.assert_close(
        control.transport.delta_index,
        original_control_table,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        original_control_table,
        Z24_DELTA_INDEX,
        rtol=0,
        atol=0,
    )
    slots = torch.randn(
        3,
        RELATION_SLOT_COUNT,
        COMPILER_WIDTH,
        requires_grad=True,
    )
    cues = torch.tensor([[0, 1, 2], [2, 1, 0], [3, 4, 5]])
    treatment_output = treatment(slots, cues)
    control_output = control(slots.detach(), cues)
    assert treatment_output.pair_logits.shape == (3, 4, 4)
    assert treatment_output.initial_particles.shape == (3, PARTICLE_COUNT)
    assert treatment_output.transported_particles.shape == (3, PARTICLE_COUNT)
    torch.testing.assert_close(
        treatment_output.initial_particles,
        control_output.initial_particles,
    )
    loss = -treatment_output.transported_particles[:, 0].log().mean()
    loss.backward()
    assert slots.grad is not None
    assert all(parameter.grad is not None for parameter in treatment.parameters())

    dense = Dense24BindingTransportControl()
    dense_output = dense(slots.detach(), cues)
    assert dense_output.transported_particles.shape == (3, PARTICLE_COUNT)


def test_binding_particle_scores_match_exhaustive_assignment_energy() -> None:
    pair_logits = torch.randn(7, ACTION_COUNT, ACTION_COUNT)
    scores = binding_particle_scores(pair_logits)
    for batch in range(pair_logits.shape[0]):
        expected = torch.tensor(
            [
                sum(
                    float(pair_logits[batch, opcode, card])
                    for opcode, card in enumerate(binding)
                )
                for binding in BINDINGS
            ]
        )
        torch.testing.assert_close(scores[batch], expected)


def test_particle_coordinates_are_biequivariant_and_invertible() -> None:
    probabilities = torch.rand(2, PARTICLE_COUNT)
    probabilities /= probabilities.sum(-1, keepdim=True)
    for opcode_order in itertools.permutations(range(ACTION_COUNT)):
        opcode_inverse = invert_permutation(opcode_order)
        for card_order in itertools.permutations(range(ACTION_COUNT)):
            card_inverse = invert_permutation(card_order)
            transformed = reindex_particle_probabilities(
                probabilities,
                opcode_order,
                card_order,
            )
            restored = reindex_particle_probabilities(
                transformed,
                opcode_inverse,
                card_inverse,
            )
            torch.testing.assert_close(restored, probabilities, rtol=0, atol=0)
            for binding in BINDINGS:
                assert transform_binding_coordinates(
                    binding,
                    opcode_order,
                    card_order,
                ) in BINDING_TO_INDEX


def test_s4_transport_is_equivariant_with_conjugated_cues() -> None:
    for binding in BINDINGS:
        initial = one_hot_group_element(BINDING_TO_INDEX[binding])
        for opcode_order in BINDINGS:
            for card_order in BINDINGS:
                transformed_initial = reindex_particle_probabilities(
                    initial,
                    opcode_order,
                    card_order,
                )
                for generator_index in S4_GENERATOR_INDICES:
                    kernel = one_hot_group_element(generator_index)
                    transformed_kernel = reindex_kernel_probabilities(
                        kernel,
                        opcode_order,
                    )
                    expected = reindex_particle_probabilities(
                        group_convolve(initial, kernel, S4_DELTA_INDEX),
                        opcode_order,
                        card_order,
                    )
                    actual = group_convolve(
                        transformed_initial,
                        transformed_kernel,
                        S4_DELTA_INDEX,
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    transformed_element = conjugate_rebinding_element(
                        BINDINGS[generator_index],
                        opcode_order,
                    )
                    assert (
                        int(transformed_kernel.argmax(-1))
                        == BINDING_TO_INDEX[transformed_element]
                    )


def test_transport_arms_accept_empty_cue_sequences() -> None:
    initial = torch.rand(3, PARTICLE_COUNT)
    initial /= initial.sum(-1, keepdim=True)
    empty = torch.empty((3, 0), dtype=torch.long)
    for transport in (
        S4TiedTransport(),
        Z24CircularTransportControl(),
        Dense24TransportControl(),
    ):
        torch.testing.assert_close(
            transport(initial, empty),
            initial,
            rtol=0,
            atol=0,
        )


def test_interleaved_workspace_updates_binding_and_state_in_event_order() -> None:
    transport = S4TiedTransport()
    with torch.no_grad():
        transport.kernel_logits.fill_(-20)
        for cue, generator_index in enumerate(S4_GENERATOR_INDICES):
            transport.kernel_logits[cue, generator_index] = 20
    particles = one_hot_group_element(IDENTITY_INDEX)
    cards = torch.tensor(
        [[[1, 0, 2], [0, 2, 1], [2, 1, 0], [0, 1, 2]]],
        dtype=torch.long,
    )
    initial = torch.tensor([[0, 1, 2]], dtype=torch.long)
    kinds = torch.tensor(
        [[ACTION_EVENT, CUE_EVENT, ACTION_EVENT, STOP_EVENT]],
        dtype=torch.long,
    )
    values = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    result = execute_interleaved_particle_ctaa(
        transport,
        particles,
        cards,
        initial,
        kinds,
        values,
        torch.tensor([0], dtype=torch.long),
    )
    assert result.halted.tolist() == [True]
    assert result.binding_trajectory.shape == (1, 5, PARTICLE_COUNT)
    assert result.full_state_trajectory.shape == (1, 5, 27)
    torch.testing.assert_close(
        result.query_distribution.sum(-1),
        torch.ones(1),
    )
    assert int(result.binding_marginals.argmax(-1)) == S4_GENERATOR_INDICES[0]

    cue_first = execute_interleaved_particle_ctaa(
        transport,
        particles,
        cards,
        initial,
        torch.tensor(
            [[CUE_EVENT, ACTION_EVENT, ACTION_EVENT, STOP_EVENT]],
            dtype=torch.long,
        ),
        values,
        torch.tensor([0], dtype=torch.long),
    )
    assert not torch.equal(
        result.full_state_marginals.argmax(-1),
        cue_first.full_state_marginals.argmax(-1),
    )


def test_interleaved_workspace_supports_action_then_stop_without_cues() -> None:
    transport = S4TiedTransport()
    result = execute_interleaved_particle_ctaa(
        transport,
        one_hot_group_element(IDENTITY_INDEX),
        torch.tensor(
            [[[1, 0, 2], [0, 2, 1], [2, 1, 0], [0, 1, 2]]],
            dtype=torch.long,
        ),
        torch.tensor([[0, 1, 2]], dtype=torch.long),
        torch.tensor([[ACTION_EVENT, STOP_EVENT]], dtype=torch.long),
        torch.tensor([[0, 0]], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    )
    assert result.halted.tolist() == [True]
    assert int(result.full_state_marginals.argmax(-1)) == 11
    assert int(result.query_distribution.argmax(-1)) == 0


def test_interleaved_workspace_matches_independent_path_enumeration() -> None:
    def oracle_compose(
        left: tuple[int, ...],
        right: tuple[int, ...],
    ) -> tuple[int, ...]:
        return tuple(left[right[index]] for index in range(ACTION_COUNT))

    transport = S4TiedTransport()
    with torch.no_grad():
        values = torch.arange(
            transport.kernel_logits.numel(),
            dtype=torch.float,
        ).reshape_as(transport.kernel_logits)
        transport.kernel_logits.copy_((values.remainder(11) - 5) / 3)
    particles = torch.arange(1, PARTICLE_COUNT + 1, dtype=torch.float)[None]
    particles /= particles.sum(-1, keepdim=True)
    cards = torch.tensor(
        [[[1, 0, 2], [0, 2, 1], [2, 1, 0], [0, 1, 2]]],
        dtype=torch.long,
    )
    initial = torch.tensor([[2, 0, 1]], dtype=torch.long)
    kinds = torch.tensor(
        [[CUE_EVENT, ACTION_EVENT, CUE_EVENT, ACTION_EVENT, STOP_EVENT]],
        dtype=torch.long,
    )
    event_values = torch.tensor([[0, 0, 1, 2, 0]], dtype=torch.long)
    result = execute_interleaved_particle_ctaa(
        transport,
        particles,
        cards,
        initial,
        kinds,
        event_values,
        torch.tensor([2], dtype=torch.long),
    )

    kernels = transport.kernel_logits.float().softmax(-1)
    paths: dict[tuple[int, tuple[int, int, int]], float] = {
        (particle, (2, 0, 1)): float(particles[0, particle])
        for particle in range(PARTICLE_COUNT)
    }
    for kind, value in zip(kinds[0].tolist(), event_values[0].tolist()):
        updated: dict[tuple[int, tuple[int, int, int]], float] = {}
        if kind == CUE_EVENT:
            for (source, state), probability in paths.items():
                for delta, kernel_probability in enumerate(
                    kernels[value].tolist()
                ):
                    destination = BINDING_TO_INDEX[
                        oracle_compose(BINDINGS[source], BINDINGS[delta])
                    ]
                    key = (destination, state)
                    updated[key] = (
                        updated.get(key, 0.0)
                        + probability * kernel_probability
                    )
        elif kind == ACTION_EVENT:
            for (binding_index, state), probability in paths.items():
                card_index = BINDINGS[binding_index][value]
                card = cards[0, card_index].tolist()
                new_state = tuple(state[index] for index in card)
                key = (binding_index, new_state)
                updated[key] = updated.get(key, 0.0) + probability
        else:
            updated = paths
        paths = updated

    expected = torch.zeros(PARTICLE_COUNT, 27)
    for (binding_index, state), probability in paths.items():
        state_index = state[0] * 9 + state[1] * 3 + state[2]
        expected[binding_index, state_index] += probability
    torch.testing.assert_close(
        result.final_joint[0],
        expected,
        rtol=2e-5,
        atol=2e-7,
    )
    expected_query = torch.zeros(3)
    for state_index, probability in enumerate(expected.sum(0)):
        expected_query[(state_index % 3)] += probability
    torch.testing.assert_close(
        result.query_distribution[0],
        expected_query,
        rtol=2e-5,
        atol=2e-7,
    )


def test_particle_executor_exposes_order_sensitive_transport_at_late_read() -> None:
    first = S4_GENERATOR_INDICES[0]
    second = S4_GENERATOR_INDICES[3]
    initial_particles = one_hot_group_element(IDENTITY_INDEX)
    first_kernel = one_hot_group_element(first)
    second_kernel = one_hot_group_element(second)
    forward = apply_kernel_sequence(
        initial_particles,
        torch.stack((first_kernel, second_kernel), dim=1),
        S4_DELTA_INDEX,
    )
    reverse = apply_kernel_sequence(
        initial_particles,
        torch.stack((second_kernel, first_kernel), dim=1),
        S4_DELTA_INDEX,
    )
    action_cards = torch.tensor(
        [
            [
                [0, 1, 2],
                [1, 0, 2],
                [0, 2, 1],
                [2, 1, 0],
            ]
        ],
        dtype=torch.long,
    )
    initial_state = torch.tensor([[0, 1, 2]], dtype=torch.long)
    schedule = torch.tensor([[0, 4]], dtype=torch.long)
    forward_execution = execute_particle_ctaa(
        forward,
        action_cards,
        initial_state,
        schedule,
    )
    reverse_execution = execute_particle_ctaa(
        reverse,
        action_cards,
        initial_state,
        schedule,
    )
    assert not torch.equal(
        forward_execution.final_state_marginals,
        reverse_execution.final_state_marginals,
    )
    assert forward_execution.halted.tolist() == [True]
    assert reverse_execution.halted.tolist() == [True]


def test_particle_executor_is_source_deleted_and_rejects_invalid_packets() -> None:
    probabilities = torch.full((1, PARTICLE_COUNT), 1 / PARTICLE_COUNT)
    cards = torch.tensor(
        [[[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1]]],
        dtype=torch.long,
    )
    initial = torch.tensor([[2, 1, 0]], dtype=torch.long)
    schedule = torch.tensor([[1, 2, 4]], dtype=torch.long)
    baseline = execute_particle_ctaa(probabilities, cards, initial, schedule)
    repeated = execute_particle_ctaa(
        probabilities.clone(),
        cards.clone(),
        initial.clone(),
        schedule.clone(),
    )
    torch.testing.assert_close(
        baseline.final_state_marginals,
        repeated.final_state_marginals,
        rtol=0,
        atol=0,
    )
    with pytest.raises(BindingCompletionError, match="values"):
        execute_particle_ctaa(
            probabilities,
            cards,
            initial,
            torch.tensor([[1, 2, 3]], dtype=torch.long),
        )
