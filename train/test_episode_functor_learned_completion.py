from __future__ import annotations

import pytest
import torch

from episode_functor_learned_completion import (
    LearnedCompletionError,
    LearnedRelationalCompletionProjector,
)
from episode_functor_witness_compiler import (
    ProofCarryingWitnessCompiler,
    collate_witness_sources,
    scan_witness_source,
)
from pipeline.episode_functor_identifiable_board import (
    GrammarFactors,
    encode_source,
    generate_machine,
    hide_one_cell_per_relation,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(83)
    return (
        torch.randn((2, 3, 8, 8), generator=generator),
        torch.randn((2, 2, 8, 4), generator=generator),
    )


def test_learned_completion_is_trainable_and_not_law_forced() -> None:
    projector = LearnedRelationalCompletionProjector(
        width=32,
        iterations=2,
    )
    assert projector.parameter_count() > 0
    for parameter in projector.parameters():
        parameter.data.zero_()
    transitions, observers = _inputs()
    soft = projector(transitions, observers).machine
    active_transitions = soft.action_next[:, :3, :8].argmax(-1)
    active_observers = soft.observer_answer[:, :2, :8].argmax(-1)
    assert torch.equal(
        active_transitions,
        torch.zeros_like(active_transitions),
    )
    assert torch.equal(
        active_observers,
        torch.zeros_like(active_observers),
    )
    assert any(
        sorted(row.tolist()) != list(range(8))
        for batch in active_transitions
        for row in batch
    )
    assert any(
        sorted(row.tolist()) != [0, 0, 1, 1, 2, 2, 3, 3]
        for batch in active_observers
        for row in batch
    )
    with pytest.raises(
        LearnedCompletionError,
        match="categorical hardening tie",
    ):
        projector.hard_project(transitions, observers)
    with pytest.raises(
        LearnedCompletionError,
        match="categorical hardening tie",
    ):
        projector(
            transitions,
            observers,
            straight_through=True,
        )


def test_learned_completion_is_state_permutation_equivariant() -> None:
    torch.manual_seed(89)
    projector = LearnedRelationalCompletionProjector(
        width=32,
        iterations=2,
    )
    transitions, observers = _inputs()
    permutation = torch.tensor((3, 0, 7, 2, 5, 1, 6, 4))
    transition_permuted = transitions[
        :,
        :,
        permutation,
    ][
        :,
        :,
        :,
        permutation,
    ]
    observer_permuted = observers[:, :, permutation]
    original = projector(transitions, observers)
    changed = projector(transition_permuted, observer_permuted)
    assert torch.allclose(
        changed.transition_transport,
        original.transition_transport[
            :,
            :,
            permutation,
        ][
            :,
            :,
            :,
            permutation,
        ],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        changed.observer_transport,
        original.observer_transport[:, :, permutation],
        atol=1e-6,
        rtol=1e-6,
    )


def test_unique_hard_completion_is_exactly_recode_equivariant() -> None:
    torch.manual_seed(109)
    projector = LearnedRelationalCompletionProjector(
        width=32,
        iterations=2,
    )
    generator = torch.Generator().manual_seed(113)
    transitions = torch.randn((2, 3, 8, 8), generator=generator)
    observers = torch.randn((2, 2, 8, 4), generator=generator)
    state_permutation = torch.tensor((3, 0, 7, 2, 5, 1, 6, 4))
    action_permutation = torch.tensor((2, 0, 1))
    observer_permutation = torch.tensor((1, 0))
    answer_permutation = torch.tensor((2, 0, 3, 1))
    inverse_state = torch.empty_like(state_permutation)
    inverse_state[state_permutation] = torch.arange(8)
    inverse_answer = torch.empty_like(answer_permutation)
    inverse_answer[answer_permutation] = torch.arange(4)

    original = projector.hard_project(transitions, observers)
    changed = projector.hard_project(
        transitions[:, action_permutation][
            :,
            :,
            state_permutation,
        ][
            :,
            :,
            :,
            state_permutation,
        ],
        observers[:, observer_permutation][
            :,
            :,
            state_permutation,
        ][
            :,
            :,
            :,
            answer_permutation,
        ],
    )
    expected_transition = inverse_state[
        original.action_next[
            :,
            :3,
        ][
            :,
            action_permutation,
        ][
            :,
            :,
            state_permutation,
        ].long()
    ]
    expected_observer = inverse_answer[
        original.observer_answer[
            :,
            :2,
        ][
            :,
            observer_permutation,
        ][
            :,
            :,
            state_permutation,
        ].long()
    ]

    assert torch.equal(
        changed.action_next[:, :3, :8].long(),
        expected_transition,
    )
    assert torch.equal(
        changed.observer_answer[:, :2, :8].long(),
        expected_observer,
    )


def test_no_host_projector_plugs_into_witness_compiler_and_backpropagates() -> None:
    torch.manual_seed(97)
    projector = LearnedRelationalCompletionProjector(
        width=32,
        iterations=2,
    )
    compiler = ProofCarryingWitnessCompiler(
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        sinkhorn_iterations=16,
        projector=projector,
    )
    assert compiler.projector is projector
    machine = generate_machine(
        seed="efc-learned-completion-forward-v1",
        split="mechanics",
        index=0,
        family="affine-f2-3",
    )
    evidence = hide_one_cell_per_relation(
        machine,
        seed="efc-learned-completion-forward-v1",
        split="mechanics",
        index=0,
    )
    source = encode_source(evidence, GrammarFactors(1, 1, 1))
    compiled = compiler(
        collate_witness_sources([scan_witness_source(source)]),
        straight_through=True,
    )
    assert compiled.projection.machine.batch_size == 1
    assert compiled.key_assignment_logits.shape == (1, 32, 32)

    transitions, observers = _inputs()
    transitions.requires_grad_()
    observers.requires_grad_()
    output = projector(
        transitions,
        observers,
        straight_through=True,
    )
    loss = (
        output.machine.action_next[:, :3, :8, :8].sum()
        + output.machine.observer_answer[:, :2, :8, :4].sum()
    )
    loss.backward()
    assert transitions.grad is not None
    assert observers.grad is not None
    assert bool(torch.isfinite(transitions.grad).all())
    assert bool(torch.isfinite(observers.grad).all())
    assert float(transitions.grad.abs().sum()) > 0.0
    assert float(observers.grad.abs().sum()) > 0.0
