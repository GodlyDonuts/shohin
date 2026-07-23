from __future__ import annotations

import math

import pytest
import torch

from contextual_relation_primitive_compiler import (
    COMPOSE_INDEX,
    CONVERSE_INDEX,
    IDENTITY_INDEX,
    INTERSECTION_INDEX,
    PRIMITIVE_COUNT,
    PROTECTED_BASE_PARAMETERS,
    STRICT_SYSTEM_CAP,
    UNION_INDEX,
    ContextualRelationPrimitiveCompiler,
    PrimitiveCompilerError,
    relation_primitive_candidates,
)


ARITIES = torch.tensor((2, 2, 2, 1, 0), dtype=torch.long)


def _permute_objects(value: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return value.index_select(-2, permutation).index_select(-1, permutation)


def _argument_mask(
    batch: int,
    slots: int,
    witnesses: int,
    arities: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(2)
    return positions.view(1, 1, 1, 2) < arities.view(
        1, slots, 1, 1
    ).expand(batch, -1, witnesses, -1)


def _identifiable_cards(
    *,
    batch: int = 2,
    witnesses: int = 8,
    objects: int = 5,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(2026072309)
    object_mask = torch.ones(batch, objects, dtype=torch.bool)
    for _ in range(100):
        left = torch.randint(
            0,
            2,
            (batch, PRIMITIVE_COUNT, witnesses, objects, objects),
            generator=generator,
        ).float()
        right = torch.randint(
            0,
            2,
            left.shape,
            generator=generator,
        ).float()
        candidates = relation_primitive_candidates(left, right, object_mask)
        primitive = torch.arange(PRIMITIVE_COUNT)
        output = candidates[
            torch.arange(batch)[:, None, None],
            torch.arange(PRIMITIVE_COUNT)[None, :, None],
            torch.arange(witnesses)[None, None, :],
            primitive[None, :, None],
        ]
        witness_mask = torch.ones(
            batch,
            PRIMITIVE_COUNT,
            witnesses,
            dtype=torch.bool,
        )
        argument_mask = _argument_mask(
            batch,
            PRIMITIVE_COUNT,
            witnesses,
            ARITIES,
        )
        compiler = ContextualRelationPrimitiveCompiler()
        compiled = compiler(
            left,
            right,
            output,
            witness_mask,
            argument_mask,
            object_mask,
            hard=True,
        )
        if bool(compiled.identifiable.all()):
            return (
                left,
                right,
                output,
                witness_mask,
                argument_mask,
                object_mask,
            )
    raise AssertionError("test fixture did not identify every primitive")


def test_exact_binding_under_fresh_slot_permutations_and_object_reindexing() -> None:
    cards = _identifiable_cards()
    left, right, output, witness_mask, argument_mask, object_mask = cards
    compiler = ContextualRelationPrimitiveCompiler()
    base = compiler(*cards, hard=True)
    assert base.identifiable.all()
    assert torch.equal(base.discrete_assignment, base.discrete_assignment.round())
    assert torch.equal(
        base.discrete_assignment.argmax(-1),
        torch.arange(PRIMITIVE_COUNT).expand(left.shape[0], -1),
    )

    slot_permutation = torch.tensor((3, 0, 4, 2, 1))
    object_permutation = torch.tensor((4, 2, 0, 3, 1))
    transformed = compiler(
        _permute_objects(left[:, slot_permutation], object_permutation),
        _permute_objects(right[:, slot_permutation], object_permutation),
        _permute_objects(output[:, slot_permutation], object_permutation),
        witness_mask[:, slot_permutation],
        argument_mask[:, slot_permutation],
        object_mask[:, object_permutation],
        hard=True,
    )
    assert transformed.identifiable.all()
    assert torch.equal(
        transformed.discrete_assignment.argmax(-1),
        slot_permutation.expand(left.shape[0], -1),
    )


def test_witness_and_card_order_invariance() -> None:
    cards = _identifiable_cards()
    left, right, output, witness_mask, argument_mask, object_mask = cards
    compiler = ContextualRelationPrimitiveCompiler()
    expected = compiler(*cards, hard=False)
    witness_permutation = torch.tensor((6, 2, 7, 0, 5, 1, 4, 3))
    card_permutation = torch.tensor((4, 1, 3, 0, 2))
    observed = compiler(
        left[:, card_permutation][:, :, witness_permutation],
        right[:, card_permutation][:, :, witness_permutation],
        output[:, card_permutation][:, :, witness_permutation],
        witness_mask[:, card_permutation][:, :, witness_permutation],
        argument_mask[:, card_permutation][:, :, witness_permutation],
        object_mask,
        hard=False,
    )
    assert torch.equal(
        observed.identifiable,
        expected.identifiable[:, card_permutation],
    )
    assert torch.allclose(
        observed.raw_compatibility_logits,
        expected.raw_compatibility_logits[:, card_permutation],
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        observed.assignment,
        expected.assignment[:, card_permutation],
        atol=1e-6,
        rtol=0.0,
    )


def test_one_witness_ambiguous_card_fails_closed() -> None:
    compiler = ContextualRelationPrimitiveCompiler()
    zeros = torch.zeros(1, 1, 1, 4, 4)
    witness_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    argument_mask = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    object_mask = torch.ones(1, 4, dtype=torch.bool)
    compiled = compiler(
        zeros,
        zeros,
        zeros,
        witness_mask,
        argument_mask,
        object_mask,
        hard=True,
    )
    assert not compiled.identifiable.item()
    assert compiled.compatible.sum().item() == 3
    assert compiled.assignment.count_nonzero().item() == 0
    assert compiled.discrete_assignment.count_nonzero().item() == 0
    with pytest.raises(PrimitiveCompilerError, match="not identifiable"):
        compiler.apply_compiled(
            compiled,
            zeros[:, :, 0],
            zeros[:, :, 0],
            argument_mask[:, :, 0],
            object_mask,
        )


def test_wrong_and_deranged_cards_fail_identifiability() -> None:
    cards = _identifiable_cards(batch=1)
    left, right, output, witness_mask, argument_mask, object_mask = cards
    compiler = ContextualRelationPrimitiveCompiler()

    wrong_output = output.clone()
    wrong_output[:, UNION_INDEX] = 0.5
    wrong = compiler(
        left,
        right,
        wrong_output,
        witness_mask,
        argument_mask,
        object_mask,
        hard=True,
    )
    assert not wrong.identifiable[:, UNION_INDEX].item()
    assert wrong.compatible[:, UNION_INDEX].sum().item() == 0
    assert wrong.assignment[:, UNION_INDEX].count_nonzero().item() == 0
    assert wrong.discrete_assignment[:, UNION_INDEX].count_nonzero().item() == 0

    deranged_output = output.roll(1, dims=1)
    deranged = compiler(
        left,
        right,
        deranged_output,
        witness_mask,
        argument_mask,
        object_mask,
        hard=True,
    )
    assert not deranged.identifiable.all()
    assert (
        deranged.assignment[~deranged.identifiable].count_nonzero().item()
        == 0
    )
    assert (
        deranged.discrete_assignment[
            ~deranged.identifiable
        ].count_nonzero().item()
        == 0
    )


def test_hard_compiled_application_is_exact_on_fresh_operands() -> None:
    cards = _identifiable_cards(batch=1)
    compiler = ContextualRelationPrimitiveCompiler()
    compiled = compiler(*cards, hard=True)
    generator = torch.Generator().manual_seed(311)
    left = torch.randint(
        0,
        2,
        (1, PRIMITIVE_COUNT, 5, 5),
        generator=generator,
    ).float()
    right = torch.randint(0, 2, left.shape, generator=generator).float()
    argument_mask = _argument_mask(
        1,
        PRIMITIVE_COUNT,
        1,
        ARITIES,
    )[:, :, 0]
    object_mask = torch.ones(1, 5, dtype=torch.bool)
    observed = compiler.apply_compiled(
        compiled,
        left,
        right,
        argument_mask,
        object_mask,
    )
    candidates = relation_primitive_candidates(left, right, object_mask)
    expected = candidates[
        torch.arange(1)[:, None],
        torch.arange(PRIMITIVE_COUNT)[None],
        torch.arange(PRIMITIVE_COUNT)[None],
    ]
    assert torch.equal(observed, expected)
    assert {
        UNION_INDEX,
        INTERSECTION_INDEX,
        COMPOSE_INDEX,
        CONVERSE_INDEX,
        IDENTITY_INDEX,
    } == set(compiled.discrete_assignment.argmax(-1).flatten().tolist())


def test_wrong_soft_assignment_carries_corrective_gradient() -> None:
    cards = _identifiable_cards(batch=1)
    compiler = ContextualRelationPrimitiveCompiler(initial_logit_scale=0.01)
    compiled = compiler(*cards, hard=False)
    generator = torch.Generator().manual_seed(997)
    left = torch.randint(
        0,
        2,
        (1, PRIMITIVE_COUNT, 5, 5),
        generator=generator,
    ).float()
    right = torch.randint(0, 2, left.shape, generator=generator).float()
    argument_mask = _argument_mask(
        1,
        PRIMITIVE_COUNT,
        1,
        ARITIES,
    )[:, :, 0]
    object_mask = torch.ones(1, 5, dtype=torch.bool)
    observed = compiler.apply_compiled(
        compiled,
        left,
        right,
        argument_mask,
        object_mask,
    )
    candidates = relation_primitive_candidates(left, right, object_mask)
    target = candidates[
        torch.arange(1)[:, None],
        torch.arange(PRIMITIVE_COUNT)[None],
        torch.arange(PRIMITIVE_COUNT)[None],
    ]
    loss = (observed - target).square().mean()
    loss.backward()
    assert loss.item() > 0.0
    assert compiler.logit_scale.grad is not None
    assert math.isfinite(compiler.logit_scale.grad.item())
    assert abs(compiler.logit_scale.grad.item()) > 0.0


def test_hard_execution_is_exact_while_surrogate_carries_gradient() -> None:
    cards = _identifiable_cards(batch=1)
    compiler = ContextualRelationPrimitiveCompiler(initial_logit_scale=0.01)
    compiled = compiler(*cards, hard=True)
    assert torch.equal(
        compiled.discrete_assignment,
        compiled.discrete_assignment.round(),
    )
    assert torch.equal(
        compiled.discrete_assignment.sum(-1),
        torch.ones_like(compiled.discrete_assignment[..., 0]),
    )
    loss = compiled.assignment[..., UNION_INDEX].sum()
    loss.backward()
    assert compiler.logit_scale.grad is not None
    assert math.isfinite(compiler.logit_scale.grad.item())
    assert abs(compiler.logit_scale.grad.item()) > 0.0


def test_unary_and_nullary_masks_are_legal_and_parameter_receipt_is_strict() -> None:
    cards = _identifiable_cards(batch=1)
    compiler = ContextualRelationPrimitiveCompiler()
    compiled = compiler(*cards, hard=False)
    assert compiled.legal[:, :3].sum(-1).eq(3).all()
    assert compiled.legal[:, CONVERSE_INDEX].sum(-1).eq(1).all()
    assert compiled.legal[:, IDENTITY_INDEX].sum(-1).eq(1).all()
    assert compiled.legal[0, CONVERSE_INDEX, CONVERSE_INDEX]
    assert compiled.legal[0, IDENTITY_INDEX, IDENTITY_INDEX]

    receipt = compiler.parameter_receipt()
    assert receipt == {
        "base": PROTECTED_BASE_PARAMETERS,
        "added": 1,
        "complete_system": PROTECTED_BASE_PARAMETERS + 1,
        "strict_cap": STRICT_SYSTEM_CAP,
        "headroom": STRICT_SYSTEM_CAP - PROTECTED_BASE_PARAMETERS - 1,
    }
    assert receipt["complete_system"] < STRICT_SYSTEM_CAP
