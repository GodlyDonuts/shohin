from __future__ import annotations

import torch

from contextual_relation_primitive_compiler import (
    ContextualRelationPrimitiveCompiler,
)
from contextual_witness_card_data import (
    generate_contextual_card_batch,
)
from contextual_witness_equivariant_binder import (
    REJECT_INDEX,
)


def test_card_batch_is_deterministic_balanced_and_structurally_valid() -> None:
    first = generate_contextual_card_batch(
        batch_size=12,
        generator=torch.Generator().manual_seed(2026072325),
    )
    second = generate_contextual_card_batch(
        batch_size=12,
        generator=torch.Generator().manual_seed(2026072325),
    )
    for field in first.__dataclass_fields__:
        assert torch.equal(getattr(first, field), getattr(second, field))
    assert first.witness_mask.sum(1).eq(5).all()
    assert first.labels.ne(-100).sum(1).eq(5).all()
    valid = first.labels[(first.labels >= 0) & (first.labels < 5)]
    assert set(valid.tolist()) == set(range(5))


def test_invalid_cards_preserve_structure_but_fail_analytic_binding() -> None:
    batch = generate_contextual_card_batch(
        batch_size=16,
        generator=torch.Generator().manual_seed(2026072326),
        invalid_fraction=1.0,
    )
    compiler = ContextualRelationPrimitiveCompiler()
    compiled = compiler(
        batch.witness_left,
        batch.witness_right,
        batch.witness_output,
        batch.witness_mask,
        batch.argument_mask,
        batch.object_mask,
        hard=True,
    )
    rejected = batch.labels.eq(REJECT_INDEX)
    assert rejected.sum() == 16
    assert not compiled.identifiable[rejected].any()
    assert compiled.discrete_assignment[rejected].count_nonzero() == 0
