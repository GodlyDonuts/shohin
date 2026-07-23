"""Deterministic train-only card batches for the learned contextual binder."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from contextual_relation_primitive_compiler import (
    PRIMITIVE_COUNT,
    relation_primitive_candidates,
)
from contextual_witness_equivariant_binder import (
    CARD_WITNESSES,
    MAX_OBJECTS,
    MAX_OPERATION_SLOTS,
    REJECT_INDEX,
)


PRIMITIVE_ARITY = torch.tensor((2, 2, 2, 1, 0), dtype=torch.long)
TRAIN_DENSITIES = (0.08, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.92)
SHIFT_DENSITIES = (0.04, 0.12, 0.28, 0.42, 0.58, 0.72, 0.88, 0.96)


class ContextualCardDataError(ValueError):
    """Raised when a synthetic card batch contract differs."""


@dataclass(frozen=True, slots=True)
class ContextualCardBatch:
    witness_left: torch.Tensor
    witness_right: torch.Tensor
    witness_output: torch.Tensor
    witness_mask: torch.Tensor
    argument_mask: torch.Tensor
    object_mask: torch.Tensor
    labels: torch.Tensor
    cardinality: torch.Tensor

    def to(self, device: torch.device | str) -> ContextualCardBatch:
        destination = torch.device(device)
        return ContextualCardBatch(
            witness_left=self.witness_left.to(destination),
            witness_right=self.witness_right.to(destination),
            witness_output=self.witness_output.to(destination),
            witness_mask=self.witness_mask.to(destination),
            argument_mask=self.argument_mask.to(destination),
            object_mask=self.object_mask.to(destination),
            labels=self.labels.to(destination),
            cardinality=self.cardinality.to(destination),
        )


def _random_relations(
    batch_size: int,
    generator: torch.Generator,
    object_mask: torch.Tensor,
    densities: tuple[float, ...],
) -> torch.Tensor:
    density = torch.tensor(densities)[
        torch.randint(
            len(densities),
            (
                batch_size,
                MAX_OPERATION_SLOTS,
                CARD_WITNESSES,
            ),
            generator=generator,
        )
    ]
    values = torch.rand(
        batch_size,
        MAX_OPERATION_SLOTS,
        CARD_WITNESSES,
        MAX_OBJECTS,
        MAX_OBJECTS,
        generator=generator,
    )
    active = (
        object_mask[:, None, None, :, None]
        & object_mask[
            :,
            None,
            None,
            None,
            :,
        ]
    )
    return (values < density[..., None, None]).float() * active.float()


def _derange_one_card(
    output: torch.Tensor,
    candidates: torch.Tensor,
    batch_index: int,
    slot: int,
) -> bool:
    for shift in range(1, CARD_WITNESSES):
        proposal = output[batch_index, slot].roll(shift, dims=0)
        compatible = (
            candidates[batch_index, slot]
            .eq(proposal[:, None])
            .flatten(2)
            .all(-1)
            .all(0)
        )
        if not bool(compatible.any()):
            output[batch_index, slot] = proposal
            return True
    return False


def generate_contextual_card_batch(
    *,
    batch_size: int,
    generator: torch.Generator,
    cardinalities: tuple[int, ...] = (3, 4, 5, 6),
    densities: tuple[float, ...] = TRAIN_DENSITIES,
    invalid_fraction: float = 0.25,
) -> ContextualCardBatch:
    """Generate balanced opaque cards; analytic laws remain data-side only."""

    if (
        batch_size < 1
        or not cardinalities
        or any(not 2 <= value <= MAX_OBJECTS for value in cardinalities)
        or len(densities) != CARD_WITNESSES
        or not 0.0 <= invalid_fraction <= 1.0
    ):
        raise ContextualCardDataError("card batch request differs")
    cardinality_choices = torch.tensor(cardinalities, dtype=torch.long)
    cardinality = cardinality_choices[
        torch.randint(
            len(cardinality_choices),
            (batch_size,),
            generator=generator,
        )
    ]
    positions = torch.arange(MAX_OBJECTS)
    object_mask = positions[None] < cardinality[:, None]
    left = _random_relations(
        batch_size,
        generator,
        object_mask,
        densities,
    )
    right = _random_relations(
        batch_size,
        generator,
        object_mask,
        tuple(reversed(densities)),
    )
    candidates = relation_primitive_candidates(left, right, object_mask)
    labels = torch.full(
        (batch_size, MAX_OPERATION_SLOTS),
        -100,
        dtype=torch.long,
    )
    active_slots = torch.zeros_like(labels, dtype=torch.bool)
    for batch_index in range(batch_size):
        slots = torch.randperm(
            MAX_OPERATION_SLOTS,
            generator=generator,
        )[:PRIMITIVE_COUNT]
        classes = torch.randperm(PRIMITIVE_COUNT, generator=generator)
        labels[batch_index, slots] = classes
        active_slots[batch_index, slots] = True

    gather = labels.clamp_min(0)[..., None, None, None, None].expand(
        -1,
        -1,
        CARD_WITNESSES,
        1,
        MAX_OBJECTS,
        MAX_OBJECTS,
    )
    output = candidates.gather(3, gather).squeeze(3)
    active_relation = active_slots[..., None, None, None].float()
    left = left * active_relation
    right = right * active_relation
    output = output * active_relation
    witness_mask = active_slots[..., None].expand(
        -1,
        -1,
        CARD_WITNESSES,
    )
    arity = torch.where(
        active_slots,
        PRIMITIVE_ARITY[labels.clamp_min(0)],
        torch.full_like(labels, -1),
    )
    argument_mask = (
        torch.arange(2)[None, None, None] < arity.clamp_min(0)[..., None, None]
    ).expand(-1, -1, CARD_WITNESSES, -1)
    left = left * argument_mask[..., 0, None, None]
    right = right * argument_mask[..., 1, None, None]

    invalid_count = round(batch_size * invalid_fraction)
    invalid_rows = torch.randperm(
        batch_size,
        generator=generator,
    )[:invalid_count]
    for batch_index in invalid_rows.tolist():
        binary = (
            (labels[batch_index] >= 0)
            & (labels[batch_index] < 3)
        ).nonzero().flatten()
        order = binary[
            torch.randperm(
                len(binary),
                generator=generator,
            )
        ]
        chosen = next(
            (
                int(slot)
                for slot in order
                if _derange_one_card(
                    output,
                    candidates,
                    batch_index,
                    int(slot),
                )
            ),
            None,
        )
        if chosen is None:
            raise ContextualCardDataError(
                "could not derange any binary training card"
            )
        labels[batch_index, chosen] = REJECT_INDEX

    return ContextualCardBatch(
        witness_left=left,
        witness_right=right,
        witness_output=output,
        witness_mask=witness_mask,
        argument_mask=argument_mask,
        object_mask=object_mask,
        labels=labels,
        cardinality=cardinality,
    )
