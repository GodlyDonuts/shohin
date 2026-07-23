"""Learned equivariant binding of opaque relation-operation witness cards.

Unlike the analytic mechanics oracle, this module never evaluates a named
primitive candidate. It learns a card classifier from aligned witness
relations, structural arity, and object-equivariant pair/triadic messages.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_OBJECTS = 8
MAX_OPERATION_SLOTS = 8
CARD_WITNESSES = 8
PRIMITIVE_COUNT = 5
REJECT_INDEX = 5
BINDER_CLASS_COUNT = 6
PROTECTED_BASE_PARAMETERS = 125_081_664
STRICT_SYSTEM_CAP = 200_000_000

_CLASS_ARITY = (2, 2, 2, 1, 0, -1)


class ContextualWitnessBinderError(ValueError):
    """Raised when learned contextual-card input violates its contract."""


@dataclass(frozen=True, slots=True)
class LearnedPrimitiveBinding:
    logits: torch.Tensor
    probabilities: torch.Tensor
    assignment: torch.Tensor
    discrete_assignment: torch.Tensor
    rejected: torch.Tensor
    arity: torch.Tensor


def _infer_arity(
    witness_mask: torch.Tensor,
    argument_mask: torch.Tensor,
) -> torch.Tensor:
    if (
        witness_mask.shape
        != (
            witness_mask.shape[0],
            MAX_OPERATION_SLOTS,
            CARD_WITNESSES,
        )
        or witness_mask.dtype != torch.bool
        or argument_mask.shape
        != (
            witness_mask.shape[0],
            MAX_OPERATION_SLOTS,
            CARD_WITNESSES,
            2,
        )
        or argument_mask.dtype != torch.bool
    ):
        raise ContextualWitnessBinderError("witness mask geometry differs")
    if bool((argument_mask[..., 1] & ~argument_mask[..., 0] & witness_mask).any()):
        raise ContextualWitnessBinderError(
            "right argument exists without left argument"
        )
    per_witness = argument_mask.long().sum(-1)
    minimum = torch.where(
        witness_mask,
        per_witness,
        torch.full_like(per_witness, 3),
    ).amin(-1)
    maximum = torch.where(
        witness_mask,
        per_witness,
        torch.full_like(per_witness, -1),
    ).amax(-1)
    active = witness_mask.any(-1)
    if bool((active & minimum.ne(maximum)).any()):
        raise ContextualWitnessBinderError("card arity changes by witness")
    return torch.where(active, maximum, torch.full_like(maximum, -1))


def _validate_relations(
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
    object_mask: torch.Tensor,
) -> torch.Tensor:
    expected = (
        left.shape[0],
        MAX_OPERATION_SLOTS,
        CARD_WITNESSES,
        MAX_OBJECTS,
        MAX_OBJECTS,
    )
    if (
        left.shape != expected
        or right.shape != expected
        or output.shape != expected
        or not left.is_floating_point()
        or not right.is_floating_point()
        or not output.is_floating_point()
        or object_mask.shape != (left.shape[0], MAX_OBJECTS)
        or object_mask.dtype != torch.bool
    ):
        raise ContextualWitnessBinderError("witness relation geometry differs")
    for value in (left, right, output):
        if (
            not bool(torch.isfinite(value).all())
            or float(value.detach().amin()) < 0.0
            or float(value.detach().amax()) > 1.0
        ):
            raise ContextualWitnessBinderError("witness relation values differ")
    active = object_mask[:, :, None] & object_mask[:, None, :]
    outside = ~active[:, None, None]
    if any(
        bool(value.detach().masked_select(outside).ne(0).any())
        for value in (left, right, output)
    ):
        raise ContextualWitnessBinderError("witness relations use inactive objects")
    return active


class _EquivariantPairRound(nn.Module):
    def __init__(self, width: int, triad_mode: str) -> None:
        super().__init__()
        if triad_mode not in {"learned", "false", "zero"}:
            raise ContextualWitnessBinderError("triadic control differs")
        self.triad_mode = triad_mode
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
        pair_mask: torch.Tensor,
        object_count: torch.Tensor,
    ) -> torch.Tensor:
        mask = pair_mask[:, None, None, :, :, None].to(state.dtype)
        row_count = object_count[:, None, None, None, None, None]
        row = (state * mask).sum(4, keepdim=True) / row_count
        row = row.expand_as(state)
        column = (state * mask).sum(3, keepdim=True) / row_count
        column = column.expand_as(state)
        left = self.triad_left(state)
        right = self.triad_right(state)
        if self.triad_mode == "learned":
            triad = torch.einsum(
                "bswikd,bswkjd->bswijd",
                left,
                right,
            )
        elif self.triad_mode == "false":
            triad = torch.einsum(
                "bswkid,bswkjd->bswijd",
                left,
                right,
            )
        else:
            triad = torch.zeros_like(state)
        triad = triad / object_count[:, None, None, None, None, None]
        update = self.update(
            torch.cat(
                (
                    state,
                    row,
                    column,
                    state.transpose(3, 4),
                    triad,
                ),
                dim=-1,
            )
        )
        return self.norm(state + update) * mask


class ContextualWitnessEquivariantBinder(nn.Module):
    """Infer fresh opaque primitive identities without analytic candidates."""

    def __init__(
        self,
        *,
        width: int = 64,
        rounds: int = 2,
        triad_mode: str = "learned",
    ) -> None:
        super().__init__()
        if (
            width < 16
            or rounds < 1
            or triad_mode not in {"learned", "false", "zero"}
        ):
            raise ContextualWitnessBinderError("binder geometry differs")
        self.width = int(width)
        self.rounds = int(rounds)
        self.triad_mode = triad_mode
        self.pair_input = nn.Linear(10, width)
        self.pair_rounds = nn.ModuleList(
            _EquivariantPairRound(width, triad_mode)
            for _ in range(rounds)
        )
        self.witness_encoder = nn.Sequential(
            nn.Linear(2 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
            nn.LayerNorm(width),
        )
        self.card_classifier = nn.Sequential(
            nn.Linear(2 * width + 3, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, BINDER_CLASS_COUNT),
        )
        self.register_buffer(
            "_class_arity",
            torch.tensor(_CLASS_ARITY, dtype=torch.long),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def added_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_receipt(
        self,
        *,
        base_parameters: int = PROTECTED_BASE_PARAMETERS,
        strict_cap: int = STRICT_SYSTEM_CAP,
    ) -> dict[str, int]:
        added = self.added_parameters
        complete = int(base_parameters) + added
        if complete >= int(strict_cap):
            raise ContextualWitnessBinderError("learned binder reaches cap")
        return {
            "base": int(base_parameters),
            "added": int(added),
            "complete_system": complete,
            "strict_cap": int(strict_cap),
            "headroom": int(strict_cap) - complete,
        }

    def forward(
        self,
        witness_left: torch.Tensor,
        witness_right: torch.Tensor,
        witness_output: torch.Tensor,
        witness_mask: torch.Tensor,
        argument_mask: torch.Tensor,
        object_mask: torch.Tensor,
        *,
        hard: bool,
    ) -> LearnedPrimitiveBinding:
        pair_mask = _validate_relations(
            witness_left,
            witness_right,
            witness_output,
            object_mask,
        )
        arity = _infer_arity(witness_mask, argument_mask)
        batch = witness_left.shape[0]
        diagonal = torch.eye(
            MAX_OBJECTS,
            device=witness_left.device,
            dtype=witness_left.dtype,
        )[None, None, None].expand(
            batch,
            MAX_OPERATION_SLOTS,
            CARD_WITNESSES,
            -1,
            -1,
        )
        arity_features = F.one_hot(
            arity.clamp_min(0),
            3,
        ).to(witness_left.dtype)
        arity_pair = arity_features[
            :,
            :,
            None,
            None,
            None,
            :,
        ].expand(-1, -1, CARD_WITNESSES, MAX_OBJECTS, MAX_OBJECTS, -1)
        features = torch.cat(
            (
                witness_left[..., None],
                witness_right[..., None],
                witness_output[..., None],
                witness_left.transpose(-1, -2)[..., None],
                witness_right.transpose(-1, -2)[..., None],
                witness_output.transpose(-1, -2)[..., None],
                diagonal[..., None],
                arity_pair,
            ),
            dim=-1,
        )
        state = self.pair_input(features)
        object_count = object_mask.sum(-1).clamp_min(1).to(state.dtype)
        for pair_round in self.pair_rounds:
            state = pair_round(state, pair_mask, object_count)

        pair_weight = pair_mask[:, None, None, :, :, None].to(state.dtype)
        pair_count = pair_weight.sum((3, 4)).clamp_min(1.0)
        pair_mean = (state * pair_weight).sum((3, 4)) / pair_count
        minimum = torch.finfo(state.dtype).min
        pair_max = state.masked_fill(~pair_weight.bool(), minimum).amax(dim=(3, 4))
        witness_state = self.witness_encoder(torch.cat((pair_mean, pair_max), dim=-1))
        witness_weight = witness_mask[..., None].to(state.dtype)
        witness_count = witness_weight.sum(2).clamp_min(1.0)
        witness_mean = (witness_state * witness_weight).sum(2) / witness_count
        witness_max = witness_state.masked_fill(
            ~witness_mask[..., None],
            minimum,
        ).amax(2)
        active = witness_mask.any(-1)
        witness_max = torch.where(
            active[..., None],
            witness_max,
            torch.zeros_like(witness_max),
        )
        logits = self.card_classifier(
            torch.cat((witness_mean, witness_max, arity_features), dim=-1)
        )
        legal = arity[..., None].eq(self._class_arity) | (
            torch.arange(
                BINDER_CLASS_COUNT,
                device=logits.device,
            )
            == REJECT_INDEX
        )
        logits = logits.masked_fill(
            ~legal,
            torch.finfo(logits.dtype).min,
        )
        logits = torch.where(active[..., None], logits, torch.zeros_like(logits))
        probabilities = logits.softmax(-1)
        probabilities = probabilities * active[..., None].to(logits.dtype)
        selected_index = logits.argmax(-1)
        selected = F.one_hot(
            selected_index,
            BINDER_CLASS_COUNT,
        ).to(logits.dtype)
        selected = selected * active[..., None].to(logits.dtype)
        if hard:
            assignment_full = selected + probabilities - probabilities.detach()
        else:
            assignment_full = probabilities
        discrete = selected[..., :PRIMITIVE_COUNT]
        assignment = assignment_full[..., :PRIMITIVE_COUNT]
        rejected = selected_index.eq(REJECT_INDEX) & active
        assignment = assignment * ~rejected[..., None]
        discrete = discrete * ~rejected[..., None]
        return LearnedPrimitiveBinding(
            logits=logits,
            probabilities=probabilities,
            assignment=assignment,
            discrete_assignment=discrete,
            rejected=rejected,
            arity=arity,
        )


class ContextualWitnessStatisticsBinder(nn.Module):
    """Marginal-statistics control with no pair or triadic topology."""

    def __init__(self, *, width: int = 64) -> None:
        super().__init__()
        if width < 16:
            raise ContextualWitnessBinderError("statistics control geometry differs")
        self.width = int(width)
        self.witness_encoder = nn.Sequential(
            nn.Linear(9, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.card_classifier = nn.Sequential(
            nn.Linear(2 * width + 3, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, BINDER_CLASS_COUNT),
        )
        self.register_buffer(
            "_class_arity",
            torch.tensor(_CLASS_ARITY, dtype=torch.long),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def added_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_receipt(
        self,
        *,
        base_parameters: int = PROTECTED_BASE_PARAMETERS,
        strict_cap: int = STRICT_SYSTEM_CAP,
    ) -> dict[str, int]:
        added = self.added_parameters
        complete = int(base_parameters) + added
        if complete >= int(strict_cap):
            raise ContextualWitnessBinderError("statistics control reaches cap")
        return {
            "base": int(base_parameters),
            "added": int(added),
            "complete_system": complete,
            "strict_cap": int(strict_cap),
            "headroom": int(strict_cap) - complete,
        }

    def forward(
        self,
        witness_left: torch.Tensor,
        witness_right: torch.Tensor,
        witness_output: torch.Tensor,
        witness_mask: torch.Tensor,
        argument_mask: torch.Tensor,
        object_mask: torch.Tensor,
        *,
        hard: bool,
    ) -> LearnedPrimitiveBinding:
        pair_mask = _validate_relations(
            witness_left,
            witness_right,
            witness_output,
            object_mask,
        )
        arity = _infer_arity(witness_mask, argument_mask)
        dtype = witness_left.dtype
        pair_weight = pair_mask[:, None, None].to(dtype)
        pair_count = pair_weight.sum((3, 4)).clamp_min(1.0)
        relations = torch.stack(
            (witness_left, witness_right, witness_output),
            dim=-1,
        )
        pair_mean = (
            relations * pair_weight[..., None]
        ).sum((3, 4)) / pair_count[..., None]
        diagonal = torch.eye(
            MAX_OBJECTS,
            device=witness_left.device,
            dtype=torch.bool,
        )[None] & pair_mask
        diagonal_weight = diagonal[:, None, None].to(dtype)
        diagonal_count = diagonal_weight.sum((3, 4)).clamp_min(1.0)
        diagonal_mean = (
            relations * diagonal_weight[..., None]
        ).sum((3, 4)) / diagonal_count[..., None]
        arity_features = F.one_hot(
            arity.clamp_min(0),
            3,
        ).to(dtype)
        per_witness_arity = arity_features[:, :, None].expand(
            -1,
            -1,
            CARD_WITNESSES,
            -1,
        )
        witness_state = self.witness_encoder(
            torch.cat(
                (pair_mean, diagonal_mean, per_witness_arity),
                dim=-1,
            )
        )
        witness_weight = witness_mask[..., None].to(dtype)
        witness_count = witness_weight.sum(2).clamp_min(1.0)
        witness_mean = (witness_state * witness_weight).sum(2) / witness_count
        minimum = torch.finfo(dtype).min
        witness_max = witness_state.masked_fill(
            ~witness_mask[..., None],
            minimum,
        ).amax(2)
        active = witness_mask.any(-1)
        witness_max = torch.where(
            active[..., None],
            witness_max,
            torch.zeros_like(witness_max),
        )
        logits = self.card_classifier(
            torch.cat((witness_mean, witness_max, arity_features), dim=-1)
        )
        legal = arity[..., None].eq(self._class_arity) | (
            torch.arange(
                BINDER_CLASS_COUNT,
                device=logits.device,
            )
            == REJECT_INDEX
        )
        logits = logits.masked_fill(~legal, torch.finfo(dtype).min)
        logits = torch.where(active[..., None], logits, torch.zeros_like(logits))
        probabilities = logits.softmax(-1)
        probabilities = probabilities * active[..., None].to(dtype)
        selected_index = logits.argmax(-1)
        selected = F.one_hot(
            selected_index,
            BINDER_CLASS_COUNT,
        ).to(dtype)
        selected = selected * active[..., None].to(dtype)
        assignment_full = (
            selected + probabilities - probabilities.detach()
            if hard
            else probabilities
        )
        rejected = selected_index.eq(REJECT_INDEX) & active
        assignment = assignment_full[..., :PRIMITIVE_COUNT]
        discrete = selected[..., :PRIMITIVE_COUNT]
        assignment = assignment * ~rejected[..., None]
        discrete = discrete * ~rejected[..., None]
        return LearnedPrimitiveBinding(
            logits=logits,
            probabilities=probabilities,
            assignment=assignment,
            discrete_assignment=discrete,
            rejected=rejected,
            arity=arity,
        )
