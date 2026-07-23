"""Contextual binding of opaque relation-operation cards to tied primitives.

The compiler receives only per-episode relation witnesses and structural
argument-presence masks. It never receives an operation name or global opcode.
Primitive semantics are tied across every card, witness, and later application.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


PROTECTED_BASE_PARAMETERS = 125_081_664
STRICT_SYSTEM_CAP = 200_000_000
PRIMITIVE_COUNT = 5

UNION_INDEX = 0
INTERSECTION_INDEX = 1
COMPOSE_INDEX = 2
CONVERSE_INDEX = 3
IDENTITY_INDEX = 4

_PRIMITIVE_ARITIES = (2, 2, 2, 1, 0)


class PrimitiveCompilerError(ValueError):
    """Raised when an opaque primitive card violates its tensor contract."""


@dataclass(frozen=True, slots=True)
class PrimitiveCompilation:
    """Auditable result of compiling opaque operation-card slots."""

    assignment: torch.Tensor
    discrete_assignment: torch.Tensor
    raw_compatibility_logits: torch.Tensor
    mean_squared_residual: torch.Tensor
    maximum_residual: torch.Tensor
    compatible: torch.Tensor
    legal: torch.Tensor
    identifiable: torch.Tensor
    arity: torch.Tensor


def _expanded_active_square(
    object_mask: torch.Tensor,
    relation_ndim: int,
) -> torch.Tensor:
    if object_mask.ndim != 2 or object_mask.dtype != torch.bool:
        raise PrimitiveCompilerError("object mask must be boolean [batch, objects]")
    singleton_axes = relation_ndim - 3
    if singleton_axes < 0:
        raise PrimitiveCompilerError("relation tensor rank differs")
    rows = object_mask.reshape(
        object_mask.shape[0],
        *(1 for _ in range(singleton_axes)),
        object_mask.shape[1],
        1,
    )
    columns = object_mask.reshape(
        object_mask.shape[0],
        *(1 for _ in range(singleton_axes)),
        1,
        object_mask.shape[1],
    )
    return rows & columns


def _validate_relations(
    left: torch.Tensor,
    right: torch.Tensor,
    object_mask: torch.Tensor,
) -> torch.Tensor:
    if (
        left.shape != right.shape
        or left.ndim < 3
        or left.shape[-1] != left.shape[-2]
        or left.shape[0] != object_mask.shape[0]
        or left.shape[-1] != object_mask.shape[1]
    ):
        raise PrimitiveCompilerError("relation operand geometry differs")
    if not left.is_floating_point() or not right.is_floating_point():
        raise PrimitiveCompilerError("relation operands must be floating tensors")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise PrimitiveCompilerError("relation operands must be finite")
    if (
        left.detach().amin().item() < 0.0
        or left.detach().amax().item() > 1.0
        or right.detach().amin().item() < 0.0
        or right.detach().amax().item() > 1.0
    ):
        raise PrimitiveCompilerError("relation operands leave [0, 1]")
    active = _expanded_active_square(object_mask, left.ndim)
    if (
        left.detach().masked_select(~active).abs().max().item()
        if (~active).any()
        else 0.0
    ) > 0.0:
        raise PrimitiveCompilerError("left relation uses inactive objects")
    if (
        right.detach().masked_select(~active).abs().max().item()
        if (~active).any()
        else 0.0
    ) > 0.0:
        raise PrimitiveCompilerError("right relation uses inactive objects")
    return active


def boolean_relation_compose(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Boolean-semiring composition for tensors in [0, 1]."""

    products = left.unsqueeze(-1) * right.unsqueeze(-3)
    return 1.0 - (1.0 - products).prod(dim=-2)


def relation_primitive_candidates(
    left: torch.Tensor,
    right: torch.Tensor,
    object_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the tied primitive bank to every leading relation position."""

    active = _validate_relations(left, right, object_mask)
    union = 1.0 - (1.0 - left) * (1.0 - right)
    intersection = left * right
    compose = boolean_relation_compose(left, right)
    converse = left.transpose(-1, -2)
    identity = torch.diag_embed(object_mask.to(dtype=left.dtype))
    identity = identity.reshape(
        identity.shape[0],
        *(1 for _ in range(left.ndim - 3)),
        identity.shape[-2],
        identity.shape[-1],
    ).expand_as(left)
    candidates = torch.stack(
        (union, intersection, compose, converse, identity),
        dim=-3,
    )
    return candidates * active.unsqueeze(-3).to(dtype=left.dtype)


def _infer_card_arity(
    argument_mask: torch.Tensor,
    witness_mask: torch.Tensor,
) -> torch.Tensor:
    if (
        argument_mask.dtype != torch.bool
        or witness_mask.dtype != torch.bool
        or argument_mask.shape[:-1] != witness_mask.shape
        or argument_mask.shape[-1] != 2
        or witness_mask.ndim != 3
    ):
        raise PrimitiveCompilerError("witness argument-mask geometry differs")
    if (argument_mask[..., 1] & ~argument_mask[..., 0] & witness_mask).any():
        raise PrimitiveCompilerError("a right argument cannot exist without a left")
    witness_arity = argument_mask.long().sum(-1)
    sentinel = torch.full_like(witness_arity, 3)
    minimum = torch.where(witness_mask, witness_arity, sentinel).amin(-1)
    maximum = torch.where(witness_mask, witness_arity, -1).amax(-1)
    has_witness = witness_mask.any(-1)
    if (has_witness & minimum.ne(maximum)).any():
        raise PrimitiveCompilerError("a card changes arity between witnesses")
    return torch.where(has_witness, maximum, torch.full_like(maximum, -1))


class ContextualRelationPrimitiveCompiler(nn.Module):
    """Compile opaque relation cards by contextual semantic identification."""

    def __init__(
        self,
        *,
        initial_logit_scale: float = 16.0,
        identifiability_tolerance: float = 1e-6,
    ) -> None:
        super().__init__()
        if initial_logit_scale <= 0.0 or not math.isfinite(initial_logit_scale):
            raise PrimitiveCompilerError("initial logit scale must be positive")
        if (
            identifiability_tolerance < 0.0
            or not math.isfinite(identifiability_tolerance)
        ):
            raise PrimitiveCompilerError("identifiability tolerance differs")
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(initial_logit_scale), dtype=torch.float32)
        )
        self.identifiability_tolerance = float(identifiability_tolerance)
        self.register_buffer(
            "_primitive_arities",
            torch.tensor(_PRIMITIVE_ARITIES, dtype=torch.long),
            persistent=False,
        )

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
            raise PrimitiveCompilerError("contextual primitive compiler reaches cap")
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
    ) -> PrimitiveCompilation:
        """Compile each opaque slot without receiving any semantic identifier."""

        if (
            witness_left.shape != witness_output.shape
            or witness_left.ndim != 5
            or witness_mask.shape != witness_left.shape[:3]
        ):
            raise PrimitiveCompilerError("witness-card geometry differs")
        active = _validate_relations(witness_left, witness_right, object_mask)
        if not witness_output.is_floating_point():
            raise PrimitiveCompilerError("witness outputs must be floating tensors")
        if not torch.isfinite(witness_output).all():
            raise PrimitiveCompilerError("witness outputs must be finite")
        if (
            witness_output.detach().amin().item() < 0.0
            or witness_output.detach().amax().item() > 1.0
        ):
            raise PrimitiveCompilerError("witness outputs leave [0, 1]")
        if (
            witness_output.detach().masked_select(~active).abs().max().item()
            if (~active).any()
            else 0.0
        ) > 0.0:
            raise PrimitiveCompilerError("witness output uses inactive objects")

        arity = _infer_card_arity(argument_mask, witness_mask)
        legal = arity[..., None].eq(self._primitive_arities)
        candidates = relation_primitive_candidates(
            witness_left,
            witness_right,
            object_mask,
        )
        difference = candidates - witness_output.unsqueeze(-3)
        evidence = (
            witness_mask[..., None, None, None]
            & active.unsqueeze(-3)
        ).to(dtype=difference.dtype)
        squared = difference.square() * evidence
        denominator = evidence.sum(dim=(2, 4, 5)).clamp_min(1.0)
        mean_squared_residual = squared.sum(dim=(2, 4, 5)) / denominator
        absolute = difference.abs() * evidence
        maximum_residual = absolute.amax(dim=(2, 4, 5))

        compatible = (
            maximum_residual.le(self.identifiability_tolerance)
            & legal
            & witness_mask.any(-1)[..., None]
        )
        identifiable = compatible.sum(-1).eq(1)
        scale = self.logit_scale.exp().clamp(min=1e-4, max=1e6)
        raw_logits = -scale * mean_squared_residual
        raw_logits = raw_logits.masked_fill(
            ~legal,
            torch.finfo(raw_logits.dtype).min,
        )
        probabilities = raw_logits.softmax(-1)
        selected = compatible.to(dtype=probabilities.dtype)
        if hard:
            assignment = selected + probabilities - probabilities.detach()
        else:
            assignment = probabilities
        assignment = assignment * identifiable[..., None].to(assignment.dtype)
        discrete_assignment = (
            selected * identifiable[..., None].to(selected.dtype)
        )
        return PrimitiveCompilation(
            assignment=assignment,
            discrete_assignment=discrete_assignment,
            raw_compatibility_logits=raw_logits,
            mean_squared_residual=mean_squared_residual,
            maximum_residual=maximum_residual,
            compatible=compatible,
            legal=legal,
            identifiable=identifiable,
            arity=arity,
        )

    def apply_compiled(
        self,
        compilation: PrimitiveCompilation,
        left: torch.Tensor,
        right: torch.Tensor,
        argument_mask: torch.Tensor,
        object_mask: torch.Tensor,
        *,
        require_identifiable: bool = True,
    ) -> torch.Tensor:
        """Apply compiled opaque slots to fresh operands using the tied bank."""

        if (
            left.ndim != 4
            or compilation.assignment.shape != (*left.shape[:2], PRIMITIVE_COUNT)
            or argument_mask.shape != (*left.shape[:2], 2)
            or argument_mask.dtype != torch.bool
        ):
            raise PrimitiveCompilerError("fresh primitive application geometry differs")
        if (argument_mask[..., 1] & ~argument_mask[..., 0]).any():
            raise PrimitiveCompilerError("fresh right argument lacks a left")
        fresh_arity = argument_mask.long().sum(-1)
        if not torch.equal(fresh_arity, compilation.arity):
            raise PrimitiveCompilerError("fresh operand arity differs from card")
        if require_identifiable and not bool(compilation.identifiable.all()):
            raise PrimitiveCompilerError("opaque primitive card is not identifiable")
        candidates = relation_primitive_candidates(left, right, object_mask)
        return torch.einsum(
            "bsp,bspij->bsij",
            compilation.assignment,
            candidates,
        )
