"""Matched neural readouts for the CTAA A4 binding-completion falsifier.

The treatment predicts four local opcode-to-card relations with one shared
slot-local head. The favorable control predicts the same four relations under
the same loss, but each output row may inspect all four slots. Both receive the
same four independently decoded source-compiler vectors. Their trainable
parameter counts are exactly equal and their multiply-accumulate counts differ
by less than 0.2%. A support-starved 24-way lookup remains an explicitly
non-decisive negative control.

This module defines experiment mechanics only. It does not authorize a board
seed, scored access, GPU job, or capability claim.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ctaa_compiler_training import TokenizedCompilerRow


ACTION_COUNT = 4
COMPILER_WIDTH = 384
RELATION_SLOT_COUNT = ACTION_COUNT * 2
PAIR_FEATURE_WIDTH = (2 + RELATION_SLOT_COUNT) * COMPILER_WIDTH
PAIR_HIDDEN = 156
WHOLE_HIDDEN = 466
SINGLE_SLOT_PROBE_HIDDEN = 312
BINDINGS = tuple(itertools.permutations(range(ACTION_COUNT)))
BINDING_TO_INDEX = {binding: index for index, binding in enumerate(BINDINGS)}
READOUT_PARAMETERS = 599_353
FACTORIZED_MACS = 9_587_136
GLOBAL_MACS = FACTORIZED_MACS
SCHEMA = "r12_ctaa_a4_binding_completion_v1"


class BindingCompletionError(ValueError):
    """The A4 neural completion experiment is malformed."""


def checked_binding(value: Sequence[int]) -> tuple[int, int, int, int]:
    binding = tuple(int(item) for item in value)
    if len(binding) != ACTION_COUNT or sorted(binding) != list(range(ACTION_COUNT)):
        raise BindingCompletionError("CTAA completion binding is not a permutation")
    return binding  # type: ignore[return-value]


def permutation_parity(value: Sequence[int]) -> int:
    binding = checked_binding(value)
    return (
        sum(
            binding[left] > binding[right]
            for left in range(ACTION_COUNT)
            for right in range(left + 1, ACTION_COUNT)
        )
        % 2
    )


def binding_class_targets(bindings: torch.Tensor) -> torch.Tensor:
    if bindings.ndim != 2 or bindings.shape[1] != ACTION_COUNT:
        raise BindingCompletionError("CTAA completion target geometry differs")
    candidates = torch.tensor(BINDINGS, dtype=bindings.dtype, device=bindings.device)
    matches = bindings[:, None].eq(candidates[None]).all(-1)
    if not bool(matches.sum(-1).eq(1).all()):
        raise BindingCompletionError("CTAA completion target leaves S4")
    return matches.to(torch.uint8).argmax(-1).long()


def materialize_factorized(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[1:] != (ACTION_COUNT, ACTION_COUNT):
        raise BindingCompletionError("CTAA factorized readout geometry differs")
    candidates = torch.tensor(BINDINGS, dtype=torch.long, device=logits.device)
    rows = torch.arange(ACTION_COUNT, device=logits.device)
    scores = logits[:, rows, candidates].sum(-1)
    return candidates[scores.argmax(-1)]


def materialize_whole(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != len(BINDINGS):
        raise BindingCompletionError("CTAA whole readout geometry differs")
    candidates = torch.tensor(BINDINGS, dtype=torch.long, device=logits.device)
    return candidates[logits.argmax(-1)]


@dataclass(frozen=True)
class A4RowSplit:
    train_even: tuple[TokenizedCompilerRow, ...]
    confirmation_odd: tuple[TokenizedCompilerRow, ...]
    audit: dict[str, object]


def audit_parity_rows(
    rows: Sequence[TokenizedCompilerRow],
    *,
    expected_parity: int,
) -> dict[str, object]:
    if expected_parity not in (0, 1) or not rows:
        raise BindingCompletionError("CTAA completion parity audit differs")
    expected_bindings = {
        binding for binding in BINDINGS if permutation_parity(binding) == expected_parity
    }
    counts = Counter(checked_binding(row.opcode_to_card) for row in rows)
    if set(counts) != expected_bindings or len(set(counts.values())) != 1:
        raise BindingCompletionError("CTAA completion parity multiplicity differs")
    per_binding = next(iter(counts.values()))
    local_marginals = tuple(
        tuple(
            sum(row.opcode_to_card[opcode] == card for row in rows)
            for card in range(ACTION_COUNT)
        )
        for opcode in range(ACTION_COUNT)
    )
    expected_marginals = ((3 * per_binding,) * ACTION_COUNT,) * ACTION_COUNT
    if local_marginals != expected_marginals:
        raise BindingCompletionError("CTAA completion parity marginals differ")
    return {
        "expected_parity": expected_parity,
        "rows": len(rows),
        "bindings": len(counts),
        "per_binding": per_binding,
        "local_marginals": local_marginals,
    }


def split_a4_rows(rows: Sequence[TokenizedCompilerRow]) -> A4RowSplit:
    """Split a globally exact S4 board into locally balanced A4/odd halves."""

    if not rows:
        raise BindingCompletionError("CTAA completion rows are empty")
    buckets: dict[tuple[int, int, int, int], list[TokenizedCompilerRow]] = {
        binding: [] for binding in BINDINGS
    }
    source_targets: dict[tuple[int, ...], set[tuple[int, int, int, int]]] = {}
    for row in rows:
        binding = checked_binding(row.opcode_to_card)
        buckets[binding].append(row)
        source_targets.setdefault(row.program_ids, set()).add(binding)
    if any(len(targets) != 1 for targets in source_targets.values()):
        raise BindingCompletionError("CTAA source maps to conflicting bindings")
    multiplicities = {len(bucket) for bucket in buckets.values()}
    if len(multiplicities) != 1 or not multiplicities or next(iter(multiplicities)) < 1:
        raise BindingCompletionError("CTAA completion S4 multiplicity differs")

    train_even = tuple(
        row
        for row in rows
        if permutation_parity(row.opcode_to_card) == 0
    )
    confirmation_odd = tuple(
        row
        for row in rows
        if permutation_parity(row.opcode_to_card) == 1
    )
    train_sources = {row.program_ids for row in train_even}
    confirmation_sources = {row.program_ids for row in confirmation_odd}
    if train_sources.intersection(confirmation_sources):
        raise BindingCompletionError("CTAA completion source partitions overlap")

    per_binding = next(iter(multiplicities))
    train_audit = audit_parity_rows(train_even, expected_parity=0)
    confirmation_audit = audit_parity_rows(
        confirmation_odd,
        expected_parity=1,
    )
    audit: dict[str, object] = {
        "schema": SCHEMA,
        "claim_boundary": "neural_binding_identification_only",
        "rows": len(rows),
        "per_binding": per_binding,
        "train_even_rows": len(train_even),
        "confirmation_odd_rows": len(confirmation_odd),
        "train_bindings": 12,
        "confirmation_bindings": 12,
        "train_local_marginals": train_audit["local_marginals"],
        "confirmation_local_marginals": confirmation_audit["local_marginals"],
        "source_partition_overlap": 0,
    }
    return A4RowSplit(train_even, confirmation_odd, audit)


class FactorizedBindingReadout(nn.Module):
    """Shared opcode/card matcher with exact ``S4 x S4`` equivariance."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(PAIR_FEATURE_WIDTH, PAIR_HIDDEN),
            nn.GELU(),
            nn.Linear(PAIR_HIDDEN, 1),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        if slots.ndim != 3 or slots.shape[1:] != (
            RELATION_SLOT_COUNT,
            COMPILER_WIDTH,
        ):
            raise BindingCompletionError("CTAA factorized slot geometry differs")
        batch = slots.shape[0]
        opcode = slots[:, :ACTION_COUNT]
        cards = slots[:, ACTION_COUNT:]
        opcode_pairs = opcode[:, :, None].expand(
            -1,
            -1,
            ACTION_COUNT,
            -1,
        )
        card_pairs = cards[:, None].expand(
            -1,
            ACTION_COUNT,
            -1,
            -1,
        )
        context = torch.zeros(
            batch,
            ACTION_COUNT,
            ACTION_COUNT,
            RELATION_SLOT_COUNT * COMPILER_WIDTH,
            dtype=slots.dtype,
            device=slots.device,
        )
        features = torch.cat((opcode_pairs, card_pairs, context), dim=-1)
        return self.network(features).squeeze(-1).float()


class GlobalStructuredBindingReadout(nn.Module):
    """Globally connected control with identical four-cell supervision.

    The treatment scorer is reused for all 16 opcode/card pairs, but its
    otherwise-zero context field receives all eight slots. Parameter count,
    call count, and dense MAC geometry are therefore identical.
    """

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(PAIR_FEATURE_WIDTH, PAIR_HIDDEN),
            nn.GELU(),
            nn.Linear(PAIR_HIDDEN, 1),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        if slots.ndim != 3 or slots.shape[1:] != (
            RELATION_SLOT_COUNT,
            COMPILER_WIDTH,
        ):
            raise BindingCompletionError("CTAA global slot geometry differs")
        opcode = slots[:, :ACTION_COUNT]
        cards = slots[:, ACTION_COUNT:]
        opcode_pairs = opcode[:, :, None].expand(
            -1,
            -1,
            ACTION_COUNT,
            -1,
        )
        card_pairs = cards[:, None].expand(
            -1,
            ACTION_COUNT,
            -1,
            -1,
        )
        context = slots.flatten(1)[:, None, None].expand(
            -1,
            ACTION_COUNT,
            ACTION_COUNT,
            -1,
        )
        features = torch.cat((opcode_pairs, card_pairs, context), dim=-1)
        return self.network(features).squeeze(-1).float()


class WholePermutationReadout(nn.Module):
    """Favorable atomic 24-class lookup over all four slot vectors."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(RELATION_SLOT_COUNT * COMPILER_WIDTH, WHOLE_HIDDEN),
            nn.GELU(),
            nn.Linear(WHOLE_HIDDEN, len(BINDINGS)),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        if slots.ndim != 3 or slots.shape[1:] != (
            RELATION_SLOT_COUNT,
            COMPILER_WIDTH,
        ):
            raise BindingCompletionError("CTAA whole slot geometry differs")
        return self.network(slots.flatten(1)).float()


class SingleSlotFullBindingProbe(nn.Module):
    """Diagnostic: recover every binding row from one selected slot."""

    def __init__(self, slot_index: int) -> None:
        super().__init__()
        if not 0 <= slot_index < ACTION_COUNT:
            raise BindingCompletionError("CTAA single-slot probe index differs")
        self.slot_index = int(slot_index)
        self.network = nn.Sequential(
            nn.Linear(
                (1 + ACTION_COUNT) * COMPILER_WIDTH + ACTION_COUNT,
                SINGLE_SLOT_PROBE_HIDDEN,
            ),
            nn.GELU(),
            nn.Linear(SINGLE_SLOT_PROBE_HIDDEN, ACTION_COUNT),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        if slots.ndim != 3 or slots.shape[1:] != (
            RELATION_SLOT_COUNT,
            COMPILER_WIDTH,
        ):
            raise BindingCompletionError("CTAA single-slot geometry differs")
        batch = slots.shape[0]
        selected = slots[:, self.slot_index]
        cards = slots[:, ACTION_COUNT:].flatten(1)
        row_queries = torch.eye(
            ACTION_COUNT,
            dtype=slots.dtype,
            device=slots.device,
        )
        features = torch.cat(
            (
                selected[:, None].expand(-1, ACTION_COUNT, -1),
                cards[:, None].expand(-1, ACTION_COUNT, -1),
                row_queries[None].expand(batch, -1, -1),
            ),
            dim=-1,
        )
        return self.network(features).float()


def factorized_loss(logits: torch.Tensor, bindings: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[1:] != (ACTION_COUNT, ACTION_COUNT):
        raise BindingCompletionError("CTAA factorized loss geometry differs")
    binding_class_targets(bindings)
    return F.cross_entropy(logits.reshape(-1, ACTION_COUNT), bindings.reshape(-1))


def whole_loss(logits: torch.Tensor, bindings: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, binding_class_targets(bindings))


def readout_resource_receipt() -> dict[str, object]:
    factorized = FactorizedBindingReadout()
    global_structured = GlobalStructuredBindingReadout()
    whole = WholePermutationReadout()
    factorized_parameters = sum(parameter.numel() for parameter in factorized.parameters())
    global_parameters = sum(
        parameter.numel() for parameter in global_structured.parameters()
    )
    whole_parameters = sum(parameter.numel() for parameter in whole.parameters())
    if (
        factorized_parameters != READOUT_PARAMETERS
        or global_parameters != READOUT_PARAMETERS
    ):
        raise AssertionError("CTAA completion readout parameter match differs")
    relative_mac_gap = abs(FACTORIZED_MACS - GLOBAL_MACS) / max(
        FACTORIZED_MACS,
        GLOBAL_MACS,
    )
    if relative_mac_gap >= 0.002:
        raise AssertionError("CTAA completion readout MAC match differs")
    return {
        "schema": "r12_ctaa_a4_readout_resource_v1",
        "factorized_parameters": factorized_parameters,
        "global_structured_parameters": global_parameters,
        "whole_parameters": whole_parameters,
        "parameter_gap": 0,
        "factorized_macs": FACTORIZED_MACS,
        "global_structured_macs": GLOBAL_MACS,
        "relative_mac_gap": relative_mac_gap,
        "whole_control_role": "support_starved_lookup_negative_only",
    }


@torch.inference_mode()
def readout_metrics(
    logits: torch.Tensor,
    bindings: torch.Tensor,
    *,
    arm: str,
) -> dict[str, float]:
    if arm == "factorized":
        predicted = materialize_factorized(logits)
    elif arm == "global_structured":
        predicted = materialize_factorized(logits)
    elif arm == "whole":
        predicted = materialize_whole(logits)
    else:
        raise BindingCompletionError("CTAA completion arm differs")
    binding_class_targets(bindings)
    parity = torch.tensor(
        [permutation_parity(row.tolist()) for row in predicted],
        device=predicted.device,
    )
    return {
        "binding_exact": float(predicted.eq(bindings).all(-1).float().mean()),
        "projected_local_cell_accuracy": float(
            predicted.eq(bindings).float().mean()
        ),
        "predicted_odd_fraction": float(parity.float().mean()),
    }
