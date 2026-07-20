"""Global-anchor closure and alpha-orbit losses for bounded S9.2."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from s8_nil_linked_graph_compiler import ROLE_INDEX, ROLE_LABELS
from s9_occurrence_quotient import compile_quotient, quotient_from_emitted_spans
from s9_occurrence_quotient_compiler import S9Example, SpanCandidate
from s9_1_alpha_closed_compiler import (
    _best_joint,
    _margin,
    _positions,
    aligned_positive_logits,
    orbit_consistency_loss,
)


ADMITTED_MODULI = (5, 7, 11)
ADMITTED_CARD_COUNTS = (2, 3, 4)
ADMITTED_EVENT_COUNTS = tuple(range(1, 9))
ANCHOR_ROLES = (
    "entity.roster",
    "position.roster",
    "state.entity",
    "card.operation",
    "entry.tag",
    "event.tag",
    "query.position",
)
DEFAULT_HARD_NEGATIVE_K = 8


@dataclass(frozen=True)
class GlobalAnchorAssignment:
    """The model-logit MAP anchor hypothesis and its audit diagnostics."""

    modulus: int
    card_count: int
    event_count: int
    score: float
    roles: tuple[str, ...]
    candidate_indices: tuple[int, ...]

    @property
    def selected_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(self.roles).items()))


@dataclass
class _Frontier:
    role: str
    scores: np.ndarray
    parents: np.ndarray
    prefix_indices: np.ndarray
    previous: _Frontier | None


def _stable_candidate_order(candidates: Sequence[SpanCandidate]) -> tuple[int, ...]:
    return tuple(sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index].end,
            candidates[index].start,
            candidates[index].end - candidates[index].start + 1,
            candidates[index].char_start,
            candidates[index].char_end,
            index,
        ),
    ))


def _prefix_argmax(scores: np.ndarray) -> np.ndarray:
    """Prefix argmax retaining the first stable candidate on exact ties."""

    prior = np.empty_like(scores)
    prior[0] = -np.inf
    if len(scores) > 1:
        prior[1:] = np.maximum.accumulate(scores[:-1])
    records = scores > prior
    choices = np.where(records, np.arange(len(scores), dtype=np.int32), -1)
    return np.maximum.accumulate(choices)


def _advance(
    previous: _Frontier | None,
    role: str,
    margins: dict[str, np.ndarray],
    predecessors: np.ndarray,
) -> _Frontier:
    role_margin = margins[role]
    parents = np.full(len(role_margin), -1, dtype=np.int32)
    if previous is None:
        scores = role_margin.copy()
    else:
        available = predecessors >= 0
        parents[available] = previous.prefix_indices[predecessors[available]]
        valid = parents >= 0
        scores = np.full(len(role_margin), -np.inf, dtype=np.float64)
        scores[valid] = previous.scores[parents[valid]] + role_margin[valid]
    return _Frontier(
        role=role,
        scores=scores,
        parents=parents,
        prefix_indices=_prefix_argmax(scores),
        previous=previous,
    )


def _recover(
    frontier: _Frontier,
    stable_to_original: Sequence[int],
) -> tuple[tuple[str, ...], tuple[int, ...], float] | None:
    index = int(frontier.prefix_indices[-1])
    if index < 0:
        return None
    score = float(frontier.scores[index])
    roles = []
    indices = []
    current: _Frontier | None = frontier
    while current is not None:
        roles.append(current.role)
        indices.append(int(stable_to_original[index]))
        index = int(current.parents[index])
        current = current.previous
    roles.reverse()
    indices.reverse()
    return tuple(roles), tuple(indices), score


def global_anchor_assignment(
    candidates: Sequence[SpanCandidate],
    logits: torch.Tensor,
) -> GlobalAnchorAssignment:
    """Select one maximum-score root-anchor structure from model logits only.

    The search sees candidate intervals and role logits. It does not receive an
    example, row metadata, gold labels, graph compiler, executor, or repair
    signal. Surface order here is the declared source grammar, not execution
    order; execution remains determined by emitted next/nil relations.
    """

    if not candidates:
        raise ValueError("S9.2 global anchor search has no candidates")
    if logits.ndim != 2 or tuple(logits.shape) != (
        len(candidates),
        len(ROLE_LABELS),
    ):
        raise ValueError("S9.2 candidate/logit shape mismatch")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("S9.2 anchor logits are not finite")

    stable_to_original = _stable_candidate_order(candidates)
    ordered = [candidates[index] for index in stable_to_original]
    ends = [candidate.end for candidate in ordered]
    predecessors = np.asarray(
        [bisect_left(ends, candidate.start) - 1 for candidate in ordered],
        dtype=np.int32,
    )
    stable_index = torch.tensor(
        stable_to_original,
        dtype=torch.long,
        device=logits.device,
    )
    ordered_logits = logits.detach().float().index_select(0, stable_index).cpu()
    none = ordered_logits[:, ROLE_INDEX["none"]]
    margins = {
        role: (
            ordered_logits[:, ROLE_INDEX[role]] - none
        ).numpy().astype(np.float64, copy=True)
        for role in ANCHOR_ROLES
    }

    best: GlobalAnchorAssignment | None = None
    for modulus in ADMITTED_MODULI:
        root: _Frontier | None = None
        for role, count in (
            ("entity.roster", modulus),
            ("position.roster", modulus),
            ("state.entity", modulus),
        ):
            for _ in range(count):
                root = _advance(root, role, margins, predecessors)
        assert root is not None

        cards = root
        for card_count in range(1, max(ADMITTED_CARD_COUNTS) + 1):
            cards = _advance(cards, "card.operation", margins, predecessors)
            if card_count not in ADMITTED_CARD_COUNTS:
                continue
            entry = _advance(cards, "entry.tag", margins, predecessors)
            events = entry
            for event_count in ADMITTED_EVENT_COUNTS:
                events = _advance(events, "event.tag", margins, predecessors)
                query = _advance(events, "query.position", margins, predecessors)
                recovered = _recover(query, stable_to_original)
                if recovered is None:
                    continue
                roles, candidate_indices, score = recovered
                proposal = GlobalAnchorAssignment(
                    modulus=modulus,
                    card_count=card_count,
                    event_count=event_count,
                    score=score,
                    roles=roles,
                    candidate_indices=candidate_indices,
                )
                # Loops are ascending and each frontier keeps the first stable
                # candidate on a tie, so strict comparison freezes all ties.
                if best is None or proposal.score > best.score:
                    best = proposal

    if best is None or not best.score > 0.0:
        raise ValueError("S9.2 global anchor assignment did not beat none")
    return best


def structured_spans_from_assignment(
    example: S9Example,
    candidates: Sequence[SpanCandidate],
    logits: torch.Tensor,
    assignment: GlobalAnchorAssignment,
) -> dict[str, dict[str, object]]:
    """Complete S9.1 local children from one irrevocable root decision."""

    if len(assignment.roles) != len(assignment.candidate_indices):
        raise ValueError("S9.2 root assignment role/index count mismatch")
    if any(not 0 <= index < len(candidates) for index in assignment.candidate_indices):
        raise ValueError("S9.2 root assignment candidate index is out of bounds")
    selected = {role: [] for role in ANCHOR_ROLES}
    occupied: set[int] = set()
    for role, candidate_index in zip(
        assignment.roles,
        assignment.candidate_indices,
        strict=True,
    ):
        candidate = candidates[candidate_index]
        selected[role].append(candidate)
        occupied.update(_positions(candidate))

    labeled: dict[str, SpanCandidate] = {}
    for role in ("entity.roster", "position.roster", "state.entity"):
        for index, candidate in enumerate(selected[role]):
            labeled[f"{role}.{index}"] = candidate

    entry = selected["entry.tag"][0]
    query = selected["query.position"][0]
    card_anchors = selected["card.operation"]
    for index, anchor in enumerate(card_anchors):
        stop = (
            card_anchors[index + 1].start
            if index + 1 < len(card_anchors)
            else entry.start
        )
        y0, y1 = _best_joint(
            candidates,
            logits,
            ("card.y0", "card.y1"),
            start=anchor.end + 1,
            end=stop,
            occupied=occupied,
        )
        labeled[f"card.{index}.operation"] = anchor
        labeled[f"card.{index}.y0"] = y0
        labeled[f"card.{index}.y1"] = y1
        occupied.update(_positions(y0) | _positions(y1))

    event_anchors = selected["event.tag"]
    for index, anchor in enumerate(event_anchors):
        stop = (
            event_anchors[index + 1].start
            if index + 1 < len(event_anchors)
            else query.start
        )
        operation, entity, successor = _best_joint(
            candidates,
            logits,
            ("event.operation", "event.entity", "event.next_or_nil"),
            start=anchor.end + 1,
            end=stop,
            occupied=occupied,
        )
        successor_index = next(
            value for value, candidate in enumerate(candidates) if candidate is successor
        )
        successor_role = (
            "next"
            if _margin(logits, successor_index, "event.next")
            >= _margin(logits, successor_index, "event.nil")
            else "nil"
        )
        labeled[f"event.{index}.tag"] = anchor
        labeled[f"event.{index}.operation"] = operation
        labeled[f"event.{index}.entity"] = entity
        labeled[f"event.{index}.{successor_role}"] = successor
        occupied.update(
            _positions(operation) | _positions(entity) | _positions(successor)
        )

    labeled["entry.tag"] = entry
    labeled["query.position"] = query
    spans = {
        label: {
            "start": candidate.char_start,
            "end": candidate.char_end,
            "text": candidate.text,
        }
        for label, candidate in labeled.items()
    }
    return spans


def global_structured_spans_from_logits(
    example: S9Example,
    candidates: Sequence[SpanCandidate],
    logits: torch.Tensor,
) -> tuple[dict[str, dict[str, object]], GlobalAnchorAssignment]:
    """Make one global root decision, then complete its local children."""

    assignment = global_anchor_assignment(candidates, logits)
    spans = structured_spans_from_assignment(
        example,
        candidates,
        logits,
        assignment,
    )
    return spans, assignment


def global_structured_decode_graph(example, candidates, logits):
    """Compile only after the model-logit anchor optimization has terminated."""

    spans, assignment = global_structured_spans_from_logits(
        example,
        candidates,
        logits,
    )
    quotient = quotient_from_emitted_spans(str(example.row["question"]), spans)
    graph = compile_quotient(quotient)
    return graph, spans, quotient, assignment


def hard_negative_profiles(
    candidate_rows: Sequence[Sequence[SpanCandidate]],
    logits: torch.Tensor,
    *,
    top_k: int = DEFAULT_HARD_NEGATIVE_K,
) -> torch.Tensor:
    """Return coordinate-free sorted hard-negative role-margin profiles."""

    if top_k <= 0:
        raise ValueError("S9.2 hard-negative top-k must be positive")
    cursor = 0
    profiles = []
    none = ROLE_INDEX["none"]
    for candidates in candidate_rows:
        width = len(candidates)
        row_logits = logits[cursor:cursor + width]
        cursor += width
        negative = torch.tensor(
            [candidate.target == none for candidate in candidates],
            dtype=torch.bool,
            device=logits.device,
        )
        if int(negative.sum().item()) < top_k:
            raise ValueError("S9.2 row has fewer negatives than hard-negative top-k")
        values = row_logits[negative]
        margins = values[:, 1:] - values[:, none].unsqueeze(-1)
        hard = torch.topk(
            margins,
            k=top_k,
            dim=0,
            largest=True,
            sorted=True,
        ).values
        profiles.append(torch.sigmoid(hard).transpose(0, 1))
    if cursor != logits.shape[0] or not profiles:
        raise ValueError("S9.2 candidate/logit hard-negative alignment failed")
    return torch.stack(profiles)


def hard_negative_orbit_loss(
    original_candidate_rows: Sequence[Sequence[SpanCandidate]],
    recoded_candidate_rows: Sequence[Sequence[SpanCandidate]],
    original_logits: torch.Tensor,
    recoded_logits: torch.Tensor,
    *,
    top_k: int = DEFAULT_HARD_NEGATIVE_K,
) -> torch.Tensor:
    """Align hard competitors without assuming equal coordinates or BPE widths."""

    original = hard_negative_profiles(
        original_candidate_rows,
        original_logits,
        top_k=top_k,
    )
    recoded = hard_negative_profiles(
        recoded_candidate_rows,
        recoded_logits,
        top_k=top_k,
    )
    if original.shape != recoded.shape:
        raise ValueError("S9.2 hard-negative orbit profile shapes differ")
    return F.mse_loss(original, recoded)


def alpha_orbit_consistency_loss(
    original_examples: Sequence[S9Example],
    recoded_examples: Sequence[S9Example],
    original_candidate_rows: Sequence[Sequence[SpanCandidate]],
    recoded_candidate_rows: Sequence[Sequence[SpanCandidate]],
    original_logits: torch.Tensor,
    recoded_logits: torch.Tensor,
    *,
    top_k: int = DEFAULT_HARD_NEGATIVE_K,
) -> torch.Tensor:
    """Combine S9.1 positive closure with coordinate-free negative closure."""

    original_positive, original_targets = aligned_positive_logits(
        original_examples,
        original_candidate_rows,
        original_logits,
    )
    recoded_positive, recoded_targets = aligned_positive_logits(
        recoded_examples,
        recoded_candidate_rows,
        recoded_logits,
    )
    positive = orbit_consistency_loss(
        original_positive,
        recoded_positive,
        original_targets,
        recoded_targets,
    )
    negative = hard_negative_orbit_loss(
        original_candidate_rows,
        recoded_candidate_rows,
        original_logits,
        recoded_logits,
        top_k=top_k,
    )
    return positive + negative


__all__ = [
    "ADMITTED_CARD_COUNTS",
    "ADMITTED_EVENT_COUNTS",
    "ADMITTED_MODULI",
    "ANCHOR_ROLES",
    "DEFAULT_HARD_NEGATIVE_K",
    "GlobalAnchorAssignment",
    "alpha_orbit_consistency_loss",
    "global_anchor_assignment",
    "global_structured_decode_graph",
    "global_structured_spans_from_logits",
    "hard_negative_orbit_loss",
    "hard_negative_profiles",
    "structured_spans_from_assignment",
]
