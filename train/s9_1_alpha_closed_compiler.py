"""Alpha-closed training and syntax-only decoding for the S9.1 compiler."""

from __future__ import annotations

from itertools import product
from typing import Sequence

import torch
import torch.nn.functional as F

from s8_nil_linked_graph_compiler import ROLE_INDEX, ROLE_LABELS
from s9_occurrence_quotient import compile_quotient, quotient_from_emitted_spans
from s9_occurrence_quotient_compiler import S9Example, SpanCandidate


ANCHOR_ROLES = (
    "entity.roster",
    "position.roster",
    "state.entity",
    "card.operation",
    "event.tag",
)
FORCED_SINGLETON_ROLES = ("entry.tag", "query.position")
STRUCTURED_BEAM = 8


def aligned_positive_logits(
    examples: Sequence[S9Example],
    candidate_rows: Sequence[Sequence[SpanCandidate]],
    logits: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return positive occurrence logits in stable source-occurrence order."""

    if len(examples) != len(candidate_rows):
        raise ValueError("S9.1 example/candidate row count mismatch")
    cursor = 0
    selected = []
    targets: list[int] = []
    for example, candidates in zip(examples, candidate_rows, strict=True):
        local = {
            (candidate.start, candidate.end): cursor + index
            for index, candidate in enumerate(candidates)
        }
        for start, end, target in example.gold:
            index = local.get((start, end))
            if index is None:
                raise ValueError("S9.1 sampled candidates omitted a positive span")
            selected.append(logits[index])
            targets.append(int(target))
        cursor += len(candidates)
    if cursor != logits.shape[0] or not selected:
        raise ValueError("S9.1 candidate/logit alignment failed")
    return torch.stack(selected), tuple(targets)


def orbit_consistency_loss(
    original_logits: torch.Tensor,
    recoded_logits: torch.Tensor,
    original_targets: Sequence[int],
    recoded_targets: Sequence[int],
) -> torch.Tensor:
    """Penalize structural-role drift across a known alpha-renaming orbit."""

    if original_logits.shape != recoded_logits.shape:
        raise ValueError("S9.1 orbit logit shapes differ")
    if tuple(original_targets) != tuple(recoded_targets):
        raise ValueError("S9.1 orbit relation labels are not aligned")
    return F.mse_loss(
        F.log_softmax(original_logits.float(), dim=-1),
        F.log_softmax(recoded_logits.float(), dim=-1),
    )


def _positions(candidate: SpanCandidate) -> set[int]:
    return set(range(candidate.start, candidate.end + 1))


def _margin(logits: torch.Tensor, index: int, role: str) -> float:
    if role == "event.next_or_nil":
        return max(
            _margin(logits, index, "event.next"),
            _margin(logits, index, "event.nil"),
        )
    return float(
        (logits[index, ROLE_INDEX[role]] - logits[index, ROLE_INDEX["none"]]).item()
    )


def _ranked(
    candidates: Sequence[SpanCandidate],
    logits: torch.Tensor,
    role: str,
    *,
    start: int = 0,
    end: int | None = None,
    occupied: set[int] | None = None,
    limit: int | None = None,
) -> list[tuple[float, int]]:
    stop = end if end is not None else 1 << 30
    blocked = occupied or set()
    values = [
        (_margin(logits, index, role), index)
        for index, candidate in enumerate(candidates)
        if start <= candidate.start < stop and not (_positions(candidate) & blocked)
    ]
    values.sort(key=lambda value: (
        -value[0],
        candidates[value[1]].end - candidates[value[1]].start,
        candidates[value[1]].start,
    ))
    return values if limit is None else values[:limit]


def _best_joint(
    candidates: Sequence[SpanCandidate],
    logits: torch.Tensor,
    roles: Sequence[str],
    *,
    start: int,
    end: int,
    occupied: set[int],
) -> tuple[SpanCandidate, ...]:
    ranked = [
        _ranked(
            candidates,
            logits,
            role,
            start=start,
            end=end,
            occupied=occupied,
            limit=STRUCTURED_BEAM,
        )
        for role in roles
    ]
    if any(not values for values in ranked):
        raise ValueError("S9.1 structured region has no legal child candidate")
    best: tuple[float, tuple[int, ...]] | None = None
    for choices in product(*ranked):
        indices = tuple(value[1] for value in choices)
        if len(set(indices)) != len(indices):
            continue
        used = set(occupied)
        legal = True
        for index in indices:
            positions = _positions(candidates[index])
            if used & positions:
                legal = False
                break
            used.update(positions)
        if legal:
            score = sum(value[0] for value in choices)
            tie = tuple(-index for index in indices)
            if best is None or (score, tie) > (best[0], tuple(-i for i in best[1])):
                best = score, indices
    if best is None:
        raise ValueError("S9.1 structured region has no non-overlapping assignment")
    return tuple(candidates[index] for index in best[1])


def _select_model_anchors(
    candidates: Sequence[SpanCandidate], logits: torch.Tensor
) -> tuple[dict[str, list[SpanCandidate]], set[int]]:
    none = ROLE_INDEX["none"]
    anchor_indices = {ROLE_INDEX[role] for role in ANCHOR_ROLES}
    scored = []
    for index, candidate in enumerate(candidates):
        role_index = int(logits[index].argmax().item())
        if role_index not in anchor_indices:
            continue
        margin = float((logits[index, role_index] - logits[index, none]).item())
        if margin <= 0:
            continue
        scored.append((margin, index, ROLE_LABELS[role_index]))
    scored.sort(key=lambda value: (
        -value[0],
        candidates[value[1]].end - candidates[value[1]].start,
        candidates[value[1]].start,
    ))
    occupied: set[int] = set()
    selected = {role: [] for role in ANCHOR_ROLES}
    for _, index, role in scored:
        positions = _positions(candidates[index])
        if occupied & positions:
            continue
        occupied.update(positions)
        selected[role].append(candidates[index])
    for values in selected.values():
        values.sort(key=lambda value: value.start)
    return selected, occupied


def structured_spans_from_logits(
    example: S9Example,
    candidates: Sequence[SpanCandidate],
    logits: torch.Tensor,
) -> dict[str, dict[str, object]]:
    """Decode the best local typed assignment around model-selected anchors.

    The decoder enforces only frozen source grammar: non-overlap, singleton
    entry/query, two card children, and three event children. It never receives
    a gold count, graph, executor result, state, answer, or retry signal.
    """

    if len(candidates) != logits.shape[0]:
        raise ValueError("S9.1 candidate/logit count mismatch")
    selected, occupied = _select_model_anchors(candidates, logits)
    if not selected["card.operation"] or not selected["event.tag"]:
        raise ValueError("S9.1 model emitted no card or event anchors")

    singleton: dict[str, SpanCandidate] = {}
    for role in FORCED_SINGLETON_ROLES:
        ranked = _ranked(candidates, logits, role, occupied=occupied, limit=1)
        if not ranked:
            raise ValueError(f"S9.1 has no legal {role} candidate")
        candidate = candidates[ranked[0][1]]
        singleton[role] = candidate
        occupied.update(_positions(candidate))

    labeled: dict[str, SpanCandidate] = {}
    for role in ("entity.roster", "position.roster", "state.entity"):
        for index, candidate in enumerate(selected[role]):
            labeled[f"{role}.{index}"] = candidate

    card_anchors = selected["card.operation"]
    event_anchors = selected["event.tag"]
    structural_stops = [
        singleton["entry.tag"].start,
        event_anchors[0].start,
        singleton["query.position"].start,
        len(example.ids),
    ]
    for index, anchor in enumerate(card_anchors):
        stop = (
            card_anchors[index + 1].start
            if index + 1 < len(card_anchors)
            else min(value for value in structural_stops if value > anchor.start)
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

    for index, anchor in enumerate(event_anchors):
        stop = (
            event_anchors[index + 1].start
            if index + 1 < len(event_anchors)
            else min(
                value
                for value in (singleton["query.position"].start, len(example.ids))
                if value > anchor.start
            )
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
            i for i, value in enumerate(candidates) if value is successor
        )
        next_margin = _margin(logits, successor_index, "event.next")
        nil_margin = _margin(logits, successor_index, "event.nil")
        successor_role = "next" if next_margin >= nil_margin else "nil"
        labeled[f"event.{index}.tag"] = anchor
        labeled[f"event.{index}.operation"] = operation
        labeled[f"event.{index}.entity"] = entity
        labeled[f"event.{index}.{successor_role}"] = successor
        occupied.update(
            _positions(operation) | _positions(entity) | _positions(successor)
        )

    labeled["entry.tag"] = singleton["entry.tag"]
    labeled["query.position"] = singleton["query.position"]
    return {
        label: {
            "start": value.char_start,
            "end": value.char_end,
            "text": value.text,
        }
        for label, value in labeled.items()
    }


def structured_decode_graph(example, candidates, logits):
    spans = structured_spans_from_logits(example, candidates, logits)
    quotient = quotient_from_emitted_spans(str(example.row["question"]), spans)
    return compile_quotient(quotient), spans, quotient


__all__ = [
    "ANCHOR_ROLES",
    "FORCED_SINGLETON_ROLES",
    "STRUCTURED_BEAM",
    "aligned_positive_logits",
    "orbit_consistency_loss",
    "structured_decode_graph",
    "structured_spans_from_logits",
]
