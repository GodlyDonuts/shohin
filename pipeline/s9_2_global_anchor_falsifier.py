#!/usr/bin/env python3
"""CPU-only pre-board falsifiers for S9.2 global anchor closure.

The permanently closed S9 development board is used only as labeled mechanics.
No checkpoint, neural score, development result, or sealed confirmation row is
read. Every score tensor below is constructed by an explicit CPU oracle or by
an adversarial control declared in the S9.2 preregistration.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
import json
import math
from pathlib import Path
import random
import statistics
from typing import Sequence
from unittest import mock

import torch
from tokenizers import Tokenizer

import s8_nil_linked_law_graph as graph_runtime
from s8_nil_linked_graph_compiler import (
    ROLE_INDEX,
    ROLE_LABELS,
    compile_row as compile_s8_row,
    recode_operation_ids,
)
import s9_2_global_anchor_compiler as s92
from s9_occurrence_quotient_compiler import (
    S9Example,
    SpanCandidate,
    all_candidates,
    compile_row,
)
from s9_occurrence_quotient_falsifier import expected_graph, semantic_key
from s9_1_alpha_closed_compiler import _select_model_anchors


EXPECTED_ROWS = 2_048
MIN_EXHAUSTIVE_CASES = 10_000
LOW_ACCURACY_CEILING = 0.10
ROOT_ROLES = frozenset(s92.ANCHOR_ROLES)


def _oracle_logits(candidates: Sequence[SpanCandidate]) -> torch.Tensor:
    logits = torch.full((len(candidates), len(ROLE_LABELS)), -20.0)
    logits[:, ROLE_INDEX["none"]] = 0.0
    for index, candidate in enumerate(candidates):
        if candidate.target != ROLE_INDEX["none"]:
            logits[index, candidate.target] = 20.0
    return logits


def _attempt_full_decode(
    example: S9Example,
    candidates: Sequence[SpanCandidate],
    logits: torch.Tensor,
):
    try:
        return s92.global_structured_decode_graph(example, candidates, logits)
    except (ValueError, IndexError):
        return None


def _is_exact(decoded, row: dict[str, object]) -> bool:
    return bool(
        decoded is not None
        and semantic_key(decoded[0]) == semantic_key(expected_graph(row))
    )


def _gold_root_indices(
    candidates: Sequence[SpanCandidate],
) -> tuple[int, ...]:
    indices = tuple(
        index
        for index, candidate in enumerate(candidates)
        if ROLE_LABELS[candidate.target] in ROOT_ROLES
    )
    return tuple(sorted(indices, key=lambda index: candidates[index].start))


def _expected_counts(row: dict[str, object]) -> tuple[int, int, int]:
    return int(row["modulus"]), len(row["cards"]), int(row["depth"])


def _assignment_counts(assignment: s92.GlobalAnchorAssignment) -> tuple[int, int, int]:
    return assignment.modulus, assignment.card_count, assignment.event_count


def _root_template(row: dict[str, object]) -> tuple[str, ...]:
    modulus, card_count, event_count = _expected_counts(row)
    return (
        ("entity.roster",) * modulus
        + ("position.roster",) * modulus
        + ("state.entity",) * modulus
        + ("card.operation",) * card_count
        + ("entry.tag",)
        + ("event.tag",) * event_count
        + ("query.position",)
    )


def _slot_bounds(
    candidates: Sequence[SpanCandidate],
    roots: Sequence[int],
    slot: int,
) -> tuple[int, int]:
    current = candidates[roots[slot]]
    if slot == 0:
        left = 0
    else:
        previous = candidates[roots[slot - 1]]
        left = (previous.end + current.start) // 2 + 1
    if slot + 1 == len(roots):
        right = max(candidate.end for candidate in candidates)
    else:
        following = candidates[roots[slot + 1]]
        right = (current.end + following.start) // 2
    return left, right


def _root_alternatives(
    candidates: Sequence[SpanCandidate],
    roots: Sequence[int],
    slot: int,
) -> tuple[int, ...]:
    left, right = _slot_bounds(candidates, roots, slot)
    gold_index = roots[slot]
    gold = candidates[gold_index]
    values = [
        index
        for index, candidate in enumerate(candidates)
        if index != gold_index
        and candidate.target == ROLE_INDEX["none"]
        and left <= candidate.start
        and candidate.end <= right
    ]
    return tuple(sorted(
        values,
        key=lambda index: (
            abs(candidates[index].start - gold.start)
            + abs(candidates[index].end - gold.end),
            candidates[index].end - candidates[index].start,
            candidates[index].start,
            index,
        ),
    ))


def _flat_positive_control(
    row: dict[str, object],
    candidates: Sequence[SpanCandidate],
    oracle: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...], int]:
    roots = _gold_root_indices(candidates)
    roles = _root_template(row)
    if len(roots) != len(roles):
        raise ValueError("S9.2 gold root/template cardinality mismatch")
    logits = oracle.clone()
    alternatives = []
    feasible_lower_bound = 1
    for slot, (gold_index, role) in enumerate(zip(roots, roles, strict=True)):
        values = _root_alternatives(candidates, roots, slot)
        if not values:
            raise ValueError("S9.2 row has no root-slot syntax alternative")
        feasible_lower_bound += len(values)
        alternative = values[0]
        alternatives.append(alternative)
        logits[gold_index, ROLE_INDEX[role]] = -20.0
        logits[alternative, ROLE_INDEX[role]] = 1.0
    return logits, tuple(alternatives), feasible_lower_bound


def _shuffled_root_control(
    candidates: Sequence[SpanCandidate],
    oracle: torch.Tensor,
    rng: random.Random,
) -> torch.Tensor:
    roots = _gold_root_indices(candidates)
    roles = [int(candidates[index].target) for index in roots]
    shuffled = roles.copy()
    rng.shuffle(shuffled)
    if shuffled == roles:
        shuffled = shuffled[1:] + shuffled[:1]
    logits = oracle.clone()
    for index, role in zip(roots, roles, strict=True):
        logits[index, role] = -20.0
    for index, role in zip(roots, shuffled, strict=True):
        logits[index, role] = 20.0
    return logits


def _extra_late_root(
    candidates: Sequence[SpanCandidate],
    roots: Sequence[int],
) -> int:
    query = candidates[roots[-1]]
    values = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.target == ROLE_INDEX["none"] and candidate.start > query.end
    ]
    if not values:
        raise ValueError("S9.2 row has no post-query spurious root candidate")
    return min(values, key=lambda index: (candidates[index].start, candidates[index].end))


def _wrong_count_logits(
    row: dict[str, object],
    candidates: Sequence[SpanCandidate],
    oracle: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    roots = _gold_root_indices(candidates)
    roles = _root_template(row)
    card_slots = [index for index, role in enumerate(roles) if role == "card.operation"]
    card_count = len(card_slots)
    logits = oracle.clone()
    if card_count < max(s92.ADMITTED_CARD_COUNTS):
        last_card = candidates[roots[card_slots[-1]]]
        entry_slot = roles.index("entry.tag")
        entry = candidates[roots[entry_slot]]
        values = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.target == ROLE_INDEX["none"]
            and candidate.start > last_card.end
            and candidate.end < entry.start
        ]
        if not values:
            raise ValueError("S9.2 row has no syntax-legal extra card anchor")
        extra = min(values, key=lambda index: (candidates[index].start, candidates[index].end))
        logits[extra, ROLE_INDEX["card.operation"]] = 100.0
        expected = int(row["modulus"]), card_count + 1, int(row["depth"])
    else:
        removed = roots[card_slots[-1]]
        logits[removed, ROLE_INDEX["none"]] = 50.0
        expected = int(row["modulus"]), card_count - 1, int(row["depth"])
    return logits, expected


def _poison_example(example: S9Example) -> S9Example:
    row = dict(example.row)
    row.update({
        "modulus": 997,
        "depth": 997,
        "cards": [{"poison": True}],
        "nodes": [{"poison": True}],
        "initial_state": [997],
        "final_state": [997],
        "answer": 997,
        "entry_node": 997,
        "query_position": 997,
    })
    return replace(example, row=row, gold=())


def _spans_key(spans: dict[str, dict[str, object]]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (label, int(value["start"]), int(value["end"]), str(value["text"]))
        for label, value in sorted(spans.items())
    )


def _instrument_decoder_boundary(
    example: S9Example,
    candidates: Sequence[SpanCandidate],
    logits: torch.Tensor,
) -> dict[str, int]:
    compile_calls = 0
    executor_calls = 0
    real_compile = s92.compile_quotient
    real_execute = graph_runtime.execute_graph

    def counted_compile(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return real_compile(*args, **kwargs)

    def counted_execute(*args, **kwargs):
        nonlocal executor_calls
        executor_calls += 1
        return real_execute(*args, **kwargs)

    with (
        mock.patch.object(s92, "compile_quotient", side_effect=counted_compile),
        mock.patch.object(graph_runtime, "execute_graph", side_effect=counted_execute),
    ):
        s92.global_anchor_assignment(candidates, logits)
        optimizer_compile_calls = compile_calls
        optimizer_executor_calls = executor_calls
        s92.global_structured_decode_graph(example, candidates, logits)

    return {
        "optimizer_compile_calls": optimizer_compile_calls,
        "optimizer_executor_calls": optimizer_executor_calls,
        "full_decode_compile_calls": compile_calls - optimizer_compile_calls,
        "full_decode_executor_calls": executor_calls - optimizer_executor_calls,
    }


def _hard_negative_gate(
    candidates: Sequence[SpanCandidate],
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(0x592A)
    original = torch.randn(
        (len(candidates), len(ROLE_LABELS)),
        generator=generator,
        dtype=torch.float32,
    )
    permutation = torch.randperm(len(candidates), generator=generator)
    permuted_candidates = tuple(candidates[int(index)] for index in permutation)
    identical = s92.hard_negative_orbit_loss(
        [candidates],
        [permuted_candidates],
        original,
        original.index_select(0, permutation),
    )

    changed_original = original.clone().requires_grad_()
    changed_recoded = original.clone()
    negative_index = next(
        index
        for index, candidate in enumerate(candidates)
        if candidate.target == ROLE_INDEX["none"]
    )
    changed_recoded[negative_index, ROLE_INDEX["entity.roster"]] += 20.0
    changed_recoded = changed_recoded.requires_grad_()
    changed = s92.hard_negative_orbit_loss(
        [candidates],
        [candidates],
        changed_original,
        changed_recoded,
    )
    changed.backward()
    gradients_finite = bool(
        torch.isfinite(changed_original.grad).all().item()
        and torch.isfinite(changed_recoded.grad).all().item()
    )
    return {
        "identical_multiset_loss": float(identical.item()),
        "changed_competitor_loss": float(changed.item()),
        "gradients_finite": gradients_finite,
        "original_gradient_l1": float(changed_original.grad.abs().sum().item()),
        "recoded_gradient_l1": float(changed_recoded.grad.abs().sum().item()),
    }


def _synthetic_span(index: int, start: int, end: int) -> SpanCandidate:
    return SpanCandidate(
        start=start,
        end=end,
        text=f"s{index}",
        char_start=3 * start,
        char_end=3 * end + 2,
        target=ROLE_INDEX["none"],
    )


def _reduced_case(
    rng: random.Random,
) -> tuple[list[SpanCandidate], torch.Tensor, list[list[int]]]:
    roles = list(
        ("entity.roster",) * 5
        + ("position.roster",) * 5
        + ("state.entity",) * 5
        + ("card.operation",) * 2
        + ("entry.tag", "event.tag", "query.position")
    )
    candidates = [
        _synthetic_span(index, 4 * index + 1, 4 * index + 1)
        for index in range(len(roles))
    ]
    options = [[index] for index in range(len(roles))]
    alternative_slots = rng.sample(range(len(roles)), k=3)
    for slot in alternative_slots:
        mode = rng.randrange(3)
        center = 4 * slot + 1
        if mode == 0:
            start, end = center - 1, center + 1
        elif mode == 1 and slot + 1 < len(roles):
            start, end = center, center + 4
        elif slot > 0:
            start, end = center - 4, center
        else:
            start, end = center - 1, center + 1
        index = len(candidates)
        candidates.append(_synthetic_span(index, start, end))
        options[slot].append(index)

    logits = torch.full((len(candidates), len(ROLE_LABELS)), -10_000.0)
    logits[:, ROLE_INDEX["none"]] = 0.0
    for slot, role in enumerate(roles):
        for index in options[slot]:
            # Continuous random scores plus a deterministic offset eliminate
            # exact ties while preserving varied negative/positive margins.
            logits[index, ROLE_INDEX[role]] = 0.25 + 3.0 * rng.random() + index * 1e-7
    return candidates, logits, options


def _exhaustive_reduced(
    candidates: Sequence[SpanCandidate],
    logits: torch.Tensor,
    options: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], float]:
    roles = (
        ("entity.roster",) * 5
        + ("position.roster",) * 5
        + ("state.entity",) * 5
        + ("card.operation",) * 2
        + ("entry.tag", "event.tag", "query.position")
    )
    best_indices: tuple[int, ...] | None = None
    best_score = -math.inf
    for indices in product(*options):
        if any(
            candidates[left].end >= candidates[right].start
            for left, right in zip(indices, indices[1:])
        ):
            continue
        score = sum(
            float(
                logits[index, ROLE_INDEX[role]]
                - logits[index, ROLE_INDEX["none"]]
            )
            for role, index in zip(roles, indices, strict=True)
        )
        if score > best_score:
            best_indices = tuple(indices)
            best_score = score
    if best_indices is None:
        raise AssertionError("reduced exhaustive case has no feasible assignment")
    return best_indices, best_score


def run_reduced_exhaustive(seed: int, cases: int) -> dict[str, object]:
    if cases <= 0:
        raise ValueError("S9.2 exhaustive case count must be positive")
    rng = random.Random(seed)
    matches = 0
    max_score_error = 0.0
    for _ in range(cases):
        candidates, logits, options = _reduced_case(rng)
        expected_indices, expected_score = _exhaustive_reduced(
            candidates, logits, options
        )
        observed = s92.global_anchor_assignment(candidates, logits)
        exact = (
            (observed.modulus, observed.card_count, observed.event_count) == (5, 2, 1)
            and observed.candidate_indices == expected_indices
        )
        matches += int(exact)
        max_score_error = max(max_score_error, abs(observed.score - expected_score))
    return {
        "cases": cases,
        "exact_matches": matches,
        "max_score_absolute_error": max_score_error,
    }


def run(
    board: Path,
    tokenizer_path: Path,
    seed: int,
    *,
    exhaustive_cases: int = MIN_EXHAUSTIVE_CASES,
) -> dict[str, object]:
    development = board / "development.jsonl"
    if not development.is_file():
        raise ValueError("S9.2 falsifier requires the closed development split")
    rows = [
        json.loads(line)
        for line in development.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError("S9.2 falsifier requires exactly 2,048 closed S9 rows")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rng = random.Random(seed)
    counts = {
        "oracle_exact": 0,
        "recoded_oracle_exact": 0,
        "low_root_global_exact": 0,
        "low_root_breaks_local_selection": 0,
        "extra_root_global_exact": 0,
        "extra_root_breaks_local_selection": 0,
        "uniform_abstentions": 0,
        "flat_positive_exact": 0,
        "flat_positive_correct_counts": 0,
        "shuffled_exact": 0,
        "rows_with_multiple_feasible_assignments": 0,
        "wrong_high_root_selected": 0,
        "wrong_high_root_exact": 0,
        "wrong_high_count_followed": 0,
        "wrong_high_count_exact": 0,
        "metadata_target_poison_identical": 0,
    }
    feasible_assignment_lower_bounds = []
    first_mechanics = None

    for row in rows:
        example = compile_row(row, tokenizer)
        candidates = all_candidates(example)
        oracle = _oracle_logits(candidates)
        roots = _gold_root_indices(candidates)
        expected_counts = _expected_counts(row)
        if len(roots) != len(_root_template(row)):
            raise ValueError("S9.2 closed row root cardinality mismatch")

        oracle_decoded = _attempt_full_decode(example, candidates, oracle)
        counts["oracle_exact"] += int(_is_exact(oracle_decoded, row))

        recoded_s8 = recode_operation_ids(compile_s8_row(row, tokenizer), tokenizer)
        recoded = compile_row(recoded_s8.row, tokenizer)
        recoded_candidates = all_candidates(recoded)
        recoded_decoded = _attempt_full_decode(
            recoded,
            recoded_candidates,
            _oracle_logits(recoded_candidates),
        )
        counts["recoded_oracle_exact"] += int(
            _is_exact(recoded_decoded, recoded.row)
        )

        low = oracle.clone()
        low_index = next(
            index
            for index in roots
            if ROLE_LABELS[candidates[index].target] == "entity.roster"
        )
        low[low_index, ROLE_INDEX["none"]] = 30.0
        local_selected, _ = _select_model_anchors(candidates, low)
        counts["low_root_breaks_local_selection"] += int(
            len(local_selected["entity.roster"]) == expected_counts[0] - 1
        )
        low_decoded = _attempt_full_decode(example, candidates, low)
        counts["low_root_global_exact"] += int(_is_exact(low_decoded, row))

        extra = oracle.clone()
        extra_index = _extra_late_root(candidates, roots)
        extra[extra_index, ROLE_INDEX["entity.roster"]] = 1.0
        local_selected, _ = _select_model_anchors(candidates, extra)
        counts["extra_root_breaks_local_selection"] += int(
            len(local_selected["entity.roster"]) == expected_counts[0] + 1
        )
        extra_decoded = _attempt_full_decode(example, candidates, extra)
        counts["extra_root_global_exact"] += int(_is_exact(extra_decoded, row))

        uniform = torch.zeros_like(oracle)
        try:
            s92.global_anchor_assignment(candidates, uniform)
        except ValueError:
            counts["uniform_abstentions"] += 1

        flat, flat_roots, feasible_lower_bound = _flat_positive_control(
            row, candidates, oracle
        )
        feasible_assignment_lower_bounds.append(feasible_lower_bound)
        counts["rows_with_multiple_feasible_assignments"] += int(
            feasible_lower_bound >= 2
        )
        flat_assignment = s92.global_anchor_assignment(candidates, flat)
        counts["flat_positive_correct_counts"] += int(
            _assignment_counts(flat_assignment) == expected_counts
            and flat_assignment.candidate_indices == flat_roots
        )
        flat_decoded = _attempt_full_decode(example, candidates, flat)
        counts["flat_positive_exact"] += int(_is_exact(flat_decoded, row))

        shuffled = _shuffled_root_control(candidates, oracle, rng)
        shuffled_decoded = _attempt_full_decode(example, candidates, shuffled)
        counts["shuffled_exact"] += int(_is_exact(shuffled_decoded, row))

        wrong_root = oracle.clone()
        query_slot = len(roots) - 1
        wrong_root_index = _root_alternatives(candidates, roots, query_slot)[0]
        wrong_root[wrong_root_index, ROLE_INDEX["query.position"]] = 100.0
        wrong_assignment = s92.global_anchor_assignment(candidates, wrong_root)
        counts["wrong_high_root_selected"] += int(
            wrong_assignment.candidate_indices[-1] == wrong_root_index
        )
        wrong_root_decoded = _attempt_full_decode(example, candidates, wrong_root)
        counts["wrong_high_root_exact"] += int(_is_exact(wrong_root_decoded, row))

        wrong_count, forced_counts = _wrong_count_logits(row, candidates, oracle)
        wrong_count_assignment = s92.global_anchor_assignment(candidates, wrong_count)
        counts["wrong_high_count_followed"] += int(
            _assignment_counts(wrong_count_assignment) == forced_counts
        )
        wrong_count_decoded = _attempt_full_decode(example, candidates, wrong_count)
        counts["wrong_high_count_exact"] += int(_is_exact(wrong_count_decoded, row))

        ordinary_spans, ordinary_assignment = s92.global_structured_spans_from_logits(
            example, candidates, oracle
        )
        poisoned_candidates = tuple(
            replace(candidate, target=(index * 7 + 3) % len(ROLE_LABELS))
            for index, candidate in enumerate(candidates)
        )
        poisoned_spans, poisoned_assignment = s92.global_structured_spans_from_logits(
            _poison_example(example), poisoned_candidates, oracle
        )
        counts["metadata_target_poison_identical"] += int(
            ordinary_assignment == poisoned_assignment
            and _spans_key(ordinary_spans) == _spans_key(poisoned_spans)
        )

        if first_mechanics is None:
            first_mechanics = example, candidates, oracle

    if first_mechanics is None:
        raise AssertionError("S9.2 falsifier did not retain an instrumentation row")
    instrumentation = _instrument_decoder_boundary(*first_mechanics)
    hard_negative = _hard_negative_gate(first_mechanics[1])
    exhaustive = run_reduced_exhaustive(seed ^ 0x5A92, exhaustive_cases)

    total = len(rows)
    gates = {
        "reduced_viterbi_matches_exhaustive_at_least_10000": (
            exhaustive["cases"] >= MIN_EXHAUSTIVE_CASES
            and exhaustive["exact_matches"] == exhaustive["cases"]
            and exhaustive["max_score_absolute_error"] < 1e-4
        ),
        "oracle_all_exact": counts["oracle_exact"] == total,
        "operation_recoded_oracle_all_exact": counts["recoded_oracle_exact"] == total,
        "one_low_root_breaks_local_on_all_rows": (
            counts["low_root_breaks_local_selection"] == total
        ),
        "one_low_root_recovered_globally_on_all_rows": (
            counts["low_root_global_exact"] == total
        ),
        "extra_low_positive_root_breaks_local_on_all_rows": (
            counts["extra_root_breaks_local_selection"] == total
        ),
        "extra_low_positive_root_ignored_globally_on_all_rows": (
            counts["extra_root_global_exact"] == total
        ),
        "uniform_logits_abstain_on_all_rows": counts["uniform_abstentions"] == total,
        "flat_positive_correct_count_control_below_10pct_exact": (
            counts["flat_positive_correct_counts"] == total
            and counts["flat_positive_exact"] / total < LOW_ACCURACY_CEILING
        ),
        "shuffled_score_distribution_below_10pct_exact": (
            counts["shuffled_exact"] / total < LOW_ACCURACY_CEILING
        ),
        "every_row_has_at_least_two_complete_syntax_assignments": (
            counts["rows_with_multiple_feasible_assignments"] == total
            and min(feasible_assignment_lower_bounds) >= 2
        ),
        "wrong_high_root_is_followed_and_never_repaired": (
            counts["wrong_high_root_selected"] == total
            and counts["wrong_high_root_exact"] == 0
        ),
        "wrong_high_count_is_followed_and_never_repaired": (
            counts["wrong_high_count_followed"] == total
            and counts["wrong_high_count_exact"] == 0
        ),
        "metadata_and_candidate_target_poisoning_is_inert": (
            counts["metadata_target_poison_identical"] == total
        ),
        "optimizer_calls_no_compiler_or_executor": (
            instrumentation["optimizer_compile_calls"] == 0
            and instrumentation["optimizer_executor_calls"] == 0
        ),
        "full_decode_calls_compiler_once_and_executor_never": (
            instrumentation["full_decode_compile_calls"] == 1
            and instrumentation["full_decode_executor_calls"] == 0
        ),
        "hard_negative_orbit_identity_and_gradient_gates": (
            hard_negative["identical_multiset_loss"] == 0.0
            and hard_negative["changed_competitor_loss"] > 0.0
            and hard_negative["gradients_finite"]
            and hard_negative["original_gradient_l1"] > 0.0
            and hard_negative["recoded_gradient_l1"] > 0.0
        ),
    }
    return {
        "schema": "r12_s9_2_global_anchor_cpu_falsifier_v1",
        "seed": seed,
        "closed_board": str(board),
        "closed_split": "development.jsonl",
        "rows": total,
        "counts": counts,
        "rates": {
            "flat_positive_exact": counts["flat_positive_exact"] / total,
            "shuffled_exact": counts["shuffled_exact"] / total,
        },
        "feasible_assignment_lower_bound": {
            "minimum": min(feasible_assignment_lower_bounds),
            "median": statistics.median(feasible_assignment_lower_bounds),
            "maximum": max(feasible_assignment_lower_bounds),
        },
        "reduced_exhaustive": exhaustive,
        "instrumentation": instrumentation,
        "hard_negative_orbit": hard_negative,
        "gates": gates,
        "decision": (
            "admit_s9_2_global_anchor_mechanics"
            if all(gates.values())
            else "reject_s9_2_global_anchor_mechanics"
        ),
        "resource_boundary": {
            "cpu_oracle_only": True,
            "neural_scoring_performed": False,
            "confirmation_rows_read": 0,
            "optimizer_inputs": ["candidate intervals", "constructed role logits"],
            "optimizer_forbidden_inputs": [
                "candidate targets",
                "row modulus, depth, cards, nodes, spans, state, or answer",
                "source bytes or exact-byte classes",
                "compiler validity, executor output, or retry feedback",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exhaustive-cases", type=int, default=MIN_EXHAUSTIVE_CASES)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S9.2 falsifier output: {args.out}")
    report = run(
        args.board,
        args.tokenizer,
        args.seed,
        exhaustive_cases=args.exhaustive_cases,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "gates": report["gates"],
        "out": str(args.out),
        "rows": report["rows"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
