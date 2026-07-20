#!/usr/bin/env python3
"""Pre-board CPU falsifiers for S9.1 alpha closure and structured decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics

import torch
from tokenizers import Tokenizer

from s8_nil_linked_graph_compiler import (
    ROLE_INDEX,
    ROLE_LABELS,
    compile_row as compile_s8_row,
    recode_operation_ids,
)
from s9_occurrence_quotient_compiler import (
    MAX_SPAN_WIDTH,
    all_candidates,
    compile_row,
    emitted_spans_from_logits,
)
from s9_occurrence_quotient_falsifier import expected_graph, semantic_key
from s9_1_alpha_closed_compiler import structured_decode_graph


def _oracle(candidates):
    logits = torch.full((len(candidates), len(ROLE_LABELS)), -20.0)
    logits[:, ROLE_INDEX["none"]] = 0.0
    for index, candidate in enumerate(candidates):
        if candidate.target:
            logits[index, candidate.target] = 20.0
    return logits


def _attempt(example, candidates, logits, structured):
    try:
        if structured:
            graph, _, _ = structured_decode_graph(example, candidates, logits)
        else:
            spans = emitted_spans_from_logits(example, candidates, logits)
            from s9_occurrence_quotient import compile_quotient, quotient_from_emitted_spans

            graph = compile_quotient(
                quotient_from_emitted_spans(str(example.row["question"]), spans)
            )
        return graph
    except (ValueError, IndexError):
        return None


def _candidate_choice_counts(example, candidates):
    spans = example.row["spans"]
    cards = sorted(
        (
            int(label.split(".")[1]),
            min(int(value) for value in span["token_positions"]),
            max(int(value) for value in span["token_positions"]),
        )
        for label, span in spans.items()
        if label.startswith("card.") and label.endswith(".operation")
    )
    events = sorted(
        (
            int(label.split(".")[1]),
            min(int(value) for value in span["token_positions"]),
            max(int(value) for value in span["token_positions"]),
        )
        for label, span in spans.items()
        if label.startswith("event.") and label.endswith(".tag")
    )
    entry = min(int(value) for value in spans["entry.tag"]["token_positions"])
    query = min(int(value) for value in spans["query.position"]["token_positions"])
    counts = []
    for index, (_, anchor_start, anchor_end) in enumerate(cards):
        stop = cards[index + 1][1] if index + 1 < len(cards) else min(entry, events[0][1], query)
        counts.append(sum(anchor_end < value.start < stop for value in candidates))
    for index, (_, anchor_start, anchor_end) in enumerate(events):
        stop = events[index + 1][1] if index + 1 < len(events) else query
        counts.append(sum(anchor_end < value.start < stop for value in candidates))
    return counts


def run(board: Path, tokenizer_path: Path, seed: int) -> dict[str, object]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows = [
        json.loads(line)
        for line in (board / "development.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != 2048:
        raise ValueError("S9.1 falsifier requires the closed 2,048-row board")
    rng = random.Random(seed)
    counts = {
        "oracle_structured_exact": 0,
        "low_margin_old_exact": 0,
        "low_margin_structured_exact": 0,
        "uniform_valid": 0,
        "wrong_child_exact": 0,
        "shuffled_exact": 0,
        "recoded_oracle_exact": 0,
        "recoded_within_width_cap": 0,
    }
    local_choices = []
    for row in rows:
        example = compile_row(row, tokenizer)
        candidates = all_candidates(example)
        expected = expected_graph(row)
        oracle = _oracle(candidates)
        graph = _attempt(example, candidates, oracle, True)
        counts["oracle_structured_exact"] += int(
            graph is not None and semantic_key(graph) == semantic_key(expected)
        )

        operation_index = next(
            index
            for index, candidate in enumerate(candidates)
            if candidate.target == ROLE_INDEX["event.operation"]
        )
        low = oracle.clone()
        low[operation_index, ROLE_INDEX["none"]] = 30.0
        old_graph = _attempt(example, candidates, low, False)
        structured_graph = _attempt(example, candidates, low, True)
        counts["low_margin_old_exact"] += int(
            old_graph is not None and semantic_key(old_graph) == semantic_key(expected)
        )
        counts["low_margin_structured_exact"] += int(
            structured_graph is not None
            and semantic_key(structured_graph) == semantic_key(expected)
        )

        uniform = torch.zeros_like(oracle)
        counts["uniform_valid"] += int(
            _attempt(example, candidates, uniform, True) is not None
        )

        operation = candidates[operation_index]
        event_tag_starts = sorted(
            candidate.start
            for candidate in candidates
            if candidate.target == ROLE_INDEX["event.tag"]
        )
        next_anchor = next(
            (value for value in event_tag_starts if value > operation.start),
            len(example.ids),
        )
        wrong_index = next(
            index
            for index, candidate in enumerate(candidates)
            if candidate.target == ROLE_INDEX["none"]
            and operation.end < candidate.start < next_anchor
        )
        wrong = oracle.clone()
        wrong[wrong_index, ROLE_INDEX["event.operation"]] = 40.0
        wrong_graph = _attempt(example, candidates, wrong, True)
        counts["wrong_child_exact"] += int(
            wrong_graph is not None and semantic_key(wrong_graph) == semantic_key(expected)
        )

        shuffled = oracle.clone()
        positive = [index for index, candidate in enumerate(candidates) if candidate.target]
        roles = [candidates[index].target for index in positive]
        rng.shuffle(roles)
        for index in positive:
            shuffled[index] = -20.0
            shuffled[index, ROLE_INDEX["none"]] = 0.0
        for index, role in zip(positive, roles, strict=True):
            shuffled[index, role] = 20.0
        shuffled_graph = _attempt(example, candidates, shuffled, True)
        counts["shuffled_exact"] += int(
            shuffled_graph is not None
            and semantic_key(shuffled_graph) == semantic_key(expected)
        )

        recoded = compile_row(
            recode_operation_ids(compile_s8_row(row, tokenizer), tokenizer).row,
            tokenizer,
        )
        recoded_candidates = all_candidates(recoded)
        counts["recoded_within_width_cap"] += int(
            max(end - start + 1 for start, end, _ in recoded.gold) <= MAX_SPAN_WIDTH
        )
        recoded_graph = _attempt(
            recoded, recoded_candidates, _oracle(recoded_candidates), True
        )
        counts["recoded_oracle_exact"] += int(
            recoded_graph is not None
            and semantic_key(recoded_graph) == semantic_key(expected_graph(recoded.row))
        )
        local_choices.extend(_candidate_choice_counts(example, candidates))

    total = len(rows)
    gates = {
        "oracle_structured_all_exact": counts["oracle_structured_exact"] == total,
        "old_greedy_fails_low_margin_intervention": counts["low_margin_old_exact"] == 0,
        "structured_recovers_low_margin_intervention": (
            counts["low_margin_structured_exact"] == total
        ),
        "uniform_logits_zero_valid": counts["uniform_valid"] == 0,
        "wrong_child_never_semantically_repaired": counts["wrong_child_exact"] == 0,
        "shuffled_roles_below_10pct_exact": counts["shuffled_exact"] / total < 0.10,
        "all_recoded_oracles_exact": counts["recoded_oracle_exact"] == total,
        "all_recodes_within_width_cap": counts["recoded_within_width_cap"] == total,
        "syntax_has_multiple_local_choices": min(local_choices) > 1,
    }
    return {
        "schema": "r12_s9_1_alpha_closed_cpu_falsifier_v1",
        "seed": seed,
        "closed_board": str(board),
        "rows": total,
        "counts": counts,
        "grammar_choice_lower_bound": {
            "regions": len(local_choices),
            "minimum_candidates": min(local_choices),
            "median_candidates": statistics.median(local_choices),
            "maximum_candidates": max(local_choices),
        },
        "gates": gates,
        "decision": (
            "admit_s9_1_alpha_closed_mechanics"
            if all(gates.values())
            else "reject_s9_1_alpha_closed_mechanics"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S9.1 falsifier output: {args.out}")
    result = run(args.board, args.tokenizer, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": result["decision"],
        "counts": result["counts"],
        "grammar_choice_lower_bound": result["grammar_choice_lower_bound"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
