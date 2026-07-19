#!/usr/bin/env python3
"""CPU sufficiency and adversarial gates for S9 occurrence quotients."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random

from s8_nil_linked_law_graph import (
    EventNode,
    LawCardNode,
    NilLinkedLawGraph,
    execute_graph,
    linked_path,
    rewire_path,
)
from s9_occurrence_quotient import (
    OccurrenceQuotient,
    compile_quotient,
    corrupt_first_relation_kind,
    merge_first_two_entities,
    permute_relation_storage,
    quotient_from_emitted_spans,
    reindex_classes,
    split_first_event_operation,
    swap_card_witnesses,
    unique_every_occurrence,
)


def semantic_key(graph: NilLinkedLawGraph) -> tuple[object, ...]:
    return (
        graph.modulus,
        graph.initial_state,
        tuple(sorted((card.operation, card.y0, card.y1) for card in graph.cards)),
        tuple((node.identity, node.operation, node.next_node) for node in graph.nodes),
        graph.entry_node,
        graph.query_position,
    )


def expected_graph(row: dict[str, object]) -> NilLinkedLawGraph:
    nodes = row["nodes"]
    tag_index = {str(node["tag"]): index for index, node in enumerate(nodes)}
    return NilLinkedLawGraph(
        modulus=int(row["modulus"]),
        initial_state=tuple(int(value) for value in row["initial_state"]),
        cards=tuple(LawCardNode(
            operation=str(card["operation"]),
            y0=int(card["y0"]),
            y1=int(card["y1"]),
        ) for card in row["cards"]),
        nodes=tuple(EventNode(
            identity=int(node["identity"]),
            operation=str(node["operation"]),
            next_node=(
                -1 if node["next_tag"] is None else tag_index[str(node["next_tag"])]
            ),
        ) for node in nodes),
        entry_node=int(row["entry_node"]),
        query_position=int(row["query_position"]),
    )


def load_generator(path: Path) -> tuple[dict[int, tuple[int, ...]], dict[int, int]]:
    cells: dict[int, dict[int, int]] = {}
    zeros: dict[int, int] = {}
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            modulus = int(row["modulus"])
            cells.setdefault(modulus, {})[int(row["current_symbol"])] = int(row["next_symbol"])
            zero = int(row["zero_symbol"])
            if modulus in zeros and zeros[modulus] != zero:
                raise ValueError("S9 generator zero disagreement")
            zeros[modulus] = zero
    successors = {
        modulus: tuple(values[index] for index in range(modulus))
        for modulus, values in cells.items()
    }
    if set(successors) != {5, 7, 11}:
        raise ValueError("S9 generator is missing an admitted modulus")
    return successors, zeros


def _new_score() -> dict[str, int]:
    return {"graph": 0, "state": 0, "answer": 0, "valid": 0, "total": 0}


def _add(score, graph, expected, expected_state, expected_answer, successor, zero):
    score["total"] += 1
    if graph is None:
        return
    score["valid"] += 1
    score["graph"] += int(semantic_key(graph) == semantic_key(expected))
    try:
        state, answer, _ = execute_graph(graph, successor, zero)
    except (ValueError, IndexError):
        return
    score["state"] += int(state == expected_state)
    score["answer"] += int(answer == expected_answer)


def _attempt(quotient: OccurrenceQuotient) -> NilLinkedLawGraph | None:
    try:
        return compile_quotient(quotient)
    except (ValueError, IndexError):
        return None


def _slot_derangement(quotient: OccurrenceQuotient) -> OccurrenceQuotient:
    records = list(quotient.relations)
    index = next(i for i, record in enumerate(records) if record.kind == "event")
    tag, operation, entity, next_tag = records[index].arguments
    records[index] = replace(
        records[index], arguments=(tag, entity, operation, next_tag)
    )
    return replace(quotient, relations=tuple(records))


def _finish(score: dict[str, int]) -> dict[str, int | float]:
    total = score["total"]
    return {
        **score,
        "valid_accuracy": score["valid"] / total,
        "graph_accuracy": score["graph"] / total,
        "state_accuracy": score["state"] / total,
        "answer_accuracy": score["answer"] / total,
    }


def run_falsifier(data_dir: Path, seed: int, limit: int | None = None) -> dict[str, object]:
    development = data_dir / "development.jsonl"
    generator = data_dir / "generator_train.jsonl"
    successors, zeros = load_generator(generator)
    rows = []
    with development.open() as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError("S9 falsifier has no rows")

    rng = random.Random(seed)
    score_names = (
        "treatment",
        "class_reindex",
        "relation_storage_reindex",
        "swapped_card_witnesses",
        "reversed_links",
        "split_reference",
        "merged_entities",
        "free_word_singletons",
        "corrupt_relation_kind",
        "slot_derangement",
    )
    scores = {name: _new_score() for name in score_names}
    rejected = {name: 0 for name in (
        "split_reference",
        "merged_entities",
        "free_word_singletons",
        "corrupt_relation_kind",
        "slot_derangement",
    )}
    for row in rows:
        # The compiler receives only source bytes and emitted spans. All other row
        # fields are isolated below as scorer/executor state.
        quotient = quotient_from_emitted_spans(str(row["question"]), row["spans"])
        expected = expected_graph(row)
        expected_state = tuple(int(value) for value in row["final_state"])
        expected_answer = int(row["answer"])
        modulus = int(row["modulus"])
        successor, zero = successors[modulus], zeros[modulus]

        treatment = _attempt(quotient)
        _add(scores["treatment"], treatment, expected, expected_state, expected_answer, successor, zero)

        class_order = list(range(len(quotient.classes)))
        relation_order = list(range(len(quotient.relations)))
        rng.shuffle(class_order)
        rng.shuffle(relation_order)
        _add(
            scores["class_reindex"],
            _attempt(reindex_classes(quotient, class_order)),
            expected, expected_state, expected_answer, successor, zero,
        )
        _add(
            scores["relation_storage_reindex"],
            _attempt(permute_relation_storage(quotient, relation_order)),
            expected, expected_state, expected_answer, successor, zero,
        )
        _add(
            scores["swapped_card_witnesses"],
            _attempt(swap_card_witnesses(quotient)),
            expected, expected_state, expected_answer, successor, zero,
        )
        reversed_graph = None
        if treatment is not None:
            reversed_graph = rewire_path(treatment, tuple(reversed(linked_path(treatment))))
        _add(
            scores["reversed_links"], reversed_graph,
            expected, expected_state, expected_answer, successor, zero,
        )
        corruptions = {
            "split_reference": split_first_event_operation(quotient),
            "merged_entities": merge_first_two_entities(quotient),
            "free_word_singletons": unique_every_occurrence(quotient),
            "corrupt_relation_kind": corrupt_first_relation_kind(quotient),
            "slot_derangement": _slot_derangement(quotient),
        }
        for name, corruption in corruptions.items():
            graph = _attempt(corruption)
            rejected[name] += int(graph is None)
            _add(scores[name], graph, expected, expected_state, expected_answer, successor, zero)

    finished = {name: _finish(score) for name, score in scores.items()}
    total = len(rows)
    gates = {
        "oracle_emitted_quotient_graph_exact": finished["treatment"]["graph_accuracy"] == 1.0,
        "oracle_emitted_quotient_state_exact": finished["treatment"]["state_accuracy"] == 1.0,
        "oracle_emitted_quotient_answer_exact": finished["treatment"]["answer_accuracy"] == 1.0,
        "class_reindex_exact": finished["class_reindex"]["graph_accuracy"] == 1.0,
        "relation_storage_reindex_exact": finished["relation_storage_reindex"]["graph_accuracy"] == 1.0,
        "split_reference_rejected": rejected["split_reference"] == total,
        "merged_entities_rejected": rejected["merged_entities"] == total,
        "free_word_singletons_rejected_without_merge": rejected["free_word_singletons"] == total,
        "corrupt_relation_kind_rejected": rejected["corrupt_relation_kind"] == total,
        "slot_derangement_rejected": rejected["slot_derangement"] == total,
        "swapped_witness_state_below_40pct": finished["swapped_card_witnesses"]["state_accuracy"] < 0.40,
        "reversed_link_state_below_40pct": finished["reversed_links"]["state_accuracy"] < 0.40,
        "compiler_input_excludes_structured_row_fields": True,
    }
    return {
        "schema": "r12_s9_occurrence_quotient_cpu_falsifier_v1",
        "seed": seed,
        "rows": total,
        "scores": finished,
        "rejected": rejected,
        "gates": gates,
        "decision": (
            "admit_s9_occurrence_quotient_theory_only"
            if all(gates.values())
            else "reject_s9_occurrence_quotient_mechanics"
        ),
        "resource_boundary": {
            "cpu_oracle_only": "frozen labeled spans stand in for future neural emissions",
            "future_model_owned": [
                "surface-island boundaries",
                "relation kinds",
                "argument slots",
                "entry, next, nil, and query relations",
            ],
            "architectural": [
                "exact equality over model-emitted nonempty surface spans",
                "class and relation validation",
                "unchanged S8 graph traversal and S7 cyclic runtime",
            ],
            "forbidden_at_inference": [
                "gold spans or candidate-name dictionaries",
                "gold relation tuples",
                "source event order or depth",
                "gold state, answer, or repaired graph",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S9 report: {args.out}")
    report = run_falsifier(args.data_dir, args.seed, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "out": str(args.out),
        "rows": report["rows"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
