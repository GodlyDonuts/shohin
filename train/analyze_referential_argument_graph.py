#!/usr/bin/env python3
"""Diagnose why a frozen R5 argument-arity intervention helped or failed."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from referential_slot_microcode import OPERATION_ARITIES


def empty_counts():
    return collections.Counter({
        "operations": 0, "raw_kind_correct": 0, "argument_kind_correct": 0,
        "kind_changed": 0, "changed_corrected": 0, "changed_harmed": 0,
        "changed_wrong_to_wrong": 0, "raw_error_crosses_arity_partition": 0,
        "raw_error_within_arity_partition": 0, "inferred_arity_correct": 0,
        "inferred_arity_unresolved": 0,
    })


def analyze_records(raw_records, candidate_records):
    regimes = collections.defaultdict(empty_counts)
    kinds = collections.defaultdict(empty_counts)
    fresh_kinds = collections.defaultdict(empty_counts)
    raw_confusion = collections.Counter()
    candidate_confusion = collections.Counter()
    answer_transitions = collections.Counter()
    program_transitions = collections.Counter()
    for raw, candidate in zip(raw_records, candidate_records):
        if raw["reference"] != candidate["reference"]:
            raise ValueError("report record order differs")
        if raw["operation_kind_predictions"] != candidate["raw_operation_kind_predictions"]:
            raise ValueError("candidate raw kinds do not reproduce raw report")
        regime = raw["regime"]
        answer_transitions[(regime, bool(raw["answer_correct"]), bool(candidate["answer_correct"]))] += 1
        program_transitions[(regime, bool(raw["program_exact"]), bool(candidate["program_exact"]))] += 1
        for target, raw_kind, argument_kind, inferred_arity in zip(
            raw["operation_kind_targets"], raw["operation_kind_predictions"],
            candidate["operation_kind_predictions"], candidate["argument_arity_predictions"],
        ):
            target = int(target)
            raw_kind = int(raw_kind)
            argument_kind = int(argument_kind)
            target_arity = OPERATION_ARITIES[target]
            raw_arity = OPERATION_ARITIES[raw_kind]
            raw_confusion[(target, raw_kind)] += 1
            candidate_confusion[(target, argument_kind)] += 1
            destinations = [regimes[regime], kinds[target]]
            if regime in {"language_ood", "full_ood"}:
                destinations.append(fresh_kinds[target])
            for counts in destinations:
                counts["operations"] += 1
                counts["raw_kind_correct"] += raw_kind == target
                counts["argument_kind_correct"] += argument_kind == target
                counts["inferred_arity_correct"] += inferred_arity == target_arity
                counts["inferred_arity_unresolved"] += inferred_arity not in (1, 2)
                if raw_kind != target:
                    counts[
                        "raw_error_crosses_arity_partition"
                        if raw_arity != target_arity else "raw_error_within_arity_partition"
                    ] += 1
                if raw_kind != argument_kind:
                    counts["kind_changed"] += 1
                    if raw_kind != target and argument_kind == target:
                        counts["changed_corrected"] += 1
                    elif raw_kind == target and argument_kind != target:
                        counts["changed_harmed"] += 1
                    else:
                        counts["changed_wrong_to_wrong"] += 1

    def finalize(counter):
        item = dict(counter)
        total = item["operations"]
        errors = total - item["raw_kind_correct"]
        item["raw_kind_accuracy"] = item["raw_kind_correct"] / total if total else 0.0
        item["argument_kind_accuracy"] = item["argument_kind_correct"] / total if total else 0.0
        item["inferred_arity_accuracy"] = item["inferred_arity_correct"] / total if total else 0.0
        item["raw_error_cross_arity_fraction"] = (
            item["raw_error_crosses_arity_partition"] / errors if errors else 0.0
        )
        return item

    fresh_counts = empty_counts()
    for regime in ("language_ood", "full_ood"):
        fresh_counts.update(regimes[regime])

    def transitions(counter, regimes_to_keep):
        return {
            "raw_wrong_argument_wrong": sum(
                count for (regime, raw_ok, candidate_ok), count in counter.items()
                if regime in regimes_to_keep and not raw_ok and not candidate_ok
            ),
            "raw_wrong_argument_right": sum(
                count for (regime, raw_ok, candidate_ok), count in counter.items()
                if regime in regimes_to_keep and not raw_ok and candidate_ok
            ),
            "raw_right_argument_wrong": sum(
                count for (regime, raw_ok, candidate_ok), count in counter.items()
                if regime in regimes_to_keep and raw_ok and not candidate_ok
            ),
            "raw_right_argument_right": sum(
                count for (regime, raw_ok, candidate_ok), count in counter.items()
                if regime in regimes_to_keep and raw_ok and candidate_ok
            ),
        }

    return {
        "by_regime": {key: finalize(value) for key, value in sorted(regimes.items())},
        "by_target_operation_kind": {str(key): finalize(value) for key, value in sorted(kinds.items())},
        "fresh_by_target_operation_kind": {
            str(key): finalize(value) for key, value in sorted(fresh_kinds.items())
        },
        "fresh_language_full": finalize(fresh_counts),
        "fresh_answer_transitions": transitions(answer_transitions, {"language_ood", "full_ood"}),
        "fresh_program_transitions": transitions(program_transitions, {"language_ood", "full_ood"}),
        "raw_kind_confusion": {
            "{}->{}".format(target, predicted): count
            for (target, predicted), count in sorted(raw_confusion.items())
        },
        "argument_kind_confusion": {
            "{}->{}".format(target, predicted): count
            for (target, predicted), count in sorted(candidate_confusion.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--argument-graph", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise SystemExit("refusing existing output")
    raw = json.load(open(args.raw))
    candidate = json.load(open(args.argument_graph))
    if raw.get("data_sha256") != candidate.get("data_sha256"):
        raise SystemExit("reports do not share data")
    result = {
        "audit": "referential_argument_graph_failure_analysis_v5",
        "raw": str(Path(args.raw).resolve()),
        "argument_graph": str(Path(args.argument_graph).resolve()),
        "data_sha256": raw["data_sha256"],
        **analyze_records(raw["records"], candidate["records"]),
        "claim_boundary": (
            "This localizes a frozen intervention's error transitions. It does not authorize "
            "threshold tuning, operator fitting, or a reasoning claim."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "fresh_language_full": result["fresh_language_full"],
        "fresh_answer_transitions": result["fresh_answer_transitions"],
        "fresh_program_transitions": result["fresh_program_transitions"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
