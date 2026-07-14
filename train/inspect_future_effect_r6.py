#!/usr/bin/env python3
"""Build a qualitative transcript from an R6 active-distinction evaluation."""

import argparse
import json
from pathlib import Path


def category(record):
    active = record["policies"]["active"]["answer_correct"]
    random = record["policies"]["random"]["answer_correct"]
    controls = (
        record["policies"]["zero"]["answer_correct"]
        or record["policies"]["shuffled"]["answer_correct"]
    )
    if active and not random and not controls:
        return "active_only_over_random_and_controls"
    if active and not random:
        return "active_only_over_random"
    if not active and random:
        return "random_only"
    if active:
        return "both_correct"
    return "both_wrong"


def compact_policy(policy):
    return {
        "answer": policy["answer"],
        "answer_correct": policy["answer_correct"],
        "exact_program": policy["exact_program"],
        "opcodes": policy["opcodes"],
        "values": policy["values"],
        "traces": policy["traces"],
    }


def build_transcript(report, rows, count):
    if report.get("protocol") != "active_counterfactual_distinction_eval_r6":
        raise ValueError("wrong R6 report protocol")
    priorities = (
        "active_only_over_random_and_controls",
        "active_only_over_random",
        "random_only",
        "both_wrong",
        "both_correct",
    )
    buckets = {name: [] for name in priorities}
    for record in report["records"]:
        if record["regime"] not in {"language", "full"}:
            continue
        buckets[category(record)].append(record)
    selected = []
    while len(selected) < count and any(buckets.values()):
        for name in priorities:
            if buckets[name] and len(selected) < count:
                selected.append((name, buckets[name].pop(0)))
    transcript = []
    for name, record in selected:
        row = rows[record["index"]]
        transcript.append({
            "category": name,
            "index": record["index"],
            "reference": record["reference"],
            "regime": record["regime"],
            "question": row["question"],
            "expected_answer": record["answer"],
            "query_prediction": record["query_prediction"],
            "query_correct": record["query_correct"],
            "active": compact_policy(record["policies"]["active"]),
            "random": compact_policy(record["policies"]["random"]),
            "zero": compact_policy(record["policies"]["zero"]),
            "shuffled": compact_policy(record["policies"]["shuffled"]),
            "oracle": compact_policy(record["policies"]["oracle"]),
        })
    return {
        "protocol": "active_counterfactual_distinction_qualitative_transcript_r6",
        "source_report": report.get("adapter_sha256"),
        "requested_cases": count,
        "selected_cases": len(transcript),
        "available_categories": {
            name: sum(
                record["regime"] in {"language", "full"} and category(record) == name
                for record in report["records"]
            )
            for name in priorities
        },
        "cases": transcript,
        "claim_boundary": (
            "These are verbatim learned-effect traces selected by a fixed category order. "
            "They support error analysis, not a benchmark or broad reasoning claim."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    if args.count <= 0:
        raise SystemExit("count must be positive")
    if Path(args.out).exists():
        raise SystemExit("refusing existing output")
    report = json.load(open(args.report))
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    if len(rows) != len(report.get("records", [])):
        raise SystemExit("report/data row count mismatch")
    transcript = build_transcript(report, rows, args.count)
    Path(args.out).write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n")
    print(json.dumps(transcript["available_categories"], sort_keys=True))
    print("wrote {} qualitative cases to {}".format(transcript["selected_cases"], args.out))


if __name__ == "__main__":
    main()
