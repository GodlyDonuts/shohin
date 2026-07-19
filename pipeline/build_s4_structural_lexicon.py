#!/usr/bin/env python3
"""Build the S4 structural decoder lexicon from training spans only."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from tokenizers import Tokenizer

from semantic_compiler_falsifier import sha256_file


def add_pattern(records, token_ids, value):
    key = tuple(map(int, token_ids))
    if key in records and records[key] != value:
        raise ValueError("structural lexicon token collision")
    records[key] = value


def build(rows):
    kinds = {}
    amounts = {}
    queries = {}
    widths = collections.Counter()
    event_references = 0
    for row in rows:
        if row.get("split") != "s4_event_tape_train":
            raise ValueError("non-training row in S4 lexicon input")
        for index in range(3):
            widths[len(row["spans"]["intro.entity{}".format(index)]["token_ids"])] += 1
        for index, operation in enumerate(row["program"]):
            add_pattern(kinds, row["spans"]["op{}.kind".format(index)]["token_ids"], operation["kind"])
            add_pattern(
                amounts,
                row["spans"]["op{}.literal".format(index)]["token_ids"],
                int(operation["amount"]),
            )
            event_references += 1
        add_pattern(
            queries,
            row["spans"]["query.position"]["token_ids"],
            int(row["query"]["position"]),
        )
    return {
        "kind_patterns": [
            {"token_ids": list(tokens), "value": value}
            for tokens, value in sorted(kinds.items())
        ],
        "amount_patterns": [
            {"token_ids": list(tokens), "value": value}
            for tokens, value in sorted(amounts.items())
        ],
        "query_patterns": [
            {"token_ids": list(tokens), "value": value}
            for tokens, value in sorted(queries.items())
        ],
        "entity_width_histogram": {str(width): count for width, count in sorted(widths.items())},
        "entity_widths": sorted(widths),
        "event_references": event_references,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing S4 structural lexicon")
    report = json.load(open(args.report))
    if not report.get("all_gates_pass"):
        raise SystemExit("S4 corpus report did not pass")
    if report["artifacts"]["train"]["sha256"] != sha256_file(args.data):
        raise SystemExit("S4 corpus report does not bind lexicon input")
    if report["tokenizer_sha256"] != sha256_file(args.tokenizer):
        raise SystemExit("S4 tokenizer mismatch")
    Tokenizer.from_file(args.tokenizer)
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    values = build(rows)
    gates = {
        "training_rows_only": all(row.get("split") == "s4_event_tape_train" for row in rows),
        "exactly_twelve_direction_patterns": len(values["kind_patterns"]) == 12,
        "both_direction_classes": {record["value"] for record in values["kind_patterns"]} == {"left", "right"},
        "both_amounts": {record["value"] for record in values["amount_patterns"]} == {1, 2},
        "all_query_positions": {record["value"] for record in values["query_patterns"]} == {0, 1, 2},
        "entity_token_widths_nonempty_and_bounded": (
            bool(values["entity_widths"])
            and min(values["entity_widths"]) >= 1
            and max(values["entity_widths"]) <= 16
        ),
        "all_intro_spans_accounted": sum(values["entity_width_histogram"].values()) == 3 * len(rows),
        "all_events_accounted": values["event_references"] == sum(len(row["program"]) for row in rows),
        "development_access_zero": True,
        "confirmation_access_zero": True,
    }
    result = {
        "schema": "r12_s4_pointer_anchored_structural_lexicon_v1",
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "data_sha256": sha256_file(args.data),
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "builder_sha256": sha256_file(__file__),
        "training_rows": len(rows),
        "development_access": 0,
        "confirmation_access": 0,
        **values,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not result["all_gates_pass"]:
        raise SystemExit("S4 structural lexicon gates failed")


if __name__ == "__main__":
    main()
