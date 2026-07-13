#!/usr/bin/env python3
"""Audit the response-format balance of a frozen SFT JSONL candidate.

Reasoning mixtures can pass syntax, provenance, and evaluation-overlap checks
while still teaching one narrow response mode.  This read-only audit records
the response contracts actually present in each immutable training group.  It
does not assess answer correctness or admit data by itself.
"""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


THINK_MARKER = "<think>"
ANSWER_MARKER = "the answer is"
STATE_PATTERN = re.compile(r"(?:^|\n)\s*(?:state=|wm:)", re.IGNORECASE)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, fraction):
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def rate(count, total):
    return count / total if total else 0.0


def audit(path, group_field="training_group", response_field="response"):
    """Return a deterministic, JSON-serializable contract report for ``path``."""
    counters = collections.defaultdict(collections.Counter)
    lengths = collections.defaultdict(list)
    malformed = 0
    missing_response = 0
    rows = 0

    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(row, dict):
                malformed += 1
                continue

            response = row.get(response_field)
            if not isinstance(response, str) or not response.strip():
                missing_response += 1
                continue

            group = str(row.get(group_field) or "default")
            lowered = response.lower()
            stat = counters[group]
            stat["rows"] += 1
            stat["think_marker"] += int(THINK_MARKER in lowered)
            stat["answer_marker"] += int(ANSWER_MARKER in lowered)
            stat["state_marker"] += int(bool(STATE_PATTERN.search(response)))
            stat["code_fence"] += int("```" in response)
            stat["short_response"] += int(len(response) < 80)
            stat["starts_think"] += int(lowered.lstrip().startswith(THINK_MARKER))
            lengths[group].append(len(response))
            rows += 1

    groups = {}
    for group in sorted(counters):
        stat = counters[group]
        total = stat["rows"]
        groups[group] = {
            "rows": total,
            "think_marker": stat["think_marker"],
            "think_marker_rate": rate(stat["think_marker"], total),
            "starts_think": stat["starts_think"],
            "starts_think_rate": rate(stat["starts_think"], total),
            "answer_marker": stat["answer_marker"],
            "answer_marker_rate": rate(stat["answer_marker"], total),
            "state_marker": stat["state_marker"],
            "state_marker_rate": rate(stat["state_marker"], total),
            "code_fence": stat["code_fence"],
            "code_fence_rate": rate(stat["code_fence"], total),
            "short_response": stat["short_response"],
            "short_response_rate": rate(stat["short_response"], total),
            "response_chars_p50": percentile(lengths[group], 0.50),
            "response_chars_p90": percentile(lengths[group], 0.90),
        }

    return {
        "schema": "shohin-response-contract-audit-v1",
        "data": str(Path(path)),
        "data_sha256": sha256_file(path),
        "rows": rows,
        "malformed_rows": malformed,
        "missing_response_rows": missing_response,
        "group_field": group_field,
        "response_field": response_field,
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="frozen JSONL candidate")
    parser.add_argument("--out", required=True, help="report JSON path")
    parser.add_argument("--group-field", default="training_group")
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--expect-sha256", default="", help="fail if the input digest differs")
    args = parser.parse_args()

    result = audit(args.data, args.group_field, args.response_field)
    if args.expect_sha256 and result["data_sha256"] != args.expect_sha256:
        raise SystemExit(
            "input SHA-256 mismatch: expected {} got {}".format(
                args.expect_sha256, result["data_sha256"]
            )
        )
    if result["malformed_rows"] or result["missing_response_rows"]:
        raise SystemExit(
            "contract audit rejected malformed={} missing_response={}".format(
                result["malformed_rows"], result["missing_response_rows"]
            )
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "[response-contract] rows={} groups={} sha256={}".format(
            result["rows"], ",".join(result["groups"]), result["data_sha256"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
