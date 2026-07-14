#!/usr/bin/env python3
"""Reject paired-answer response grammar from a direct operator-trace mix.

The first operator-trace SFT mixed ordinary single-problem rows with minimal
pairs.  That taught a visible ``Problem A`` / ``Problem B`` answer mode which
then leaked into otherwise ordinary direct prompts.  A broad mix can pass
structural, quality, and evaluation-overlap audits while retaining that mode,
so this source-level audit is a separate admission boundary.

It is deliberately narrow: it makes no claim about answer quality.  It only
binds a frozen JSONL to the requirement that its operator-trace rows are
single-problem ``direct`` rows with none of the paired-answer markers in any
field consumed by SFT.
"""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


PAIR_MARKERS = {
    "problem_a": re.compile(r"\bproblem\s+a\s*:", re.IGNORECASE),
    "problem_b": re.compile(r"\bproblem\s+b\s*:", re.IGNORECASE),
    "answers_are_a": re.compile(r"\bthe\s+answers\s+are\s+a\s*=", re.IGNORECASE),
}
CONSUMED_FIELDS = ("question", "completion_prompt", "response")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(path, group="operator_trace_contrast"):
    """Return a deterministic source-contract report for ``path``."""
    marker_rows = collections.Counter()
    marker_fields = collections.defaultdict(collections.Counter)
    contracts = collections.Counter()
    operator_rows = malformed_rows = missing_contract_rows = 0

    with open(path, errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue
            if not isinstance(row, dict) or str(row.get("training_group") or "") != group:
                continue
            operator_rows += 1
            contract = str(row.get("contract") or "").strip()
            if not contract:
                missing_contract_rows += 1
            else:
                contracts[contract] += 1
            for field in CONSUMED_FIELDS:
                value = row.get(field)
                if not isinstance(value, str):
                    continue
                for name, pattern in PAIR_MARKERS.items():
                    if pattern.search(value):
                        marker_rows[name] += 1
                        marker_fields[field][name] += 1

    return {
        "schema": "shohin-operator-trace-directness-v1",
        "data": str(Path(path)),
        "data_sha256": sha256_file(path),
        "group": group,
        "operator_rows": operator_rows,
        "malformed_rows": malformed_rows,
        "missing_contract_rows": missing_contract_rows,
        "contract_rows": dict(contracts),
        "pair_marker_rows": dict(marker_rows),
        "pair_marker_rows_by_field": {
            field: dict(counter) for field, counter in sorted(marker_fields.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--group", default="operator_trace_contrast")
    parser.add_argument(
        "--require-only-contract",
        default="",
        help="fail unless every row in --group has exactly this contract",
    )
    parser.add_argument("--expect-sha256", default="")
    args = parser.parse_args()

    report = audit(args.data, args.group)
    if args.expect_sha256 and report["data_sha256"] != args.expect_sha256:
        raise SystemExit("input SHA-256 mismatch")
    if report["malformed_rows"] or report["missing_contract_rows"]:
        raise SystemExit("operator directness audit found malformed or contractless rows")
    if not report["operator_rows"]:
        raise SystemExit("operator directness audit found no operator-trace rows")
    if args.require_only_contract and report["contract_rows"] != {
        args.require_only_contract: report["operator_rows"]
    }:
        raise SystemExit("operator directness audit found a non-direct contract")
    if any(report["pair_marker_rows"].values()):
        raise SystemExit("operator directness audit found paired-answer response grammar")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "[operator-directness] rows={} contracts={} sha256={}".format(
            report["operator_rows"], report["contract_rows"], report["data_sha256"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
