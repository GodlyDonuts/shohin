#!/usr/bin/env python3
"""Audit text-only structural spans and training-only mention supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from referential_slot_microcode import compile_referential_example  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(path, tokenizer, expected_rows):
    counts = Counter()
    errors = []
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            counts["rows"] += 1
            try:
                row = json.loads(line)
                example = compile_referential_example(row, tokenizer)
                if set(example.intro_slot_targets[0]) & set(example.intro_slot_targets[1]):
                    raise ValueError("introductory slot targets overlap")
                if not all(example.intro_slot_targets):
                    raise ValueError("introductory slot target is empty")
                counts["intro_slot_mentions"] += 2
                for targets in example.operation_mention_targets:
                    counts["role_operations" if targets else "role_free_operations"] += 1
                counts["role_queries" if example.query_mention_target else "role_free_queries"] += 1
                counts["operation_mention_tokens"] += sum(map(len, example.operation_mention_targets))
                counts["query_mention_tokens"] += len(example.query_mention_target)
                counts["intro_mention_tokens"] += sum(map(len, example.intro_slot_targets))
                counts["max_tokens"] = max(counts["max_tokens"], len(example.compiled.ids))
            except Exception as error:
                if len(errors) < 20:
                    errors.append({"line": line_number, "error": str(error)})
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256(path),
        "expected_rows": expected_rows,
        "counts": dict(sorted(counts.items())),
        "errors": errors,
        "all_checks_pass": not errors and counts["rows"] == expected_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--manual", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-rows", type=int, default=288000)
    parser.add_argument("--eval-rows", type=int, default=896)
    parser.add_argument("--manual-rows", type=int, default=8)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise SystemExit("refusing existing output")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    datasets = {
        "train": audit(args.train, tokenizer, args.train_rows),
        "eval": audit(args.eval, tokenizer, args.eval_rows),
        "manual": audit(args.manual, tokenizer, args.manual_rows),
    }
    report = {
        "audit": "referential_slot_label_admission_v1",
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "tokenizer_sha256": sha256(args.tokenizer),
        "datasets": datasets,
        "all_checks_pass": all(item["all_checks_pass"] for item in datasets.values()),
        "claim_boundary": (
            "Keys generate attention supervision and scoring labels only. Model inference receives token "
            "states and formatting-derived intro/event/query spans, never structured key identities."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
