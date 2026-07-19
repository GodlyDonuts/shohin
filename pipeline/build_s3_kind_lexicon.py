#!/usr/bin/env python3
"""Build an exact-token direction lexicon from frozen training spans only."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from referential_literal_pointer_compiler import compile_row, sha256_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise SystemExit("refusing existing kind lexicon")
    report = json.load(open(args.report))
    if not report.get("all_gates_pass"):
        raise SystemExit("kind lexicon requires admitted factorized training data")
    if report["artifacts"]["train"]["sha256"] != sha256_file(args.data):
        raise SystemExit("factorized report does not bind kind-lexicon data")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    patterns = {}
    counts = collections.Counter()
    rows = 0
    references = 0
    with open(args.data) as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("kind lexicon received non-training row")
            example = compile_row(row, tokenizer, keep_evidence=True)
            rows += 1
            for index, operation in enumerate(row["program"]):
                token_ids = tuple(
                    example.ids[position]
                    for position in example.target_positions["op{}.kind".format(index)]
                )
                previous = patterns.get(token_ids)
                if previous is not None and previous != operation["kind"]:
                    raise ValueError("kind token sequence has cross-class collision")
                patterns[token_ids] = operation["kind"]
                counts[(token_ids, operation["kind"])] += 1
                references += 1
    records = [{
        "kind": kind,
        "token_ids": list(token_ids),
        "count": counts[(token_ids, kind)],
        "decoded": tokenizer.decode(list(token_ids), skip_special_tokens=False),
    } for token_ids, kind in sorted(patterns.items(), key=lambda item: (item[1], item[0]))]
    by_kind = collections.Counter(record["kind"] for record in records)
    gates = {
        "training_rows_exactly_96000": rows == 96_000,
        "operation_references_exactly_192000": references == 192_000,
        "six_patterns_per_kind": by_kind == {"left": 6, "right": 6},
        "no_cross_class_token_sequence": len(records) == 12,
        "no_development_or_confirmation_access": True,
    }
    result = {
        "schema": "r12_s3_training_kind_lexicon_v1",
        "patterns": records,
        "rows": rows,
        "references": references,
        "patterns_by_kind": dict(sorted(by_kind.items())),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "data_sha256": sha256_file(args.data),
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "fit_updates": 0,
        "development_access": 0,
        "confirmation_access": 0,
        "claim_boundary": (
            "Training-span lexical relation table only; no development, confirmation, "
            "execution, language generalization, or reasoning claim."
        ),
    }
    if not result["all_gates_pass"]:
        raise SystemExit("kind lexicon gates failed: {}".format(gates))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(Path(args.out).resolve()),
        "patterns_by_kind": result["patterns_by_kind"],
        "references": references,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
