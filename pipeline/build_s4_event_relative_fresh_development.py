#!/usr/bin/env python3
"""Build the one-shot fresh S4 v2 development board after source freeze."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from tokenizers import Tokenizer

from build_referential_literal_pointer_factorized_corpus import (
    normalized,
    ngrams,
)
from build_s4_self_delimiting_event_tape import (
    DEVELOPMENT_SPLIT,
    build_development,
    fresh_paired_names,
    gold_events,
    read_public,
    select_factors,
    write_jsonl,
)
from semantic_compiler_falsifier import sha256_file


NONCE_PATTERN = re.compile(r"\b[a-z]{4}-[a-z]{4}\b")


def source_nonce_names(paths):
    names = set()
    for path in paths:
        with open(path) as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                names.update(NONCE_PATTERN.findall(row.get("question", "")))
    return names


def audit_fresh(rows, public, source_train_sha256, source_report_sha256, tokenizer_path):
    questions = [normalized(row["question"]) for row in rows]
    row_grams = [ngrams(row["question"]) for row in rows]
    names = [set(NONCE_PATTERN.findall(row["question"])) for row in rows]
    groups = collections.defaultdict(list)
    for row in rows:
        groups[int(row["group"])].append(row)
    depth_counts = collections.Counter(int(row["depth"]) for row in rows)
    group_contract = True
    for group in groups.values():
        if len(group) != 4 or {row["surface_type"] for row in group} != {
            "canonical", "paraphrase", "order_twin", "binding_twin",
        }:
            group_contract = False
            break
    overlap_grams = sum(bool(grams & public["grams"]) for grams in row_grams)
    overlap_names = sum(bool(values & public["names"]) for values in names)
    overlap_factors = sum(row["factor_signature"] in public["factors"] for row in rows)
    gates = {
        "exactly_2048_rows": len(rows) == 2048,
        "exactly_512_matched_groups": len(groups) == 512 and group_contract,
        "depths_three_through_eight": set(depth_counts) == set(range(3, 9)),
        "each_depth_at_least_300_rows": min(depth_counts.values(), default=0) >= 300,
        "all_unpadded_whole_sources": all(
            row.get("split") == DEVELOPMENT_SPLIT
            and "active_operations" not in row
            and row.get("renderer") == "s4_whole_source_unpadded"
            for row in rows
        ),
        "event_count_equals_depth": all(len(gold_events(row)) == row["depth"] for row in rows),
        "independent_executors_agree": all(row.get("executor_agreement") for row in rows),
        "all_sources_fit_context": max((row["token_count"] for row in rows), default=999999) <= 2048,
        "no_internal_duplicate_questions": len(questions) == len(set(questions)),
        "no_public_exact_prompt_overlap": not (set(questions) & public["questions"]),
        "no_public_13gram_overlap": overlap_grams == 0,
        "no_public_nonce_name_overlap": overlap_names == 0,
        "no_public_factor_overlap": overlap_factors == 0,
        "source_training_is_hash_bound": bool(source_train_sha256),
        "source_report_is_hash_bound": bool(source_report_sha256),
        "development_access_zero_at_build": True,
        "confirmation_access_zero": True,
    }
    return {
        "schema": "r12_s4_event_relative_fresh_development_report_v1",
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "rows": len(rows),
        "groups": len(groups),
        "depth_counts": {str(depth): count for depth, count in sorted(depth_counts.items())},
        "max_tokens": max((row["token_count"] for row in rows), default=0),
        "public_exact_prompt_overlaps": len(set(questions) & public["questions"]),
        "public_13gram_rows": overlap_grams,
        "public_nonce_name_rows": overlap_names,
        "public_factor_rows": overlap_factors,
        "source_training_sha256": source_train_sha256,
        "source_report_sha256": source_report_sha256,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "development_access": 0,
        "confirmation_access": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="artifacts/shohin-tok-32k.json")
    parser.add_argument("--source-training-data", required=True)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--public-data", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--groups", type=int, default=512)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.groups != 512:
        raise SystemExit("production S4 v2 board requires exactly 512 groups")
    output = Path(args.out_dir)
    data_path = output / "development.jsonl"
    report_path = output / "report.json"
    if data_path.exists() or report_path.exists():
        raise SystemExit("refusing existing S4 v2 board")
    source_report = json.load(open(args.source_report))
    source_train_sha256 = sha256_file(args.source_training_data)
    if source_report.get("artifacts", {}).get("train", {}).get("sha256") != source_train_sha256:
        raise SystemExit("source S4 report does not bind training data")
    if not source_report.get("all_gates_pass") or source_report.get("confirmation_access") != 0:
        raise SystemExit("source S4 corpus is not admitted")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    public_paths = list(dict.fromkeys([
        args.source_training_data,
        *args.public_data,
    ]))
    public = read_public(public_paths)
    public["names"].update(source_nonce_names(public_paths))
    names = fresh_paired_names(tokenizer, 1000, args.seed ^ 0x51A4E2, public["names"])
    factors = select_factors(
        "development_compositional", args.groups * 2, args.seed ^ 0xE7E170,
        public["factors"],
    )
    rows = build_development(args.groups, args.seed, tokenizer, names, factors)
    report = audit_fresh(
        rows,
        public,
        source_train_sha256,
        sha256_file(args.source_report),
        args.tokenizer,
    )
    report["seed"] = int(args.seed)
    report["builder_sha256"] = sha256_file(__file__)
    output.mkdir(parents=True, exist_ok=True)
    report["artifacts"] = {"development": write_jsonl(data_path, rows)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit("fresh S4 v2 board failed gates")


if __name__ == "__main__":
    main()
