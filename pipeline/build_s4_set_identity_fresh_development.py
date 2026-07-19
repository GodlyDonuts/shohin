#!/usr/bin/env python3
"""Build the one-shot fresh S4 v3 set-identity development board."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from build_s4_event_relative_fresh_development import (
    audit_fresh as audit_event_relative_fresh,
    source_nonce_names,
)
from build_s4_self_delimiting_event_tape import (
    build_development,
    fresh_paired_names,
    read_public,
    select_factors,
    source_rows,
    write_jsonl,
)
from semantic_compiler_falsifier import sha256_file


def entity_multisets(row, tokenizer):
    if not row.get("question"):
        return ()
    encoding = tokenizer.encode(row["question"])
    values = []
    for identity in range(3):
        positions = row.get("spans", {}).get(
            "intro.entity{}".format(identity), {},
        ).get("token_positions", ())
        if not positions:
            return ()
        values.append(tuple(sorted(encoding.ids[int(position)] for position in positions)))
    return tuple(values)


def public_entity_multisets(paths, tokenizer):
    values = set()
    for path in paths:
        with open(path) as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                for item in source_rows(row):
                    values.update(entity_multisets(item, tokenizer))
    return values


def audit_fresh(
    rows,
    public,
    source_train_sha256,
    source_report_sha256,
    tokenizer_path,
    public_multisets=None,
):
    result = audit_event_relative_fresh(
        rows, public, source_train_sha256, source_report_sha256, tokenizer_path,
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    row_multisets = [entity_multisets(row, tokenizer) for row in rows]
    external = set(public_multisets or ())
    internal_unique = all(len(values) == 3 and len(set(values)) == 3 for values in row_multisets)
    overlap_rows = sum(bool(set(values) & external) for values in row_multisets)
    result["schema"] = "r12_s4_set_identity_fresh_development_report_v1"
    result["gates"]["three_unique_roster_token_multisets_per_row"] = internal_unique
    result["gates"]["no_public_roster_token_multiset_overlap"] = overlap_rows == 0
    result["public_roster_token_multiset_rows"] = overlap_rows
    result["all_gates_pass"] = all(result["gates"].values())
    return result


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
        raise SystemExit("production S4 v3 board requires exactly 512 groups")
    output = Path(args.out_dir)
    data_path = output / "development.jsonl"
    report_path = output / "report.json"
    if data_path.exists() or report_path.exists():
        raise SystemExit("refusing existing S4 v3 board")
    source_report = json.load(open(args.source_report))
    source_train_sha256 = sha256_file(args.source_training_data)
    if source_report.get("artifacts", {}).get("train", {}).get("sha256") != source_train_sha256:
        raise SystemExit("source S4 report does not bind training data")
    if not source_report.get("all_gates_pass") or source_report.get("confirmation_access") != 0:
        raise SystemExit("source S4 corpus is not admitted")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    public_paths = list(dict.fromkeys([args.source_training_data, *args.public_data]))
    public = read_public(public_paths)
    public["names"].update(source_nonce_names(public_paths))
    public_multisets = public_entity_multisets(public_paths, tokenizer)
    names = fresh_paired_names(tokenizer, 1000, args.seed ^ 0x51E7B5, public["names"])
    factors = select_factors(
        "development_compositional",
        args.groups * 2,
        args.seed ^ 0xE7E171,
        public["factors"],
    )
    rows = build_development(args.groups, args.seed, tokenizer, names, factors)
    report = audit_fresh(
        rows,
        public,
        source_train_sha256,
        sha256_file(args.source_report),
        args.tokenizer,
        public_multisets,
    )
    report["seed"] = int(args.seed)
    report["builder_sha256"] = sha256_file(__file__)
    output.mkdir(parents=True, exist_ok=True)
    report["artifacts"] = {"development": write_jsonl(data_path, rows)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit("fresh S4 v3 board failed gates")


if __name__ == "__main__":
    main()
