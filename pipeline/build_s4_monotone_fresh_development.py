#!/usr/bin/env python3
"""Build the one-shot fresh S4 v4 monotone event-region development board."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from build_s4_self_delimiting_event_tape import (
    build_development,
    fresh_paired_names,
    read_public,
    select_factors,
    write_jsonl,
)
from build_s4_set_identity_fresh_development import (
    audit_fresh as audit_set_identity_fresh,
    public_entity_multisets,
)
from build_s4_event_relative_fresh_development import source_nonce_names
from semantic_compiler_falsifier import sha256_file


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
        raise SystemExit("production S4 v4 board requires exactly 512 groups")
    output = Path(args.out_dir)
    data_path = output / "development.jsonl"
    report_path = output / "report.json"
    if data_path.exists() or report_path.exists():
        raise SystemExit("refusing existing S4 v4 board")
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
    names = fresh_paired_names(tokenizer, 1000, args.seed ^ 0x4D4F4E4F, public["names"])
    factors = select_factors(
        "development_compositional",
        args.groups * 2,
        args.seed ^ 0x52454749,
        public["factors"],
    )
    rows = build_development(args.groups, args.seed, tokenizer, names, factors)
    report = audit_set_identity_fresh(
        rows,
        public,
        source_train_sha256,
        sha256_file(args.source_report),
        args.tokenizer,
        public_multisets,
    )
    report["schema"] = "r12_s4_monotone_event_region_fresh_development_report_v1"
    report["seed"] = int(args.seed)
    report["builder_sha256"] = sha256_file(__file__)
    output.mkdir(parents=True, exist_ok=True)
    report["artifacts"] = {"development": write_jsonl(data_path, rows)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit("fresh S4 v4 board failed gates")


if __name__ == "__main__":
    main()
