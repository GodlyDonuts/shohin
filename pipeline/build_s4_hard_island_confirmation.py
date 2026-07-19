#!/usr/bin/env python3
"""Build one disjoint S4 v5 confirmation board after development qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from build_s4_event_relative_fresh_development import source_nonce_names
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
from semantic_compiler_falsifier import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="artifacts/shohin-tok-32k.json")
    parser.add_argument("--source-training-data", required=True)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--public-data", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--groups", type=int, default=512)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.groups != 512:
        raise SystemExit("S4 v5 confirmation requires exactly 512 groups")
    qualification = json.load(open(args.qualification))
    if qualification.get("decision") != "qualify_s4_v5_for_fresh_confirmation":
        raise SystemExit("S4 v5 development did not qualify")
    output = Path(args.out_dir)
    data_path = output / "confirmation.jsonl"
    report_path = output / "report.json"
    if data_path.exists() or report_path.exists():
        raise SystemExit("refusing existing S4 v5 confirmation board")
    source_report = json.load(open(args.source_report))
    source_train_sha256 = sha256_file(args.source_training_data)
    if source_report.get("artifacts", {}).get("train", {}).get("sha256") != source_train_sha256:
        raise SystemExit("source S4 report does not bind training data")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    public_paths = list(dict.fromkeys([args.source_training_data, *args.public_data]))
    public = read_public(public_paths)
    public["names"].update(source_nonce_names(public_paths))
    public_multisets = public_entity_multisets(public_paths, tokenizer)
    names = fresh_paired_names(tokenizer, 1000, args.seed ^ 0x434F4E46, public["names"])
    factors = select_factors(
        "development_compositional",
        args.groups * 2,
        args.seed ^ 0x49524D21,
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
    report["schema"] = "r12_s4_hard_island_confirmation_report_v1"
    report["board_role"] = "confirmation"
    report["row_split_label"] = "s4_event_tape_development"
    report["qualification_sha256"] = sha256_file(args.qualification)
    report["gates"]["development_qualified_before_confirmation"] = True
    report["all_gates_pass"] = all(report["gates"].values())
    report["seed"] = int(args.seed)
    report["builder_sha256"] = sha256_file(__file__)
    output.mkdir(parents=True, exist_ok=True)
    receipt = write_jsonl(data_path, rows)
    report["artifacts"] = {
        "confirmation": receipt,
        "development": dict(receipt),
    }
    report["development_artifact_is_confirmation_compatibility_alias"] = True
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit("S4 v5 confirmation board failed gates")


if __name__ == "__main__":
    main()
