#!/usr/bin/env python3
"""Generate the FQRB source-tuple factor with familiar wording.

The main FQRB held-out suite changes source tuples and language together. This
factor holds the train wording/query family fixed while requiring unseen normal
source bundles. It deliberately reports wording n-gram overlap instead of
pretending a familiar-template factor is a language-OOD split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from generate_finite_query_residual_basis_v1 import (
    QUERY_KINDS,
    TWO_DIGIT_VALUES,
    audit,
    build,
    consumer_support,
    render_bundle,
    source_bundle_key,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise SystemExit("refusing to overwrite {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def choose_unseen_groups(train_rows: list[dict], groups: int, seed: int) -> list[dict]:
    train_bundles = {source_bundle_key(row) for row in train_rows}
    for offset in range(1_000):
        candidate = build(groups, seed + offset, "factor_core", TWO_DIGIT_VALUES, 90)
        candidate_bundles = {source_bundle_key(row) for row in candidate}
        if not (candidate_bundles & train_bundles):
            return candidate
    raise RuntimeError("could not sample a source-disjoint FQRB core factor")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--groups", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026071421)
    args = parser.parse_args()
    train_path, out_path, report_path = (Path(path) for path in (args.train, args.out, args.report))
    if args.groups <= 1 or out_path.exists() or report_path.exists():
        raise SystemExit("groups must exceed one and output paths must be fresh")
    train_rows = load_rows(train_path)
    rows = choose_unseen_groups(train_rows, args.groups, args.seed)
    report = audit(train_rows, rows)
    train_bundles = {source_bundle_key(row) for row in train_rows}
    factor_bundles = {source_bundle_key(row) for row in rows}
    supported = set().union(*consumer_support().values())
    answers = {row[key] for row in rows for key in ("response", "counterfactual_response")}
    if report["train_heldout_exact_source_bundle_hits"] or report["bad_heldout_group_cardinality"]:
        raise SystemExit("FQRB core factor source/group audit failed")
    if not answers <= supported or {row["query_kind"] for row in rows} != set(QUERY_KINDS):
        raise SystemExit("FQRB core factor finite-consumer audit failed")
    write_jsonl(out_path, rows)
    report.update({
        "audit": "finite_query_residual_basis_v1_core_factor",
        "claim_boundary": "Familiar-wording source-tuple factor; it does not test language generalization.",
        "train_sha256": sha256_file(train_path),
        "factor_sha256": sha256_file(out_path),
        "factor_rows": len(rows),
        "factor_groups": len(factor_bundles),
        "query_kinds": list(QUERY_KINDS),
        "answer_labels": sorted(answers),
        "heldout_axes": ["source_tuple"],
        "familiar_wording_13gram_overlap_is_expected": bool(report["train_heldout_13gram_hits"]),
    })
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
