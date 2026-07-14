#!/usr/bin/env python3
"""Generate FQRB primary-magnitude factor data with a fixed answer alphabet.

The base, edited, and donor *primary* fields are three-digit signed values.
The secondary field remains two-digit so every relation consumer can still
receive a counterfactual answer change. This is a primary-state transport test,
not a claim that every source field has a three-digit distribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

from generate_finite_query_residual_basis_v1 import (
    BOUNDARY_TARGETS,
    QUERY_KINDS,
    TWO_DIGIT_VALUES,
    audit,
    consumer_support,
    label,
    make_row,
    render_bundle,
    source_bundle_key,
)


PRIMARY_MAGNITUDES = tuple(range(-999, -99)) + tuple(range(100, 1000))


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


def choose_counter_target(target: int, donor_primary: int, base_primary: int, donor_secondary: int, kind: str, rng: random.Random) -> tuple[int, int] | None:
    normal = label(target, donor_secondary, kind)
    candidates = []
    for counter_target in BOUNDARY_TARGETS:
        if counter_target == target:
            continue
        counter_delta = counter_target - donor_primary
        if base_primary + counter_delta not in PRIMARY_MAGNITUDES:
            continue
        counter_label = label(counter_target, donor_secondary, kind)
        if counter_label != normal and counter_label in consumer_support()[kind]:
            candidates.append((counter_target, counter_delta))
    return rng.choice(candidates) if candidates else None


def make_group(rng: random.Random, basis_id: str) -> list[dict]:
    for _ in range(100_000):
        target = rng.choice(BOUNDARY_TARGETS)
        donor_primary = rng.choice(PRIMARY_MAGNITUDES)
        normal_delta = target - donor_primary
        base_primary = rng.choice(PRIMARY_MAGNITUDES)
        if base_primary + normal_delta not in PRIMARY_MAGNITUDES:
            continue
        base_secondary = rng.choice(TWO_DIGIT_VALUES)
        relation_mode = rng.choice(("less", "equal", "greater"))
        if relation_mode == "equal":
            donor_secondary = target
        elif relation_mode == "less":
            donor_secondary = rng.choice([value for value in TWO_DIGIT_VALUES if value > target])
        else:
            donor_secondary = rng.choice([value for value in TWO_DIGIT_VALUES if value < target])
        if (base_primary, base_secondary) == (donor_primary, donor_secondary):
            continue
        counters = {kind: choose_counter_target(target, donor_primary, base_primary, donor_secondary, kind, rng) for kind in QUERY_KINDS}
        if any(value is None for value in counters.values()):
            continue
        source_index, paraphrase_index = rng.randrange(2), rng.randrange(2)
        return [
            make_row(
                "factor_magnitude", basis_id, kind, base_primary, base_secondary, donor_primary,
                donor_secondary, normal_delta, int(counters[kind][1]), source_index, paraphrase_index,
                False, True,
            )
            for kind in QUERY_KINDS
        ]
    raise RuntimeError("could not sample a valid FQRB magnitude group")


def build(groups: int, seed: int) -> list[dict]:
    rng, rows, prompts, bundles = random.Random(seed), [], set(), set()
    attempts = 0
    while len(rows) < groups * len(QUERY_KINDS):
        attempts += 1
        if attempts > groups * 100:
            raise RuntimeError("could not make enough unique FQRB magnitude groups")
        basis_id = "factor_magnitude-{:06d}".format(len(rows) // len(QUERY_KINDS))
        group = make_group(rng, basis_id)
        bundle = source_bundle_key(group[0])
        group_prompts = {render_bundle(row) for row in group}
        if bundle in bundles or len(group_prompts) != len(QUERY_KINDS) or prompts & group_prompts:
            continue
        bundles.add(bundle)
        prompts.update(group_prompts)
        rows.extend(group)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--groups", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026071422)
    args = parser.parse_args()
    train_path, out_path, report_path = (Path(path) for path in (args.train, args.out, args.report))
    if args.groups <= 1 or out_path.exists() or report_path.exists():
        raise SystemExit("groups must exceed one and output paths must be fresh")
    train_rows = [json.loads(line) for line in train_path.open() if line.strip()]
    rows = build(args.groups, args.seed)
    report = audit(train_rows, rows)
    supported = set().union(*consumer_support().values())
    answers = {row[key] for row in rows for key in ("response", "counterfactual_response")}
    primary_values = [row["state"][field]["primary"] for row in rows for field in ("base", "donor")]
    edited_primary_values = [row["state"]["base"]["primary"] + row["delta"] for row in rows]
    if report["train_heldout_exact_source_bundle_hits"] or report["bad_heldout_group_cardinality"]:
        raise SystemExit("FQRB magnitude source/group audit failed")
    if not answers <= supported or not all(abs(value) >= 100 for value in primary_values + edited_primary_values):
        raise SystemExit("FQRB magnitude primary/answer audit failed")
    write_jsonl(out_path, rows)
    group_counts = Counter(row["basis_id"] for row in rows)
    report.update({
        "audit": "finite_query_residual_basis_v1_magnitude_factor",
        "claim_boundary": "Three-digit primary-source factor with two-digit secondary fields for relation falsifiability.",
        "train_sha256": sha256_file(train_path), "factor_sha256": sha256_file(out_path),
        "factor_rows": len(rows), "factor_groups": len(group_counts), "query_kinds": list(QUERY_KINDS),
        "answer_labels": sorted(answers), "heldout_axes": ["primary_magnitude"],
        "primary_source_absolute_range": [min(abs(value) for value in primary_values + edited_primary_values), max(abs(value) for value in primary_values + edited_primary_values)],
        "secondary_source_value_range": [min(TWO_DIGIT_VALUES), max(TWO_DIGIT_VALUES)],
        "familiar_wording_13gram_overlap_is_expected": bool(report["train_heldout_13gram_hits"]),
    })
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
