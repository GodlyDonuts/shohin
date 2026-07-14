#!/usr/bin/env python3
"""Build a CRA value-shift probe with answer strings held in training support.

The ordinary value factor shifts both source numbers and target answer strings.
This probe holds the target answer vocabulary fixed while every source number is
outside the CRA training range.  It therefore distinguishes source-state
transport from an inability to emit unseen integer strings.  It is an
evaluation-only CPU artifact: no checkpoint or training data is modified.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from generate_counterfactual_residual_algebra_v1 import (
    TRAIN_PARAPHRASES,
    TRAIN_SOURCES,
    TRAIN_QUERIES,
    factor_audit_sets,
    readout,
    render_bundle,
    sha256_file,
    validate_row,
    write_jsonl,
)


OUT_OF_RANGE = tuple(range(-9, -4)) + tuple(range(5, 10))
EDITABLE = (-8, -7, -6, 6, 7, 8)


def train_answer_support(rows: list[dict]) -> set[str]:
    return {row[key] for row in rows for key in ("response", "counterfactual_response")}


def same_sign_values(value: int) -> tuple[int, ...]:
    return tuple(item for item in OUT_OF_RANGE if (item < 0) == (value < 0))


def make_row(rng: random.Random, split: str, index: int) -> dict:
    """Make a source-OOD row whose two target answers stay in train support."""
    primary = rng.choice(EDITABLE)
    donor_primary = rng.choice(EDITABLE)
    delta, counter_delta = rng.sample((-1, 1), 2)
    base_secondary = rng.choice(OUT_OF_RANGE)
    donor_secondary = rng.choice(same_sign_values(donor_primary))
    while (donor_primary, donor_secondary) == (primary, base_secondary):
        donor_primary = rng.choice(EDITABLE)
        donor_secondary = rng.choice(same_sign_values(donor_primary))
    source_index = index % len(TRAIN_SOURCES)
    paraphrase_index = (index // len(TRAIN_QUERIES)) % len(TRAIN_PARAPHRASES)
    target_primary = donor_primary + delta
    counter_target_primary = donor_primary + counter_delta
    if not all(value in OUT_OF_RANGE for value in (
        primary, primary + delta, primary + counter_delta, donor_primary,
        target_primary, counter_target_primary, base_secondary, donor_secondary,
    )):
        raise AssertionError("support probe left the source-OOD value range")

    def source(template: str, first: int, second: int) -> str:
        return template.format(primary=first, secondary=second)

    # Difference keeps both normal and counterfactual answers within [-4, 4],
    # a strict subset of the frozen train answer vocabulary [-8, 8].
    kind, question = ("difference", TRAIN_QUERIES[2][1])
    row = {
        "schema": "counterfactual_residual_algebra_v1",
        "split": split,
        "episode_id": "{}-{:06d}".format(split, index),
        "base_source": source(TRAIN_SOURCES[source_index], primary, base_secondary),
        "edited_source": source(TRAIN_SOURCES[(source_index + 1) % len(TRAIN_SOURCES)], primary + delta, base_secondary),
        "counterfactual_edited_source": source(TRAIN_SOURCES[(source_index + 1) % len(TRAIN_SOURCES)], primary + counter_delta, base_secondary),
        "donor_source": source(TRAIN_SOURCES[source_index], donor_primary, donor_secondary),
        "paraphrase_base_source": source(TRAIN_PARAPHRASES[paraphrase_index], primary, base_secondary),
        "paraphrase_edited_source": source(TRAIN_PARAPHRASES[(paraphrase_index + 1) % len(TRAIN_PARAPHRASES)], primary + delta, base_secondary),
        "paraphrase_counterfactual_edited_source": source(TRAIN_PARAPHRASES[(paraphrase_index + 1) % len(TRAIN_PARAPHRASES)], primary + counter_delta, base_secondary),
        "paraphrase_donor_source": source(TRAIN_PARAPHRASES[paraphrase_index], donor_primary, donor_secondary),
        "suffix_prompt": "Question: {}\nAnswer:".format(question),
        "response": "answer={}".format(readout(target_primary, donor_secondary, kind)),
        "counterfactual_response": "answer={}".format(readout(counter_target_primary, donor_secondary, kind)),
        "query_kind": kind,
        "delta": delta,
        "counterfactual_delta": counter_delta,
        "state": {
            "base": {"primary": primary, "secondary": base_secondary},
            "donor": {"primary": donor_primary, "secondary": donor_secondary},
            "target": {"primary": target_primary, "secondary": donor_secondary},
            "counterfactual_target": {"primary": counter_target_primary, "secondary": donor_secondary},
        },
        "axes": {
            "language_heldout": False,
            "values_heldout": True,
            "delta_heldout": False,
            "query_heldout": False,
            "answer_support_matched": True,
        },
    }
    validate_row(row)
    return row


def build(count: int, seed: int, split: str = "factor_values_answer_supported") -> list[dict]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng, rows, prompts = random.Random(seed), [], set()
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("could not create enough distinct supported-value rows")
        row = make_row(rng, split, len(rows))
        prompt = render_bundle(row)
        if prompt in prompts:
            continue
        prompts.add(prompt)
        rows.append(row)
    return rows


def source_numbers(row: dict) -> set[int]:
    state = row["state"]
    return {
        state[world][field]
        for world in ("base", "donor", "target", "counterfactual_target")
        for field in ("primary", "secondary")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-train", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026071411)
    args = parser.parse_args()
    train_path, out_path, report_path = map(Path, (args.existing_train, args.out, args.report))
    if not train_path.is_file():
        raise SystemExit("existing train file is missing")
    if any(path.exists() for path in (out_path, report_path)):
        raise SystemExit("output and report paths must be fresh")
    train = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
    if not train:
        raise SystemExit("existing train file is empty")
    for row in train:
        validate_row(row)
    rows = build(args.count, args.seed)
    support = train_answer_support(train)
    if any(row[key] not in support for row in rows for key in ("response", "counterfactual_response")):
        raise SystemExit("supported-value probe introduced an unseen answer string")
    if any(any(value not in OUT_OF_RANGE for value in source_numbers(row)) for row in rows):
        raise SystemExit("supported-value probe leaked an in-range source value")
    audit = factor_audit_sets(train, rows)
    if audit["train_factor_exact_bundle_hits"] or audit["train_factor_exact_state_hits"]:
        raise SystemExit("supported-value factor overlaps training: {}".format(audit))
    write_jsonl(out_path, rows)
    report = {
        "audit": "counterfactual_residual_algebra_v1_supported_values",
        "claim_boundary": "CPU-only evaluation-factor construction; no model or reasoning claim is created.",
        "rows": len(rows),
        "train_sha256": sha256_file(train_path),
        "data_sha256": sha256_file(out_path),
        "train_answer_strings": len(support),
        "factor_answer_strings": len({row[key] for row in rows for key in ("response", "counterfactual_response")}),
        "factor_answer_support_subset": True,
        "source_value_range": list(OUT_OF_RANGE),
        "query_kinds": ["difference"],
        **audit,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
