#!/usr/bin/env python3
"""Build an unseen two-edit FQRB composition factor.

The FQRB training corpus contains only one residual edit. This held-out factor
requires ``Z(donor) + Z(primary_edit) + Z(secondary_edit) - 2*Z(base)`` while
retaining the same five finite consumers and answer alphabet. It is evaluation
only: no two-edit row is admitted to FQRB training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from generate_finite_query_residual_basis_v1 import (
    ANCHOR,
    BOUNDARY_TARGETS,
    QUERY_KINDS,
    TRAIN_PARAPHRASES,
    TRAIN_SOURCES,
    TWO_DIGIT_VALUES,
    consumer_support,
    label,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source(template: str, primary: int, secondary: int) -> str:
    return template.format(primary=primary, secondary=secondary)


def render_bundle(row: dict) -> str:
    return "Base world:\n{}\nPrimary edit:\n{}\nSecondary edit:\n{}\nDonor world:\n{}\n{}".format(
        row["base_source"], row["primary_edited_source"], row["secondary_edited_source"],
        row["donor_source"], row["suffix_prompt"],
    )


def render_one_edit_bundle(row: dict) -> str:
    return "Base world:\n{}\nEdited world:\n{}\nDonor world:\n{}\n{}".format(
        row["base_source"], row["edited_source"], row["donor_source"], row["suffix_prompt"],
    )


def ngrams(text: str, width: int = 13) -> set[tuple[str, ...]]:
    words = text.split()
    return {tuple(words[index:index + width]) for index in range(max(0, len(words) - width + 1))}


def source_bundle_key(row: dict) -> tuple[int, int, int, int, int, int]:
    state = row["state"]
    return (
        state["base"]["primary"], state["base"]["secondary"], state["donor"]["primary"],
        state["donor"]["secondary"], row["primary_delta"], row["secondary_delta"],
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise SystemExit("refusing to overwrite {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def choose_counter_delta(target: int, donor_primary: int, base_primary: int, secondary_delta: int, donor_secondary: int, kind: str, rng: random.Random) -> int | None:
    normal = label(target, donor_secondary, kind)
    candidates = []
    for counter_primary_delta in tuple(range(-15, 0)) + tuple(range(1, 16)):
        counter_target = donor_primary + counter_primary_delta + secondary_delta
        if counter_target not in BOUNDARY_TARGETS or base_primary + counter_primary_delta not in TWO_DIGIT_VALUES:
            continue
        candidate = label(counter_target, donor_secondary, kind)
        if candidate != normal and candidate in consumer_support()[kind]:
            candidates.append(counter_primary_delta)
    return rng.choice(candidates) if candidates else None


def make_group(rng: random.Random, basis_id: str) -> list[dict]:
    for _ in range(100_000):
        target = rng.choice(BOUNDARY_TARGETS)
        primary_delta = rng.choice(tuple(range(-15, 0)) + tuple(range(1, 16)))
        secondary_delta = rng.choice(tuple(range(-15, 0)) + tuple(range(1, 16)))
        donor_primary = target - primary_delta - secondary_delta
        if donor_primary not in TWO_DIGIT_VALUES:
            continue
        base_primary = rng.choice(TWO_DIGIT_VALUES)
        if base_primary + primary_delta not in TWO_DIGIT_VALUES or base_primary + secondary_delta not in TWO_DIGIT_VALUES:
            continue
        base_secondary = rng.choice(TWO_DIGIT_VALUES)
        relation_mode = rng.choice(("less", "equal", "greater"))
        if relation_mode == "equal":
            donor_secondary = target
        elif relation_mode == "less":
            donor_secondary = rng.choice([value for value in TWO_DIGIT_VALUES if value > target])
        else:
            donor_secondary = rng.choice([value for value in TWO_DIGIT_VALUES if value < target])
        counters = {
            kind: choose_counter_delta(target, donor_primary, base_primary, secondary_delta, donor_secondary, kind, rng)
            for kind in QUERY_KINDS
        }
        if any(value is None for value in counters.values()):
            continue
        source_index, paraphrase_index = rng.randrange(2), rng.randrange(2)
        rows = []
        for kind in QUERY_KINDS:
            counter_primary_delta = int(counters[kind])
            counter_target = donor_primary + counter_primary_delta + secondary_delta
            row = {
                "schema": "counterfactual_residual_algebra_v1", "mode": "two_edit", "split": "two_edit",
                "basis_mode": "multi_consumer_two_edit", "basis_id": basis_id,
                "episode_id": "{}:{}".format(basis_id, kind), "query_kind": kind,
                "base_source": source(TRAIN_SOURCES[source_index], base_primary, base_secondary),
                "primary_edited_source": source(TRAIN_SOURCES[(source_index + 1) % 2], base_primary + primary_delta, base_secondary),
                "counterfactual_primary_edited_source": source(TRAIN_SOURCES[(source_index + 1) % 2], base_primary + counter_primary_delta, base_secondary),
                "secondary_edited_source": source(TRAIN_SOURCES[source_index], base_primary + secondary_delta, base_secondary),
                "donor_source": source(TRAIN_SOURCES[(source_index + 1) % 2], donor_primary, donor_secondary),
                "paraphrase_base_source": source(TRAIN_PARAPHRASES[paraphrase_index], base_primary, base_secondary),
                "paraphrase_primary_edited_source": source(TRAIN_PARAPHRASES[(paraphrase_index + 1) % 2], base_primary + primary_delta, base_secondary),
                "paraphrase_counterfactual_primary_edited_source": source(TRAIN_PARAPHRASES[(paraphrase_index + 1) % 2], base_primary + counter_primary_delta, base_secondary),
                "paraphrase_secondary_edited_source": source(TRAIN_PARAPHRASES[paraphrase_index], base_primary + secondary_delta, base_secondary),
                "paraphrase_donor_source": source(TRAIN_PARAPHRASES[(paraphrase_index + 1) % 2], donor_primary, donor_secondary),
                "suffix_prompt": "Question: {}\nAnswer:".format({
                    "ones": "Which named class gives the ones place of amber?",
                    "tens": "Which named class gives the tens place of amber?",
                    "sign": "Which sign class describes amber?",
                    "parity": "Which parity class describes amber?",
                    "relation": "Which relation class compares amber with cobalt?",
                }[kind]),
                "response": label(target, donor_secondary, kind),
                "counterfactual_response": label(counter_target, donor_secondary, kind),
                "primary_delta": primary_delta, "secondary_delta": secondary_delta,
                "counterfactual_primary_delta": counter_primary_delta,
                "state": {
                    "base": {"primary": base_primary, "secondary": base_secondary},
                    "donor": {"primary": donor_primary, "secondary": donor_secondary},
                    "target": {"primary": target, "secondary": donor_secondary},
                    "counterfactual_target": {"primary": counter_target, "secondary": donor_secondary},
                },
                "axes": {"two_edit_composition": True, "finite_query_basis": True, "language_heldout": False},
            }
            validate_row(row)
            rows.append(row)
        return rows
    raise RuntimeError("could not sample a valid two-edit FQRB group")


def validate_row(row: dict) -> None:
    state = row["state"]
    target, counter, donor = state["target"], state["counterfactual_target"], state["donor"]
    if target["primary"] != donor["primary"] + row["primary_delta"] + row["secondary_delta"]:
        raise ValueError("two-edit normal target is not solver-derived")
    if counter["primary"] != donor["primary"] + row["counterfactual_primary_delta"] + row["secondary_delta"]:
        raise ValueError("two-edit counterfactual target is not solver-derived")
    if row["response"] != label(target["primary"], target["secondary"], row["query_kind"]):
        raise ValueError("two-edit normal label is invalid")
    if row["counterfactual_response"] != label(counter["primary"], counter["secondary"], row["query_kind"]):
        raise ValueError("two-edit counterfactual label is invalid")
    if row["response"] == row["counterfactual_response"]:
        raise ValueError("two-edit counterfactual must change its consumer answer")
    source_numbers = set(re.findall(r"(?<!\d)-?\d+(?!\d)", "\n".join(
        row[field] for field in ("base_source", "primary_edited_source", "secondary_edited_source", "donor_source")
    )))
    if source_numbers & set(re.findall(r"(?<!\d)-?\d+(?!\d)", row["suffix_prompt"])):
        raise ValueError("suffix leaked a two-edit source number")
    if not all(text.endswith(ANCHOR) for text in (
        row["base_source"], row["primary_edited_source"], row["secondary_edited_source"], row["donor_source"],
    )):
        raise ValueError("two-edit source lost its native anchor")


def build(groups: int, seed: int) -> list[dict]:
    rng, rows, prompts, bundles = random.Random(seed), [], set(), set()
    attempts = 0
    while len(rows) < groups * len(QUERY_KINDS):
        attempts += 1
        if attempts > groups * 100:
            raise RuntimeError("could not build enough two-edit FQRB groups")
        group = make_group(rng, "two-edit-{:06d}".format(len(rows) // len(QUERY_KINDS)))
        bundle, group_prompts = source_bundle_key(group[0]), {render_bundle(row) for row in group}
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
    parser.add_argument("--seed", type=int, default=2026071423)
    args = parser.parse_args()
    train_path, out_path, report_path = (Path(path) for path in (args.train, args.out, args.report))
    if args.groups <= 1 or out_path.exists() or report_path.exists():
        raise SystemExit("groups must exceed one and output paths must be fresh")
    train = [json.loads(line) for line in train_path.open() if line.strip()]
    rows = build(args.groups, args.seed)
    prompts = {render_bundle(row) for row in rows}
    labels = {row[key] for row in rows for key in ("response", "counterfactual_response")}
    groups = Counter(row["basis_id"] for row in rows)
    train_prompts = {render_one_edit_bundle(row) for row in train}
    train_grams = set().union(*(ngrams(prompt) for prompt in train_prompts))
    factor_grams = set().union(*(ngrams(prompt) for prompt in prompts))
    report = {
        "audit": "finite_query_residual_basis_v1_two_edit_factor", "train_sha256": sha256_file(train_path),
        "factor_rows": len(rows), "factor_groups": len(groups), "factor_sha256": None,
        "duplicate_factor_prompts": len(rows) - len(prompts),
        "bad_group_cardinality": sum(value != len(QUERY_KINDS) for value in groups.values()),
        "train_exact_prompt_hits": len(train_prompts & prompts),
        "train_surface_13gram_hits": len(train_grams & factor_grams),
        "train_surface_13gram_overlap_expected": bool(train_grams & factor_grams),
        "query_kinds": list(QUERY_KINDS), "answer_labels": sorted(labels),
        "claim_boundary": "Unseen two-edit source-free composition factor; it is never FQRB training data.",
    }
    if report["duplicate_factor_prompts"] or report["bad_group_cardinality"] or report["train_exact_prompt_hits"] or not labels <= set().union(*consumer_support().values()):
        raise SystemExit("two-edit FQRB factor audit failed")
    write_jsonl(out_path, rows)
    report["factor_sha256"] = sha256_file(out_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
