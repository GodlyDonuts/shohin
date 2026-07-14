#!/usr/bin/env python3
"""Generate an audited finite-query residual-basis corpus (FQRB).

FQRB retains CRA's source-free ``donor + edited - base`` tape, but replaces an
open-ended integer readout with independent finite consumers of the same target
state.  This is designed to test numeric source transport without confusing it
with unseen output numeral strings.  Construction is CPU-only and every row is
solver-derived; this script neither loads nor modifies a model.
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

from generate_counterfactual_residual_algebra_v1 import ANCHOR


DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
QUERY_KINDS = ("ones", "tens", "sign", "parity", "relation")
BOUNDARY_TARGETS = (-11, -10, -9, -1, 0, 1, 9, 10, 11)
TWO_DIGIT_VALUES = tuple(range(-99, 100))
THREE_DIGIT_VALUES = tuple(range(-999, 1000))

TRAIN_SOURCES = (
    "The amber account holds {primary}, and the cobalt account holds {secondary}." + ANCHOR,
    "A ledger records amber={primary}; cobalt={secondary}." + ANCHOR,
)
TRAIN_PARAPHRASES = (
    "A clerk counted {primary} amber marks and {secondary} cobalt marks." + ANCHOR,
    "For the two inventory fields, amber is {primary} while cobalt is {secondary}." + ANCHOR,
)
HELDOUT_SOURCES = (
    "The north register contains {primary} credits; the south register contains {secondary} credits." + ANCHOR,
    "An archive lists north at {primary} and south at {secondary}." + ANCHOR,
)
HELDOUT_PARAPHRASES = (
    "Records show {primary} tokens in the northern cache and {secondary} in the southern cache." + ANCHOR,
    "For two vault balances, north={primary} and south={secondary}." + ANCHOR,
)

QUERY_PROMPTS = {
    "ones": "Which named class gives the ones place of amber?",
    "tens": "Which named class gives the tens place of amber?",
    "sign": "Which sign class describes amber?",
    "parity": "Which parity class describes amber?",
    "relation": "Which relation class compares amber with cobalt?",
}
HELDOUT_QUERY_PROMPTS = {
    "ones": "Return the unit-symbol class for north.",
    "tens": "Return the decade-symbol class for north.",
    "sign": "Return the polarity class for north.",
    "parity": "Return the evenness class for north.",
    "relation": "Return the ordering class between north and south.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit("refusing to overwrite {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(partial, path)


def label(primary: int, secondary: int, kind: str) -> str:
    if kind == "ones":
        return "answer=ones_{}".format(DIGITS[abs(primary) % 10])
    if kind == "tens":
        return "answer=tens_{}".format(DIGITS[(abs(primary) // 10) % 10])
    if kind == "sign":
        return "answer={}".format("negative" if primary < 0 else "positive" if primary > 0 else "zero")
    if kind == "parity":
        return "answer={}".format("even" if primary % 2 == 0 else "odd")
    if kind == "relation":
        return "answer=relation_{}".format("less" if primary < secondary else "greater" if primary > secondary else "equal")
    raise ValueError("unknown FQRB query kind: {}".format(kind))


def consumer_support() -> dict[str, set[str]]:
    """Fixed answer alphabet deliberately covered by normal training targets."""
    return {
        "ones": {"answer=ones_zero", "answer=ones_one", "answer=ones_nine"},
        "tens": {"answer=tens_zero", "answer=tens_one"},
        "sign": {"answer=negative", "answer=zero", "answer=positive"},
        "parity": {"answer=even", "answer=odd"},
        "relation": {"answer=relation_less", "answer=relation_equal", "answer=relation_greater"},
    }


def source(template: str, primary: int, secondary: int) -> str:
    return template.format(primary=primary, secondary=secondary)


def validate_fqrb_row(row: dict) -> None:
    """Validate FQRB semantics without inheriting CRA's numeric readout names."""
    if row.get("schema") != "counterfactual_residual_algebra_v1":
        raise ValueError("invalid FQRB schema")
    if row.get("query_kind") not in QUERY_KINDS:
        raise ValueError("unknown FQRB query kind")
    state = row.get("state", {})
    base, donor = state.get("base", {}), state.get("donor", {})
    target, counter = state.get("target", {}), state.get("counterfactual_target", {})
    if target.get("primary") != donor.get("primary") + row.get("delta"):
        raise ValueError("normal target does not implement residual edit")
    if counter.get("primary") != donor.get("primary") + row.get("counterfactual_delta"):
        raise ValueError("counterfactual target does not implement residual edit")
    kind = row["query_kind"]
    expected, counter_expected = label(target["primary"], target["secondary"], kind), label(counter["primary"], counter["secondary"], kind)
    if row.get("response") != expected or row.get("counterfactual_response") != counter_expected:
        raise ValueError("FQRB response is not solver-derived")
    if expected == counter_expected:
        raise ValueError("FQRB counterfactual must change the supervised answer")
    source_values = set(re.findall(r"(?<!\d)-?\d+(?!\d)", "\n".join(
        row[key] for key in ("base_source", "edited_source", "donor_source")
    )))
    suffix_values = set(re.findall(r"(?<!\d)-?\d+(?!\d)", row["suffix_prompt"]))
    if source_values & suffix_values:
        raise ValueError("FQRB suffix leaked a source number")
    if len({row[key] for key in ("base_source", "edited_source", "donor_source")}) != 3:
        raise ValueError("FQRB source worlds must be textually distinct")
    if base.get("primary") is None or base.get("secondary") is None:
        raise ValueError("FQRB base state is incomplete")


def render_bundle(row: dict, paraphrase: bool = False, counterfactual: bool = False) -> str:
    prefix = "paraphrase_" if paraphrase else ""
    edited = "counterfactual_edited_source" if counterfactual else "edited_source"
    return "Base world:\n{}\nEdited world:\n{}\nDonor world:\n{}\n{}".format(
        row[prefix + "base_source"], row[prefix + edited], row[prefix + "donor_source"], row["suffix_prompt"],
    )


def templates(language_heldout: bool) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (HELDOUT_SOURCES, HELDOUT_PARAPHRASES) if language_heldout else (TRAIN_SOURCES, TRAIN_PARAPHRASES)


def allowed_deltas(primary: int, donor_primary: int, values: tuple[int, ...], delta_bound: int) -> list[int]:
    membership = set(values)
    return [delta for delta in range(-delta_bound, delta_bound + 1) if delta and primary + delta in membership and donor_primary + delta in membership]


def choose_counter_delta(
    primary: int,
    donor_primary: int,
    secondary: int,
    normal_delta: int,
    kind: str,
    values: tuple[int, ...],
    delta_bound: int,
    rng: random.Random,
) -> int | None:
    normal = label(donor_primary + normal_delta, secondary, kind)
    candidates = [
        delta for delta in allowed_deltas(primary, donor_primary, values, delta_bound)
        if delta != normal_delta
        and label(donor_primary + delta, secondary, kind) != normal
        and label(donor_primary + delta, secondary, kind) in consumer_support()[kind]
    ]
    return rng.choice(candidates) if candidates else None


def make_row(
    split: str,
    basis_id: str,
    query_kind: str,
    primary: int,
    base_secondary: int,
    donor_primary: int,
    donor_secondary: int,
    normal_delta: int,
    counter_delta: int,
    source_index: int,
    paraphrase_index: int,
    language_heldout: bool,
    magnitude_heldout: bool,
) -> dict:
    sources, paraphrases = templates(language_heldout)
    target, counter_target = donor_primary + normal_delta, donor_primary + counter_delta
    row = {
        # The existing CRA trainer/evaluator is deliberately reused: no new model
        # parameters, controller, parser, or residual objective is introduced.
        "schema": "counterfactual_residual_algebra_v1",
        "split": split,
        "episode_id": "{}:{}".format(basis_id, query_kind),
        "basis_id": basis_id,
        "basis_mode": "multi_consumer",
        "base_source": source(sources[source_index % len(sources)], primary, base_secondary),
        "edited_source": source(sources[(source_index + 1) % len(sources)], primary + normal_delta, base_secondary),
        "counterfactual_edited_source": source(sources[(source_index + 1) % len(sources)], primary + counter_delta, base_secondary),
        "donor_source": source(sources[source_index % len(sources)], donor_primary, donor_secondary),
        "paraphrase_base_source": source(paraphrases[paraphrase_index % len(paraphrases)], primary, base_secondary),
        "paraphrase_edited_source": source(paraphrases[(paraphrase_index + 1) % len(paraphrases)], primary + normal_delta, base_secondary),
        "paraphrase_counterfactual_edited_source": source(paraphrases[(paraphrase_index + 1) % len(paraphrases)], primary + counter_delta, base_secondary),
        "paraphrase_donor_source": source(paraphrases[paraphrase_index % len(paraphrases)], donor_primary, donor_secondary),
        "suffix_prompt": "Question: {}\nAnswer:".format(
            (HELDOUT_QUERY_PROMPTS if language_heldout else QUERY_PROMPTS)[query_kind]
        ),
        "response": label(target, donor_secondary, query_kind),
        "counterfactual_response": label(counter_target, donor_secondary, query_kind),
        "query_kind": query_kind,
        "delta": normal_delta,
        "counterfactual_delta": counter_delta,
        "state": {
            "base": {"primary": primary, "secondary": base_secondary},
            "donor": {"primary": donor_primary, "secondary": donor_secondary},
            "target": {"primary": target, "secondary": donor_secondary},
            "counterfactual_target": {"primary": counter_target, "secondary": donor_secondary},
        },
        "axes": {
            "language_heldout": language_heldout,
            "values_heldout": magnitude_heldout,
            "delta_heldout": False,
            "query_heldout": False,
            "finite_query_basis": True,
        },
    }
    validate_fqrb_row(row)
    if row["response"] == row["counterfactual_response"]:
        raise AssertionError("counterfactual did not change FQRB consumer answer")
    return row


def make_multi_group(
    rng: random.Random,
    split: str,
    basis_id: str,
    values: tuple[int, ...],
    delta_bound: int,
    language_heldout: bool,
    magnitude_heldout: bool,
) -> list[dict]:
    """One source triple, five independent consumers, five distinct counteredits."""
    membership = set(values)
    target_options = [item for item in BOUNDARY_TARGETS if item in membership]
    for _ in range(20_000):
        target = rng.choice(target_options)
        donor_primary = rng.choice(values)
        normal_delta = target - donor_primary
        if not normal_delta or abs(normal_delta) > delta_bound:
            continue
        primary = rng.choice(values)
        base_secondary = rng.choice(values)
        relation_mode = rng.choice(("less", "equal", "greater"))
        if relation_mode == "equal":
            donor_secondary = target
        elif relation_mode == "less":
            donor_secondary = rng.choice([value for value in values if value > target])
        else:
            donor_secondary = rng.choice([value for value in values if value < target])
        if (primary, base_secondary) == (donor_primary, donor_secondary):
            continue
        candidates = allowed_deltas(primary, donor_primary, values, delta_bound)
        if normal_delta not in candidates:
            continue
        counters = {
            kind: choose_counter_delta(primary, donor_primary, donor_secondary, normal_delta, kind, values, delta_bound, rng)
            for kind in QUERY_KINDS
        }
        if any(value is None for value in counters.values()):
            continue
        source_index, paraphrase_index = rng.randrange(2), rng.randrange(2)
        return [
            make_row(
                split, basis_id, kind, primary, base_secondary, donor_primary, donor_secondary,
                normal_delta, int(counters[kind]), source_index, paraphrase_index,
                language_heldout, magnitude_heldout,
            )
            for kind in QUERY_KINDS
        ]
    raise RuntimeError("could not sample a valid multi-consumer FQRB group")


def build(groups: int, seed: int, split: str, values: tuple[int, ...], delta_bound: int, language_heldout: bool = False, magnitude_heldout: bool = False) -> list[dict]:
    if groups <= 0:
        raise ValueError("groups must be positive")
    rng, rows, prompts, state_keys = random.Random(seed), [], set(), set()
    attempts = 0
    while len(rows) < groups * len(QUERY_KINDS):
        attempts += 1
        if attempts > groups * 100:
            raise RuntimeError("could not make enough unique FQRB groups")
        basis_id = "{}-{:06d}".format(split, len(rows) // len(QUERY_KINDS))
        group = make_multi_group(rng, split, basis_id, values, delta_bound, language_heldout, magnitude_heldout)
        state = group[0]["state"]
        state_key = (
            state["base"]["primary"], state["base"]["secondary"], state["donor"]["primary"],
            state["donor"]["secondary"], group[0]["delta"],
        )
        group_prompts = {render_bundle(row) for row in group}
        if state_key in state_keys or len(group_prompts) != len(QUERY_KINDS) or prompts & group_prompts:
            continue
        state_keys.add(state_key)
        prompts.update(group_prompts)
        rows.extend(group)
    return rows


def ngrams(text: str, width: int = 13) -> set[tuple[str, ...]]:
    words = text.split()
    return {tuple(words[index:index + width]) for index in range(max(0, len(words) - width + 1))}


def source_bundle_key(row: dict) -> tuple[int, int, int, int, int]:
    """The full normal source triple, independent of which suffix consumes it."""
    state = row["state"]
    return (
        state["base"]["primary"], state["base"]["secondary"], state["donor"]["primary"],
        state["donor"]["secondary"], row["delta"],
    )


def audit(train: list[dict], heldout: list[dict]) -> dict:
    train_prompts = {render_bundle(row) for row in train}
    held_prompts = {render_bundle(row) for row in heldout}
    train_grams = set().union(*(ngrams(prompt) for prompt in train_prompts))
    held_grams = set().union(*(ngrams(prompt) for prompt in held_prompts))
    train_source_bundles = {source_bundle_key(row) for row in train}
    held_source_bundles = {source_bundle_key(row) for row in heldout}
    train_groups = Counter(row["basis_id"] for row in train)
    held_groups = Counter(row["basis_id"] for row in heldout)
    return {
        "duplicate_train_prompts": len(train) - len(train_prompts),
        "duplicate_heldout_prompts": len(heldout) - len(held_prompts),
        "train_heldout_exact_prompt_hits": len(train_prompts & held_prompts),
        "train_heldout_13gram_hits": len(train_grams & held_grams),
        "train_heldout_exact_source_bundle_hits": len(train_source_bundles & held_source_bundles),
        "train_groups": len(train_groups),
        "heldout_groups": len(held_groups),
        "bad_train_group_cardinality": sum(count != len(QUERY_KINDS) for count in train_groups.values()),
        "bad_heldout_group_cardinality": sum(count != len(QUERY_KINDS) for count in held_groups.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-groups", type=int, default=12_000)
    parser.add_argument("--heldout-groups", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026071412)
    args = parser.parse_args()
    paths = tuple(Path(path) for path in (args.train_out, args.heldout_out, args.report))
    if args.train_groups <= 0 or args.heldout_groups <= 0:
        raise SystemExit("group counts must be positive")
    if any(path.exists() for path in paths):
        raise SystemExit("all output paths must be fresh")
    train = build(args.train_groups, args.seed, "train", TWO_DIGIT_VALUES, 90)
    heldout = build(args.heldout_groups, args.seed + 1, "heldout", TWO_DIGIT_VALUES, 90, language_heldout=True)
    report = audit(train, heldout)
    required_zero = (
        "duplicate_train_prompts", "duplicate_heldout_prompts", "train_heldout_exact_prompt_hits",
        "train_heldout_13gram_hits", "train_heldout_exact_source_bundle_hits", "bad_train_group_cardinality",
        "bad_heldout_group_cardinality",
    )
    if any(report[key] for key in required_zero):
        raise SystemExit("FQRB split audit failed: {}".format(report))
    supported = set().union(*consumer_support().values())
    train_answers = {row["response"] for row in train}
    held_answers = {row[key] for row in heldout for key in ("response", "counterfactual_response")}
    if train_answers != supported or not held_answers <= train_answers:
        raise SystemExit("FQRB answer-support audit failed")
    write_jsonl(paths[0], train)
    write_jsonl(paths[1], heldout)
    report.update({
        # The generic CRA trainer requires this exact audit name and fields. The
        # mechanism field preserves the stronger FQRB interpretation contract.
        "audit": "counterfactual_residual_algebra_v1",
        "mechanism": "finite_query_residual_basis_v1",
        "claim_boundary": "CPU-only FQRB data admission; no model or reasoning result is created.",
        "train_rows": len(train),
        "heldout_rows": len(heldout),
        "train_sha256": sha256_file(paths[0]),
        "heldout_sha256": sha256_file(paths[1]),
        "query_kinds": list(QUERY_KINDS),
        "answer_labels": sorted(train_answers),
        "normal_source_value_range": [min(TWO_DIGIT_VALUES), max(TWO_DIGIT_VALUES)],
        "heldout_source_value_range": [min(TWO_DIGIT_VALUES), max(TWO_DIGIT_VALUES)],
        "combined_heldout_axes": ["source_tuple", "language"],
    })
    paths[2].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
