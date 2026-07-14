#!/usr/bin/env python3
"""Build an audited source-free corpus for Counterfactual Residual Algebra.

Each row supplies three natural-language worlds.  A trainer encodes their
native residual tapes and must answer from ``Z(donor) + Z(edited) - Z(base)``.
The suffix has no source values, source tokens, or source cache.  Construction
is CPU-only and solver-verifies every target.
"""
import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path


ANCHOR = "\nEnd state record:"
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

TRAIN_QUERIES = (
    ("primary", "What is the amber value?"),
    ("sum", "What is amber plus cobalt?"),
    ("difference", "What is amber minus cobalt?"),
)
HELDOUT_QUERIES = (
    ("primary", "Report the north quantity."),
    ("sum", "Give the total over north and south."),
    ("difference", "Give north less south."),
)
QUERY_HELDOUT_QUERIES = (
    ("primary", "State the amber figure in the account."),
    ("sum", "How many marks are in the two accounts altogether?"),
    ("difference", "By how much does amber exceed or trail cobalt?"),
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path, rows):
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit("refusing to overwrite {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(str(partial), str(path))


def readout(primary, secondary, kind):
    if kind == "primary":
        return primary
    if kind == "secondary":
        return secondary
    if kind == "sum":
        return primary + secondary
    if kind == "difference":
        return primary - secondary
    raise ValueError("unknown query kind: {}".format(kind))


def _range_for(values_heldout):
    if not values_heldout:
        return tuple(range(-4, 5))
    return tuple(range(-9, -4)) + tuple(range(5, 10))


def _valid_edits(values, deltas):
    return [(value, delta) for value in values for delta in deltas if value + delta in values]


def _choose_worlds(rng, values, deltas):
    candidates = [value for value in values if sum(value + delta in values for delta in deltas) >= 2]
    if not candidates:
        raise ValueError("value and delta ranges have no paired counterfactual edits")
    primary = rng.choice(candidates)
    valid_deltas = [delta for delta in deltas if primary + delta in values]
    delta, counter_delta = rng.sample(valid_deltas, 2)
    donor_primary = rng.choice([item for item in candidates if item + delta in values and item + counter_delta in values])
    base_secondary = rng.choice(values)
    donor_secondary = rng.choice(values)
    while (donor_primary, donor_secondary) == (primary, base_secondary):
        donor_primary = rng.choice([item for item in candidates if item + delta in values and item + counter_delta in values])
        donor_secondary = rng.choice(values)
    return primary, base_secondary, donor_primary, donor_secondary, delta, counter_delta


def _templates(language_heldout, query_heldout=False):
    if language_heldout:
        sources, paraphrases, queries = HELDOUT_SOURCES, HELDOUT_PARAPHRASES, HELDOUT_QUERIES
    else:
        sources, paraphrases, queries = TRAIN_SOURCES, TRAIN_PARAPHRASES, TRAIN_QUERIES
    if query_heldout:
        queries = QUERY_HELDOUT_QUERIES
    return sources, paraphrases, queries


def render_bundle(row, paraphrase=False, counterfactual=False):
    prefix = "paraphrase_" if paraphrase else ""
    edited_key = "counterfactual_edited_source" if counterfactual else "edited_source"
    return "Base world:\n{}\nEdited world:\n{}\nDonor world:\n{}\n{}".format(
        row[prefix + "base_source"], row[prefix + edited_key], row[prefix + "donor_source"], row["suffix_prompt"],
    )


def make_row(rng, split, index, language_heldout=False, values_heldout=False, delta_heldout=False, query_heldout=False):
    values = _range_for(values_heldout)
    deltas = (-2, 2) if delta_heldout else (-1, 1)
    primary, base_secondary, donor_primary, donor_secondary, delta, counter_delta = _choose_worlds(rng, values, deltas)
    sources, paraphrases, queries = _templates(language_heldout, query_heldout)
    source_index = index % len(sources)
    paraphrase_index = (index // len(queries)) % len(paraphrases)
    kind, question = queries[index % len(queries)]

    def source(template, first, second):
        return template.format(primary=first, secondary=second)

    target_primary, counter_target_primary = donor_primary + delta, donor_primary + counter_delta
    row = {
        "schema": "counterfactual_residual_algebra_v1",
        "split": split,
        "episode_id": "{}-{:06d}".format(split, index),
        "base_source": source(sources[source_index], primary, base_secondary),
        "edited_source": source(sources[(source_index + 1) % len(sources)], primary + delta, base_secondary),
        "counterfactual_edited_source": source(sources[(source_index + 1) % len(sources)], primary + counter_delta, base_secondary),
        "donor_source": source(sources[source_index], donor_primary, donor_secondary),
        "paraphrase_base_source": source(paraphrases[paraphrase_index], primary, base_secondary),
        "paraphrase_edited_source": source(paraphrases[(paraphrase_index + 1) % len(paraphrases)], primary + delta, base_secondary),
        "paraphrase_counterfactual_edited_source": source(paraphrases[(paraphrase_index + 1) % len(paraphrases)], primary + counter_delta, base_secondary),
        "paraphrase_donor_source": source(paraphrases[paraphrase_index], donor_primary, donor_secondary),
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
        "axes": {"language_heldout": language_heldout, "values_heldout": values_heldout,
                 "delta_heldout": delta_heldout, "query_heldout": query_heldout},
    }
    validate_row(row)
    return row


def validate_row(row):
    if row.get("schema") != "counterfactual_residual_algebra_v1":
        raise ValueError("invalid schema")
    state = row.get("state", {})
    donor, target, counter = state.get("donor", {}), state.get("target", {}), state.get("counterfactual_target", {})
    if target.get("primary") != donor.get("primary") + row.get("delta"):
        raise ValueError("normal target does not implement residual edit")
    if counter.get("primary") != donor.get("primary") + row.get("counterfactual_delta"):
        raise ValueError("counterfactual target does not implement residual edit")
    expected = "answer={}".format(readout(target["primary"], target["secondary"], row["query_kind"]))
    counter_expected = "answer={}".format(readout(counter["primary"], counter["secondary"], row["query_kind"]))
    if row.get("response") != expected or row.get("counterfactual_response") != counter_expected:
        raise ValueError("response is not solver-verified")
    if expected == counter_expected:
        raise ValueError("counterfactual must change the supervised answer")
    source_numbers = set(re.findall(r"(?<!\d)-?\d+(?!\d)", "\n".join(
        row[key] for key in ("base_source", "edited_source", "donor_source")
    )))
    suffix_numbers = set(re.findall(r"(?<!\d)-?\d+(?!\d)", row["suffix_prompt"]))
    if source_numbers & suffix_numbers:
        raise ValueError("suffix leaked a source number")
    if len({row[key] for key in ("base_source", "edited_source", "donor_source")}) != 3:
        raise ValueError("source worlds must be textually distinct")


def ngrams(text, width=13):
    words = text.split()
    return {tuple(words[index:index + width]) for index in range(max(0, len(words) - width + 1))}


def build(split, count, seed, language_heldout=False, values_heldout=False, delta_heldout=False, query_heldout=False):
    rng, rows, prompts = random.Random(seed), [], set()
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("could not create enough unique CRA rows")
        row = make_row(rng, split, len(rows), language_heldout, values_heldout, delta_heldout, query_heldout)
        prompt = render_bundle(row)
        if prompt in prompts:
            continue
        prompts.add(prompt)
        rows.append(row)
    return rows


def audit_sets(train, heldout):
    train_prompts = {render_bundle(row) for row in train}
    heldout_prompts = {render_bundle(row) for row in heldout}
    train_grams = set().union(*(ngrams(prompt) for prompt in train_prompts))
    heldout_grams = set().union(*(ngrams(prompt) for prompt in heldout_prompts))
    return {
        "duplicate_train_prompts": len(train) - len(train_prompts),
        "duplicate_heldout_prompts": len(heldout) - len(heldout_prompts),
        "train_heldout_exact_prompt_hits": len(train_prompts & heldout_prompts),
        "train_heldout_13gram_hits": len(train_grams & heldout_grams),
    }


def factor_audit_sets(train, factor):
    """Audit controlled factors without confusing shared scaffolding for leakage.

    A delta-only or value-only factor intentionally retains the train wording.
    It must have no exact three-source bundle or exact latent-world transition
    in training; shared fixed wording is reported, not silently called a hard
    decontamination pass.
    """
    def state_key(row):
        state = row["state"]
        return (
            state["base"]["primary"], state["base"]["secondary"],
            state["donor"]["primary"], state["donor"]["secondary"],
            row["delta"], row["counterfactual_delta"], row["query_kind"],
        )
    train_prompts = {render_bundle(row) for row in train}
    factor_prompts = {render_bundle(row) for row in factor}
    train_grams = set().union(*(ngrams(prompt) for prompt in train_prompts))
    factor_grams = set().union(*(ngrams(prompt) for prompt in factor_prompts))
    train_states = {state_key(row) for row in train}
    factor_states = {state_key(row) for row in factor}
    return {
        "train_factor_exact_bundle_hits": len(train_prompts & factor_prompts),
        "train_factor_exact_state_hits": len(train_states & factor_states),
        "train_factor_surface_13gram_hits": len(train_grams & factor_grams),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="")
    parser.add_argument("--heldout-out", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--existing-train", default="", help="frozen train JSONL used only to build factor sets")
    parser.add_argument("--train-count", type=int, default=30000)
    parser.add_argument("--heldout-count", type=int, default=2000)
    parser.add_argument("--factor-dir", default="", help="write fresh single-axis factor sets under this directory")
    parser.add_argument("--factor-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    paths = tuple(Path(path) for path in (args.train_out, args.heldout_out, args.report) if path)
    if args.train_count <= 0 or args.heldout_count <= 0 or args.factor_count <= 0:
        raise SystemExit("counts must be positive")
    if args.existing_train:
        if not args.factor_dir or any((args.train_out, args.heldout_out, args.report)):
            raise SystemExit("--existing-train requires --factor-dir and no primary output paths")
        if not Path(args.existing_train).is_file():
            raise SystemExit("existing train file is missing")
    elif len(paths) != 3 or any(path.exists() for path in paths):
        raise SystemExit("primary output paths are required and must be fresh")
    factor_dir = Path(args.factor_dir) if args.factor_dir else None
    if factor_dir is not None and factor_dir.exists():
        raise SystemExit("factor directory must be fresh")
    if args.existing_train:
        train = [json.loads(line) for line in open(args.existing_train) if line.strip()]
        if not train or any(row.get("schema") != "counterfactual_residual_algebra_v1" or row.get("split") != "train" for row in train):
            raise SystemExit("existing train file has an invalid CRA schema/split")
        for row in train:
            validate_row(row)
        report = {"audit": "counterfactual_residual_algebra_v1", "train_rows": len(train), "train_sha256": sha256_file(Path(args.existing_train))}
    else:
        train = build("train", args.train_count, args.seed)
        heldout = build("heldout", args.heldout_count, args.seed + 1, True, True, True)
        audit = audit_sets(train, heldout)
        if any(audit.values()):
            raise SystemExit("cross-split audit failed: {}".format(audit))
        write_jsonl(paths[0], train)
        write_jsonl(paths[1], heldout)
        report = {
            "audit": "counterfactual_residual_algebra_v1",
            "train_rows": len(train),
            "heldout_rows": len(heldout),
            "train_sha256": sha256_file(paths[0]),
            "heldout_sha256": sha256_file(paths[1]),
            "axes": {"heldout": ["language", "values", "delta"], "pending_before_training": ["two_edit_commutativity"]},
            "claim_boundary": "CPU-only solver-verified corpus construction; no model result is created.",
        }
        report.update(audit)
        paths[2].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if factor_dir is not None:
        factor_dir.mkdir(parents=True)
        factors = {
            "language": {"language_heldout": True},
            "values": {"values_heldout": True},
            "delta": {"delta_heldout": True},
            "query": {"query_heldout": True},
        }
        factor_report = {"audit": "counterfactual_residual_algebra_v1_factors", "train_sha256": report["train_sha256"], "factors": {}}
        for offset, (name, axes) in enumerate(sorted(factors.items())):
            rows = build("factor_{}".format(name), args.factor_count, args.seed + 100 + offset, **axes)
            factor_audit = factor_audit_sets(train, rows)
            allow_state_overlap = name in ("language", "query")
            if factor_audit["train_factor_exact_bundle_hits"] or (factor_audit["train_factor_exact_state_hits"] and not allow_state_overlap):
                raise SystemExit("factor audit failed for {}: {}".format(name, factor_audit))
            output = factor_dir / (name + ".jsonl")
            write_jsonl(output, rows)
            factor_report["factors"][name] = dict(
                factor_audit, rows=len(rows), sha256=sha256_file(output), axes=axes,
                state_overlap_admitted_for_query_wording=allow_state_overlap,
            )
        (factor_dir / "audit.json").write_text(json.dumps(factor_report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(factor_report, sort_keys=True))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
