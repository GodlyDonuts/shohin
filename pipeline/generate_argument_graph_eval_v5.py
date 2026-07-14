#!/usr/bin/env python3
"""Freeze a fresh lexical/domain board for the R5 argument-graph hypothesis.

The existing fit/depth rows remain pinned preservation controls. Language and
full OOD rows are generated only after the R4 error analysis, with new domains
and several unseen templates per operation. The report audits the fresh rows
against both the R4 training corpus and the development board; only these new
rows may support an R5 generalization claim.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from generate_latent_operator_v1 import (
    apply_operation,
    copy_values,
    normalized,
    ngrams,
    sample_operation,
)


FRESH_DOMAINS = (
    ("greenhouse", ("seedlings", "planters"), "plants"),
    ("depot", ("parcels", "pallets"), "packages"),
    ("laboratory", ("samples", "vials"), "specimens"),
    ("library", ("folios", "maps"), "documents"),
)

OPERATION_TEMPLATES = {
    "add": (
        "Put {value} more {item} into {target}.",
        "The tally for {target} receives {value} additional {item}.",
        "Increase {target} by {value} {item}.",
    ),
    "sub": (
        "The tally for {target} gives up {value} {item}.",
        "Decrease {target} by {value} {item}.",
        "Remove {value} {item} from the tally for {target}.",
    ),
    "move": (
        "Shift {value} {item} out of {source} and into {target}.",
        "The tally for {target} receives {value} {item} taken from {source}.",
        "Transfer {value} {item} from {source} over to {target}.",
    ),
    "merge": (
        "Add the complete {source} amount into {target}.",
        "Let {target} include all of {source} in addition to its own amount.",
        "Copy the entire tally for {source} into {target} as an addition.",
    ),
    "swap": (
        "Interchange the tallies for {left} and {right}.",
        "Let {left} take the old {right} amount and {right} take the old {left} amount.",
        "Trade the recorded amounts of {left} and {right}.",
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_operation(rng, operation, item):
    template = rng.choice(OPERATION_TEMPLATES[operation["kind"]])
    return template.format(item=item, **operation)


def build_query(rng, values, keys, item):
    kind = rng.choice(("read", "sum", "difference"))
    left, right = keys
    if kind == "read":
        key = rng.choice(keys)
        text = rng.choice((
            "Report the final tally for {key}.",
            "What amount remains assigned to {key}?",
        )).format(key=key)
        return {"kind": kind, "key": key, "answer": int(values[key]), "text": text}
    if kind == "sum":
        text = rng.choice((
            "Report the combined {item} across {left} and {right}.",
            "What is the sum of the final {left} and {right} tallies?",
        )).format(item=item, left=left, right=right)
        return {
            "kind": kind, "answer": int(values[left] + values[right]), "text": text,
        }
    high, low = (left, right) if values[left] >= values[right] else (right, left)
    text = rng.choice((
        "Report the amount by which {high} exceeds {low}.",
        "How much larger is the final {high} tally than the final {low} tally?",
    )).format(high=high, low=low)
    return {
        "kind": kind, "high": high, "low": low,
        "answer": int(values[high] - values[low]), "text": text,
    }


def make_row(index, rng, regime, depth):
    domain = FRESH_DOMAINS[index % len(FRESH_DOMAINS)]
    place, keys, item = domain
    values = {keys[0]: rng.randint(37, 79), keys[1]: rng.randint(37, 79)}
    initial = copy_values(values)
    operations = []
    for _ in range(depth):
        operation = sample_operation(rng, values, keys)
        operations.append(operation)
        values = apply_operation(values, operation)
    query = build_query(rng, values, keys, item)
    intro = (
        "Scenario: The {place} ledger begins with {left} at {left_value} and {right} at "
        "{right_value}; identifier words have no numerical value."
    ).format(
        place=place, left=keys[0], left_value=initial[keys[0]],
        right=keys[1], right_value=initial[keys[1]],
    )
    events = "\n".join(
        "Event {}: {}".format(step + 1, render_operation(rng, operation, item))
        for step, operation in enumerate(operations)
    )
    question = "{}\n{}\nRequest: {}\nResult:".format(intro, events, query["text"])
    return {
        "question": question,
        "response": "The answer is {}.".format(query["answer"]),
        "answer": str(query["answer"]),
        "source": "referential_argument_graph_v5_fresh",
        "training_group": "referential_argument_graph_v5",
        "family": place,
        "depth": int(depth),
        "heldout": True,
        "eval_regime": regime,
        "reference": "R5-{}-{:06d}".format(place, index),
        "initial": initial,
        "keys": list(keys),
        "operations": operations,
        "query": query,
    }


def build_fresh(count, regime, depths, seed):
    rng = random.Random(seed)
    rows, seen = [], set()
    index = 0
    while len(rows) < count:
        row = make_row(index, rng, regime, depths[index % len(depths)])
        key = normalized(row["question"])
        index += 1
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def scan_overlap(paths, fresh_questions, fresh_grams):
    exact_hits, gram_hits = [], []
    for path in paths:
        with open(path) as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                question = json.loads(line)["question"]
                key = normalized(question)
                if key in fresh_questions:
                    exact_hits.append("{}:{}".format(path, line_number))
                overlap = fresh_grams.intersection(ngrams(question))
                if overlap:
                    gram_hits.append({
                        "source": "{}:{}".format(path, line_number),
                        "grams": sorted(overlap)[:3],
                    })
    return exact_hits, gram_hits


def counts(rows):
    return {
        "rows": len(rows),
        "regimes": dict(sorted(collections.Counter(row["eval_regime"] for row in rows).items())),
        "depths": dict(sorted(collections.Counter(str(row["depth"]) for row in rows).items())),
        "families": dict(sorted(collections.Counter(row["family"] for row in rows).items())),
        "operation_kinds": dict(sorted(collections.Counter(
            operation["kind"] for row in rows for operation in row["operations"]
        ).items())),
        "query_kinds": dict(sorted(collections.Counter(row["query"]["kind"] for row in rows).items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--development", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--language-cases", type=int, default=256)
    parser.add_argument("--full-cases", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    for path in (args.train, args.development):
        if not Path(path).is_file():
            raise SystemExit("missing input {}".format(path))
    for path in (args.out, args.report):
        if Path(path).exists():
            raise SystemExit("refusing existing output {}".format(path))

    development = [json.loads(line) for line in open(args.development) if line.strip()]
    controls = [
        row for row in development if row.get("eval_regime") in {"fit_iid", "depth_ood"}
    ]
    if collections.Counter(row.get("eval_regime") for row in controls) != {
        "fit_iid": 256, "depth_ood": 192,
    }:
        raise SystemExit("development preservation controls have unexpected shape")
    language = build_fresh(args.language_cases, "language_ood", (1, 2, 3, 4), args.seed)
    full = build_fresh(args.full_cases, "full_ood", (5, 6, 8), args.seed + 1)
    fresh = language + full
    fresh_questions = {normalized(row["question"]) for row in fresh}
    if len(fresh_questions) != len(fresh):
        raise SystemExit("duplicate fresh question")
    fresh_grams = set().union(*(ngrams(row["question"]) for row in fresh))
    exact_hits, gram_hits = scan_overlap(
        (args.train, args.development), fresh_questions, fresh_grams,
    )
    if exact_hits or gram_hits:
        raise SystemExit("fresh overlap exact={} gram13={}".format(len(exact_hits), len(gram_hits)))

    combined = controls + fresh
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in combined))
    report = {
        "audit": "referential_argument_graph_eval_v5_build",
        "all_checks_pass": True,
        "train": str(Path(args.train).resolve()),
        "train_sha256": sha256(args.train),
        "development": str(Path(args.development).resolve()),
        "development_sha256": sha256(args.development),
        "out": str(Path(args.out).resolve()),
        "out_sha256": sha256(args.out),
        "seed": args.seed,
        "preservation_controls": counts(controls),
        "fresh": counts(fresh),
        "fresh_exact_train_or_development_hits": len(exact_hits),
        "fresh_13gram_train_or_development_hits": len(gram_hits),
        "claim_boundary": (
            "Only fresh language/full rows may support R5 generalization; fit/depth rows are pinned "
            "development controls."
        ),
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
