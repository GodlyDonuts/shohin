#!/usr/bin/env python3
"""Build paired, training-only language views for semantic microcode compilation."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


DOMAINS = (
    ("depot", ("bolts", "washers"), "components"),
    ("aviary", ("hawks", "owls"), "birds"),
    ("library", ("atlases", "manuals"), "books"),
    ("garden", ("tulips", "lilies"), "flowers"),
    ("studio", ("brushes", "canvases"), "supplies"),
    ("bakery", ("loaves", "rolls"), "bakes"),
    ("garage", ("wrenches", "pliers"), "tools"),
    ("theater", ("masks", "props"), "pieces"),
    ("school", ("notebooks", "folders"), "materials"),
    ("market", ("melons", "plums"), "produce"),
    ("museum", ("prints", "statues"), "objects"),
    ("station", ("tickets", "passes"), "documents"),
    ("factory", ("gears", "springs"), "parts"),
    ("kitchen", ("plates", "bowls"), "dishes"),
    ("nursery", ("saplings", "shrubs"), "plants"),
    ("archive", ("folios", "maps"), "records"),
)

INTRO_A = (
    "In the {place} ledger, {left} holds {a} {item}, while {right} holds {b} {item}.",
    "The {place} log assigns {a} {item} to {left} and {b} {item} to {right}.",
    "At the {place}, the starting tally is {left} with {a} {item} and {right} with {b} {item}.",
)
INTRO_B = (
    "A {place} tally begins with {a} {item} under {left} and {b} {item} under {right}.",
    "For the {place} account, {left} starts at {a} {item}; {right} starts at {b} {item}.",
    "The opening {place} sheet records {a} {item} for {left} and {b} {item} for {right}.",
)

OPS_A = {
    "add": (
        "Place {value} additional {item} with {target}.",
        "Give {target} another {value} {item}.",
        "Credit {target} with {value} extra {item}.",
    ),
    "sub": (
        "Remove {value} {item} belonging to {target}.",
        "Deduct {value} {item} from {target}.",
        "Lower the {target} tally by {value} {item}.",
    ),
    "move": (
        "Transfer {value} {item} out of {source} and into {target}.",
        "Send {value} {item} from {source} over to {target}.",
        "Move a batch of {value} {item} from {source} across to {target}.",
    ),
    "merge": (
        "Combine the current {source} amount into {target} without emptying {source}.",
        "Add the entire {source} tally onto {target}, leaving {source} unchanged.",
        "Copy the full {source} amount into an addition for {target}.",
    ),
    "swap": (
        "Let {left} and {right} trade their totals.",
        "Make the totals for {left} and {right} switch places.",
        "Transpose the amounts stored by {left} and {right}.",
    ),
}
OPS_B = {
    "add": (
        "{target} receives {value} more {item}.",
        "Append {value} {item} to the amount at {target}.",
        "Raise {target} by an extra {value} {item}.",
    ),
    "sub": (
        "Reduce {target} by {value} {item}.",
        "Withdraw {value} {item} from {target}.",
        "Debit {value} {item} against {target}.",
    ),
    "move": (
        "Shift {value} {item} from {source} across to {target}.",
        "Route {value} {item} away from {source} and toward {target}.",
        "Pass {value} {item} out of {source} into {target}.",
    ),
    "merge": (
        "Augment {target} with the full existing total in {source}; retain {source}.",
        "Duplicate the {source} total as an addition to {target}.",
        "Put one extra copy of all {source} into {target}.",
    ),
    "swap": (
        "Interchange the totals stored under {left} and {right}.",
        "Reverse which of {left} and {right} owns each total.",
        "Trade the recorded amounts between {left} and {right}.",
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remap_operation(operation, key_map):
    result = dict(operation)
    for field in ("target", "source", "left", "right"):
        if field in result:
            result[field] = key_map[result[field]]
    return result


def remap_query(query, key_map):
    result = {key: value for key, value in query.items() if key != "text"}
    for field in ("key", "high", "low"):
        if field in result:
            result[field] = key_map[result[field]]
    return result


def render_operation(operation, templates, item, choice):
    options = templates[operation["kind"]]
    return options[choice % len(options)].format(item=item, **operation)


def render_query(query, keys, item, view, choice):
    if view == 0:
        options = {
            "read": (
                "Report the ending total for {key}.",
                "State the final amount kept by {key}.",
                "Give the last recorded count at {key}.",
            ),
            "sum": (
                "Report the final total across {left} and {right}.",
                "State the ending combined amount for {left} and {right}.",
                "Give the sum of the final {left} and {right} counts.",
            ),
            "difference": (
                "Report how much {high} exceeds {low}.",
                "State the final positive gap from {low} up to {high}.",
                "Give the amount by which {high} finishes above {low}.",
            ),
        }
    else:
        options = {
            "read": (
                "Which final count belongs to {key}?",
                "What ending amount is associated with {key}?",
                "How many {item} remain assigned to {key}?",
            ),
            "sum": (
                "What do {left} and {right} contain altogether at the end?",
                "How many {item} are present across both {left} and {right} finally?",
                "What is the ending aggregate of {left} with {right}?",
            ),
            "difference": (
                "By how many {item} does {high} finish above {low}?",
                "What positive distance separates final {high} from final {low}?",
                "How much larger is ending {high} than ending {low}?",
            ),
        }
    values = {"left": keys[0], "right": keys[1], "item": item, **query}
    selected = options[query["kind"]]
    return selected[choice % len(selected)].format(**values)


def make_view(source, equivalence_index, view):
    source_keys = source["keys"]
    # Every domain appears in both views across the corpus; the offset keeps
    # paired views distinct without making vocabulary identify the view.
    domain = DOMAINS[(equivalence_index + view * 7) % len(DOMAINS)]
    place, keys, item = domain
    key_map = dict(zip(source_keys, keys))
    initial = {key_map[key]: int(source["initial"][key]) for key in source_keys}
    operations = [remap_operation(operation, key_map) for operation in source["operations"]]
    query = remap_query(source["query"], key_map)
    intro_templates = INTRO_A if view == 0 else INTRO_B
    op_templates = OPS_A if view == 0 else OPS_B
    choice = equivalence_index + view
    intro = intro_templates[choice % len(intro_templates)].format(
        place=place, left=keys[0], right=keys[1], a=initial[keys[0]], b=initial[keys[1]], item=item,
    )
    events = "\n".join(
        "Step {}: {}".format(index + 1, render_operation(operation, op_templates, item, choice + index))
        for index, operation in enumerate(operations)
    )
    query_text = render_query(query, keys, item, view, choice)
    question = "{}\n{}\nRequest: {}\nAnswer:".format(intro, events, query_text)
    return {
        "question": question,
        "response": "The answer is {}.".format(source["answer"]),
        "answer": str(source["answer"]),
        "source": "categorical_microcode_equivalence_v2",
        "training_group": "categorical_microcode_equivalence",
        "family": place,
        "depth": int(source["depth"]),
        "heldout": False,
        "reference": "CMBQ-{:06d}-{}".format(equivalence_index, view),
        "initial": initial,
        "keys": list(keys),
        "operations": operations,
        "query": query,
        "equivalence_id": "cmbq-{:06d}".format(equivalence_index),
        "equivalence_view": int(view),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--pairs", type=int, default=48000)
    args = parser.parse_args()
    if args.pairs <= 0:
        raise SystemExit("pairs must be positive")
    if Path(args.out).exists() or Path(args.report).exists():
        raise SystemExit("refusing existing output")
    source_rows = [json.loads(line) for line in Path(args.source).read_text().splitlines() if line.strip()]
    if len(source_rows) < args.pairs:
        raise SystemExit("source has fewer rows than requested pairs")
    rows = [make_view(source_rows[index], index, view) for index in range(args.pairs) for view in (0, 1)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = {
        "build": "categorical_microcode_equivalence_v2",
        "source": str(Path(args.source).resolve()),
        "source_sha256": sha256(args.source),
        "pairs": args.pairs,
        "rows": len(rows),
        "views": dict(collections.Counter(str(row["equivalence_view"]) for row in rows)),
        "depths": dict(collections.Counter(str(row["depth"]) for row in rows)),
        "data_sha256": sha256(args.out),
        "heldout_phrase_policy": "training-only templates; latent-operator heldout render strings excluded",
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
