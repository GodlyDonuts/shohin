#!/usr/bin/env python3
"""Create a solver-verified answer-only curriculum for continuous latent rollouts.

The model sees an initial two-register record and a sequence of symbolic events,
then predicts only the final answer.  There is no text chain of thought or
serialized state to imitate.  A latent-rollout model must therefore carry its
own intermediate state through continuous embeddings.  The held-out split
changes domains, field names, event wording, values, and operation length.
"""

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path


WORD = re.compile(r"\w+")
TRAIN_DOMAINS = (
    ("workshop", ("copper", "silver"), "parts"),
    ("orchard", ("apples", "pears"), "fruit"),
    ("pantry", ("jars", "tins"), "supplies"),
)
HELDOUT_DOMAINS = (
    ("harbor", ("crates", "lanterns"), "items"),
    ("clinic", ("cabinets", "carts"), "bandages"),
    ("observatory", ("cases", "drawers"), "lenses"),
)


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def ngrams(text, width=13):
    words = normalized(text).split()
    return {" ".join(words[index:index + width]) for index in range(max(0, len(words) - width + 1))}


def copy_values(values):
    return {key: int(value) for key, value in values.items()}


def apply_operation(values, operation):
    result = copy_values(values)
    kind = operation["kind"]
    if kind == "add":
        result[operation["target"]] += operation["value"]
    elif kind == "sub":
        result[operation["target"]] -= operation["value"]
    elif kind == "move":
        result[operation["source"]] -= operation["value"]
        result[operation["target"]] += operation["value"]
    elif kind == "merge":
        result[operation["target"]] += result[operation["source"]]
    elif kind == "swap":
        left, right = operation["left"], operation["right"]
        result[left], result[right] = result[right], result[left]
    else:
        raise ValueError("unknown operation {}".format(kind))
    return result


def sample_operation(rng, values, keys):
    left, right = keys
    kind = rng.choice(("add", "sub", "move", "merge", "swap"))
    if kind == "add":
        return {"kind": kind, "target": rng.choice(keys), "value": rng.randint(1, 9)}
    if kind == "sub":
        target = rng.choice(keys)
        return {"kind": kind, "target": target, "value": rng.randint(1, max(1, values[target] // 3))}
    if kind == "move":
        source, target = (left, right) if rng.randrange(2) else (right, left)
        return {
            "kind": kind,
            "source": source,
            "target": target,
            "value": rng.randint(1, max(1, values[source] // 3)),
        }
    if kind == "merge":
        source, target = (left, right) if rng.randrange(2) else (right, left)
        return {"kind": kind, "source": source, "target": target}
    return {"kind": kind, "left": left, "right": right}


def render_operation(operation, heldout, item):
    kind = operation["kind"]
    if kind == "add":
        if heldout:
            return "The {} count gains {} {}.".format(operation["target"], operation["value"], item)
        return "Add {} {} to {}.".format(operation["value"], item, operation["target"])
    if kind == "sub":
        if heldout:
            return "Take {} {} away from {}.".format(operation["value"], item, operation["target"])
        return "Subtract {} {} from {}.".format(operation["value"], item, operation["target"])
    if kind == "move":
        if heldout:
            return "Relocate {} {} from {} into {}.".format(
                operation["value"], item, operation["source"], operation["target"]
            )
        return "Move {} {} from {} to {}.".format(
            operation["value"], item, operation["source"], operation["target"]
        )
    if kind == "merge":
        if heldout:
            return "Increase {} by everything currently in {}.".format(operation["target"], operation["source"])
        return "Set {} to its value plus {}.".format(operation["target"], operation["source"])
    if heldout:
        return "Exchange the values assigned to {} and {}.".format(operation["left"], operation["right"])
    return "Swap {} with {}.".format(operation["left"], operation["right"])


def query_for(rng, values, keys, item, heldout):
    kind = rng.choice(("read", "sum", "difference"))
    left, right = keys
    if kind == "read":
        key = rng.choice(keys)
        answer = values[key]
        text = "What is the final {} total?".format(key)
        detail = {"key": key}
    elif kind == "sum":
        answer = values[left] + values[right]
        text = "What is the combined number of {} in {} and {}?".format(item, left, right)
        detail = {}
    else:
        high, low = (left, right) if values[left] >= values[right] else (right, left)
        answer = values[high] - values[low]
        text = "How many more {} are in {} than in {}?".format(item, high, low)
        detail = {"high": high, "low": low}
    if heldout:
        text = "After all updates, " + text[0].lower() + text[1:]
    return {"kind": kind, "text": text, "answer": int(answer), **detail}


def render_question(domain, values, operations, query, heldout, reference):
    place, keys, item = domain
    left, right = keys
    if heldout:
        intro = (
            "Task: A {} inventory record marked {} lists {} {} under {} and {} {} under {}. "
            "The reference is not a quantity."
        ).format(place, reference, values[left], item, left, values[right], item, right)
        events = "\n".join("Event {}: {}".format(index + 1, render_operation(op, True, item))
                           for index, op in enumerate(operations))
        return "{}\n{}\nRequest: {}\nAnswer:".format(intro, events, query["text"])
    intro = (
        "Question: In a {} record {} has {} {} and {} has {} {}. "
        "The record label is not a quantity."
    ).format(place, left, values[left], item, right, values[right], item)
    events = "\n".join("Step {}: {}".format(index + 1, render_operation(op, False, item))
                       for index, op in enumerate(operations))
    return "{}\n{}\nQuestion: {}\nAnswer:".format(intro, events, query["text"])


def make_row(index, rng, domain, depth, heldout, initial_range=None):
    _, keys, _ = domain
    low, high = initial_range or ((37, 79) if heldout else (3, 29))
    if low > high:
        raise ValueError("initial_range must be ordered")
    values = {keys[0]: rng.randint(low, high), keys[1]: rng.randint(low, high)}
    initial = copy_values(values)
    operations = []
    for _ in range(depth):
        operation = sample_operation(rng, values, keys)
        operations.append(operation)
        values = apply_operation(values, operation)
    query = query_for(rng, values, keys, domain[2], heldout)
    reference = "{}-{}-{:06d}".format("H" if heldout else "T", domain[0], index)
    question = render_question(domain, initial, operations, query, heldout, reference)
    return {
        "question": question,
        "response": "The answer is {}.".format(query["answer"]),
        "answer": str(query["answer"]),
        "source": "latent_operator_v1_{}".format("heldout" if heldout else "train"),
        "training_group": "latent_operator",
        "family": domain[0],
        "depth": int(depth),
        "heldout": bool(heldout),
        "reference": reference,
        "initial": initial,
        "keys": list(keys),
        "operations": operations,
        "query": query,
    }


def build_rows(count, depths, seed, heldout):
    rng = random.Random(seed)
    domains = HELDOUT_DOMAINS if heldout else TRAIN_DOMAINS
    rows = []
    seen = set()
    for index in range(count):
        # Train prompts intentionally do not expose the synthetic reference
        # identifier, so random operation/value collisions are possible at
        # corpus scale.  Reject and resample them here rather than relying on a
        # downstream audit to discover that duplicate examples consumed a
        # fraction of the curriculum.
        for _ in range(10_000):
            row = make_row(index, rng, domains[index % len(domains)], depths[index % len(depths)], heldout)
            prompt = normalized(row["question"])
            if prompt not in seen:
                rows.append(row)
                seen.add(prompt)
                break
        else:
            raise RuntimeError("could not create a unique latent-operator prompt at row {}".format(index))
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("refusing to overwrite {}".format(path))
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def summary(rows):
    return {
        "rows": len(rows),
        "families": dict(sorted(collections.Counter(row["family"] for row in rows).items())),
        "depths": dict(sorted(collections.Counter(row["depth"] for row in rows).items())),
        "answers": {"min": min(int(row["answer"]) for row in rows), "max": max(int(row["answer"]) for row in rows)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--train-rows", type=int, default=96_000)
    parser.add_argument("--eval-per-depth", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    if args.train_rows <= 0 or args.eval_per_depth <= 0:
        raise SystemExit("row counts must be positive")
    train = build_rows(args.train_rows, (1, 2, 3, 4), args.seed, False)
    eval_rows = build_rows(args.eval_per_depth * 3, (5, 6, 8), args.seed + 1, True)
    train_prompts = {normalized(row["question"]) for row in train}
    eval_prompts = {normalized(row["question"]) for row in eval_rows}
    if train_prompts & eval_prompts:
        raise SystemExit("train-heldout exact prompt overlap")
    write_jsonl(args.train_out, train)
    write_jsonl(args.eval_out, eval_rows)
    print(json.dumps({
        "schema": "latent_operator_v1",
        "train": summary(train),
        "heldout": summary(eval_rows),
        "train_sha256": sha256(args.train_out),
        "heldout_sha256": sha256(args.eval_out),
        "initial_exact_overlap": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
