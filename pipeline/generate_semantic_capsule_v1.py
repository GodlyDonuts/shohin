#!/usr/bin/env python3
"""Build a solver-verified semantic-capsule curriculum and held-out episodes.

The earlier compact-state experiments exposed their answer in a fixed template:
they measured protocol imitation, not context transfer.  A semantic capsule is
different.  It stores multiple named facts from a natural-language record,
then a controller removes the original record and repeatedly supplies only the
model-produced capsule plus a new event.  The model must preserve the facts,
apply each event, and answer a later query.

This generator creates training rows for write/update/readout/repair modes and
held-out controller episodes.  Train and held-out data use disjoint domains,
field names, number bands, event wording, and episode lengths.  It is data
generation only; a separate audit and isolated SFT/evaluation are required.
"""

import argparse
import collections
import hashlib
import json
import os
import random
import re
from pathlib import Path


WORD = re.compile(r"\w+")
TRAIN_DOMAINS = (
    ("workshop", ("copper", "silver"), "parts"),
    ("bakery", ("trays", "racks"), "pastries"),
    ("library", ("folios", "returns"), "books"),
)
HELDOUT_DOMAINS = (
    ("harbor", ("crates", "lockers"), "lanterns"),
    ("observatory", ("cases", "drawers"), "lenses"),
    ("clinic", ("cabinets", "carts"), "bandages"),
)


def normalized_question(text):
    return " ".join(WORD.findall(text.lower()))


def canonical_capsule(values, keys):
    return "capsule:" + ";".join("{}={}".format(key, int(values[key])) for key in keys)


def copy_values(values):
    return {key: int(value) for key, value in values.items()}


def apply_operation(values, operation):
    result = copy_values(values)
    kind = operation["kind"]
    if kind == "add":
        result[operation["target"]] += operation["value"]
    elif kind == "sub":
        result[operation["target"]] -= operation["value"]
    elif kind == "transfer":
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


def operation_for(rng, values, keys):
    left, right = keys
    kinds = ("add", "sub", "transfer", "merge", "swap")
    kind = rng.choice(kinds)
    if kind == "add":
        return {"kind": kind, "target": rng.choice(keys), "value": rng.randint(3, 29)}
    if kind == "sub":
        target = rng.choice(keys)
        return {"kind": kind, "target": target, "value": rng.randint(1, max(1, values[target] // 3))}
    if kind == "transfer":
        source, target = (left, right) if rng.randrange(2) else (right, left)
        return {
            "kind": kind, "source": source, "target": target,
            "value": rng.randint(1, max(1, values[source] // 3)),
        }
    if kind == "merge":
        target, source = (left, right) if rng.randrange(2) else (right, left)
        return {"kind": kind, "target": target, "source": source}
    return {"kind": kind, "left": left, "right": right}


def render_event(operation, heldout, item):
    kind = operation["kind"]
    if kind == "add":
        if heldout:
            return "The {} tally receives {} more {}.".format(operation["target"], operation["value"], item)
        return "Add {} {} to {}.".format(operation["value"], item, operation["target"])
    if kind == "sub":
        if heldout:
            return "Remove {} {} from the {} tally.".format(operation["value"], item, operation["target"])
        return "Subtract {} {} from {}.".format(operation["value"], item, operation["target"])
    if kind == "transfer":
        if heldout:
            return "Relocate {} {} from {} into {}.".format(
                operation["value"], item, operation["source"], operation["target"]
            )
        return "Move {} {} from {} to {}.".format(
            operation["value"], item, operation["source"], operation["target"]
        )
    if kind == "merge":
        if heldout:
            return "Increase {} by the full current amount stored in {}.".format(
                operation["target"], operation["source"]
            )
        return "Set {} to {} plus {}.".format(operation["target"], operation["target"], operation["source"])
    if heldout:
        return "Exchange the quantities recorded for {} and {}.".format(operation["left"], operation["right"])
    return "Swap the values of {} and {}.".format(operation["left"], operation["right"])


def trace_for(before, operation, after, keys):
    kind = operation["kind"]
    if kind == "add":
        key = operation["target"]
        return "{} changes from {} to {}.".format(key, before[key], after[key])
    if kind == "sub":
        key = operation["target"]
        return "{} changes from {} to {}.".format(key, before[key], after[key])
    if kind == "transfer":
        return "{} becomes {} and {} becomes {}.".format(
            operation["source"], after[operation["source"]], operation["target"], after[operation["target"]]
        )
    if kind == "merge":
        return "{} becomes {} while {} stays {}.".format(
            operation["target"], after[operation["target"]], operation["source"], after[operation["source"]]
        )
    return "{} and {} exchange their values.".format(keys[0], keys[1])


def write_prompt(domain, keys, values, heldout, reference):
    place, _, item = domain
    left, right = keys
    if heldout:
        return (
            "Task: Compress the factual record below before the source is discarded.\n"
            "Record reference {}: a record from the {} lists {} {} under {} and {} {} under {}. "
            "The reference is not a quantity, and the two labels describe separate quantities.\n"
            "Reason inside <think>, then emit only capsule:{}=<integer>;{}=<integer> on the last line.\nResult:"
        ).format(reference, place, values[left], item, left, values[right], item, right, left, right)
    record = (
        "At a {}, record {} says the {} ledger holds {} {} and the {} ledger holds {} {}. "
        "The record identifier is not a quantity; keep both ledger facts."
    ).format(place, reference, left, values[left], item, right, values[right], item)
    return (
        "Question: Build a compact semantic capsule from this record.\n{}\n"
        "Inside <think> identify both named quantities. On the last line return only "
        "capsule:{}=<integer>;{}=<integer>.\nAnswer:"
    ).format(record, left, right)


def update_prompt(capsule, event, keys, heldout=False, reference="", revision=0):
    if heldout:
        return (
            "Task: A previous record has been discarded after being compressed. Reference {} revision {} is not a quantity.\n"
            "Retained facts: {}\nNew record: {}\n"
            "Reason inside <think>, then emit only one final line in the form "
            "capsule:{}=<integer>;{}=<integer>.\nResult:"
        ).format(reference, revision, capsule, event, keys[0], keys[1])
    return (
        "Question: Continue record {} revision {} from a compact semantic capsule; identifiers are not quantities.\n"
        "Capsule: {}\nEvent: {}\n"
        "Inside <think> apply the event while retaining both named facts. On the last line return only "
        "capsule:{}=<integer>;{}=<integer>.\nAnswer:"
    ).format(reference, revision, capsule, event, keys[0], keys[1])


def query_for(rng, values, keys, item, heldout):
    kind = rng.choice(("read", "sum", "difference"))
    left, right = keys
    if kind == "read":
        key = rng.choice(keys)
        text = "Report the current {} quantity.".format(key)
        answer = values[key]
        trace = "The capsule states {}={}.".format(key, answer)
        detail = {"key": key}
    elif kind == "sum":
        text = "What is the combined number of {} in {} and {}?".format(item, left, right)
        answer = values[left] + values[right]
        trace = "Add {} and {} to get {}.".format(values[left], values[right], answer)
        detail = {}
    else:
        high, low = (left, right) if values[left] >= values[right] else (right, left)
        text = "How many more {} are in {} than in {}?".format(item, high, low)
        answer = values[high] - values[low]
        trace = "Subtract {} from {} to get {}.".format(values[low], values[high], answer)
        detail = {"high": high, "low": low}
    if heldout:
        text = "Using only the retained capsule, " + text[0].lower() + text[1:]
    return {"kind": kind, "text": text, "answer": int(answer), "trace": trace, **detail}


def query_prompt(capsule, query, heldout=False, reference=""):
    if heldout:
        return (
            "Task: The original record is unavailable; reference {} is not a quantity. Use the retained facts below.\n"
            "Retained facts: {}\nRequest: {}\n"
            "Reason inside <think>, then finish with 'The answer is <integer>.'.\nResult:"
        ).format(reference, capsule, query["text"])
    return (
        "Question: Answer a query using record {} and only this compact semantic capsule. The record identifier is not a quantity.\n"
        "Capsule: {}\nQuery: {}\n"
        "Inside <think> use the named facts. Then end with 'The answer is <integer>.'.\nAnswer:"
    ).format(reference, capsule, query["text"])


def capsule_response(trace, values, keys):
    return "<think>{}</think>\n{}".format(trace, canonical_capsule(values, keys))


def answer_response(trace, answer):
    return "<think>{}</think>\nThe answer is {}.".format(trace, answer)


def corrupt(values, keys):
    result = copy_values(values)
    key = keys[0]
    result[key] += 1
    return result


def make_episode(index, rng, domain, heldout):
    place, keys, item = domain
    value_low, value_high = (101, 199) if heldout else (9, 79)
    values = {keys[0]: rng.randint(value_low, value_high), keys[1]: rng.randint(value_low, value_high)}
    if heldout:
        regime = ("semantic_len4", "semantic_len8", "semantic_len12")[index % 3]
        steps = {"semantic_len4": 4, "semantic_len8": 8, "semantic_len12": 12}[regime]
    else:
        regime = "train_len{}".format(2 + (index % 3))
        steps = 2 + (index % 3)
    initial = copy_values(values)
    reference = "{}-{}-{:05d}".format("H" if heldout else "T", place, index)
    operations = []
    for revision in range(1, steps + 1):
        operation = operation_for(rng, values, keys)
        next_values = apply_operation(values, operation)
        operations.append({
            "operation": operation,
            "revision": revision,
            "instruction": render_event(operation, heldout, item),
            "before": copy_values(values),
            "expected": copy_values(next_values),
            "trace": trace_for(values, operation, next_values, keys),
        })
        values = next_values
    query = query_for(rng, values, keys, item, heldout)
    return {
        "id": "{}-{}-{}".format("heldout" if heldout else "train", place, index),
        "heldout": bool(heldout),
        "reference": reference,
        "family": place,
        "regime": regime,
        "keys": list(keys),
        "item": item,
        "question": write_prompt(domain, keys, initial, heldout, reference),
        "initial": {
            "values": initial,
            "prompt": write_prompt(domain, keys, initial, heldout, reference),
            "trace": "The record gives {}={} and {}={}.".format(keys[0], initial[keys[0]], keys[1], initial[keys[1]]),
        },
        "operations": operations,
        "query": query,
    }


def rows_for_episode(episode):
    keys = tuple(episode["keys"])
    rows = []
    initial = episode["initial"]
    rows.append({
        "question": initial["prompt"],
        "completion_prompt": initial["prompt"],
        "response": capsule_response(initial["trace"], initial["values"], keys),
        "answer": canonical_capsule(initial["values"], keys),
        "source": "semantic_capsule_v1_train",
        "training_group": "semantic_capsule",
        "family": episode["family"],
        "mode": "write",
    })
    current = copy_values(initial["values"])
    for operation in episode["operations"]:
        prompt = update_prompt(canonical_capsule(current, keys), operation["instruction"], keys,
                              reference=episode["reference"], revision=operation["revision"])
        rows.append({
            "question": prompt,
            "completion_prompt": prompt,
            "response": capsule_response(operation["trace"], operation["expected"], keys),
            "answer": canonical_capsule(operation["expected"], keys),
            "source": "semantic_capsule_v1_train",
            "training_group": "semantic_capsule",
            "family": episode["family"],
            "mode": "update",
        })
        bad = corrupt(operation["expected"], keys)
        repair_prompt = (
            "Question: Repair a proposed semantic capsule for record {} revision {}; identifiers are not quantities.\nPrevious capsule: {}\nEvent: {}\n"
            "Proposed capsule: {}\nInside <think> recompute from the previous capsule and event. "
            "On the last line return only capsule:{}=<integer>;{}=<integer>.\nAnswer:"
        ).format(episode["reference"], operation["revision"], canonical_capsule(current, keys), operation["instruction"], canonical_capsule(bad, keys), keys[0], keys[1])
        rows.append({
            "question": repair_prompt,
            "completion_prompt": repair_prompt,
            "response": capsule_response(operation["trace"], operation["expected"], keys),
            "answer": canonical_capsule(operation["expected"], keys),
            "source": "semantic_capsule_v1_train",
            "training_group": "semantic_capsule",
            "family": episode["family"],
            "mode": "repair",
        })
        current = copy_values(operation["expected"])
    prompt = query_prompt(canonical_capsule(current, keys), episode["query"], reference=episode["reference"])
    rows.append({
        "question": prompt,
        "completion_prompt": prompt,
        "response": answer_response(episode["query"]["trace"], episode["query"]["answer"]),
        "answer": str(episode["query"]["answer"]),
        "source": "semantic_capsule_v1_train",
        "training_group": "semantic_capsule",
        "family": episode["family"],
        "mode": "readout",
    })
    return rows


def build_episodes(per_domain, seed, heldout):
    if per_domain <= 0:
        raise ValueError("per_domain must be positive")
    rng = random.Random(seed)
    domains = HELDOUT_DOMAINS if heldout else TRAIN_DOMAINS
    episodes, seen, counts = [], set(), collections.Counter()
    for domain in domains:
        place = domain[0]
        attempts = 0
        while counts[place] < per_domain:
            attempts += 1
            if attempts > per_domain * 40:
                raise RuntimeError("could not generate enough unique {} episodes".format(place))
            episode = make_episode(attempts, rng, domain, heldout)
            key = normalized_question(episode["initial"]["prompt"])
            if key in seen:
                continue
            seen.add(key)
            episodes.append(episode)
            counts[place] += 1
    rng.shuffle(episodes)
    return episodes


def write_jsonl(path, rows):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit("refusing to overwrite existing artifact: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    os.replace(partial, path)


def summarize(episodes, rows=None):
    result = {
        "episodes": len(episodes),
        "families": dict(sorted(collections.Counter(item["family"] for item in episodes).items())),
        "regimes": dict(sorted(collections.Counter(item["regime"] for item in episodes).items())),
    }
    if rows is not None:
        result.update({
            "rows": len(rows),
            "modes": dict(sorted(collections.Counter(item["mode"] for item in rows).items())),
            "all_have_think": all(item["response"].startswith("<think>") for item in rows),
            "all_have_answer_or_capsule": all("capsule:" in item["response"] or "The answer is" in item["response"] for item in rows),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--train-per-domain", type=int, default=15000)
    parser.add_argument("--heldout-per-domain", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    train_episodes = build_episodes(args.train_per_domain, args.seed, heldout=False)
    heldout_episodes = build_episodes(args.heldout_per_domain, args.seed + 1, heldout=True)
    train_rows = [row for episode in train_episodes for row in rows_for_episode(episode)]
    train_prompts = {normalized_question(row["question"]) for row in train_rows}
    heldout_prompts = {normalized_question(episode["initial"]["prompt"]) for episode in heldout_episodes}
    if train_prompts & heldout_prompts:
        raise RuntimeError("train/held-out initial prompt overlap")
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.heldout_out, heldout_episodes)
    print(json.dumps({
        "schema": "semantic_capsule_v1",
        "train": summarize(train_episodes, train_rows),
        "heldout": summarize(heldout_episodes),
        "train_heldout_initial_prompt_overlap": 0,
        "train_rows_sha256": hashlib.sha256(Path(args.train_out).read_bytes()).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
