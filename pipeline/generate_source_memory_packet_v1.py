#!/usr/bin/env python3
"""Generate answer-only data for source-dropping continuous memory packets.

Each example contains a record split across chunks and a later query. The
trainer may read chunks only through fixed continuous memory slots; source
tokens must be absent at answer decoding. Evaluation separates fit, longer
chunk sequences, language transfer, and the combined OOD condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from generate_latent_operator_v1 import apply_operation, sample_operation


WORD = re.compile(r"\w+")
TRAIN_DOMAINS = (
    ("workshop", ("copper", "silver"), "parts"),
    ("orchard", ("apples", "pears"), "fruit"),
    ("pantry", ("jars", "tins"), "supplies"),
    ("library", ("shelves", "carts"), "books"),
    ("garage", ("bolts", "washers"), "pieces"),
    ("studio", ("brushes", "paints"), "tools"),
    ("greenhouse", ("seeds", "pots"), "items"),
    ("kitchen", ("plates", "cups"), "dishes"),
)
HELDOUT_DOMAINS = (
    ("harbor", ("crates", "lanterns"), "items"),
    ("clinic", ("cabinets", "carts"), "bandages"),
    ("observatory", ("cases", "drawers"), "lenses"),
    ("theater", ("props", "costumes"), "pieces"),
)
TRAIN_STYLES = (0, 1, 2)
HELDOUT_STYLES = (3, 4)


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def source_key(row):
    return normalized("\n".join(row["chunks"]) + "\n" + row["query"])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_initial(domain, values, style):
    place, keys, item = domain
    left, right = keys
    if style == 0:
        return "Record: the {} has {} {} in {} and {} {} in {}.".format(
            place, values[left], item, left, values[right], item, right
        )
    if style == 1:
        return "Opening ledger for {}: {}={} {}; {}={} {}.".format(
            place, left, values[left], item, right, values[right], item
        )
    if style == 2:
        return "At the {} store, count {} {} and {} {}.".format(
            place, values[left], left, values[right], right
        )
    if style == 3:
        return "Archive entry for {} lists {} {} under {} and {} {} under {}.".format(
            place, values[left], item, left, values[right], item, right
        )
    return "Inventory note, {}: {} begins at {} {} while {} begins at {} {}.".format(
        place, left, values[left], item, right, values[right], item
    )


def render_operation(operation, item, style):
    kind = operation["kind"]
    if kind == "add":
        if style in (0, 1):
            return "Add {} {} to {}.".format(operation["value"], item, operation["target"])
        return "{} receives {} more {}.".format(operation["target"], operation["value"], item)
    if kind == "sub":
        if style in (0, 1):
            return "Subtract {} {} from {}.".format(operation["value"], item, operation["target"])
        return "Remove {} {} from {}.".format(operation["value"], item, operation["target"])
    if kind == "move":
        if style in (0, 1):
            return "Move {} {} from {} to {}.".format(
                operation["value"], item, operation["source"], operation["target"]
            )
        return "Transfer {} {} out of {} and into {}.".format(
            operation["value"], item, operation["source"], operation["target"]
        )
    if kind == "merge":
        if style in (0, 1):
            return "Set {} to its value plus {}.".format(operation["target"], operation["source"])
        return "Increase {} by all of {}.".format(operation["target"], operation["source"])
    if style in (0, 1):
        return "Swap {} with {}.".format(operation["left"], operation["right"])
    return "Exchange the values of {} and {}.".format(operation["left"], operation["right"])


def build_query(rng, values, keys, item, style):
    left, right = keys
    kind = rng.choice(("read", "sum", "difference"))
    if kind == "read":
        key = rng.choice(keys)
        answer = values[key]
        text = "What is the final {} count?".format(key)
        detail = {"key": key}
    elif kind == "sum":
        answer = values[left] + values[right]
        text = "What is the combined number of {}?".format(item)
        detail = {}
    else:
        high, low = (left, right) if values[left] >= values[right] else (right, left)
        answer = values[high] - values[low]
        text = "How many more {} are in {} than {}?".format(item, high, low)
        detail = {"high": high, "low": low}
    if style >= 3:
        text = "After every update, " + text[0].lower() + text[1:]
    return {"kind": kind, "text": text, "answer": int(answer), **detail}


def make_row(index, rng, domain, chunks, style, initial_range, heldout):
    _, keys, item = domain
    low, high = initial_range
    values = {keys[0]: rng.randint(low, high), keys[1]: rng.randint(low, high)}
    initial = dict(values)
    operations = []
    for _ in range(chunks):
        operation = sample_operation(rng, values, keys)
        operations.append(operation)
        values = apply_operation(values, operation)
    query = build_query(rng, values, keys, item, style)
    source_chunks = []
    for step, operation in enumerate(operations):
        event = "Update {}: {}".format(step + 1, render_operation(operation, item, style))
        source_chunks.append((render_initial(domain, initial, style) + "\n" + event) if step == 0 else event)
    return {
        "chunks": source_chunks,
        "query": query["text"],
        "response": "The answer is {}.".format(query["answer"]),
        "answer": str(query["answer"]),
        "source": "source_memory_packet_v1_{}".format("heldout" if heldout else "train"),
        "training_group": "source_memory_packet",
        "family": domain[0],
        "chunk_count": int(chunks),
        "heldout": bool(heldout),
        "initial": initial,
        "keys": list(keys),
        "operations": operations,
        "query_spec": query,
        "style": int(style),
        "reference": "M-{}-{:07d}".format(domain[0], index),
    }


def build_rows(count, chunk_counts, domains, styles, initial_range, heldout, seed, forbidden=()):
    rng = random.Random(seed)
    rows, seen = [], set(forbidden)
    for index in range(count):
        for attempt in range(10_000):
            row = make_row(
                index * 10_000 + attempt,
                rng,
                domains[index % len(domains)],
                chunk_counts[index % len(chunk_counts)],
                styles[(index // len(domains)) % len(styles)],
                initial_range,
                heldout,
            )
            key = source_key(row)
            if key not in seen:
                rows.append(row)
                seen.add(key)
                break
        else:
            raise RuntimeError("could not create a unique source-memory row {}".format(index))
    return rows


def write_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("refusing to overwrite {}".format(target))
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--train-rows", type=int, default=192_000)
    parser.add_argument("--eval-per-chunks", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.train_rows <= 0 or args.eval_per_chunks <= 0:
        raise SystemExit("row counts must be positive")
    train = build_rows(args.train_rows, (2, 3, 4), TRAIN_DOMAINS, TRAIN_STYLES, (3, 55), False, args.seed)
    train_keys = {source_key(row) for row in train}
    specs = (
        ("fit_iid", (2, 3, 4), TRAIN_DOMAINS, TRAIN_STYLES, (3, 55)),
        ("length_ood", (5, 6, 8), TRAIN_DOMAINS, TRAIN_STYLES, (3, 55)),
        ("language_ood", (2, 3, 4), HELDOUT_DOMAINS, HELDOUT_STYLES, (3, 55)),
        ("full_ood", (5, 6, 8), HELDOUT_DOMAINS, HELDOUT_STYLES, (56, 99)),
    )
    eval_rows = []
    for offset, (regime, counts, domains, styles, value_range) in enumerate(specs):
        rows = build_rows(
            args.eval_per_chunks * len(counts), counts, domains, styles, value_range, True,
            args.seed + 100 + offset, train_keys | {source_key(row) for row in eval_rows},
        )
        for row in rows:
            row["eval_regime"] = regime
        eval_rows.extend(rows)
    write_jsonl(args.train_out, train)
    write_jsonl(args.eval_out, eval_rows)
    print(json.dumps({
        "schema": "source_memory_packet_v1",
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "train_sha256": sha256(args.train_out),
        "eval_sha256": sha256(args.eval_out),
        "train_exact_eval_overlap": len(train_keys & {source_key(row) for row in eval_rows}),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
