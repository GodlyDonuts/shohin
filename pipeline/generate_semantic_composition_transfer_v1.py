#!/usr/bin/env python3
"""Build a held-out semantic composition transfer suite for V10A.

This file creates *evaluation-only* solver-verified rows.  It intentionally
combines operations that appear separately in semantic_bridge_v1 and includes
a source-dropped named-state task.  It must never be included in SFT data.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path


WORD = re.compile(r"\w+")
FAMILIES = (
    "product_to_chain",
    "base_then_adjust",
    "fact_to_chain",
    "repair_to_chain",
    "source_dropped_named_state",
)
TERMS = (
    ("rail depot", "wagons", "signal lamps"),
    ("field station", "racks", "sample vials"),
    ("conservatory", "cabinets", "seed packets"),
    ("shipyard", "bays", "tool crates"),
    ("survey camp", "lockers", "map tubes"),
)


def normalized_question(text):
    return " ".join(WORD.findall(text.lower()))


def render(question, trace, answer, family):
    return {
        "question": question,
        "response": f"<think>{trace}</think>\nThe answer is {answer}.",
        "answer": str(answer),
        "family": family,
        "source": "semantic_composition_transfer_v1_eval_only",
    }


def product_to_chain(rng):
    place, containers, items = rng.choice(TERMS)
    groups = rng.randint(211, 299)
    per_group = rng.randint(31, 67)
    removed = rng.randint(101, 197)
    scale = rng.randint(2, 5)
    product = groups * per_group
    after_removed = product - removed
    answer = after_removed * scale
    question = (
        f"At a {place}, {groups} {containers} each hold {per_group} {items}. After {removed} {items} "
        f"are removed, the remaining count is packed into {scale} identical shipments. What total number "
        "of items is represented across those shipments?"
    )
    trace = (
        f"First {groups}*{per_group}={product}. Remove {removed}: {product}-{removed}={after_removed}. "
        f"Then {after_removed}*{scale}={answer}."
    )
    return render(question, trace, answer, "product_to_chain")


def base_then_adjust(rng):
    base = rng.choice((3, 4))
    digits = [rng.randint(1, base - 1), rng.randint(0, base - 1), rng.randint(0, base - 1), rng.randint(0, base - 1)]
    numeral = "".join(map(str, digits))
    value = sum(digit * (base ** power) for digit, power in zip(digits, (3, 2, 1, 0)))
    add = rng.randint(211, 299)
    subtract = rng.randint(31, 97)
    answer = value + add - subtract
    terms = " + ".join(f"{digit}*{base}^{power}" for digit, power in zip(digits, (3, 2, 1, 0)))
    question = (
        f"Interpret {numeral} as a base-{base} numeral. Add {add} to its decimal value and then subtract "
        f"{subtract}. What final decimal integer results?"
    )
    trace = f"Convert first: {terms}={value}. Then {value}+{add}={value + add}; subtract {subtract} to get {answer}."
    return render(question, trace, answer, "base_then_adjust")


def fact_to_chain(rng):
    place, containers, items = rng.choice(TERMS)
    product = rng.randint(7001, 9999)
    added = rng.randint(211, 299)
    multiplier = rng.randint(2, 5)
    answer = (product + added) * multiplier
    question = (
        f"A verified note from a {place} says inventory={product} {items}; the original counting story is unavailable. "
        f"Add {added} replacement {items}, then multiply that updated inventory by {multiplier} for a forecast. "
        "What is the forecast integer?"
    )
    trace = f"Use the verified inventory {product}. Update: {product}+{added}={product + added}. Forecast: {product + added}*{multiplier}={answer}."
    return render(question, trace, answer, "fact_to_chain")


def repair_to_chain(rng):
    left = rng.randint(211, 299)
    right = rng.randint(31, 67)
    offset = rng.randint(101, 197)
    product = left * right
    wrong = product + rng.choice((-83, -47, 29, 61, 113))
    answer = product - offset
    question = (
        f"A draft states that {left} times {right} equals {wrong}, then says to subtract {offset}. "
        "Repair the multiplication independently and give the corrected final integer."
    )
    trace = f"The draft product is wrong: {left}*{right}={product}. Then {product}-{offset}={answer}."
    return render(question, trace, answer, "repair_to_chain")


def source_dropped_named_state(rng):
    first, second = rng.choice((("cedar", "basalt"), ("silver", "amber"), ("north", "south"), ("lilac", "ochre")))
    left = rng.randint(401, 699)
    right = rng.randint(211, 399)
    change = rng.randint(101, 197)
    answer = left + change - right
    question = (
        f"The original record is unavailable. Its compact ledger is {first}={left};{second}={right}. "
        f"Increase only {first} by {change}. What is the resulting difference {first} minus {second}?"
    )
    trace = f"Update only {first}: {left}+{change}={left + change}; keep {second}={right}. Difference {left + change}-{right}={answer}."
    return render(question, trace, answer, "source_dropped_named_state")


BUILDERS = {
    "product_to_chain": product_to_chain,
    "base_then_adjust": base_then_adjust,
    "fact_to_chain": fact_to_chain,
    "repair_to_chain": repair_to_chain,
    "source_dropped_named_state": source_dropped_named_state,
}


def build(per_family, seed):
    if per_family <= 0:
        raise ValueError("per_family must be positive")
    rng = random.Random(seed)
    rows, seen = [], set()
    counts = Counter()
    for family in FAMILIES:
        attempts = 0
        while counts[family] < per_family:
            attempts += 1
            if attempts > per_family * 100:
                raise RuntimeError("could not build enough unique {} rows".format(family))
            row = BUILDERS[family](rng)
            key = normalized_question(row["question"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            counts[family] += 1
    rng.shuffle(rows)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit("refusing to overwrite transfer suite: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    rows = build(args.per_family, args.seed)
    write_jsonl(args.out, rows)
    print(json.dumps({"rows": len(rows), "families": dict(sorted(Counter(row["family"] for row in rows).items()))}, sort_keys=True))


if __name__ == "__main__":
    main()
