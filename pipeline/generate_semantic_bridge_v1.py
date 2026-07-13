#!/usr/bin/env python3
"""Generate a solver-verified semantic bridge curriculum for small-model reasoning.

The prior primitive and working-memory experiments establish that Shohin can
imitate a narrow format without transferring it to ordinary questions.  This
generator instead binds short, natural-language problems to explicit verified
intermediates in three useful modes:

* solve a word problem through its intermediate calculation,
* continue from a correct compact fact supplied in natural language, and
* repair a plausible but wrong calculation before answering.

Every numeric field is computed in this program.  Train and held-out splits use
disjoint value bands and distinct question templates.  This produces an
isolated SFT candidate only; separate decontamination, packing, and capability
gates are required before it can be mixed or trained.
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path


WORD = re.compile(r"\w+")
FAMILIES = (
    "product_adjust",
    "state_chain",
    "base_conversion",
    "fact_continue",
    "trace_repair",
)

TRAIN_TERMS = (
    ("workshop", "boxes", "gears"),
    ("bakery", "trays", "pastries"),
    ("library", "shelves", "books"),
    ("garden", "rows", "seedlings"),
    ("factory", "bins", "bolts"),
    ("school", "tables", "markers"),
)
HELDOUT_TERMS = (
    ("harbor", "crates", "lanterns"),
    ("museum", "cabinets", "tokens"),
    ("theater", "racks", "costumes"),
    ("clinic", "carts", "bandages"),
    ("observatory", "cases", "lenses"),
    ("archive", "drawers", "maps"),
)


def normalized_question(text):
    return " ".join(WORD.findall(text.lower()))


def response(trace, answer):
    return f"<think>{trace}</think>\nThe answer is {answer}."


def row(question, trace, answer, family, mode):
    return {
        "question": question,
        "response": response(trace, answer),
        "answer": str(answer),
        "source": "semantic_bridge_v1_train",
        "training_group": "semantic_bridge",
        "family": family,
        "mode": mode,
    }


def settings(heldout):
    if heldout:
        return {
            "a": (101, 199), "b": (27, 59), "delta": (31, 67), "base": (5, 9),
            "state": (101, 199), "add": (31, 67), "mult": (3, 9), "sub": (29, 73),
            "terms": HELDOUT_TERMS,
        }
    return {
        "a": (11, 79), "b": (4, 24), "delta": (2, 29), "base": (5, 8),
        "state": (7, 79), "add": (2, 25), "mult": (2, 7), "sub": (2, 29),
        "terms": TRAIN_TERMS,
    }


def product_adjust(rng, heldout):
    s = settings(heldout)
    place, containers, items = rng.choice(s["terms"])
    a = rng.randint(*s["a"])
    b = rng.randint(*s["b"])
    delta = rng.randint(*s["delta"])
    product = a * b
    if rng.randrange(2):
        final = product - delta
        change = f"{delta} {items} are removed"
        operation = f"{product} - {delta} = {final}"
    else:
        final = product + delta
        change = f"{delta} spare {items} are added"
        operation = f"{product} + {delta} = {final}"
    if heldout:
        question = (
            f"At a {place}, each of {a} {containers} holds {b} {items}. Later, {change}. "
            "How many items are there now?"
        )
    else:
        question = rng.choice((
            f"A {place} fills {a} {containers} with {b} {items} each. Then {change}. How many {items} remain?",
            f"There are {a} {containers} of {b} {items} at the {place}; afterward {change}. Give the final count.",
        ))
    trace = f"First multiply: {a} * {b} = {product}. Then apply the change: {operation}."
    return row(question, trace, final, "product_adjust", "solve")


def state_chain(rng, heldout):
    s = settings(heldout)
    start = rng.randint(*s["state"])
    add = rng.randint(*s["add"])
    mult = rng.randint(*s["mult"])
    sub = rng.randint(*s["sub"])
    after_add = start + add
    after_mult = after_add * mult
    final = after_mult - sub
    if heldout:
        question = (
            f"A counter begins at {start}. Increase it by {add}; scale that result by {mult}; "
            f"then reduce it by {sub}. What value does the counter finish with?"
        )
    else:
        question = rng.choice((
            f"Start with n={start}. Add {add}, multiply by {mult}, then subtract {sub}. What is n?",
            f"Let n begin at {start}. First add {add}; next multiply by {mult}; finally remove {sub}. Give final n.",
        ))
    trace = (
        f"After addition, n={start}+{add}={after_add}. "
        f"After multiplication, n={after_add}*{mult}={after_mult}. "
        f"After subtraction, n={after_mult}-{sub}={final}."
    )
    return row(question, trace, final, "state_chain", "solve")


def base_conversion(rng, heldout):
    s = settings(heldout)
    base = rng.randint(*s["base"])
    length = rng.randint(3, 5)
    digits = [rng.randint(1, base - 1)] + [rng.randint(0, base - 1) for _ in range(length - 1)]
    numeral = "".join(map(str, digits))
    powers = list(range(length - 1, -1, -1))
    contributions = [digit * (base ** power) for digit, power in zip(digits, powers)]
    final = sum(contributions)
    if heldout:
        question = f"Read {numeral} as a numeral written in base {base}. What is its decimal value?"
    else:
        question = rng.choice((
            f"Convert {numeral} from base {base} into base 10.",
            f"What decimal integer is represented by the base-{base} numeral {numeral}?",
        ))
    terms = ", ".join(
        f"{digit}*{base}^{power}={value}"
        for digit, power, value in zip(digits, powers, contributions)
    )
    trace = f"The place values are {terms}. Their sum is {'+'.join(map(str, contributions))}={final}."
    return row(question, trace, final, "base_conversion", "solve")


def fact_continue(rng, heldout):
    s = settings(heldout)
    place, containers, items = rng.choice(s["terms"])
    a = rng.randint(*s["a"])
    b = rng.randint(*s["b"])
    delta = rng.randint(*s["delta"])
    product = a * b
    final = product - delta
    if heldout:
        question = (
            f"A verified inventory note for a {place} states that {a} {containers} of {b} {items} each "
            f"make product={product}. Then {delta} {items} are removed. Use the verified note to give the final count."
        )
    else:
        question = rng.choice((
            f"Verified fact: {a} groups of {b} {items} give product={product}. If {delta} {items} are removed, what remains?",
            f"Use this checked state: product={product} for the {place} inventory. Subtract {delta} {items}. Give the final count.",
        ))
    trace = f"Use the verified product={product}. Subtracting gives {product}-{delta}={final}."
    return row(question, trace, final, "fact_continue", "continue")


def trace_repair(rng, heldout):
    s = settings(heldout)
    a = rng.randint(*s["a"])
    b = rng.randint(*s["b"])
    delta = rng.randint(*s["delta"])
    product = a * b
    wrong = product + rng.choice((-13, -7, 5, 11, 17))
    final = product + delta
    if heldout:
        question = (
            f"A draft claims {a} times {b} gives product={wrong}, then adds {delta}. "
            "Check the draft independently and give the corrected final integer."
        )
    else:
        question = rng.choice((
            f"A previous solution says {a}*{b}={wrong}. Correct that calculation, then add {delta}. What is the answer?",
            f"Repair this computation: it used product={wrong} for {a} times {b}, followed by +{delta}. Return the corrected result.",
        ))
    trace = f"The draft product is wrong: {a}*{b}={product}. Then {product}+{delta}={final}."
    return row(question, trace, final, "trace_repair", "repair")


BUILDERS = {
    "product_adjust": product_adjust,
    "state_chain": state_chain,
    "base_conversion": base_conversion,
    "fact_continue": fact_continue,
    "trace_repair": trace_repair,
}


def build_split(per_family, seed, heldout, excluded=None):
    if per_family <= 0:
        raise ValueError("per_family must be positive")
    rng = random.Random(seed)
    excluded = set(excluded or ())
    rows, seen = [], set()
    counts = Counter()
    for family in FAMILIES:
        attempts = 0
        while counts[family] < per_family:
            attempts += 1
            if attempts > per_family * 80:
                raise RuntimeError(f"could not generate enough unique rows for {family}")
            candidate = BUILDERS[family](rng, heldout)
            key = normalized_question(candidate["question"])
            if key in seen or key in excluded:
                continue
            seen.add(key)
            rows.append(candidate)
            counts[family] += 1
    rng.shuffle(rows)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite bridge artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(temporary, "w") as output:
        for item in rows:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def summary(rows):
    return {
        "rows": len(rows),
        "families": dict(sorted(Counter(item["family"] for item in rows).items())),
        "modes": dict(sorted(Counter(item["mode"] for item in rows).items())),
        "all_have_think": all(item["response"].startswith("<think>") for item in rows),
        "all_have_final": all("The answer is" in item["response"] for item in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--train-per-family", type=int, default=40_000)
    parser.add_argument("--eval-per-family", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    train = build_split(args.train_per_family, args.seed, heldout=False)
    heldout = build_split(
        args.eval_per_family,
        args.seed + 1,
        heldout=True,
        excluded={normalized_question(item["question"]) for item in train},
    )
    train_keys = {normalized_question(item["question"]) for item in train}
    heldout_keys = {normalized_question(item["question"]) for item in heldout}
    if train_keys & heldout_keys:
        raise RuntimeError("train/held-out normalized prompt overlap")
    write_jsonl(args.train_out, train)
    write_jsonl(args.eval_out, heldout)
    print(json.dumps({
        "schema": "semantic_bridge_v1",
        "train": summary(train),
        "heldout": summary(heldout),
        "normalized_prompt_overlap": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
