#!/usr/bin/env python3
"""Generate a verified, train/held-out curriculum for reasoning primitives.

This deliberately fills operations that the first broad procedural mix scarcely
covered: compact arithmetic/state transitions, base conversion, list and string
transforms, syllogisms, and correction after a wrong candidate answer.  Every
answer is calculated by this script, and the evaluation split uses disjoint
seeds and value ranges.  It is an isolated SFT ablation source, never a public
benchmark or live-pretraining shard.
"""
import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path


WORDS = ["sequoia", "marigold", "cobalt", "pioneer", "safflower", "vermilion"]
INSERTS = ["QR", "uv", "JK", "rs", "LM", "wx"]
NOUNS = ["norp", "zelk", "favin", "quor", "bex", "tul", "mep", "sarn"]
CONSONANTS = "bcdfghjklmnprstvwxz"
VOWELS = "aeiou"


def answer_response(reasoning, answer):
    return f"<think>{reasoning}</think>\nThe answer is {answer}."


def verified_row(question, reasoning, answer):
    """Keep the SFT completion and held-out exact-answer fields in one place."""
    return {
        "question": question,
        "response": answer_response(reasoning, answer),
        "answer": str(answer),
    }


def digits_for_base(rng, base, length):
    digits = [rng.randrange(base) for _ in range(length)]
    digits[0] = rng.randrange(1, base)
    return digits


def made_up_word(rng, syllables=4):
    """Generate many short ASCII strings without relying on a fixed word list."""
    return "".join(rng.choice(CONSONANTS) + rng.choice(VOWELS) for _ in range(syllables))


def numeral_value(digits, base):
    value = 0
    for digit in digits:
        value = value * base + digit
    return value


def make_case(family, rng, heldout):
    """Return a source-verified supervision row for one compact operation."""
    if heldout:
        lo, hi = 211, 499
        state_lo, state_hi = 121, 399
        list_hi = 199
        base_length = 5
    else:
        lo, hi = 41, 199
        state_lo, state_hi = 23, 119
        list_hi = 99
        base_length = 5

    if family == "arithmetic":
        a, b, c = rng.randint(lo, hi), rng.randint(19, 47), rng.randint(11, 97)
        value = a * b + c
        question = rng.choice([
            f"Compute {a} times {b}, then add {c}. Return only the final integer.",
            f"First multiply {a} by {b}. Then add {c}. What is the result? Give only the integer.",
        ])
        return verified_row(f"{question}", f"{a} times {b} is {a * b}; {a * b} plus {c} is {value}", value)

    if family == "base_conversion":
        base = rng.randint(5, 9)
        digits = digits_for_base(rng, base, base_length)
        numeral = "".join(str(digit) for digit in digits)
        value = numeral_value(digits, base)
        expanded = " + ".join(
            f"{digit} times {base}^{power}"
            for power, digit in zip(range(base_length - 1, -1, -1), digits)
        )
        question = rng.choice([
            f"Convert the base-{base} numeral {numeral} to base 10. Return only the integer.",
            f"What decimal number is represented by {numeral} in base {base}? Give only the integer.",
        ])
        return verified_row(question, f"Use place values: {expanded} = {value}", value)

    if family == "state_update":
        start = rng.randint(state_lo, state_hi)
        add = rng.randint(13, 61)
        mult = rng.randint(2, 7)
        sub = rng.randint(11, 83)
        after_add = start + add
        after_mult = after_add * mult
        value = after_mult - sub
        question = rng.choice([
            f"Start with n = {start}. Add {add}, multiply by {mult}, then subtract {sub}. What is n? Return only the integer.",
            f"Let n begin at {start}. Increase it by {add}, multiply the result by {mult}, and decrease it by {sub}. Give only final n.",
        ])
        return verified_row(
            question,
            f"After adding, n is {after_add}; after multiplying, n is {after_mult}; subtracting gives {value}",
            value,
        )

    if family == "sort_unique":
        values = [rng.randint(0, list_hi) for _ in range(8)]
        answer = sorted(set(values))
        question = rng.choice([
            f"Sort {values} in ascending order and remove duplicates. Return only the resulting list.",
            f"Remove repeated values from {values}, then order the remaining values from low to high. Give only a list.",
        ])
        return verified_row(question, f"The distinct values in ascending order are {answer}", answer)

    if family == "string_insert":
        word = rng.choice(WORDS) + made_up_word(rng, syllables=2)
        insert = rng.choice(INSERTS)
        position = rng.randint(2, len(word) - 2)
        answer = word[:position] + insert + word[position:]
        question = rng.choice([
            f"Put '{insert}' immediately after character {position} of '{word}'. Return only the resulting string.",
            f"Split '{word}' after its first {position} characters, insert '{insert}' there, and give only the new string.",
        ])
        return verified_row(
            question,
            f"The prefix is '{word[:position]}' and the suffix is '{word[position:]}', so joining them gives {answer}",
            answer,
        )

    if family == "syllogism":
        names = set()
        while len(names) < 3:
            names.add(rng.choice(NOUNS) + made_up_word(rng, syllables=2))
        subject, middle, target = sorted(names)
        if rng.randrange(2):
            question = (
                f"Every {subject} is a {middle}. Every {middle} is a {target}. "
                f"Can a {subject} be a {target}? Answer yes or no only."
            )
            answer = "yes"
            reasoning = f"A {subject} must be a {middle}, and every {middle} can be a {target}; the answer is yes"
        else:
            question = (
                f"Every {subject} is a {middle}. No {middle} is a {target}. "
                f"Can a {subject} be a {target}? Answer yes or no only."
            )
            answer = "no"
            reasoning = f"Any {subject} is a {middle}, while no {middle} is a {target}; the answer is no"
        return verified_row(question, reasoning, answer)

    if family == "correction":
        start = rng.randint(state_lo, state_hi)
        add = rng.randint(13, 61)
        mult = rng.randint(2, 7)
        sub = rng.randint(11, 83)
        after_add = start + add
        after_mult = after_add * mult
        value = after_mult - sub
        wrong = value + rng.choice([-19, -11, 7, 13, 29])
        question = (
            f"A prior answer says this is {wrong}: start with n = {start}, add {add}, multiply by {mult}, "
            f"then subtract {sub}. Recompute independently and return only the corrected final integer."
        )
        return verified_row(
            question,
            f"The prior value is wrong. {start} plus {add} is {after_add}; times {mult} is {after_mult}; minus {sub} is {value}",
            value,
        )
    raise ValueError(f"unknown family: {family}")


def build_split(n_per_family, seed, heldout, excluded_questions=None):
    families = ["arithmetic", "base_conversion", "state_update", "sort_unique", "string_insert", "syllogism", "correction"]
    rng = random.Random(seed)
    rows, questions = [], set()
    excluded_questions = excluded_questions or set()
    counts = Counter()
    for family in families:
        attempts = 0
        while counts[family] < n_per_family:
            attempts += 1
            if attempts > n_per_family * 30:
                raise RuntimeError(f"could not generate enough unique {family} rows")
            row = make_case(family, rng, heldout)
            if row["question"] in questions or row["question"] in excluded_questions:
                continue
            questions.add(row["question"])
            rows.append({
                **row,
                "source": "primitives_v1_heldout" if heldout else "primitives_v1_train",
                "training_group": "primitives",
                "family": family,
            })
            counts[family] += 1
    rng.shuffle(rows)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    if path.exists() or tmp.exists():
        raise SystemExit(f"refusing to overwrite primitive curriculum artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--eval-out", required=True)
    ap.add_argument("--train-per-family", type=int, default=30_000)
    ap.add_argument("--eval-per-family", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260712)
    args = ap.parse_args()
    if args.train_per_family <= 0 or args.eval_per_family <= 0:
        raise ValueError("split sizes must be positive")

    train = build_split(args.train_per_family, args.seed, heldout=False)
    heldout = build_split(
        args.eval_per_family, args.seed + 1, heldout=True,
        excluded_questions={row["question"] for row in train},
    )
    train_questions = {row["question"] for row in train}
    eval_questions = {row["question"] for row in heldout}
    if train_questions & eval_questions:
        raise RuntimeError("train/held-out primitive prompt overlap")
    write_jsonl(args.train_out, train)
    write_jsonl(args.eval_out, heldout)
    print(json.dumps({
        "train_out": args.train_out,
        "eval_out": args.eval_out,
        "train_rows": len(train),
        "eval_rows": len(heldout),
        "train_families": dict(Counter(row["family"] for row in train)),
        "eval_families": dict(Counter(row["family"] for row in heldout)),
        "prompt_overlap": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
