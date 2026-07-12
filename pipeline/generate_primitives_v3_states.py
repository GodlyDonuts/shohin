#!/usr/bin/env python3
"""Build a typed, solver-verified compact-state curriculum.

V6 established that a model can imitate free-form state prompts without
producing a usable state.  This source makes the intermediate representation a
small deterministic protocol: the state string, its corruption, and its final
answer are generated together.  Held-out evaluation can therefore require both
an exact valid state and a correct answer.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
from pathlib import Path


FAMILIES = (
    "arithmetic", "base_conversion", "state_update", "sort_unique",
    "string_insert", "syllogism", "correction",
)
CONTRACTS = ("write", "repair", "reuse")
WORDS = ("sequoia", "marigold", "cobalt", "pioneer", "safflower", "vermilion")
INSERTS = ("QR", "uv", "JK", "rs", "LM", "wx")
NOUNS = ("norp", "zelk", "favin", "quor", "bex", "tul", "mep", "sarn")
CONSONANTS = "bcdfghjklmnprstvwxz"
VOWELS = "aeiou"


def normalized(text):
    return re.sub(r"\s+", "", str(text).strip().lower())


def made_up_word(rng, syllables=3):
    return "".join(rng.choice(CONSONANTS) + rng.choice(VOWELS) for _ in range(syllables))


def digits_for_base(rng, base, length):
    digits = [rng.randrange(base) for _ in range(length)]
    digits[0] = rng.randrange(1, base)
    return digits


def base_value(digits, base):
    value = 0
    for digit in digits:
        value = value * base + digit
    return value


def completion(state, answer):
    return f"{state}\nThe answer is {answer}."


def make_case(family, rng, heldout):
    if heldout:
        number_lo, number_hi = 211, 499
        state_lo, state_hi = 121, 399
        list_hi, digits_len = 199, 5
    else:
        number_lo, number_hi = 41, 199
        state_lo, state_hi = 23, 119
        list_hi, digits_len = 99, 4

    if family == "arithmetic":
        left = rng.randint(number_lo, number_hi)
        right = rng.randint(19, 47)
        sub = rng.randint(11, 97)
        product = left * right
        answer = product - sub
        state = f"state=mul:{left}*{right}={product};sub:{product}-{sub}={answer}"
        corrupt = f"state=mul:{left}*{right}={product};sub:{product}-{sub}={answer + 1}"
        question = rng.choice((
            f"Compute {left} times {right}, then subtract {sub}. Return only the final integer.",
            f"First multiply {left} by {right}. Then subtract {sub}. What is the result? Give only the integer.",
        ))
        template = "state=mul:A*B=P;sub:P-C=R"
    elif family == "base_conversion":
        base = rng.randint(5, 9)
        digits = digits_for_base(rng, base, digits_len)
        numeral = "".join(str(digit) for digit in digits)
        answer = base_value(digits, base)
        state = f"state=base:{base};digits:{numeral};value:{answer}"
        corrupt = f"state=base:{base};digits:{numeral};value:{answer + 1}"
        question = rng.choice((
            f"Convert the base-{base} numeral {numeral} to base 10. Return only the integer.",
            f"What decimal number is represented by {numeral} in base {base}? Give only the integer.",
        ))
        template = "state=base:B;digits:D;value:V"
    elif family in {"state_update", "correction"}:
        start = rng.randint(state_lo, state_hi)
        add = rng.randint(13, 61)
        mult = rng.randint(2, 7)
        sub = rng.randint(11, 83)
        after_add = start + add
        after_mult = after_add * mult
        answer = after_mult - sub
        state = f"state=n:{start}>{after_add}>{after_mult}>{answer}"
        corrupt = f"state=n:{start}>{after_add}>{after_mult}>{answer + 1}"
        if family == "state_update":
            question = rng.choice((
                f"Start with n = {start}. Add {add}, multiply by {mult}, then subtract {sub}. What is n? Return only the integer.",
                f"Let n begin at {start}. Increase it by {add}, multiply by {mult}, and decrease it by {sub}. Give only final n.",
            ))
        else:
            wrong = answer + rng.choice((-19, -11, 7, 13, 29))
            question = (
                f"A prior answer says this is {wrong}: start with n = {start}, add {add}, multiply by {mult}, "
                f"then subtract {sub}. Recompute independently and return only the corrected final integer."
            )
        template = "state=n:S>A>M>R"
    elif family == "sort_unique":
        values = [rng.randint(0, list_hi) for _ in range(8)]
        ordered = sorted(set(values))
        encoded = ",".join(str(value) for value in ordered)
        answer = f"[{encoded}]"
        state = f"state=unique:{encoded};sorted:{encoded}"
        corrupt = f"state=unique:{encoded};sorted:{','.join(str(value) for value in reversed(ordered))}"
        question = rng.choice((
            f"Sort {values} in ascending order and remove duplicates. Return only the resulting list.",
            f"Remove repeated values from {values}, then order the remaining values from low to high. Give only a list.",
        ))
        template = "state=unique:U;sorted:O"
    elif family == "string_insert":
        word = rng.choice(WORDS) + made_up_word(rng, syllables=2)
        insert = rng.choice(INSERTS)
        position = rng.randint(2, len(word) - 2)
        prefix, suffix = word[:position], word[position:]
        answer = prefix + insert + suffix
        state = f"state=split:{prefix}|{suffix};insert:{insert};result:{answer}"
        corrupt = f"state=split:{prefix}|{suffix};insert:{insert};result:{answer[::-1]}"
        question = rng.choice((
            f"Put '{insert}' immediately after character {position} of '{word}'. Return only the resulting string.",
            f"Split '{word}' after its first {position} characters, insert '{insert}' there, and give only the new string.",
        ))
        template = "state=split:P|S;insert:I;result:R"
    elif family == "syllogism":
        names = set()
        while len(names) < 3:
            names.add(rng.choice(NOUNS) + made_up_word(rng, syllables=2))
        subject, middle, target = sorted(names)
        if rng.randrange(2):
            answer = "yes"
            state = f"state={subject}>{middle};{middle}>{target};answer:yes"
            corrupt = f"state={subject}>{middle};{middle}>{target};answer:no"
            question = (
                f"Every {subject} is a {middle}. Every {middle} is a {target}. "
                f"Can a {subject} be a {target}? Answer yes or no only."
            )
        else:
            answer = "no"
            state = f"state={subject}>{middle};{middle}!{target};answer:no"
            corrupt = f"state={subject}>{middle};{middle}!{target};answer:yes"
            question = (
                f"Every {subject} is a {middle}. No {middle} is a {target}. "
                f"Can a {subject} be a {target}? Answer yes or no only."
            )
        template = "state=A>B;B>C-or-B!C;answer:yes-or-no"
    else:
        raise ValueError(f"unknown family: {family}")
    return {
        "family": family,
        "question": question,
        "answer": str(answer),
        "state": state,
        "corrupt_state": corrupt,
        "template": template,
    }


def render(case, contract):
    question, answer, state = case["question"], case["answer"], case["state"]
    if contract == "write":
        prompt = (
            f"Question: {question}\nWrite exactly one compact state line using "
            f"the template `{case['template']}`, then on a new line write "
            "'The answer is <final answer>.'.\nAnswer:"
        )
        return prompt, completion(state, answer)
    if contract == "repair":
        prompt = (
            f"Question: {question}\nThis proposed state may be wrong:\n{case['corrupt_state']}\n\n"
            f"Independently recompute. Write exactly one corrected state line using `{case['template']}`, "
            "then on a new line write 'The answer is <final answer>.'.\nAnswer:"
        )
        return prompt, completion(state, answer)
    if contract == "reuse":
        prompt = (
            f"Question: {question}\nA verified compact state is:\n{state}\n\n"
            "Use that state. Return only 'The answer is <final answer>.'.\nAnswer:"
        )
        return prompt, f"The answer is {answer}."
    raise ValueError(f"unknown contract: {contract}")


def build_rows(per_family, seed, heldout, excluded_questions=None):
    rng = random.Random(seed)
    rows = []
    seen = set(excluded_questions or ())
    for family in FAMILIES:
        count, attempts = 0, 0
        while count < per_family:
            attempts += 1
            if attempts > per_family * 30:
                raise RuntimeError(f"could not generate enough unique {family} rows")
            case = make_case(family, rng, heldout)
            question_key = normalized(case["question"])
            if question_key in seen:
                continue
            seen.add(question_key)
            for contract in CONTRACTS:
                prompt, response = render(case, contract)
                rows.append({
                    "question": prompt,
                    "completion_prompt": prompt,
                    "response": response,
                    "answer": case["answer"],
                    "expected_state": case["state"],
                    "source_question": case["question"],
                    "source": "primitives_v3_state_heldout" if heldout else "primitives_v3_state_train",
                    "training_group": "state_protocol",
                    "family": family,
                    "contract": contract,
                    "source_index": count,
                })
            count += 1
    rng.shuffle(rows)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    os.replace(partial, path)


def report(rows):
    return {
        "rows": len(rows),
        "families": dict(sorted(collections.Counter(row["family"] for row in rows).items())),
        "contracts": dict(sorted(collections.Counter(row["contract"] for row in rows).items())),
        "prompt_sha256": hashlib.sha256(
            "\n".join(sorted(row["completion_prompt"] for row in rows)).encode()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--train-per-family", type=int, default=15000)
    parser.add_argument("--eval-per-family", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.train_per_family <= 0 or args.eval_per_family <= 0:
        raise ValueError("per-family counts must be positive")
    train = build_rows(args.train_per_family, args.seed, heldout=False)
    heldout = build_rows(
        args.eval_per_family, args.seed + 1, heldout=True,
        excluded_questions={normalized(row["source_question"]) for row in train},
    )
    train_source = {normalized(row["source_question"]) for row in train}
    heldout_source = {normalized(row["source_question"]) for row in heldout}
    if train_source & heldout_source:
        raise RuntimeError("train and heldout source prompts overlap")
    train_prompts = {normalized(row["completion_prompt"]) for row in train}
    heldout_prompts = {normalized(row["completion_prompt"]) for row in heldout}
    if train_prompts & heldout_prompts:
        raise RuntimeError("rendered train and heldout prompts overlap")
    write_jsonl(args.train_out, train)
    write_jsonl(args.eval_out, heldout)
    print(json.dumps({
        "train": report(train),
        "heldout": report(heldout),
        "train_heldout_prompt_overlap": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
