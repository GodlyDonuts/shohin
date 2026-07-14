#!/usr/bin/env python3
"""Build a counterfactual operator-binding curriculum and factorized held-out set.

The retained SFT candidate can sometimes emit a correct arithmetic trace, but
its OOD transcripts often copy numbers while replacing an instructed operation
with a different one.  This generator targets that failure directly.  Every
example is solver-derived and teaches an executable trace with two components:

* a canonical operation plan, and
* the state after each operation.

Training includes direct single-problem prompts (the transfer target),
minimal-pair prompts where exactly one operation changes, and wording-pair
prompts where the computation is invariant.  The held-out suite separately
measures wording transfer, value transfer, and their conjunction.  It is a
data candidate only: it does not submit training or change a checkpoint.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from random import Random
import re


WORD = re.compile(r"\w+")
OPS = ("add", "subtract", "multiply", "divide")
TRAIN_TEMPLATES = {
    "add_multiply_subtract": (
        "A register opens with {start}. Credit {a}, apply a factor of {b} to the new balance, and debit {c}.",
        "Begin with t={start}. Combine in {a}; take {b} copies of that total; finally take away {c}.",
    ),
    "subtract_multiply_add": (
        "Open a register r at {start}. Deduct {a}, multiply the remainder by {b}, and finally add {c}.",
        "Let r start at {start}. First subtract {a}, next multiply by {b}, then add {c}.",
    ),
    "double_add_divide": (
        "Let v begin at {start}. Make two copies of v, combine in {a}, and split the total evenly into {b} shares.",
        "Start v={start}. Form twice v, include {a}, and take one exact {b}th of the resulting amount.",
    ),
}
HELDOUT_TEMPLATES = {
    "add_multiply_subtract": (
        "A counter begins at {start}. Raise it by {a}; enlarge that amount {b}-fold; then reduce it by {c}.",
        "Take an accumulator equal to {start}. Put in {a}, apply a factor of {b}, and take away {c}.",
    ),
    "subtract_multiply_add": (
        "An inventory starts at {start}. Withdraw {a}, make the remainder {b} times as large, then restore {c}.",
        "With a balance of {start}, remove {a}, scale what remains by {b}, and append {c}.",
    ),
    "double_add_divide": (
        "A value begins as {start}. Make two copies of it, combine in {a}, then split the total evenly into {b} parts.",
        "For x={start}, form twice x, increase it by {a}, and take one exact {b}th of the result.",
    ),
}


def normalize(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Episode:
    family: str
    start: int
    a: int
    b: int
    c: int
    operations: tuple[str, str, str]
    states: tuple[int, int, int, int]

    @property
    def answer(self) -> int:
        return self.states[-1]


def apply(value: int, operation: str, operand: int) -> int:
    if operation == "add":
        return value + operand
    if operation == "subtract":
        return value - operand
    if operation == "multiply":
        return value * operand
    if operation == "divide":
        if value % operand:
            raise ValueError("division must be exact")
        return value // operand
    raise ValueError("unknown operation")


def episode_for(family: str, rng: Random, heldout_values: bool) -> Episode:
    if heldout_values:
        start_lo, start_hi, add_lo, add_hi, factors = 421, 997, 71, 199, range(8, 13)
    else:
        start_lo, start_hi, add_lo, add_hi, factors = 17, 359, 3, 79, range(2, 10)
    if family == "add_multiply_subtract":
        start = rng.randint(start_lo, start_hi)
        a, b, c = rng.randint(add_lo, add_hi), rng.choice(tuple(factors)), rng.randint(add_lo, add_hi)
        ops = ("add", "multiply", "subtract")
    elif family == "subtract_multiply_add":
        start = rng.randint(start_lo + add_hi, start_hi + add_hi)
        a, b, c = rng.randint(add_lo, add_hi), rng.choice(tuple(factors)), rng.randint(add_lo, add_hi)
        ops = ("subtract", "multiply", "add")
    elif family == "double_add_divide":
        b = rng.choice(tuple(factors))
        a = rng.randint(add_lo, add_hi)
        # Choose the start so (2 * start + a) is exactly divisible by b.
        base = rng.randint(start_lo, start_hi)
        remainder = (2 * base + a) % b
        start = base + ((b - remainder) * pow(2, -1, b) % b) if b % 2 else base
        # For even divisors, select a compatible addend instead of inverting 2.
        if b % 2 == 0:
            a += (-a) % 2
            start = base + ((-((2 * base + a) % b)) // 2) % b
        ops = ("multiply", "add", "divide")
        a, b, c = a, b, 2
        operands = (2, a, b)
        values = [start]
        for operation, operand in zip(ops, operands):
            values.append(apply(values[-1], operation, operand))
        return Episode(family, start, a, b, c, ops, tuple(values))
    else:
        raise ValueError("unknown family")
    operands = (a, b, c)
    values = [start]
    for operation, operand in zip(ops, operands):
        values.append(apply(values[-1], operation, operand))
    return Episode(family, start, a, b, c, ops, tuple(values))


def render_question(episode: Episode, template_pool: dict[str, tuple[str, ...]], variant: int) -> str:
    template = template_pool[episode.family][variant % len(template_pool[episode.family])]
    return template.format(start=episode.start, a=episode.a, b=episode.b, c=episode.c)


def operator_name(operation: str) -> str:
    return {"add": "ADD", "subtract": "SUBTRACT", "multiply": "MULTIPLY", "divide": "DIVIDE"}[operation]


def equation(left: int, operation: str, operand: int, result: int) -> str:
    symbol = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[operation]
    return f"{left} {symbol} {operand} = {result}"


def direct_response(episode: Episode) -> str:
    operands = (episode.a, episode.b, episode.c)
    plan = ",".join(operator_name(operation) for operation in episode.operations)
    transitions = "; ".join(
        equation(episode.states[index], operation, operands[index], episode.states[index + 1])
        for index, operation in enumerate(episode.operations)
    )
    return f"<think>plan={plan}; {transitions}</think>\nThe answer is {episode.answer}."


def direct_row(episode: Episode, question: str, source: str) -> dict:
    prompt = (
        f"Question: {question} Inside <think>, first name the three operations as "
        "plan=OP1,OP2,OP3, then show every exact state transition. "
        "End with 'The answer is <integer>.'.\nAnswer:"
    )
    return {
        "question": prompt,
        "completion_prompt": prompt,
        "response": direct_response(episode),
        "answer": str(episode.answer),
        "source": source,
        "training_group": "operator_trace_contrast",
        "family": episode.family,
        "contract": "direct",
        "operations": list(episode.operations),
        "states": list(episode.states),
    }


def swapped_episode(episode: Episode, swap_index: int) -> Episode:
    ops = list(episode.operations)
    old = ops[swap_index]
    replacements = {"add": "subtract", "subtract": "add"}
    if old not in replacements:
        raise ValueError("minimal pairs only swap an add/subtract operation")
    replacement = replacements[old]
    operands = (episode.a, episode.b, episode.c)
    values = [episode.start]
    for operation, operand in zip((*ops[:swap_index], replacement, *ops[swap_index + 1:]), operands):
        values.append(apply(values[-1], operation, operand))
    ops[swap_index] = replacement
    return Episode(episode.family, episode.start, episode.a, episode.b, episode.c, tuple(ops), tuple(values))


def pair_question(episode: Episode) -> str:
    operands = (episode.a, episode.b, episode.c)
    clauses = [f"{operation} {operand}" for operation, operand in zip(episode.operations, operands)]
    return "Start at {}. Then {}.".format(episode.start, ", then ".join(clauses))


def minimal_pair_row(episode: Episode, question: str) -> dict:
    swap_index = 0
    alternate = swapped_episode(episode, swap_index)
    left_question = question
    right_question = pair_question(alternate)
    response = "<think>Problem A: " + direct_response(episode).split("<think>", 1)[1].split("</think>", 1)[0]
    response += "; Problem B: " + direct_response(alternate).split("<think>", 1)[1].split("</think>", 1)[0]
    response += f"</think>\nThe answers are A={episode.answer}; B={alternate.answer}."
    prompt = (
        "Problem A: " + left_question + "\nProblem B: " + right_question + "\n"
        "The two problems use the same numbers but one operation differs. Inside <think>, write a separate "
        "plan and all exact transitions for A and B. End with 'The answers are A=<integer>; B=<integer>.'.\nAnswer:"
    )
    return {
        "question": prompt,
        "completion_prompt": prompt,
        "response": response,
        "answer": f"A={episode.answer};B={alternate.answer}",
        "source": "operator_trace_contrast_v1_train",
        "training_group": "operator_trace_contrast",
        "family": episode.family,
        "contract": "minimal_pair",
        "operations": list(episode.operations),
        "alternate_operations": list(alternate.operations),
        "states": list(episode.states),
        "alternate_states": list(alternate.states),
        "swap_index": swap_index,
    }


def build_train(per_family: int, seed: int) -> list[dict]:
    rng = Random(seed)
    rows, seen = [], set()
    for family in TRAIN_TEMPLATES:
        attempts = count = 0
        while count < per_family:
            attempts += 1
            if attempts > per_family * 80:
                raise RuntimeError("could not create unique training rows")
            episode = episode_for(family, rng, heldout_values=False)
            for variant in range(2):
                question = render_question(episode, TRAIN_TEMPLATES, variant)
                row = direct_row(episode, question, "operator_trace_contrast_v1_train")
                key = normalize(row["completion_prompt"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
            if family != "double_add_divide":
                pair = minimal_pair_row(episode, pair_question(episode))
                key = normalize(pair["completion_prompt"])
                if key not in seen:
                    seen.add(key)
                    rows.append(pair)
            count += 1
    rng.shuffle(rows)
    return rows


def case_from_episode(episode: Episode, question: str, regime: str, item_id: str) -> dict:
    operands = (episode.a, episode.b, episode.c)
    markers = [[f"after_{index + 1}", state] for index, state in enumerate(episode.states[1:])]
    alternatives = [[re.escape(equation(episode.states[index], operation, operands[index], episode.states[index + 1]))]
                    for index, operation in enumerate(episode.operations)]
    return {
        "id": item_id,
        "family": f"{regime}_{episode.family}",
        "regime": regime,
        "question": (
            question + " Inside <think>, show each exact equation. End with 'The answer is <integer>.'."
        ),
        "answer": episode.answer,
        "markers": markers,
        "alternate_patterns": alternatives,
        "operations": list(episode.operations),
        "states": list(episode.states),
    }


def build_eval(per_regime_family: int, seed: int, train_prompts: set[str]) -> list[dict]:
    rng = Random(seed)
    cases, seen = [], set(train_prompts)
    regimes = (("wording", False, HELDOUT_TEMPLATES), ("value", True, TRAIN_TEMPLATES), ("full", True, HELDOUT_TEMPLATES))
    for regime, high_values, templates in regimes:
        for family in TRAIN_TEMPLATES:
            count = attempts = 0
            while count < per_regime_family:
                attempts += 1
                if attempts > per_regime_family * 100:
                    raise RuntimeError("could not create unique held-out cases")
                episode = episode_for(family, rng, heldout_values=high_values)
                question = render_question(episode, templates, count)
                case = case_from_episode(episode, question, regime, f"{regime}_{family}_{count:03d}")
                key = normalize(case["question"])
                if key in seen:
                    continue
                seen.add(key)
                cases.append(case)
                count += 1
    return cases


def write_jsonl(path: Path, rows: list[dict]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as target:
        for row in rows:
            target.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(partial, path)


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--train-per-family", type=int, default=40_000)
    parser.add_argument("--eval-per-regime-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.train_per_family <= 0 or args.eval_per_regime_family <= 0:
        raise SystemExit("all requested counts must be positive")
    train_path, eval_path, report_path = map(Path, (args.train_out, args.eval_out, args.report_out))
    if any(path.exists() for path in (train_path, eval_path, report_path)):
        raise SystemExit("refusing to overwrite an operator-trace candidate")
    train = build_train(args.train_per_family, args.seed)
    train_prompts = {normalize(row["completion_prompt"]) for row in train}
    cases = build_eval(args.eval_per_regime_family, args.seed + 1, train_prompts)
    eval_prompts = {normalize(case["question"]) for case in cases}
    if train_prompts & eval_prompts:
        raise RuntimeError("exact train/eval prompt overlap")
    write_jsonl(train_path, train)
    write_json(eval_path, {"schema": "operator_trace_contrast_eval_v1", "cases": cases})
    report = {
        "schema": "operator_trace_contrast_v1",
        "seed": args.seed,
        "train_rows": len(train),
        "train_by_family": dict(sorted(collections.Counter(row["family"] for row in train).items())),
        "train_by_contract": dict(sorted(collections.Counter(row["contract"] for row in train).items())),
        "eval_cases": len(cases),
        "eval_by_regime_family": dict(sorted(collections.Counter(case["family"] for case in cases).items())),
        "train_eval_exact_prompt_overlap": 0,
        "train_sha256": sha256(train_path),
        "eval_sha256": sha256(eval_path),
        "claim_boundary": (
            "Data-only candidate. Correct outputs on this suite would demonstrate bounded operator-to-state "
            "transfer, not general reasoning or a flagship-promotion condition."
        ),
    }
    write_json(report_path, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
