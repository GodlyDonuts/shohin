#!/usr/bin/env python3
"""Build fresh, verifier-derived state-trace cases after a candidate is frozen."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def add_multiply_subtract(rng, band, index):
    if band == "mid":
        start, add, multiplier, subtract = (
            rng.randint(121, 399), rng.randint(13, 61), rng.randint(2, 7), rng.randint(11, 83)
        )
    else:
        start, add, multiplier, subtract = (
            rng.randint(401, 899), rng.randint(63, 127), rng.randint(8, 12), rng.randint(91, 179)
        )
    after_add = start + add
    after_multiply = after_add * multiplier
    answer = after_multiply - subtract
    return {
        "id": f"{band}_add_multiply_subtract_{index}",
        "family": f"{band}_add_multiply_subtract",
        "question": (
            f"Set a running total t to {start}. Increase t by {add}, scale the updated total by {multiplier}, "
            f"then remove {subtract}. In <think> emit after_add=<integer> and after_multiply=<integer>. "
            "End with 'The answer is <integer>.'."
        ),
        "answer": answer,
        "markers": [["after_add", after_add], ["after_multiply", after_multiply]],
    }


def subtract_multiply_add(rng, band, index):
    if band == "mid":
        start, subtract, multiplier, add = (
            rng.randint(171, 499), rng.randint(11, 83), rng.randint(2, 7), rng.randint(13, 61)
        )
    else:
        start, subtract, multiplier, add = (
            rng.randint(501, 999), rng.randint(91, 179), rng.randint(8, 12), rng.randint(63, 127)
        )
    after_subtract = start - subtract
    after_multiply = after_subtract * multiplier
    answer = after_multiply + add
    return {
        "id": f"{band}_subtract_multiply_add_{index}",
        "family": f"{band}_subtract_multiply_add",
        "question": (
            f"Open a register r at {start}. Deduct {subtract}, multiply the remainder by {multiplier}, "
            f"and finally add {add}. In <think> write after_subtract=<integer> and after_multiply=<integer>. "
            "Finish with 'The answer is <integer>.'."
        ),
        "answer": answer,
        "markers": [["after_subtract", after_subtract], ["after_multiply", after_multiply]],
    }


def double_add_divide(rng, band, index):
    if band == "mid":
        start_range, add_range, divisor_range = (121, 399), (13, 127), (2, 7)
    else:
        start_range, add_range, divisor_range = (401, 899), (101, 359), (3, 11)
    for _ in range(1_000):
        start = rng.randint(*start_range)
        add = rng.randint(*add_range)
        divisor = rng.randint(*divisor_range)
        after_double = 2 * start
        after_add = after_double + add
        if after_add % divisor == 0:
            return {
                "id": f"{band}_double_add_divide_{index}",
                "family": f"{band}_double_add_divide",
                "question": (
                    f"Initialize v as {start}. Double v, add {add}, then divide the result by {divisor}; "
                    "the division is exact. In <think> state after_double=<integer> and after_add=<integer>. "
                    "Conclude with 'The answer is <integer>.'."
                ),
                "answer": after_add // divisor,
                "markers": [["after_double", after_double], ["after_add", after_add]],
            }
    raise RuntimeError("could not construct an exact division case")


def build_cases(seed, per_template):
    rng = random.Random(seed)
    cases = []
    builders = (add_multiply_subtract, subtract_multiply_add, double_add_divide)
    for band in ("mid", "high"):
        for builder in builders:
            cases.extend(builder(rng, band, index) for index in range(per_template))
    questions = [case["question"] for case in cases]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate generated questions")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=2026071401)
    parser.add_argument("--per-template", type=int, default=12)
    args = parser.parse_args()
    if args.per_template <= 0:
        raise ValueError("--per-template must be positive")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output")
    cases = build_cases(args.seed, args.per_template)
    payload = {
        "audit": "trace_state_ood_v1",
        "seed": args.seed,
        "per_template": args.per_template,
        "cases": cases,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "cases": len(cases)}, sort_keys=True))


if __name__ == "__main__":
    main()
