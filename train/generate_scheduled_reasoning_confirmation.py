#!/usr/bin/env python3
"""Generate the frozen 256-case Source-Scheduled Reasoning confirmation board."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


SCHEMA = "source_scheduled_reasoning_confirmation_v1"
SEED = 2026071502
PER_FAMILY = 64
FAMILIES = (
    "multiply_subtract",
    "base_conversion",
    "sequential_state",
    "modular_update",
)
EXPECTED_CASES_SHA256 = (
    "4afc6c4b0c271ea2f723078ab183e8d1ac1851fd1728898384ef52275887b0e4"
)
EXPECTED_BOARD_SHA256 = (
    "19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474"
)


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )


def digest_rows(rows):
    return hashlib.sha256(canonical_bytes(rows)).hexdigest()


def build_rows():
    rng = random.Random(SEED)
    rows = []

    for index in range(PER_FAMILY):
        a = rng.randint(20, 99)
        multiplier = rng.randint(2, 9) if index < 32 else rng.randint(10, 19)
        product = a * multiplier
        subtractor = rng.randint(1, min(50, product - 1))
        rows.append(
            {
                "id": f"multiply_subtract_{index:03d}",
                "family": "multiply_subtract",
                "question": f"Compute {a} times {multiplier}, then subtract {subtractor}.",
                "initial_state": a,
                "schedule": [["multiply", multiplier], ["subtract", subtractor]],
                "answer": product - subtractor,
                "stratum": "small_multiplier" if index < 32 else "two_digit_multiplier",
            }
        )

    for index in range(PER_FAMILY):
        base = rng.randint(2, 9) if index < 32 else rng.randint(10, 12)
        max_digit = min(base - 1, 9)
        digits = [
            rng.randint(1, max_digit),
            rng.randint(0, max_digit),
            rng.randint(0, max_digit),
        ]
        numeral = "".join(map(str, digits))
        rows.append(
            {
                "id": f"base_conversion_{index:03d}",
                "family": "base_conversion",
                "question": f"Convert the base-{base} numeral {numeral} to base 10.",
                "initial_state": digits[0],
                "schedule": [
                    ["multiply", base],
                    ["add", digits[1]],
                    ["multiply", base],
                    ["add", digits[2]],
                ],
                "answer": digits[0] * base * base + digits[1] * base + digits[2],
                "stratum": "base_2_9" if index < 32 else "base_10_12",
            }
        )

    for index in range(PER_FAMILY):
        start = rng.randint(5, 50)
        addend = rng.randint(1, 25)
        multiplier = rng.randint(2, 5) if index < 32 else rng.randint(6, 7)
        before_subtract = (start + addend) * multiplier
        subtractor = rng.randint(1, min(40, before_subtract - 1))
        rows.append(
            {
                "id": f"sequential_state_{index:03d}",
                "family": "sequential_state",
                "question": (
                    f"Start at {start}, add {addend}, multiply by {multiplier}, "
                    f"then subtract {subtractor}."
                ),
                "initial_state": start,
                "schedule": [
                    ["add", addend],
                    ["multiply", multiplier],
                    ["subtract", subtractor],
                ],
                "answer": before_subtract - subtractor,
                "stratum": "multiplier_2_5" if index < 32 else "multiplier_6_7",
            }
        )

    for index in range(PER_FAMILY):
        left = rng.randint(10, 99)
        right = rng.randint(10, 99)
        modulus = rng.randint(3, 14) if index < 32 else rng.randint(15, 25)
        rows.append(
            {
                "id": f"modular_update_{index:03d}",
                "family": "modular_update",
                "question": (
                    f"Add {left} and {right}, then give the remainder after division by {modulus}."
                ),
                "initial_state": left,
                "schedule": [["add", right], ["remainder", modulus]],
                "answer": (left + right) % modulus,
                "stratum": "modulus_3_14" if index < 32 else "modulus_15_25",
            }
        )

    questions = [row["question"] for row in rows]
    if len(rows) != len(FAMILIES) * PER_FAMILY or len(set(questions)) != len(questions):
        raise RuntimeError("frozen board is incomplete or contains duplicate questions")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = build_rows()
    cases_sha256 = digest_rows(rows)
    if cases_sha256 != EXPECTED_CASES_SHA256:
        raise RuntimeError("generator no longer reproduces the frozen cases hash")
    result = {
        "schema": SCHEMA,
        "seed": SEED,
        "per_family": PER_FAMILY,
        "case_count": len(rows),
        "family_order": list(FAMILIES),
        "cases_sha256": cases_sha256,
        "rows": rows,
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("ascii")
    if hashlib.sha256(payload).hexdigest() != EXPECTED_BOARD_SHA256:
        raise RuntimeError("generator no longer reproduces the frozen board artifact")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(payload)
        sink.flush()
        os.fsync(sink.fileno())
        os.fchmod(sink.fileno(), 0o444)
    print(
        json.dumps(
            {
                "case_count": result["case_count"],
                "cases_sha256": result["cases_sha256"],
                "board_sha256": EXPECTED_BOARD_SHA256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
