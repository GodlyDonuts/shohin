#!/usr/bin/env python3
"""Generate a coverage-complete DRS transition-basis candidate.

Unlike v2's magnitude-banded episodes, v3 constructs complete arithmetic
episodes whose designated transition covers every reachable tuple of width,
operation, digit position, carry/borrow, and operand digits.  Full result
traces remain solver-derived, so the model must preserve the actual recurrent
tape rather than learn a disconnected truth table.

This generator only creates a candidate.  It does not submit an SFT or touch
the flagship path.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "pipeline"))
from digitwise_basis_protocol import WIDTHS, context_label, local_context, reachable_contexts
from digitwise_protocol import apply_microstep, initial_state
from generate_digitwise_recurrent_v1 import (
    counterfactual_episode,
    deduplicate_rows,
    episode_from_operands,
    episode_prompts,
    episode_signature,
    normalized,
    rows_from_episode,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digits_to_value(digits):
    return sum(int(digit) * (10 ** index) for index, digit in enumerate(digits))


def _random_digits(rng, width):
    return [rng.randrange(10) for _ in range(width)]


def operands_for_context(context, rng):
    """Construct a full non-negative arithmetic episode hitting one local state."""
    width, operation, position, carry, left_digit, right_digit = context
    top = width - 1
    left, right = _random_digits(rng, width), _random_digits(rng, width)
    left[position], right[position] = left_digit, right_digit

    if position:
        # The immediate lower position fixes the carry/borrow into `position`;
        # farther-lower digits may vary without changing that outgoing bit.
        lower = position - 1
        if operation == "add":
            if carry:
                left[lower], right[lower] = 9, 9
            else:
                left[lower], right[lower] = 0, 0
            for index in range(lower):
                left[index], right[index] = rng.randrange(10), rng.randrange(10)
        elif carry:
            left[lower], right[lower] = 0, 1
            for index in range(lower):
                left[index], right[index] = rng.randrange(10), rng.randrange(10)
        else:
            # Equal lower digits guarantee no borrow reaches `position`.
            for index in range(lower + 1):
                digit = rng.randrange(10)
                left[index] = right[index] = digit

    if operation == "sub":
        if position < top:
            # This higher digit keeps the complete subtraction non-negative
            # while allowing every local pair at the target position.
            left[top], right[top] = 9, 0
        else:
            # reachable_contexts already excludes globally negative terminal cases.
            if left_digit < right_digit or (left_digit == right_digit and carry):
                raise ValueError("requested unreachable subtraction context")
            if carry == 0:
                # Preserve non-negativity when terminal digits tie.
                for index in range(position):
                    digit = rng.randrange(10)
                    left[index] = right[index] = digit

    left_value, right_value = digits_to_value(left), digits_to_value(right)
    if operation == "sub" and left_value < right_value:
        raise AssertionError("basis constructor produced negative subtraction")
    state = initial_state(operation, left_value, right_value, width)
    for _ in range(position):
        state = apply_microstep(state)
    if local_context(state) != context:
        raise AssertionError("basis constructor missed requested local context")
    return left_value, right_value


def context_episode(context, variant, rng):
    width, operation, _position, _carry, _left, _right = context
    left, right = operands_for_context(context, rng)
    return episode_from_operands(
        "basis-{}-v{:02d}".format(context_label(context), variant), "train", width, operation, left, right, "core",
    )


def random_heldout_episode(split, width, rng, reserved):
    maximum = 10 ** width - 1
    for attempt in range(1, 100000):
        operation = rng.choice(("add", "sub"))
        left, right = rng.randint(0, maximum), rng.randint(0, maximum)
        if operation == "sub" and left < right:
            left, right = right, left
        episode = episode_from_operands(
            "{}-{:05d}".format(split, attempt), split, width, operation, left, right, "heldout",
        )
        counterfactual = counterfactual_episode(episode)
        signatures = {episode_signature(episode), episode_signature(counterfactual)}
        if signatures & reserved:
            continue
        reserved.update(signatures)
        episode["counterfactual"] = counterfactual
        return episode
    raise RuntimeError("unable to sample heldout basis episode")


def write_jsonl(path, rows):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit("refusing to overwrite existing output: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(partial, path)


def context_counts(rows):
    counts = Counter()
    for row in rows:
        if row.get("kind") != "transition":
            continue
        from digitwise_protocol import parse_state

        context = local_context(parse_state(row["state"]))
        if context is not None:
            counts[context] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--heldout-per-regime", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    if args.variants <= 0 or args.heldout_per_regime <= 0:
        raise SystemExit("variants and heldout-per-regime must be positive")
    destinations = tuple(Path(path) for path in (args.train_out, args.heldout_out, args.report))
    if any(path.exists() for path in destinations):
        raise SystemExit("refusing to overwrite an existing basis candidate")

    rng = random.Random(args.seed)
    required = sorted(reachable_contexts(WIDTHS))
    train_episodes = [context_episode(context, variant, rng) for context in required for variant in range(args.variants)]
    train_rows, duplicate_prompts_dropped = deduplicate_rows(
        [row for episode in train_episodes for row in rows_from_episode(episode)]
    )
    actual_contexts = context_counts(train_rows)
    missing_contexts = [context for context in required if actual_contexts[context] == 0]
    if missing_contexts:
        raise RuntimeError("basis candidate lost required local contexts after deduplication")

    reserved = {episode_signature(episode) for episode in train_episodes}
    heldout_specs = (("recombine_w4", 4), ("recombine_w6", 6), ("width_ood_w8", 8))
    heldout = []
    for split, width in heldout_specs:
        for _ in range(args.heldout_per_regime):
            heldout.append(random_heldout_episode(split, width, rng, reserved))
    train_prompts = {normalized(row["completion_prompt"]) for row in train_rows}
    heldout_prompts = {normalized(prompt) for episode in heldout for prompt in episode_prompts(episode)}
    if train_prompts & heldout_prompts:
        raise RuntimeError("exact train/heldout prompt overlap in basis candidate")
    rng.shuffle(train_rows)
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.heldout_out, heldout)
    report = {
        "schema": "shohin-digitwise-basis-v3",
        "seed": args.seed,
        "variants": args.variants,
        "required_local_contexts": len(required),
        "covered_local_contexts": len(actual_contexts),
        "missing_local_contexts": len(missing_contexts),
        "train_episodes": len(train_episodes),
        "train_rows": len(train_rows),
        "duplicate_train_prompts_dropped": duplicate_prompts_dropped,
        "heldout_episodes": len(heldout),
        "heldout_by_regime": dict(sorted(Counter(episode["split"] for episode in heldout).items())),
        "train_sha256": sha256_file(args.train_out),
        "heldout_sha256": sha256_file(args.heldout_out),
        "claim_boundary": (
            "CPU-only data candidate. Complete local-context coverage is a learnability control, not proof of "
            "model reasoning, language compilation, or context scaling."
        ),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
