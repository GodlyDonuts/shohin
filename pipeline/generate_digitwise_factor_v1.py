#!/usr/bin/env python3
"""Generate a coverage-complete static-tape / recurrent-register DRS candidate.

The model receives immutable operand evidence on every turn but only emits the
small register that changes.  Complete solver-derived episodes cover every
reachable local decimal context across width 4/6.  This creates only a CPU
candidate; it neither submits SFT nor touches pretraining.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from digitwise_basis_protocol import WIDTHS, context_label, reachable_contexts
from digitwise_factor_protocol import (
    apply_microstep,
    canonical_register,
    canonical_tape,
    digit_prompt,
    final_prompt,
    initial_register,
    initial_tape,
    local_context,
    microstep_prompt,
    parse_register,
    parse_tape,
    register_answer,
    register_digit,
)


WORD = re.compile(r"\w+")


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digits_to_value(digits):
    return sum(int(digit) * (10 ** index) for index, digit in enumerate(digits))


def random_digits(rng, width):
    return [rng.randrange(10) for _ in range(width)]


def operands_for_context(context, rng):
    """Build a complete nonnegative arithmetic tape containing ``context``."""
    width, operation, position, carry, left_digit, right_digit = context
    top = width - 1
    left, right = random_digits(rng, width), random_digits(rng, width)
    left[position], right[position] = left_digit, right_digit
    if position:
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
            for index in range(lower + 1):
                digit = rng.randrange(10)
                left[index] = right[index] = digit
    if operation == "sub":
        if position < top:
            left[top], right[top] = 9, 0
        else:
            if left_digit < right_digit or (left_digit == right_digit and carry):
                raise ValueError("requested unreachable terminal subtraction context")
            if carry == 0:
                for index in range(position):
                    digit = rng.randrange(10)
                    left[index] = right[index] = digit
    left_value, right_value = digits_to_value(left), digits_to_value(right)
    if operation == "sub" and left_value < right_value:
        raise AssertionError("factor constructor produced negative subtraction")
    tape = initial_tape(operation, left_value, right_value, width)
    register = initial_register(tape)
    for _ in range(position):
        register = apply_microstep(tape, register)
    if local_context(tape, register) != context:
        raise AssertionError("factor constructor missed requested local context")
    return left_value, right_value


def episode_from_operands(episode_id, split, width, operation, left, right, prompt_style):
    tape = initial_tape(operation, left, right, width)
    register = initial_register(tape)
    expected = []
    for _ in range(width):
        register = apply_microstep(tape, register)
        expected.append(canonical_register(tape, register))
    return {
        "id": episode_id,
        "split": split,
        "prompt_style": prompt_style,
        "operation": operation,
        "width": width,
        "left": left,
        "right": right,
        "tape": canonical_tape(tape),
        "initial_register": canonical_register(tape, initial_register(tape)),
        "expected_registers": expected,
        "expected_answer": register_answer(tape, register),
    }


def counterfactual_episode(episode):
    maximum = 10 ** int(episode["width"]) - 1
    left, right = int(episode["left"]), int(episode["right"])
    if episode["operation"] == "add":
        left = left + 1 if left < maximum else left - 1
    elif left < maximum:
        left += 1
    elif right > 0:
        right -= 1
    else:
        left -= 1
    if episode["operation"] == "sub" and left < right:
        raise AssertionError("counterfactual violated subtraction invariant")
    result = episode_from_operands(
        episode["id"] + "-cf", episode["split"], episode["width"], episode["operation"],
        left, right, episode["prompt_style"],
    )
    if result["expected_answer"] == episode["expected_answer"]:
        raise AssertionError("counterfactual did not change answer")
    return result


def episode_signature(episode):
    return int(episode["width"]), int(episode["left"]), int(episode["right"])


def episode_prompts(episode):
    tape = initial_tape(episode["operation"], int(episode["left"]), int(episode["right"]), int(episode["width"]))
    register = parse_register(episode["initial_register"], tape)
    if register is None:
        raise ValueError("invalid initial factor register")
    prompts = []
    for line in episode["expected_registers"]:
        prompts.append(microstep_prompt(tape, register, style=episode["prompt_style"]))
        register = parse_register(line, tape)
        if register is None:
            raise ValueError("invalid expected factor register")
    prompts.append(final_prompt(tape, register, style=episode["prompt_style"]))
    if "counterfactual" in episode:
        prompts.extend(episode_prompts(episode["counterfactual"]))
    return prompts


def rows_from_episode(episode):
    tape = initial_tape(episode["operation"], int(episode["left"]), int(episode["right"]), int(episode["width"]))
    register = parse_register(episode["initial_register"], tape)
    if register is None:
        raise ValueError("invalid initial factor register")
    rows = []
    for index, expected_line in enumerate(episode["expected_registers"]):
        expected = parse_register(expected_line, tape)
        if expected is None:
            raise ValueError("invalid expected factor register")
        prompt = microstep_prompt(tape, register, style=episode["prompt_style"])
        rows.append({
            "question": prompt,
            "completion_prompt": prompt,
            "response": expected_line,
            "source": "digitwise_factor_transition_v1",
            "training_group": "digitwise_factor",
            "kind": "transition",
            "episode_id": episode["id"],
            "width": episode["width"],
            "operation": episode["operation"],
            "transition_index": index,
            "tape": canonical_tape(tape),
            "register": canonical_register(tape, register),
            "expected_register": expected_line,
            "prompt_style": episode["prompt_style"],
        })
        register = expected
        digit = register_digit(tape, register, index)
        prompt = digit_prompt(tape, register, index, style=episode["prompt_style"])
        rows.append({
            "question": prompt,
            "completion_prompt": prompt,
            "response": "digit={}".format(digit),
            "source": "digitwise_factor_readout_v1",
            "training_group": "digitwise_factor",
            "kind": "digit",
            "episode_id": episode["id"],
            "width": episode["width"],
            "operation": episode["operation"],
            "transition_index": index,
            "digit_index": index,
            "tape": canonical_tape(tape),
            "register": canonical_register(tape, register),
            "expected_digit": digit,
            "prompt_style": episode["prompt_style"],
        })
    prompt = final_prompt(tape, register, style=episode["prompt_style"])
    rows.append({
        "question": prompt,
        "completion_prompt": prompt,
        "response": "answer={}".format(register_answer(tape, register)),
        "source": "digitwise_factor_final_v1",
        "training_group": "digitwise_factor",
        "kind": "final",
        "episode_id": episode["id"],
        "width": episode["width"],
        "operation": episode["operation"],
        "transition_index": episode["width"],
        "tape": canonical_tape(tape),
        "register": canonical_register(tape, register),
        "expected_answer": register_answer(tape, register),
        "prompt_style": episode["prompt_style"],
    })
    return rows


def deduplicate_rows(rows):
    unique, seen, dropped = [], set(), 0
    for row in rows:
        key = normalized(row["completion_prompt"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, dropped


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--variants", type=int, default=8)
    parser.add_argument("--heldout-per-regime", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    if args.variants <= 0 or args.heldout_per_regime <= 0:
        raise SystemExit("variants and heldout-per-regime must be positive")
    destinations = tuple(Path(path) for path in (args.train_out, args.heldout_out, args.report))
    if any(path.exists() for path in destinations):
        raise SystemExit("refusing to overwrite an existing factor candidate")

    rng = random.Random(args.seed)
    required = sorted(reachable_contexts(WIDTHS))
    train_episodes = []
    for context in required:
        for variant in range(args.variants):
            left, right = operands_for_context(context, rng)
            width, operation = context[:2]
            train_episodes.append(episode_from_operands(
                "factor-{}-v{:02d}".format(context_label(context), variant), "train", width, operation,
                left, right, "core",
            ))
    train_rows, duplicate_prompts_dropped = deduplicate_rows(
        [row for episode in train_episodes for row in rows_from_episode(episode)]
    )
    observed = Counter()
    for row in train_rows:
        if row["kind"] != "transition":
            continue
        tape = parse_tape(row["tape"])
        register = parse_register(row["register"], tape) if tape is not None else None
        context = local_context(tape, register) if register is not None else None
        if context is not None:
            observed[context] += 1
    missing = [context for context in required if observed[context] == 0]
    if missing:
        raise RuntimeError("factor candidate lost required contexts after deduplication")

    reserved = {episode_signature(episode) for episode in train_episodes}
    heldout_specs = (("recombine_w4", 4), ("recombine_w6", 6), ("width_ood_w8", 8))
    heldout = []
    for split, width in heldout_specs:
        maximum = 10 ** width - 1
        for index in range(args.heldout_per_regime):
            for _attempt in range(100000):
                operation = rng.choice(("add", "sub"))
                left, right = rng.randint(0, maximum), rng.randint(0, maximum)
                if operation == "sub" and left < right:
                    left, right = right, left
                episode = episode_from_operands(
                    "{}-{:05d}".format(split, index), split, width, operation, left, right, "heldout",
                )
                counterfactual = counterfactual_episode(episode)
                signatures = {episode_signature(episode), episode_signature(counterfactual)}
                if signatures & reserved:
                    continue
                reserved.update(signatures)
                episode["counterfactual"] = counterfactual
                heldout.append(episode)
                break
            else:
                raise RuntimeError("unable to sample factor heldout episode")
    train_prompts = {normalized(row["completion_prompt"]) for row in train_rows}
    heldout_prompts = {normalized(prompt) for episode in heldout for prompt in episode_prompts(episode)}
    if train_prompts & heldout_prompts:
        raise RuntimeError("exact train/heldout prompt overlap in factor candidate")
    rng.shuffle(train_rows)
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.heldout_out, heldout)
    report = {
        "schema": "shohin-digitwise-factor-v1",
        "seed": args.seed,
        "variants": args.variants,
        "required_local_contexts": len(required),
        "covered_local_contexts": len(observed),
        "missing_local_contexts": len(missing),
        "train_episodes": len(train_episodes),
        "train_rows": len(train_rows),
        "duplicate_train_prompts_dropped": duplicate_prompts_dropped,
        "heldout_episodes": len(heldout),
        "heldout_by_regime": dict(sorted(Counter(episode["split"] for episode in heldout).items())),
        "train_sha256": sha256_file(args.train_out),
        "heldout_sha256": sha256_file(args.heldout_out),
        "claim_boundary": (
            "CPU-only factorized-register data candidate. Removing repeated immutable-tape rewriting is a "
            "representation control, not evidence of reasoning, language compilation, or context scaling."
        ),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
