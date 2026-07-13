#!/usr/bin/env python3
"""Generate a solver-verified digitwise recurrent-scratchpad curriculum.

This first screen intentionally contains no natural-language compiler.  The
model receives a canonical machine state and must author one local decimal
transition at a time.  Held-out episodes reserve larger values, wider tapes,
and a distinct lexical wrapper.  A controller later forwards only the model's
emitted state; it never executes or repairs the arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from digitwise_protocol import (apply_microstep, canonical_state, digit_prompt, final_prompt,
                                initial_state, microstep_prompt, state_answer, state_digit)


WORD = re.compile(r"\w+")


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_operands(rng, operation, minimum, maximum):
    if minimum < 0 or maximum < minimum:
        raise ValueError("invalid operand band")
    left, right = rng.randint(minimum, maximum), rng.randint(minimum, maximum)
    if operation == "sub" and left < right:
        left, right = right, left
    return left, right


def episode_from_operands(episode_id, split, width, operation, left, right, prompt_style):
    state = initial_state(operation, left, right, width)
    expected_states = []
    for _ in range(width):
        state = apply_microstep(state)
        expected_states.append(canonical_state(state))
    return {
        "id": episode_id,
        "split": split,
        "prompt_style": prompt_style,
        "operation": operation,
        "width": width,
        "left": left,
        "right": right,
        "initial_state": canonical_state(initial_state(operation, left, right, width)),
        "expected_states": expected_states,
        "expected_answer": state_answer(state),
    }


def make_episode(rng, episode_id, split, width, minimum, maximum, prompt_style):
    operation = rng.choice(("add", "sub"))
    left, right = make_operands(rng, operation, minimum, maximum)
    return episode_from_operands(episode_id, split, width, operation, left, right, prompt_style)


def counterfactual_episode(episode):
    """Change one operand while keeping opcode, width, and prompt form fixed."""
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
    counterfactual = episode_from_operands(
        episode["id"] + "-cf", episode["split"], episode["width"], episode["operation"],
        left, right, episode["prompt_style"],
    )
    if counterfactual["expected_answer"] == episode["expected_answer"]:
        raise AssertionError("counterfactual did not change answer")
    return counterfactual


def episode_signature(episode):
    # The initial state fully determines every state prompt in this deterministic
    # protocol, so reusing it across splits would invalidate the heldout surface.
    return episode["initial_state"]


def _episode_prompts(episode):
    """Return every prompt the recurrent evaluator may send for an episode."""
    from digitwise_protocol import parse_state

    style = episode["prompt_style"]
    state = parse_state(episode["initial_state"])
    if state is None:
        raise ValueError("invalid initial state")
    prompts = []
    for expected_line in episode["expected_states"]:
        prompts.append(microstep_prompt(state, style=style))
        state = parse_state(expected_line)
        if state is None:
            raise ValueError("invalid expected state")
    prompts.append(final_prompt(state, style=style))
    return prompts


def episode_prompts(episode):
    prompts = _episode_prompts(episode)
    if "counterfactual" in episode:
        prompts.extend(_episode_prompts(episode["counterfactual"]))
    return prompts


def rows_from_episode(episode):
    from digitwise_protocol import parse_state

    style = episode["prompt_style"]
    state = parse_state(episode["initial_state"])
    rows = []
    for index, expected_line in enumerate(episode["expected_states"]):
        expected = parse_state(expected_line)
        prompt = microstep_prompt(state, style=style)
        rows.append({
            "question": prompt,
            "completion_prompt": prompt,
            "response": expected_line,
            "source": "digitwise_recurrent_transition_v1",
            "training_group": "digitwise_recurrent",
            "kind": "transition",
            "episode_id": episode["id"],
            "width": episode["width"],
            "operation": episode["operation"],
            "transition_index": index,
            "state": canonical_state(state),
            "expected_state": expected_line,
            "prompt_style": style,
        })
        state = expected
        digit = state_digit(state, index)
        prompt = digit_prompt(state, index, style=style)
        rows.append({
            "question": prompt,
            "completion_prompt": prompt,
            "response": "digit={}".format(digit),
            "source": "digitwise_recurrent_readout_v1",
            "training_group": "digitwise_recurrent",
            "kind": "digit",
            "episode_id": episode["id"],
            "width": episode["width"],
            "operation": episode["operation"],
            "transition_index": index,
            "digit_index": index,
            "state": canonical_state(state),
            "expected_digit": digit,
            "prompt_style": style,
        })
    prompt = final_prompt(state, style=style)
    rows.append({
        "question": prompt,
        "completion_prompt": prompt,
        "response": "answer={}".format(state_answer(state)),
        "source": "digitwise_recurrent_final_v1",
        "training_group": "digitwise_recurrent",
        "kind": "final",
        "episode_id": episode["id"],
        "width": episode["width"],
        "operation": episode["operation"],
        "transition_index": episode["width"],
        "state": canonical_state(state),
        "expected_answer": state_answer(state),
        "prompt_style": style,
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
    parser.add_argument("--train-episodes", type=int, default=40000)
    parser.add_argument("--heldout-per-regime", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    if args.train_episodes <= 0 or args.heldout_per_regime <= 0:
        raise SystemExit("episode counts must be positive")
    if any(Path(path).exists() for path in (args.train_out, args.heldout_out, args.report)):
        raise SystemExit("refusing to overwrite an existing artifact")

    rng = random.Random(args.seed)
    train_episodes = []
    for index in range(args.train_episodes):
        width = 4 if index % 2 == 0 else 6
        maximum = 2999 if width == 4 else 299999
        train_episodes.append(make_episode(
            rng, "train-{:06d}".format(index), "train", width, 0, maximum, "core"
        ))
    train_rows_all = [row for episode in train_episodes for row in rows_from_episode(episode)]
    train_rows, duplicate_prompts_dropped = deduplicate_rows(train_rows_all)
    if not train_rows or any(row["question"] != row["completion_prompt"] or not row["response"] for row in train_rows):
        raise RuntimeError("malformed digitwise training row")

    train_signatures = {episode_signature(episode) for episode in train_episodes}
    heldout_specs = (
        ("fit_w4", 4, 0, 2999),
        ("fit_w6", 6, 0, 299999),
        ("value_ood_w4", 4, 7000, 9999),
        ("value_ood_w6", 6, 700000, 999999),
        ("width_ood_w8", 8, 80000000, 99999999),
    )
    heldout = []
    for split, width, minimum, maximum in heldout_specs:
        attempts = 0
        while sum(row["split"] == split for row in heldout) < args.heldout_per_regime:
            attempts += 1
            if attempts > args.heldout_per_regime * 200:
                raise RuntimeError("could not build disjoint heldout regime: {}".format(split))
            index = sum(row["split"] == split for row in heldout)
            episode = make_episode(
                rng, "{}-{:05d}".format(split, index), split, width, minimum, maximum, "heldout"
            )
            if episode_signature(episode) in train_signatures:
                continue
            episode["counterfactual"] = counterfactual_episode(episode)
            heldout.append(episode)
    train_prompts = {normalized(row["completion_prompt"]) for row in train_rows}
    heldout_prompts = set().union(*(set(map(normalized, episode_prompts(episode))) for episode in heldout))
    if train_prompts & heldout_prompts:
        raise RuntimeError("exact train/heldout controller prompt overlap")

    rng.shuffle(train_rows)
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.heldout_out, heldout)
    report = {
        "schema": "shohin-digitwise-recurrent-v1",
        "seed": args.seed,
        "train_episodes": len(train_episodes),
        "train_rows": len(train_rows),
        "duplicate_train_prompts_dropped": duplicate_prompts_dropped,
        "heldout_episodes": len(heldout),
        "heldout_counterfactual_pairs": len(heldout),
        "heldout_by_regime": dict(sorted(Counter(row["split"] for row in heldout).items())),
        "train_widths": [4, 6],
        "heldout_widths": [4, 6, 8],
        "train_prompt_style": "core",
        "heldout_prompt_style": "heldout",
        "train_sha256": sha256_file(args.train_out),
        "heldout_sha256": sha256_file(args.heldout_out),
        "protocol": (
            "model emits one discrete state per digit; controller forwards only that emitted state; "
            "the first screen begins from a canonical state and has no language-to-program compiler"
        ),
        "claim_boundary": (
            "Passing can establish narrow local algorithmic execution and bounded recurrent state transport only; "
            "it is not a broad-reasoning or long-context claim."
        ),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
