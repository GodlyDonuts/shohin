#!/usr/bin/env python3
"""Generate the Dual-Code Reversible Deliberation curriculum.

Each episode has two codebooks for the same decimal machine state.  The model
is supervised on forward A transitions, A-to-B transcodes, reverse B
transitions, B-to-A transcodes, and terminal readout.  Codebook construction
and solver transitions are generation-only; a future runtime controller may
transport only model text and exact-string equality decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from digitwise_protocol import apply_microstep, initial_state, state_answer
from dual_code_reversible_protocol import (
    codebook_prompt, codebook_record, encode_state, forward_prompt, invert_microstep,
    make_codebook, readout_prompt, reverse_prompt, transcode_prompt,
)


WORD = re.compile(r"\w+")


def normalized(text: str) -> str:
    return " ".join(WORD.findall(str(text).lower()))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_operands(rng: random.Random, operation: str, minimum: int, maximum: int) -> tuple[int, int]:
    left, right = rng.randint(minimum, maximum), rng.randint(minimum, maximum)
    if operation == "sub" and left < right:
        left, right = right, left
    return left, right


def episode_from_operands(
    episode_id: str,
    split: str,
    width: int,
    operation: str,
    left: int,
    right: int,
    code_seed: str,
    vocabulary: str,
):
    states = [initial_state(operation, left, right, width)]
    for _ in range(width):
        states.append(apply_microstep(states[-1]))
    a_book = make_codebook(code_seed, "A", vocabulary=vocabulary)
    b_book = make_codebook(code_seed, "B", vocabulary=vocabulary)
    return {
        "id": episode_id,
        "split": split,
        "width": width,
        "operation": operation,
        "left": left,
        "right": right,
        "code_seed": code_seed,
        "prompt_style": "train" if vocabulary == "train" else "heldout",
        "codebooks": {"A": codebook_record(a_book), "B": codebook_record(b_book)},
        "initial_a": encode_state(states[0], a_book),
        "expected_a_states": [encode_state(state, a_book) for state in states[1:]],
        "expected_b_states": [encode_state(state, b_book) for state in states[1:]],
        "expected_answer": state_answer(states[-1]),
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
    counterfactual = episode_from_operands(
        episode_id=str(episode["id"]) + "-cf", split=str(episode["split"]), width=int(episode["width"]),
        operation=str(episode["operation"]), left=left, right=right, code_seed=str(episode["code_seed"]),
        vocabulary=str(episode["codebooks"]["A"]["vocabulary"]),
    )
    if int(counterfactual["expected_answer"]) == int(episode["expected_answer"]):
        raise AssertionError("counterfactual did not change answer")
    return counterfactual


def episode_signature(episode) -> tuple[int, int, int]:
    return int(episode["width"]), int(episode["left"]), int(episode["right"])


def rows_from_episode(episode):
    """Build all supervision tasks without exposing canonical DWS text."""
    from dual_code_reversible_protocol import codebook_from_record, parse_state

    a_book = codebook_from_record(episode["codebooks"]["A"])
    b_book = codebook_from_record(episode["codebooks"]["B"])
    style = str(episode["prompt_style"])
    states_a = [str(episode["initial_a"])] + [str(value) for value in episode["expected_a_states"]]
    states_b = []
    for a_line in states_a:
        state = parse_state(a_line, a_book)
        if state is None:
            raise ValueError("episode A state cannot be decoded")
        states_b.append(encode_state(state, b_book))
    rows = []
    for index in range(int(episode["width"])):
        a_current, a_next = states_a[index], states_a[index + 1]
        b_current, b_next = states_b[index], states_b[index + 1]
        tasks = (
            ("forward_a", forward_prompt(a_book, a_current, style), a_next),
            ("a_to_b", transcode_prompt(a_book, b_book, a_next, style), b_next),
            ("reverse_b", reverse_prompt(b_book, b_next, style), b_current),
            ("b_to_a", transcode_prompt(b_book, a_book, b_current, style), a_current),
        )
        for kind, prompt, response in tasks:
            rows.append({
                "question": prompt,
                "completion_prompt": prompt,
                "response": response,
                "source": "dual_code_reversible_v1",
                "training_group": "dual_code_reversible",
                "kind": kind,
                "episode_id": episode["id"],
                "split": episode["split"],
                "width": episode["width"],
                "operation": episode["operation"],
                "transition_index": index,
                "code_seed": episode["code_seed"],
                "codebook_vocabulary": a_book.vocabulary,
                "prompt_style": style,
            })
    final_prompt = readout_prompt(a_book, states_a[-1], style)
    rows.append({
        "question": final_prompt,
        "completion_prompt": final_prompt,
        "response": "answer={}".format(int(episode["expected_answer"])),
        "source": "dual_code_reversible_readout_v1",
        "training_group": "dual_code_reversible",
        "kind": "readout",
        "episode_id": episode["id"],
        "split": episode["split"],
        "width": episode["width"],
        "operation": episode["operation"],
        "transition_index": episode["width"],
        "code_seed": episode["code_seed"],
        "codebook_vocabulary": a_book.vocabulary,
        "prompt_style": style,
    })
    return rows


def controller_prompts(episode):
    """Enumerate solver-known prompts for split-overlap auditing only."""
    return [row["completion_prompt"] for row in rows_from_episode(episode)]


def deduplicate_rows(rows):
    kept, seen, dropped = [], set(), 0
    for row in rows:
        key = normalized(row["completion_prompt"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept, dropped


def write_jsonl(path: str | Path, rows) -> None:
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
    destinations = tuple(Path(path) for path in (args.train_out, args.heldout_out, args.report))
    if any(path.exists() for path in destinations):
        raise SystemExit("refusing to overwrite an existing artifact")

    rng = random.Random(args.seed)
    train_episodes = []
    for index in range(args.train_episodes):
        width = 4 if index % 2 == 0 else 6
        maximum = 2999 if width == 4 else 299999
        operation = rng.choice(("add", "sub"))
        left, right = make_operands(rng, operation, 0, maximum)
        train_episodes.append(episode_from_operands(
            "train-{:06d}".format(index), "train", width, operation, left, right,
            "train-code-{:06d}".format(index), "train",
        ))
    train_rows, duplicate_dropped = deduplicate_rows([row for episode in train_episodes for row in rows_from_episode(episode)])
    if not train_rows:
        raise RuntimeError("empty dual-code training rows")
    if any(row["question"] != row["completion_prompt"] or not row["response"] for row in train_rows):
        raise RuntimeError("malformed dual-code training row")

    reserved = {episode_signature(episode) for episode in train_episodes}
    specs = (
        ("fit_w4", 4, 0, 2999), ("fit_w6", 6, 0, 299999),
        ("value_ood_w4", 4, 7000, 9999), ("value_ood_w6", 6, 700000, 999999),
        ("width_ood_w8", 8, 80000000, 99999999),
    )
    heldout = []
    for split, width, minimum, maximum in specs:
        attempts = 0
        while sum(item["split"] == split for item in heldout) < args.heldout_per_regime:
            attempts += 1
            if attempts > args.heldout_per_regime * 300:
                raise RuntimeError("unable to build disjoint heldout regime: {}".format(split))
            index = sum(item["split"] == split for item in heldout)
            operation = rng.choice(("add", "sub"))
            left, right = make_operands(rng, operation, minimum, maximum)
            episode = episode_from_operands(
                "{}-{:05d}".format(split, index), split, width, operation, left, right,
                "heldout-code-{}-{:05d}".format(split, index), "heldout",
            )
            episode["counterfactual"] = counterfactual_episode(episode)
            signatures = {episode_signature(episode), episode_signature(episode["counterfactual"])}
            if signatures & reserved:
                continue
            reserved.update(signatures)
            heldout.append(episode)

    train_prompts = {normalized(row["completion_prompt"]) for row in train_rows}
    heldout_prompts = {normalized(prompt) for episode in heldout for prompt in controller_prompts(episode)}
    if train_prompts & heldout_prompts:
        raise RuntimeError("exact train/heldout prompt overlap")
    rng.shuffle(train_rows)
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.heldout_out, heldout)
    report = {
        "schema": "shohin-dual-code-reversible-v1",
        "seed": args.seed,
        "train_episodes": len(train_episodes),
        "train_rows": len(train_rows),
        "duplicate_train_prompts_dropped": duplicate_dropped,
        "heldout_episodes": len(heldout),
        "heldout_counterfactual_pairs": len(heldout),
        "heldout_by_regime": dict(sorted(Counter(item["split"] for item in heldout).items())),
        "train_vocabularies": ["train"],
        "heldout_vocabularies": ["heldout"],
        "train_sha256": sha256_file(args.train_out),
        "heldout_sha256": sha256_file(args.heldout_out),
        "protocol": (
            "model learns forward A, A-to-B, reverse B, B-to-A, and terminal readout; "
            "runtime must never invoke the solver and may only transport text and compare exact recovered state"
        ),
        "claim_boundary": (
            "This is an untrained isolated curriculum. Passing future codebook-OOD state closure can establish "
            "narrow reversible local execution only, not broad reasoning or context scaling."
        ),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
