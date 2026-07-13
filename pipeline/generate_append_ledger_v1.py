#!/usr/bin/env python3
"""Generate a solver-verified append-only delta-ledger curriculum.

The model writes short local deltas, then compacts only its own emitted deltas
into fixed-size blocks.  The first version tests a bounded first-level ledger;
recursive block-of-block compression is intentionally gated on this result.
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
from append_ledger_protocol import (canonical_base, canonical_block, canonical_delta, compact_prompt,
                                    expected_answer, expected_block, expected_delta, final_prompt,
                                    initial_base, transition_prompt)


WORD = re.compile(r"\w+")


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def operands(rng, operation, minimum, maximum):
    left, right = rng.randint(minimum, maximum), rng.randint(minimum, maximum)
    if operation == "sub" and left < right:
        left, right = right, left
    return left, right


def episode_from_operands(identifier, split, width, operation, left, right, prompt_style, block_size):
    base = initial_base(operation, left, right, width)
    carry, live, deltas, blocks = 0, [], [], []
    for step in range(width):
        delta = expected_delta(base, step, carry)
        carry = delta["c"]
        line = canonical_delta(delta)
        live.append(line)
        deltas.append(line)
        if len(live) == block_size or step + 1 == width:
            blocks.append(canonical_block(expected_block(len(blocks), live)))
            live = []
    return {
        "id": identifier, "split": split, "prompt_style": prompt_style, "block_size": int(block_size),
        "operation": operation, "width": int(width), "left": int(left), "right": int(right),
        "base": canonical_base(base), "expected_deltas": deltas, "expected_blocks": blocks,
        "expected_answer": expected_answer(base),
    }


def counterfactual(episode):
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
    result = episode_from_operands(episode["id"] + "-cf", episode["split"], episode["width"], episode["operation"],
                                   left, right, episode["prompt_style"], episode["block_size"])
    if result["expected_answer"] == episode["expected_answer"]:
        raise AssertionError("counterfactual answer did not change")
    return result


def signature(episode):
    # Operands remain reserved across operations because an op token can fall
    # outside a short n-gram window in the serialized base record.
    return int(episode["width"]), int(episode["left"]), int(episode["right"])


def prompts_for_episode(episode):
    base = initial_base(episode["operation"], int(episode["left"]), int(episode["right"]), int(episode["width"]))
    blocks, live, prompts, block_index = [], [], [], 0
    for step, line in enumerate(episode["expected_deltas"]):
        prompts.append(transition_prompt(base, blocks, live, step, style=episode["prompt_style"]))
        live.append(line)
        if len(live) == episode["block_size"] or step + 1 == len(episode["expected_deltas"]):
            prompts.append(compact_prompt(base, blocks, live, block_index, style=episode["prompt_style"]))
            blocks.append(episode["expected_blocks"][block_index])
            live, block_index = [], block_index + 1
    prompts.append(final_prompt(base, blocks, style=episode["prompt_style"]))
    return prompts


def rows_from_episode(episode):
    base = initial_base(episode["operation"], int(episode["left"]), int(episode["right"]), int(episode["width"]))
    blocks, live, rows, block_index = [], [], [], 0
    common = {key: episode[key] for key in ("id", "split", "prompt_style", "block_size", "operation", "width", "left", "right", "base", "expected_answer")}
    for step, line in enumerate(episode["expected_deltas"]):
        prompt = transition_prompt(base, blocks, live, step, style=episode["prompt_style"])
        rows.append({"question": prompt, "completion_prompt": prompt, "response": line,
                     "source": "append_ledger_delta_v1", "training_group": "append_ledger",
                     "kind": "delta", "step": step, "blocks": list(blocks), "live": list(live), **common})
        live.append(line)
        if len(live) == episode["block_size"] or step + 1 == len(episode["expected_deltas"]):
            prompt = compact_prompt(base, blocks, live, block_index, style=episode["prompt_style"])
            block = episode["expected_blocks"][block_index]
            rows.append({"question": prompt, "completion_prompt": prompt, "response": block,
                         "source": "append_ledger_compaction_v1", "training_group": "append_ledger",
                         "kind": "block", "block": block_index, "blocks": list(blocks), "live": list(live), **common})
            blocks.append(block)
            live, block_index = [], block_index + 1
    prompt = final_prompt(base, blocks, style=episode["prompt_style"])
    rows.append({"question": prompt, "completion_prompt": prompt, "response": "answer={}".format(episode["expected_answer"]),
                 "source": "append_ledger_final_v1", "training_group": "append_ledger", "kind": "final",
                 "blocks": list(blocks), "live": [], **common})
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit("refusing existing output: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as target:
        for row in rows:
            target.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-episodes", type=int, default=24000)
    parser.add_argument("--heldout-per-regime", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    if args.train_episodes <= 0 or args.heldout_per_regime <= 0 or args.block_size <= 1:
        raise SystemExit("counts must be positive and block-size must exceed one")
    if any(Path(path).exists() for path in (args.train_out, args.heldout_out, args.report)):
        raise SystemExit("refusing existing output")
    rng = random.Random(args.seed)
    train = []
    for index in range(args.train_episodes):
        width = 8 if index % 2 == 0 else 16
        maximum = 4999999 if width == 8 else 499999999999999
        operation = rng.choice(("add", "sub"))
        left, right = operands(rng, operation, 0, maximum)
        train.append(episode_from_operands("train-{:06d}".format(index), "train", width, operation, left, right, "core", args.block_size))
    rows = [row for episode in train for row in rows_from_episode(episode)]
    unique, seen, dropped = [], set(), 0
    for row in rows:
        key = normalized(row["question"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        unique.append(row)
    reserved = {signature(episode) for episode in train}
    specs = (("fit_w8", 8, 0, 4999999), ("fit_w16", 16, 0, 499999999999999),
             ("value_ood_w8", 8, 70000000, 99999999), ("value_ood_w16", 16, 7000000000000000, 9999999999999999),
             ("width_ood_w32", 32, 10 ** 30, 10 ** 32 - 1))
    heldout = []
    for split, width, minimum, maximum in specs:
        attempts = 0
        while sum(item["split"] == split for item in heldout) < args.heldout_per_regime:
            attempts += 1
            if attempts > args.heldout_per_regime * 300:
                raise RuntimeError("could not build disjoint heldout regime")
            operation = rng.choice(("add", "sub"))
            left, right = operands(rng, operation, minimum, maximum)
            episode = episode_from_operands("{}-{:05d}".format(split, sum(item["split"] == split for item in heldout)),
                                            split, width, operation, left, right, "heldout", args.block_size)
            episode["counterfactual"] = counterfactual(episode)
            signatures = {signature(episode), signature(episode["counterfactual"])}
            if signatures & reserved:
                continue
            reserved.update(signatures)
            heldout.append(episode)
    train_prompts = {normalized(row["question"]) for row in unique}
    heldout_prompts = set()
    for episode in heldout:
        heldout_prompts.update(normalized(prompt) for prompt in prompts_for_episode(episode))
        heldout_prompts.update(normalized(prompt) for prompt in prompts_for_episode(episode["counterfactual"]))
    if train_prompts & heldout_prompts:
        raise RuntimeError("exact prompt overlap")
    rng.shuffle(unique)
    write_jsonl(args.train_out, unique)
    write_jsonl(args.heldout_out, heldout)
    report = {"schema": "append-ledger-v1", "seed": args.seed, "block_size": args.block_size,
              "train_episodes": len(train), "train_rows": len(unique), "duplicate_train_prompts_dropped": dropped,
              "heldout_episodes": len(heldout), "counterfactual_pairs": len(heldout),
              "heldout_by_regime": dict(sorted(Counter(item["split"] for item in heldout).items())),
              "train_sha256": sha256_file(args.train_out), "heldout_sha256": sha256_file(args.heldout_out),
              "claim_boundary": "This is model-authored local execution and first-level ledger compaction only; it is not language reasoning or general long-context capability."}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
